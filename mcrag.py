import os
import re
import glob
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
import concurrent.futures
from tqdm import tqdm
import torch
from dataclasses import dataclass, field
from datetime import datetime
import gc
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCRAG")

# Configuration class with updated embedding configuration
@dataclass
class MCRAGConfig:
    data_dir: str
    output_dir: str
    cache_dir: str = "./cache"
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    generator_model: str = "stanford-crfm/BioMedLM"
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_cache: bool = True
    batch_size: int = 32
    max_workers: int = 4
    device: Optional[str] = None
    quantization: bool = True
    retrieval_k: int = 5
    retrieval_mode: str = "hybrid"
    mrsty_path: Optional[str] = None 
    use_umls_linking: bool = True  # New parameter to control UMLS linking
    umls_confidence_threshold: float = 0.7  # Threshold for UMLS entity confidence
    
    def __post_init__(self):
        # Determine optimal device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

# Lazy imports for heavy dependencies to reduce startup time and memory usage
def import_text_splitter():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def import_document_loaders():
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
    return PyMuPDFLoader, TextLoader

def import_langchain_vectorstores():
    from langchain_community.vectorstores import FAISS as LangchainFAISS
    from langchain.schema import Document as LangchainDocument
    from langchain_huggingface import HuggingFaceEmbeddings
    return LangchainFAISS, LangchainDocument, HuggingFaceEmbeddings

def import_transformers():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoTokenizer, AutoModelForCausalLM

def import_datasets():
    from datasets import load_dataset
    return load_dataset

# New import function for scispaCy components
def import_scispacy():
    from scispacy.umls_linking import UmlsEntityLinker
    from scispacy.abbreviation import AbbreviationDetector
    return UmlsEntityLinker, AbbreviationDetector


# Optimized document processor fully integrated with Langchain
class OptimizedDocumentProcessor:
    def __init__(self, config: MCRAGConfig):
        self.config = config
        RecursiveCharacterTextSplitter = import_text_splitter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Use only Langchain embeddings
        self._langchain_embeddings = None
        
        # New attributes for UMLS and scispaCy components
        self._spacy_nlp = None
        self._umls_linker = None

        self.semantic_type_cache = {}
        
        # Pre-compile regex patterns for faster metadata extraction
        self._compile_regex_patterns()
        
        # Initialize UMLS dataset loader (lazy loaded)
        self._umls_dataset = None
        self._umls_mapping = None
    
    @property
    def langchain_embeddings(self):
        """Lazy load Langchain's embeddings"""
        if self._langchain_embeddings is None:
            _, _, HuggingFaceEmbeddings = import_langchain_vectorstores()
            self._langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.config.device}
            )
        return self._langchain_embeddings
    
    def _load_spacy_model(self):
        """Lazy-load the spaCy model"""
        if self._spacy_nlp is None:
            import spacy
            try:
                print("Loading en_core_sci_lg model...")
                self._spacy_nlp = spacy.load("en_core_sci_lg")
                print("Successfully loaded en_core_sci_lg model")
                
                # Add pipeline components using the updated spaCy API
                if self.config.use_umls_linking:
                    self._setup_scispacy_components()
                
                return True
            except OSError:
                print("Model not found, attempting to download...")
                try:
                    spacy.cli.download("en_core_sci_sm")
                    self._spacy_nlp = spacy.load("en_core_sci_sm")
                    print("Successfully downloaded and loaded en_core_sci_sm model")
                    
                    # Add pipeline components using the updated spaCy API
                    if self.config.use_umls_linking:
                        self._setup_scispacy_components()
                    
                    return True
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    self._spacy_nlp = None
                    return False
        return True
    
    def _setup_scispacy_components(self):
        """Set up scispaCy components with proper factory registration"""
        try:
            # Import the components
            from scispacy.abbreviation import AbbreviationDetector
            from scispacy.linking import EntityLinker   
            import spacy

            self._spacy_nlp.add_pipe("abbreviation_detector")

            self._spacy_nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            self._umls_linker = self._spacy_nlp.get_pipe("scispacy_linker")
            
            return True
        except Exception as e:
            print(f"Error setting up scispaCy components: {e}")
            return False
    
    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for faster metadata extraction"""
        # Common conditions and medications
        self.condition_patterns = [
            "diabetes", "hypertension", "asthma", "depression", "cancer", "heart failure", 
            "COPD", "arthritis", "dementia", "stroke", "myocardial infarction", "pneumonia", 
            "chronic kidney disease", "obesity", "anxiety", "osteoporosis"
        ]
        
        self.medication_patterns = [
            "aspirin", "metformin", "insulin", "statin", "lisinopril", "atorvastatin", 
            "simvastatin", "amlodipine", "metoprolol", "losartan", "omeprazole", 
            "albuterol", "fluticasone", "warfarin", "clopidogrel", "furosemide"
        ]
        
        # Recommendation patterns
        self.recommendation_patterns = [
            r"recommend(?:ed|s|ing)?",
            r"should",
            r"advis(?:e|ed|ing)",
            r"indicat(?:e|ed|ing)",
            r"suggest(?:ed|s|ing)?",
            r"must",
            r"require[sd]?"
        ]
        
        # Compile patterns for faster matching
        self.condition_regex = re.compile(r'\b(' + '|'.join(self.condition_patterns) + r')\b', re.IGNORECASE)
        self.medication_regex = re.compile(r'\b(' + '|'.join(self.medication_patterns) + r')\b', re.IGNORECASE)
        self.recommendation_regex = re.compile('|'.join(self.recommendation_patterns), re.IGNORECASE)
    
    def process_documents(self, file_paths, metadata=None, use_parallel=True):
        """Process multiple documents in parallel for better performance"""
        if use_parallel and self.config.max_workers > 1:
            # Process in parallel using ThreadPoolExecutor
            chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_document, file_path, metadata): file_path 
                    for file_path in file_paths
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                                  total=len(file_paths), 
                                  desc="Processing documents"):
                    file_path = future_to_file[future]
                    try:
                        file_chunks = future.result()
                        chunks.extend(file_chunks)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
            return chunks
        else:
            # Process sequentially
            chunks = []
            for file_path in tqdm(file_paths, desc="Processing documents"):
                try:
                    file_chunks = self.process_document(file_path, metadata)
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            return chunks
    
    def process_document(self, file_path, metadata=None):
        """Process a single document with improved error handling"""
        PyMuPDFLoader, TextLoader = import_document_loaders()
        
        if file_path.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata
        file_metadata = {"source": os.path.basename(file_path)}
        if metadata:
            file_metadata.update(metadata)
        
        for chunk in chunks:
            chunk.metadata.update(file_metadata)
        
        # Enrich metadata efficiently in batches
        self._enrich_metadata_batch(chunks)
        
        return chunks
    
    def _enrich_metadata_batch(self, chunks, batch_size=None):
        """Process metadata in batches for better memory efficiency"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # First, initialize the spaCy model and UMLS components if UMLS linking is enabled
        if self.config.use_umls_linking and self._spacy_nlp is None:
            self._load_spacy_model()
        
        # Try to use advanced NLP with direct UMLS linking
        success = self._enrich_with_advanced_nlp(chunks, batch_size)
        
        # Fall back to simple pattern matching if advanced NLP fails
        if not success:
            print("Advanced NLP failed, falling back to pattern matching")
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self._enrich_clinical_metadata_simple(batch)
    
    def _enrich_with_advanced_nlp(self, chunks, batch_size):
        """Enhanced metadata enrichment using advanced NLP with direct UMLS linking"""
        try:
            # Check if spaCy model is ready
            if self._spacy_nlp is None and not self._load_spacy_model():
                return False
            
            # Define category mappings based on official UMLS semantic groups
            category_mappings = {
                # Conditions/Diseases (DISO - Disorders semantic group)
                "conditions": [
                    "T020",  # Acquired Abnormality
                    "T190",  # Anatomical Abnormality
                    "T049",  # Cell or Molecular Dysfunction
                    "T019",  # Congenital Abnormality
                    "T047",  # Disease or Syndrome
                    "T050",  # Experimental Model of Disease
                    "T033",  # Finding
                    "T037",  # Injury or Poisoning
                    "T048",  # Mental or Behavioral Dysfunction
                    "T191",  # Neoplastic Process
                    "T046",  # Pathologic Function
                    "T184",  # Sign or Symptom
                ],
                
                # Medications (CHEM - Chemicals & Drugs semantic group)
                "medications": [
                    "T116",  # Amino Acid, Peptide, or Protein
                    "T195",  # Antibiotic
                    "T123",  # Biologically Active Substance
                    "T122",  # Biomedical or Dental Material
                    "T103",  # Chemical
                    "T120",  # Chemical Viewed Functionally
                    "T104",  # Chemical Viewed Structurally
                    "T200",  # Clinical Drug
                    "T196",  # Element, Ion, or Isotope
                    "T126",  # Enzyme
                    "T131",  # Hazardous or Poisonous Substance
                    "T125",  # Hormone
                    "T129",  # Immunologic Factor
                    "T130",  # Indicator, Reagent, or Diagnostic Aid
                    "T197",  # Inorganic Chemical
                    "T114",  # Nucleic Acid, Nucleoside, or Nucleotide
                    "T109",  # Organic Chemical
                    "T121",  # Pharmacologic Substance
                    "T192",  # Receptor
                    "T127",  # Vitamin
                ],
                
                # Add the rest of the categories...
                "procedures": ["T060", "T065", "T058", "T059", "T063", "T062", "T061"],
                "lab_tests": ["T034", "T059", "T201", "T067"],
                "anatomy": ["T017", "T029", "T023", "T030", "T031", "T022", "T025", "T026", "T018", "T021", "T024"],
                "demographics": ["T100", "T099", "T096", "T016", "T101", "T098", "T097"],
                "devices": ["T203", "T074", "T075"],
                "physiology": ["T043", "T201", "T045", "T041", "T044", "T032", "T040", "T042", "T039"],
                "behaviors": ["T052", "T053", "T056", "T051", "T064", "T055", "T066", "T057", "T054"],
            }
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [chunk.page_content for chunk in batch]
                
                # Process with NLP pipeline in batches
                try:
                    docs = list(self._spacy_nlp.pipe(texts, batch_size=min(batch_size, 8)))
                except Exception as e:
                    print(f"Error in NLP processing: {e}")
                    continue
                
                # Extract and categorize entities
                for j, doc in enumerate(docs):
                    if j >= len(batch):  # Safety check
                        break
                        
                    chunk = batch[j]
                    
                    # Initialize metadata categories
                    for category in category_mappings:
                        chunk.metadata[category] = []
                    
                    # Add a container for all UMLS entities
                    chunk.metadata["umls_entities"] = []
                    
                    # Track found entities to avoid duplicates
                    entities_found = {category: set() for category in category_mappings}
                    
                    # Handle abbreviations if present
                    if hasattr(doc._, 'abbreviations'):
                        abbreviations = {}
                        for abrv in doc._.abbreviations:
                            abbreviations[abrv.text] = abrv._.long_form.text
                        chunk.metadata['abbreviations'] = abbreviations
                    
                    # Process entities with direct UMLS linking
                    for entity in doc.ents:
                        entity_text = entity.text
                        entity_text_lower = entity_text.lower()
                        
                        # Check if UMLS linker is available and entity has UMLS links
                        if self.config.use_umls_linking:
                            umls_linked = False
                            
                            # Process top UMLS matches
                            for umls_ent in entity._.kb_ents[:3]:  # Get top 3 matches
                                cui = umls_ent[0]
                                score = umls_ent[1]
                                
                                # Only use high confidence matches
                                if score > self.config.umls_confidence_threshold:
                                    try:
                                        # Get entity details from linker
                                        umls_entity = self._umls_linker.kb.cui_to_entity[cui]
                                        
                                        # Extract semantic types
                                        semantic_types = []
                                        for type_string in umls_entity.types:
                                            # Format is typically "T047_Disease_or_Syndrome"
                                            parts = type_string.split('_')
                                            if parts:
                                                semantic_types.append(parts[0])  # Extract "T047"
                                        
                                        # Create entity info
                                        entity_info = {
                                            "text": entity_text,
                                            "cui": cui,
                                            "name": umls_entity.canonical_name,
                                            "definition": umls_entity.definition if umls_entity.definition else "",
                                            "score": float(score),
                                            "semantic_types": semantic_types,
                                            "aliases": list(umls_entity.aliases)[:5]  # Limit to 5 aliases
                                        }
                                        
                                        # Add to umls_entities list
                                        chunk.metadata["umls_entities"].append(entity_info)
                                        
                                        # Categorize based on semantic types
                                        for category, type_codes in category_mappings.items():
                                            if any(t in type_codes for t in semantic_types):
                                                if entity_text_lower not in entities_found[category]:
                                                    chunk.metadata[category].append(entity_info)
                                                    entities_found[category].add(entity_text_lower)
                                                    umls_linked = True
                                    except Exception as e:
                                        print(f"Error processing UMLS entity {cui}: {e}")
                                        continue
                            
                            # If no categories matched but we have UMLS data, add to appropriate category based on entity label
                            if not umls_linked:
                                entity_label = entity.label_
                                category = self._map_entity_label_to_category(entity_label)
                                
                                if category and entity_text_lower not in entities_found[category]:
                                    # Use the entity info we already created, if available
                                    entity_info = next((e for e in chunk.metadata["umls_entities"] 
                                                      if e["text"].lower() == entity_text_lower), 
                                                      {"text": entity_text, "source": "spacy"})
                                    
                                    chunk.metadata[category].append(entity_info)
                                    entities_found[category].add(entity_text_lower)
                        else:
                            # Fallback to basic entity type mapping
                            entity_label = entity.label_
                            category = self._map_entity_label_to_category(entity_label)
                            
                            if category and entity_text_lower not in entities_found[category]:
                                chunk.metadata[category].append({"text": entity_text, "source": "spacy"})
                                entities_found[category].add(entity_text_lower)
                    
                    # Create simplified lists for each category
                    for category in category_mappings:
                        chunk.metadata[f"{category}_simple"] = [item["text"] for item in chunk.metadata[category]]
                    
                    # Identify recommendations
                    self._detect_recommendations(chunk)
            
            return True
            
        except Exception as e:
            print(f"Advanced NLP enrichment failed: {e}")
            return False
    
    def _map_entity_label_to_category(self, entity_label):
        """Map spaCy entity labels to our categories based on UMLS semantic groups"""
        # Map from standard scispaCy/spaCy entity types to our categories
        label_to_category = {
            # Common biomedical entity types
            "DISEASE": "conditions",
            "PROBLEM": "conditions",
            "DX": "conditions",  # Diagnosis abbreviation
            "DIAGNOSIS": "conditions",
            "DISORDER": "conditions",
            "FINDING": "conditions",
            "SYMPTOM": "conditions",
            
            "CHEMICAL": "medications",
            "DRUG": "medications",
            "MEDICATION": "medications",
            "SUBSTANCE": "medications",
            "RX": "medications",  # Prescription abbreviation
            
            "PROCEDURE": "procedures",
            "TREATMENT": "procedures", 
            "THERAPY": "procedures",
            "SURGERY": "procedures",
            
            "TEST": "lab_tests",
            "LAB": "lab_tests",
            "LABORATORY": "lab_tests",
            "MEASUREMENT": "lab_tests",
            
            "ANATOMY": "anatomy",
            "BODY": "anatomy",
            "ORGAN": "anatomy",
            "TISSUE": "anatomy",
            "CELL": "anatomy",
            
            "DEVICE": "devices",
            "EQUIPMENT": "devices",
            
            "PHYSIOLOGY": "physiology",
            "FUNCTION": "physiology",
            
            "BEHAVIOR": "behaviors",
            "ACTIVITY": "behaviors",
            
            # General entity types that might appear
            "PERSON": "demographics",
            "GPE": "demographics",  # Geo-political entity
            "ORG": "demographics",   # Organization
            "NORP": "demographics",  # Nationalities, religious, political groups
        }
        
        # Return the mapped category or None if no mapping exists
        return label_to_category.get(entity_label.upper(), None)
    
    def _detect_recommendations(self, chunk):
        """Detect if a chunk contains clinical recommendations"""
        text = chunk.page_content.lower()
        
        # Check for recommendation patterns
        recommendation_evidence = []
        for pattern in self.recommendation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract context around match
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                recommendation_evidence.append({
                    "pattern": pattern,
                    "text": text[context_start:context_end],
                    "position": (match.start(), match.end())
                })
        
        # Analyze recommendation strength
        recommendation_strength = "unknown"
        strength_patterns = [
            r"(strong(?:ly)?|moderate(?:ly)?|weak(?:ly)?)\s+recommend",
            r"(grade|class|level)\s+([ABCD1234I]+)",
            r"(high|moderate|low)\s+quality\s+evidence"
        ]
        
        for pattern in strength_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract strength indicator
                if "strong" in match.group().lower():
                    recommendation_strength = "strong"
                elif "moderate" in match.group().lower():
                    recommendation_strength = "moderate"
                elif "weak" in match.group().lower():
                    recommendation_strength = "weak"
                elif "grade" in match.group().lower() or "class" in match.group().lower() or "level" in match.group().lower():
                    # Extract grade/class level
                    grade_match = re.search(r"[ABCD1234I]+", match.group())
                    if grade_match:
                        recommendation_strength = f"grade_{grade_match.group()}"
                break
        
        # Set metadata
        if recommendation_evidence:
            chunk.metadata["content_type"] = "recommendation"
            chunk.metadata["recommendation_evidence"] = recommendation_evidence
            chunk.metadata["recommendation_strength"] = recommendation_strength
        else:
            chunk.metadata["content_type"] = "information"
    
    def _enrich_clinical_metadata_simple(self, chunks):
        """Simplified metadata extraction using pre-compiled regex patterns (fallback method)"""
        for chunk in chunks:
            text = chunk.page_content.lower()
            
            # Extract conditions and medications using pre-compiled patterns
            conditions = [{"text": match.group(0)} for match in self.condition_regex.finditer(text)]
            medications = [{"text": match.group(0)} for match in self.medication_regex.finditer(text)]
            
            # Store metadata
            chunk.metadata["conditions"] = conditions
            chunk.metadata["medications"] = medications
            chunk.metadata["conditions_simple"] = [item["text"] for item in conditions]
            chunk.metadata["medications_simple"] = [item["text"] for item in medications]
            
            # Initialize all categories for consistency
            all_categories = [
                "procedures", "lab_tests", "anatomy", "devices", 
                "demographics", "physiology", "behaviors"
            ]
            
            # Initialize empty lists for all categories
            for category in all_categories:
                chunk.metadata[category] = []
                chunk.metadata[f"{category}_simple"] = []
            
            # Initialize UMLS entities container
            chunk.metadata["umls_entities"] = []
            
            # Check for recommendations
            is_recommendation = bool(self.recommendation_regex.search(text))
            chunk.metadata["content_type"] = "recommendation" if is_recommendation else "information"
            if is_recommendation:
                chunk.metadata["recommendation_strength"] = "unknown"
    
    def generate_embeddings(self, chunks, batch_size=None):
        """Generate embeddings using Langchain's embedding model"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        print(f"Generating embeddings with batch size {batch_size}")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i+batch_size]
            texts = [chunk.page_content for chunk in batch]
            
            # Generate embeddings using Langchain's embedding model
            batch_embeddings = self.langchain_embeddings.embed_documents(texts)
            all_embeddings.extend(batch_embeddings)
            
            # Force garbage collection to free memory
            del texts
            gc.collect()
        
        # Convert to numpy array for operations like clustering
        embeddings_array = np.array(all_embeddings)
        
        # Clean up to free memory
        del all_embeddings
        gc.collect()
        
        return embeddings_array
    
    def build_index(self, chunks, embeddings=None):
        """Build a Langchain vector store index using FAISS"""
        LangchainFAISS, LangchainDocument, _ = import_langchain_vectorstores()
        
        # Convert chunks to Langchain Documents if needed
        langchain_docs = []
        for chunk in chunks:
            # Check if it's already a Langchain Document
            if not isinstance(chunk, LangchainDocument):
                langchain_doc = LangchainDocument(
                    page_content=chunk.page_content,
                    metadata=chunk.metadata
                )
                langchain_docs.append(langchain_doc)
            else:
                langchain_docs.append(chunk)
        
        print(f"Building Langchain FAISS index with {len(langchain_docs)} documents")
        
        # If pre-computed embeddings are provided, use them
        if embeddings is not None:
            print("Using pre-computed embeddings")
            # Convert numpy array to list of lists if needed
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
                
            # Create FAISS from embeddings and documents
            vectorstore = LangchainFAISS.from_embeddings(
                embedding_pairs=list(zip([doc.page_content for doc in langchain_docs], embeddings_list)),
                embedding=self.langchain_embeddings,
                metadatas=[doc.metadata for doc in langchain_docs]
            )
        else:
            # Create FAISS index directly from documents
            print("Computing embeddings and building index")
            vectorstore = LangchainFAISS.from_documents(
                documents=langchain_docs,
                embedding=self.langchain_embeddings
            )
        
        print(f"Successfully built Langchain FAISS index")
        return vectorstore

    def release_resources(self):
        """Release memory for heavy components"""
        # Clear langchain embeddings
        if self._langchain_embeddings is not None:
            del self._langchain_embeddings
            self._langchain_embeddings = None
        
        # Clear spaCy and UMLS components
        if self._spacy_nlp is not None:
            del self._spacy_nlp
            self._spacy_nlp = None
        
        # Force garbage collection
        gc.collect()

# Optimized retriever with Langchain vector stores
class OptimizedRetriever:
    def __init__(self, vectorstore, chunks=None, embeddings=None, embedding_model=None):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        
        # Lazy initialization for clustering
        self._kmeans = None
        self._cluster_labels = None
    
    def retrieve(self, query, mode="hybrid", k=5, diversity_factor=2):
        """
        Retrieve relevant chunks with multiple retrieval strategies
        
        Args:
            query: The clinical query
            mode: Retrieval mode (random, top, diversity, cluster, hybrid)
            k: Number of chunks to retrieve
            diversity_factor: Factor to increase diverse results
        """
        if mode == "random":
            return self._random_mode(k)
        elif mode == "top":
            return self._top_mode(query, k)
        elif mode == "diversity":
            return self._diversity_mode(query, k, diversity_factor)
        elif mode == "class" or mode == "cluster":
            return self._cluster_mode(query, k)
        elif mode == "hybrid":
            return self._hybrid_mode(query, k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
    
    def _random_mode(self, k):
        """Random selection with unique results"""
        import random
        if self.chunks:
            indices = random.sample(range(len(self.chunks)), min(k, len(self.chunks)))
            return [self.chunks[i] for i in indices]
        else:
            # If chunks aren't stored directly, get all documents from the vectorstore
            all_docs = self.vectorstore.similarity_search("", k=1000)  # Get a large sample
            indices = random.sample(range(len(all_docs)), min(k, len(all_docs)))
            return [all_docs[i] for i in indices]
    
    def _top_mode(self, query, k):
        """Basic similarity search using Langchain's vector store"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def _diversity_mode(self, query, k, diversity_factor):
        """Maximum Marginal Relevance for diverse results using Langchain's vector store"""
        # Use Langchain's MMR retrieval
        fetch_k = min(k * diversity_factor, 100)  # Avoid fetching too many
        return self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k
        )
    
    def _cluster_mode(self, query, k):
        """Retrieve from different clusters for better coverage"""
        try:
            from sklearn.cluster import KMeans
            
            # We need embeddings for clustering
            if self.embeddings is None or self.chunks is None:
                print("Cluster mode requires stored embeddings and chunks. Falling back to diversity mode.")
                return self._diversity_mode(query, k, 3)
            
            # Create clusters if not already done
            if self._kmeans is None or self._cluster_labels is None:
                n_clusters = min(int(np.sqrt(len(self.chunks))), 100)
                self._kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self._cluster_labels = self._kmeans.fit_predict(self.embeddings)
            
            # Get top k*3 docs using Langchain's similarity search
            top_docs = self.vectorstore.similarity_search(query, k=k*3)
            
            # Map docs back to original chunks to get indices
            top_indices = []
            for doc in top_docs:
                # Find matching chunk by content and metadata
                for i, chunk in enumerate(self.chunks):
                    if (doc.page_content == chunk.page_content and 
                        doc.metadata.get("source") == chunk.metadata.get("source")):
                        top_indices.append(i)
                        break
            
            # Get clusters of retrieved documents
            retrieved_clusters = [self._cluster_labels[i] for i in top_indices if i < len(self._cluster_labels)]
            unique_clusters = list(set(retrieved_clusters))
            
            # Select top document from each cluster
            selected_docs = []
            for cluster in unique_clusters:
                cluster_indices = [idx for i, idx in enumerate(top_indices) 
                               if i < len(retrieved_clusters) and retrieved_clusters[i] == cluster]
                if cluster_indices:
                    selected_docs.append(top_docs[top_indices.index(cluster_indices[0])])
                    if len(selected_docs) >= k:
                        break
            
            # If we need more, add from top results
            if len(selected_docs) < k:
                remaining_docs = [doc for doc in top_docs if doc not in selected_docs]
                selected_docs.extend(remaining_docs[:k-len(selected_docs)])
            
            return selected_docs
        except ImportError:
            # Fallback if sklearn is not available
            print("Clustering not available, falling back to diversity mode")
            return self._diversity_mode(query, k, 3)
    
    def _hybrid_mode(self, query, k):
        """Combine top and diversity modes for balanced results"""
        # Use half for top results, half for diverse results
        half_k = max(1, k // 2)
        
        # Get top results
        top_docs = self._top_mode(query, half_k)
        
        # Get diverse results using MMR
        diversity_docs = self._diversity_mode(query, k - half_k, 3)
        
        # Combine results, removing duplicates
        seen = set()
        combined_docs = []
        
        for doc in top_docs + diversity_docs:
            doc_id = hash(doc.page_content)  # Use content as hash key
            if doc_id not in seen:
                seen.add(doc_id)
                combined_docs.append(doc)
                if len(combined_docs) >= k:
                    break
        
        return combined_docs[:k]

# Optimized cache for Langchain vector stores
class OptimizedCache:
    """Efficient caching system with compression and incremental updates"""
    
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        
        # Create subdirectories
        self.chunks_dir = os.path.join(cache_dir, "chunks")
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.index_dir = os.path.join(cache_dir, "index")
        
        for d in [self.chunks_dir, self.embeddings_dir, self.index_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Maintain in-memory cache for frequently accessed small items
        self.metadata_cache = {}
        
        print(f"Cache initialized at {cache_dir}")
    
    def generate_key(self, files, metadata=None):
        """Generate a cache key with efficient hashing"""
        # Use only filenames and modified times for faster processing
        file_info = []
        
        # Process files in batches to avoid memory issues with large file lists
        batch_size = 1000
        sorted_files = sorted(files)
        
        for i in range(0, len(sorted_files), batch_size):
            batch = sorted_files[i:i+batch_size]
            for file_path in batch:
                if os.path.exists(file_path):
                    mtime = int(os.path.getmtime(file_path))
                    file_size = os.path.getsize(file_path)
                    # Use filename, size and mtime in hash
                    file_info.append(f"{os.path.basename(file_path)}:{file_size}:{mtime}")
        
        # Add metadata hash if provided
        if metadata:
            if isinstance(metadata, dict):
                # Sort keys for consistent hashing
                metadata_str = json.dumps(
                    {k: metadata[k] for k in sorted(metadata.keys())}, 
                    sort_keys=True
                )
                file_info.append(metadata_str)
            else:
                file_info.append(str(metadata))
        
        # Create hash
        key_string = "|".join(file_info)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def save_chunks(self, chunks, files, metadata=None):
        """Save chunks to cache with compression"""
        import pickle
        import gzip
        
        # Generate key
        cache_key = self.generate_key(files, metadata)
        cache_path = os.path.join(self.chunks_dir, f"{cache_key}.pkl.gz")
        
        # Save metadata separately for quick access
        meta_data = {
            "count": len(chunks),
            "sources": list(set(chunk.metadata.get("source", "unknown") for chunk in chunks)),
            "timestamp": datetime.now().isoformat()
        }
        meta_path = os.path.join(self.chunks_dir, f"{cache_key}.meta.json")
        
        # Save compressed chunks
        with gzip.open(cache_path, "wb", compresslevel=5) as f:
            pickle.dump(chunks, f)
        
        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)
        
        # Update in-memory cache
        self.metadata_cache[cache_key] = meta_data
        
        print(f"Cached {len(chunks)} chunks with key {cache_key}")
        return cache_key
    
    def load_chunks(self, files, metadata=None):
        """Load chunks from cache with compression support"""
        import pickle
        import gzip
        
        # Generate key
        cache_key = self.generate_key(files, metadata)
        cache_path = os.path.join(self.chunks_dir, f"{cache_key}.pkl.gz")
        
        if os.path.exists(cache_path):
            try:
                # Load compressed chunks
                with gzip.open(cache_path, "rb") as f:
                    chunks = pickle.load(f)
                
                print(f"Loaded {len(chunks)} chunks from cache (key: {cache_key})")
                return chunks, True
            except Exception as e:
                print(f"Error loading cached chunks: {e}")
        
        return None, False
    
    def save_embeddings(self, embeddings, cache_key):
        """Save embeddings with memory-efficient approach"""
        # Save in compressed npz format
        cache_path = os.path.join(self.embeddings_dir, f"{cache_key}.npz")
        
        # Convert to numpy array if it's a list
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
            
        # Save embeddings
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"Cached embeddings with shape {embeddings.shape}")
    
    def load_embeddings(self, cache_key):
        """Load embeddings with memory-efficient approach"""
        cache_path = os.path.join(self.embeddings_dir, f"{cache_key}.npz")
        
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                embeddings = data['embeddings']
                print(f"Loaded embeddings with shape {embeddings.shape} from cache")
                return embeddings, True
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        
        return None, False
    
    # Updated to handle Langchain vector stores
    def save_index(self, vectorstore, cache_key):
        """Save Langchain FAISS vector store"""
        cache_path = os.path.join(self.index_dir, f"{cache_key}")
        
        try:
            # Save the vector store using Langchain's built-in functionality
            vectorstore.save_local(cache_path)
            print(f"Cached Langchain FAISS vector store with key {cache_key}")
            return True
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False
    
    def load_index(self, cache_key, embedding_model):
        """Load Langchain FAISS vector store"""
        LangchainFAISS, _, _ = import_langchain_vectorstores()
        cache_path = os.path.join(self.index_dir, f"{cache_key}")
        
        if os.path.exists(cache_path):
            try:
                # Load the vector store with the embedding model
                vectorstore = LangchainFAISS.load_local(cache_path, embedding_model, allow_dangerous_deserialization=True)
                print(f"Loaded Langchain FAISS vector store from cache (key: {cache_key})")
                return vectorstore, True
            except Exception as e:
                print(f"Error loading cached vector store: {e}")
        
        return None, False
    
    def clear_cache(self, older_than_days=None):
        """Clear cache files with option to keep recent files"""
        if older_than_days is not None:
            # Calculate cutoff time
            cutoff_time = datetime.now().timestamp() - (older_than_days * 86400)
            
            # Clear old files from each cache directory
            for cache_dir in [self.chunks_dir, self.embeddings_dir, self.index_dir]:
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                        try:
                            os.remove(filepath)
                            print(f"Removed old cache file: {filepath}")
                        except Exception as e:
                            print(f"Error removing cache file {filepath}: {e}")
        else:
            # Clear all cache
            for cache_dir in [self.chunks_dir, self.embeddings_dir, self.index_dir]:
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    if os.path.isfile(filepath):
                        try:
                            os.remove(filepath)
                        except Exception as e:
                            print(f"Error removing cache file {filepath}: {e}")
            
            # Clear in-memory cache
            self.metadata_cache.clear()
            
            print("Cache cleared")

# Text generation remains the same
class OptimizedGenerator:
    def __init__(self, config: MCRAGConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            AutoTokenizer, _ = import_transformers()
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model)
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            # Lazy loading of model to save memory until needed
            _, AutoModelForCausalLM = import_transformers()
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load model with optimized settings for the target device"""
        _, AutoModelForCausalLM = import_transformers()
        device = self.config.device
        
        print(f"Loading generator model on {device}...")
        
        # Configure model loading based on device
        if device == "cpu":
            # CPU configuration - optimize for memory efficiency
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.generator_model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
        elif device == "mps":
            # Apple Silicon configuration
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.generator_model,
                torch_dtype=torch.float16
            ).to("mps")
        else:
            # CUDA configuration with memory efficiency options
            if self.config.quantization:
                # Use 8-bit quantization to reduce memory footprint
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.generator_model,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except ImportError:
                    print("8-bit quantization not available, falling back to 16-bit")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.config.generator_model,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                # Standard 16-bit precision
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.generator_model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
    
    def generate_recommendation(self, query, retrieved_chunks, max_new_tokens=512):
        """Generate clinical recommendation with improved prompt and error handling"""
        # Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)
        
        # Create prompt with better structure for improved responses
        prompt = self._create_prompt(query, context)
        
        # Generate response with error handling
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_length = len(inputs["input_ids"][0])
            
            print(f"Input length: {input_length} tokens")
            
            # Move to appropriate device
            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generation parameters optimized for clinical text - use max_new_tokens instead of max_length
            generation_config = {
                "max_new_tokens": max_new_tokens,  # Generate this many new tokens regardless of input length
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Generate with proper error handling
            with torch.no_grad():
                try:
                    output = self.model.generate(**inputs, **generation_config)
                except RuntimeError as e:
                    error_msg = str(e)
                    if "CUDA out of memory" in error_msg or "MPS backend" in error_msg:
                        # Handle out-of-memory by reducing generation parameters
                        print(f"Memory error: {e}. Retrying with reduced parameters...")
                        
                        # Check if we need to fall back to CPU for MPS-specific errors
                        if "MPS backend" in error_msg and device == "mps":
                            print("Falling back to CPU for MPS-specific error")
                            if hasattr(self._model, "to"):
                                self._model = self._model.to("cpu")
                            inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        
                        # Reduce parameters to save memory
                        generation_config["max_new_tokens"] = 256
                        generation_config["do_sample"] = False
                        
                        # Retry generation
                        output = self.model.generate(**inputs, **generation_config)
                    else:
                        raise
            
            # Decode output, trimming the input prompt
            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the recommendation part
            recommendation = self._extract_recommendation(full_response, prompt)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            # Fallback extraction from retrieved chunks
            return self._generate_fallback_response(retrieved_chunks)
    
    def _create_prompt(self, query, context):
        """Create an optimized prompt for better medical responses"""
        return f"""
You are a clinical decision support system trained to provide evidence-based medical recommendations.

PATIENT QUERY:
{query}

RELEVANT CLINICAL EVIDENCE:
{context}

Based solely on the provided evidence, generate a comprehensive clinical recommendation.
Include:
1. Specific guidelines cited
2. Appropriate dosages if relevant
3. Potential contraindications
4. Reference to the source of each recommendation

CLINICAL RECOMMENDATION:
"""
    
    def _format_context(self, chunks):
        """Format chunks with metadata to provide better context"""
        formatted_chunks = []
        
        # Define categories to display in headers with friendly names
        display_categories = {
            "conditions": "Conditions",
            "medications": "Medications",
            "procedures": "Procedures",
            "lab_tests": "Lab Tests",
            "anatomy": "Anatomy",
            "devices": "Devices",
            "demographics": "Demographics",
            "physiology": "Physiology",
            "behaviors": "Behaviors"
        }
        
        for i, chunk in enumerate(chunks):
            # Extract metadata with fallbacks
            source = chunk.metadata.get("source", "Unknown")
            doc_type = chunk.metadata.get("document_type", "Unknown")
            
            # Create header with source info
            header = f"[{i+1}] Source: {source} (Type: {doc_type})"
            
            # Add entity information for each category if available
            for category_key, display_name in display_categories.items():
                simple_key = f"{category_key}_simple"
                if simple_key in chunk.metadata and chunk.metadata[simple_key]:
                    entities = ", ".join(chunk.metadata[simple_key][:5])  # Limit to 5 entities
                    if entities:
                        header += f" | {display_name}: {entities}"
            
            # Include UMLS entities if available
            umls_entities = chunk.metadata.get("umls_entities", [])
            umls_info = ""
            if umls_entities:
                top_entities = [f"{e['name']} (CUI: {e['cui']})" for e in umls_entities[:3]]
                umls_info = f" | Key Concepts: {', '.join(top_entities)}"
            
            # Include abbreviations if available
            abbreviations = chunk.metadata.get("abbreviations", {})
            abbr_info = ""
            if abbreviations:
                abbr_list = [f"{abbr} = {long_form}" for abbr, long_form in list(abbreviations.items())[:3]]
                abbr_info = f" | Abbreviations: {', '.join(abbr_list)}"
            
            # Add UMLS and abbreviation info
            header += umls_info + abbr_info
            
            # Format the chunk with its header
            formatted_chunk = f"{header}\n{chunk.page_content}"
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n".join(formatted_chunks)
    
    def _extract_recommendation(self, full_response, prompt):
        """Extract recommendation from model output"""
        # Look for recommendation section
        prompt_parts = prompt.split("CLINICAL RECOMMENDATION:")
        prompt_prefix = prompt_parts[0] + "CLINICAL RECOMMENDATION:"
        
        # Remove prompt prefix if it exists in the response
        if full_response.startswith(prompt_prefix):
            recommendation = full_response[len(prompt_prefix):].strip()
        else:
            # Try splitting by the section marker
            recommendation_parts = full_response.split("CLINICAL RECOMMENDATION:")
            if len(recommendation_parts) > 1:
                recommendation = recommendation_parts[1].strip()
            else:
                recommendation = full_response.strip()
        
        return recommendation
    
    def _generate_fallback_response(self, chunks):
        """Generate a simple response from chunks when model generation fails"""
        response = "Based on the available evidence, the following information may be relevant:\n\n"
        
        # Extract the most important chunks (up to 3)
        for i, chunk in enumerate(chunks[:3]):
            # Extract key metadata
            source = chunk.metadata.get("source", "Unknown Source")
            content_type = chunk.metadata.get("content_type", "information")
            
            # Add emphasis if it's a recommendation
            emphasis = "**" if content_type == "recommendation" else ""
            
            # Add the chunk content
            text = chunk.page_content.strip()
            response += f"{emphasis}Source {i+1} ({source}):{emphasis}\n{text}\n\n"
        
        return response
    
    def release_memory(self):
        """Release memory by clearing model and tokenizer"""
        if self._model is not None:
            # Delete model and move to CPU first if on GPU
            if hasattr(self._model, "to") and self.config.device not in ["cpu", "mps"]:
                self._model.to("cpu")
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Force garbage collection
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

# Main system class updated for Langchain vector stores
class OptimizedMCRAG:
    def __init__(self, config: MCRAGConfig):
        self.config = config
        
        # Initialize cache
        self.cache = OptimizedCache(config.cache_dir) if config.use_cache else None
        
        # Initialize components
        self.document_processor = OptimizedDocumentProcessor(config)
        
        # These will be populated when data is loaded
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.retriever = None
        
        # Lazy initialization for generator to save memory
        self._generator = None
    
    @property
    def generator(self):
        """Lazy initialization of generator to save memory"""
        if self._generator is None:
            self._generator = OptimizedGenerator(self.config)
        return self._generator
    
    def load_data(self):
        """Load and process clinical documents with caching and parallel processing"""
        print("Loading and processing clinical documents...")
        
        # Collect file paths
        guideline_files = glob.glob(os.path.join(self.config.data_dir, "guidelines/**/*.pdf"), recursive=True)
        literature_files = glob.glob(os.path.join(self.config.data_dir, "literature/**/*.pdf"), recursive=True)
        text_files = glob.glob(os.path.join(self.config.data_dir, "**/*.txt"), recursive=True)
        
        all_files = guideline_files + literature_files + text_files
        print(f"Found {len(all_files)} files to process")
        
        # Try to load chunks from cache
        cache_key = None
        if self.config.use_cache and self.cache:
            self.chunks, cache_hit = self.cache.load_chunks(all_files)
            if cache_hit:
                cache_key = self.cache.generate_key(all_files)
            else:
                self.chunks = []
        else:
            self.chunks = []
            cache_hit = False
        
        # Process files if not found in cache
        if not cache_hit:
            # Group files by type for specialized processing
            file_batches = {
                "guidelines": [(f, {"document_type": "guideline"}) for f in guideline_files],
                "literature": [(f, {"document_type": "literature"}) for f in literature_files],
                "text": [(f, {"document_type": "text"}) for f in text_files]
            }
            
            # Process each batch
            for batch_name, files_with_meta in file_batches.items():
                if not files_with_meta:
                    continue
                    
                files = [f[0] for f in files_with_meta]
                metadata_template = files_with_meta[0][1] if files_with_meta else {}
                
                print(f"Processing {len(files)} {batch_name} files...")
                batch_chunks = self.document_processor.process_documents(
                    files, 
                    metadata=metadata_template,
                    use_parallel=self.config.max_workers > 1
                )
                self.chunks.extend(batch_chunks)
            
            # Cache processed chunks
            if self.config.use_cache and self.cache:
                cache_key = self.cache.save_chunks(self.chunks, all_files)
        
        print(f"Loaded {len(self.chunks)} document chunks")
        
        # Try to load vector store from cache first
        vector_store_loaded = False
        if self.config.use_cache and self.cache and cache_key:
            self.index, vector_store_loaded = self.cache.load_index(
                cache_key, 
                self.document_processor.langchain_embeddings
            )
        
        # If vector store wasn't in cache or cache is disabled
        if not vector_store_loaded:
            # Generate embeddings if needed for clustering/diversity search
            if self.config.retrieval_mode in ["cluster", "diversity"]:
                if self.config.use_cache and self.cache and cache_key:
                    # Try to load embeddings for clustering
                    self.embeddings, emb_cache_hit = self.cache.load_embeddings(cache_key)
                    
                    if not emb_cache_hit:
                        # Generate and cache embeddings
                        print("Generating embeddings for advanced search modes...")
                        self.embeddings = self.document_processor.generate_embeddings(
                            self.chunks, 
                            batch_size=self.config.batch_size
                        )
                        self.cache.save_embeddings(self.embeddings, cache_key)
                else:
                    # Generate embeddings
                    print("Generating embeddings for advanced search modes...")
                    self.embeddings = self.document_processor.generate_embeddings(
                        self.chunks, 
                        batch_size=self.config.batch_size
                    )
            
            # Build Langchain vector store
            print("Building Langchain vector store...")
            if self.embeddings is not None:
                # Use pre-computed embeddings
                self.index = self.document_processor.build_index(self.chunks, self.embeddings)
            else:
                # Let Langchain compute embeddings
                self.index = self.document_processor.build_index(self.chunks)
            
            # Cache the vector store
            if self.config.use_cache and self.cache and cache_key:
                self.cache.save_index(self.index, cache_key)
        
        # Initialize retriever with Langchain vector store
        self.retriever = OptimizedRetriever(
            vectorstore=self.index,
            chunks=self.chunks,
            embeddings=self.embeddings,
            embedding_model=self.document_processor.langchain_embeddings
        )
        
        print("Data loading complete!")
    
    def process_query(self, query, mode=None, k=None):
        """Process a clinical query and generate recommendation"""
        if self.retriever is None:
            raise ValueError("System not initialized. Call load_data() first.")
        
        # Use config defaults if not specified
        if mode is None:
            mode = self.config.retrieval_mode
        if k is None:
            k = self.config.retrieval_k
        
        print(f"Processing query using {mode} retrieval mode...")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, mode=mode, k=k)
        
        # Generate recommendation
        print("Generating recommendation...")
        recommendation = self.generator.generate_recommendation(query, retrieved_chunks)
        
        # Format result
        result = {
            "query": query,
            "retrieval_mode": mode,
            "retrieved_chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in retrieved_chunks
            ],
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def process_batch_queries(self, queries, mode=None, k=None):
        """Process multiple queries in batch"""
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}")
            try:
                result = self.process_query(query, mode=mode, k=k)
                results.append(result)
            except Exception as e:
                print(f"Error processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def save_result(self, result, query_id=None):
        """Save result to output directory"""
        if query_id is None:
            query_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"result_{query_id}.json")
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Result saved to {output_path}")
        
        return output_path
    
    def release_memory(self):
        """Free memory by releasing heavy components"""
        # Release generator
        if self._generator is not None:
            self._generator.release_memory()
            del self._generator
            self._generator = None
        
        # Release document processor resources
        self.document_processor.release_resources()
        
        # Clear retriever components
        if self.retriever is not None:
            if hasattr(self.retriever, '_kmeans') and self.retriever._kmeans is not None:
                del self.retriever._kmeans
                self.retriever._kmeans = None
            
            if hasattr(self.retriever, '_cluster_labels') and self.retriever._cluster_labels is not None:
                del self.retriever._cluster_labels
                self.retriever._cluster_labels = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Optimized MC-RAG Clinical Decision Support System with Langchain")
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the system with data")
    setup_parser.add_argument("--data-dir", default="./data", help="Data directory")
    setup_parser.add_argument("--output-dir", default="./results", help="Results directory")
    setup_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    setup_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    setup_parser.add_argument("--no-umls", action="store_true", help="Disable UMLS linking")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process a clinical query")
    query_parser.add_argument("--query", required=True, help="Clinical query to process")
    query_parser.add_argument("--data-dir", default="./data", help="Data directory")
    query_parser.add_argument("--output-dir", default="./results", help="Results directory")
    query_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    query_parser.add_argument("--mode", default="hybrid", 
                            choices=["random", "top", "diversity", "cluster", "hybrid"], 
                            help="Retrieval mode")
    query_parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("--no-umls", action="store_true", help="Disable UMLS linking")
    
    # Batch query command
    batch_parser = subparsers.add_parser("batch", help="Process queries from a file")
    batch_parser.add_argument("--file", required=True, help="File with queries (one per line)")
    batch_parser.add_argument("--data-dir", default="./data", help="Data directory")
    batch_parser.add_argument("--output-dir", default="./results", help="Results directory")
    batch_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    batch_parser.add_argument("--mode", default="hybrid", help="Retrieval mode")
    
    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage cache")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--older-than", type=int, help="Clear cache files older than N days")
    cache_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "setup":
        # Create config
        config = MCRAGConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            use_umls_linking=not args.no_umls
        )
        
        # Initialize system
        mcrag = OptimizedMCRAG(config)
        
        # Load data
        mcrag.load_data()
        
        print("System setup complete")
        
    elif args.command == "query":
        # Create config
        config = MCRAGConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            use_umls_linking=not args.no_umls
        )
        
        # Initialize system
        mcrag = OptimizedMCRAG(config)
        
        # Load data
        mcrag.load_data()
        
        # Process query
        result = mcrag.process_query(args.query, mode=args.mode, k=args.k)
        
        # Save result
        mcrag.save_result(result)
        
        # Display recommendation
        print("\n=== Clinical Recommendation ===")
        print(result["recommendation"])
        
        # Release memory
        mcrag.release_memory()
        
    elif args.command == "batch":
        # Create config
        config = MCRAGConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir
        )
        
        # Load queries
        with open(args.file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(queries)} queries from {args.file}")
        
        # Initialize system
        mcrag = OptimizedMCRAG(config)
        
        # Load data
        mcrag.load_data()
        
        # Process queries
        results = mcrag.process_batch_queries(queries, mode=args.mode)
        
        # Save results
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.output_dir, f"batch_results_{batch_id}.json")
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch results saved to {output_path}")
        
        # Release memory
        mcrag.release_memory()
        
    elif args.command == "cache":
        # Create cache manager
        cache = OptimizedCache(args.cache_dir)
        
        if args.clear:
            # Clear cache
            cache.clear_cache(older_than_days=args.older_than)
            print("Cache cleared")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Print a banner indicating this is the Langchain version
    print("="*80)
    print("MCRAG - Medical Context Retrieval Augmented Generation")
    print("Langchain Vector Store Edition")
    print("="*80)
    main()