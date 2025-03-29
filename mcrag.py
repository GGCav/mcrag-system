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
import warnings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCRAG")

# Configuration class for better organization and flexibility
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

def import_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def import_faiss():
    import faiss
    return faiss

def import_transformers():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return AutoTokenizer, AutoModelForCausalLM

# Optimized document processor
class OptimizedDocumentProcessor:
    def __init__(self, config: MCRAGConfig):
        self.config = config
        RecursiveCharacterTextSplitter = import_text_splitter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # Lazy load embedding model only when needed
        self._embedding_model = None
        self.semantic_type_cache = {}
        
        # Pre-compile regex patterns for faster metadata extraction
        self._compile_regex_patterns()
    
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
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            SentenceTransformer = import_sentence_transformer()
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model
    
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
        
        # Try to use advanced NLP first
        success = self._enrich_with_advanced_nlp(chunks, batch_size)
        
        # Fall back to simple pattern matching if advanced NLP fails
        if not success:
            print("Advanced NLP failed, falling back to pattern matching")
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                self._enrich_clinical_metadata_simple(batch)
    
    def _enrich_with_advanced_nlp(self, chunks, batch_size):
        """
        Enhanced metadata enrichment using advanced NLP with en_core_sci_lg and UMLS linking
        
        Args:
            chunks: List of document chunks to process
            batch_size: Number of chunks to process at once
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Import required libraries
            import spacy
            import re
            from spacy.tokens.span import Span
            
            # Load specialized biomedical NLP model
            try:
                print("Loading en_core_sci_lg model...")
                nlp = spacy.load("en_core_sci_lg")
                print("Successfully loaded en_core_sci_lg model")
            except OSError:
                # If model not found, try to download it
                print("Model not found, attempting to download...")
                try:
                    spacy.cli.download("en_core_sci_lg")
                    nlp = spacy.load("en_core_sci_lg")
                    print("Successfully downloaded and loaded en_core_sci_lg model")
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    return False
            
            # Add UMLS entity linker
            try:
                from scispacy.linking import EntityLinker
                
                # Register the entity linker if not already present
                if "scispacy_linker_umls" not in nlp.pipe_names:
                    # Define a factory for the entity linker
                    @spacy.Language.factory("scispacy_linker_umls")
                    def create_entity_linker(nlp, name):
                        return EntityLinker(nlp, name="umls", resolve_abbreviations=True)
                    
                    # Add the entity linker to the pipeline
                    nlp.add_pipe("scispacy_linker_umls")
                    print("Added UMLS entity linker to pipeline")
            except ImportError as e:
                print(f"EntityLinker not available: {e}")
                # Continue without entity linking
            
            # Set up custom extensions for negation if not already present
            if not Span.has_extension("is_negated"):
                Span.set_extension("is_negated", default=False)
            
            # Add custom negation detection
            # First, create a proper component class instead of a direct function
            @spacy.Language.factory("negation_detector")
            class NegationDetector:
                def __init__(self, nlp, name):
                    self.name = name
                    self.nlp = nlp
                    # Common medical negation terms
                    self.negation_terms = ["no", "not", "none", "negative", "deny", "denies", 
                                          "denied", "absent", "without", "non", "ruled out"]
                    self.window_size = 5  # Words to look before/after entity
                
                def __call__(self, doc):
                    """Process the document, marking entities as negated where appropriate"""
                    for ent in doc.ents:
                        # Default: not negated
                        ent._.set("is_negated", False)
                        
                        # Check before entity
                        start = max(0, ent.start - self.window_size)
                        prefix = doc[start:ent.start]
                        
                        # Check after entity
                        end = min(len(doc), ent.end + self.window_size)
                        suffix = doc[ent.end:end]
                        
                        # Mark as negated if negation term found
                        for token in list(prefix) + list(suffix):
                            if token.lower_ in self.negation_terms:
                                ent._.set("is_negated", True)
                                break
                    
                    return doc
            
            # Add negation component to pipeline if not already there
            if "negation_detector" not in nlp.pipe_names:
                nlp.add_pipe("negation_detector")
                print("Added custom negation detector")
            
            # Define semantic type mappings
            category_mappings = {
                # Conditions/Diseases
                "conditions": [
                    "T047",  # Disease or Syndrome
                    "T048",  # Mental/Behavioral Dysfunction
                    "T019",  # Congenital Abnormality
                    "T046",  # Pathologic Function
                    "T184",  # Sign or Symptom
                    "T050",  # Experimental Model of Disease
                    "T033",  # Finding
                ],
                
                # Medications
                "medications": [
                    "T121",  # Pharmacologic Substance
                    "T122",  # Biomedical or Dental Material
                    "T123",  # Biologically Active Substance
                    "T131",  # Hazardous or Poisonous Substance
                ],
                
                # Procedures
                "procedures": [
                    "T060",  # Diagnostic Procedure
                    "T061",  # Therapeutic or Preventive Procedure
                    "T059",  # Laboratory Procedure
                ],
                
                # Lab tests
                "lab_tests": [
                    "T034",  # Laboratory or Test Result
                    "T201",  # Clinical Attribute
                    "T196",  # Element, Ion, or Isotope
                ],
            }
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [chunk.page_content for chunk in batch]
                
                # Process with NLP pipeline in batches
                try:
                    docs = list(nlp.pipe(texts, batch_size=min(batch_size, 8)))
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
                    
                    # Track found entities to avoid duplicates
                    entities_found = {category: set() for category in category_mappings}
                    
                    # Process entities
                    for ent in doc.ents:
                        # Skip negated entities
                        if ent._.get("is_negated", False):
                            continue
                        
                        entity_text = ent.text.lower()
                        entity_label = ent.label_
                        linked = False
                        
                        # Try UMLS linking if available
                        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                            kb_entities = ent._.kb_ents
                            
                            if kb_entities and len(kb_entities) > 0:
                                # Get top UMLS entity
                                umls_entity = kb_entities[0]
                                cui = umls_entity[0]  # Concept ID
                                score = umls_entity[1]  # Confidence score
                                
                                # Only use high confidence matches
                                if score > 0.7:
                                    # Get semantic types for the concept
                                    semantic_types = self._get_semantic_types(cui)
                                    
                                    # Categorize based on semantic type
                                    for category, type_codes in category_mappings.items():
                                        if any(t in type_codes for t in semantic_types):
                                            # Add if not already found
                                            if entity_text not in entities_found[category]:
                                                chunk.metadata[category].append({
                                                    "text": entity_text,
                                                    "cui": cui,
                                                    "confidence": float(score)
                                                })
                                                entities_found[category].add(entity_text)
                                                linked = True
                        
                        # For entities not linked to UMLS, use spaCy entity labels
                        if not linked:
                            # Map spaCy labels to our categories
                            if entity_label in ["DISEASE", "PROBLEM"]:
                                category = "conditions"
                            elif entity_label in ["CHEMICAL", "DRUG"]:
                                category = "medications"
                            elif entity_label in ["PROCEDURE"]:
                                category = "procedures"
                            else:
                                continue  # Skip other entity types
                                
                            # Add entity if not already found
                            if entity_text not in entities_found[category]:
                                chunk.metadata[category].append({"text": entity_text})
                                entities_found[category].add(entity_text)
                    
                    # Create simplified lists
                    for category in category_mappings:
                        chunk.metadata[f"{category}_simple"] = [item["text"] for item in chunk.metadata[category]]
                    
                    # Identify recommendations
                    self._detect_recommendations(chunk)
            
            return True
            
        except Exception as e:
            print(f"Advanced NLP enrichment failed: {e}")
            return False
    
    def _get_semantic_types(self, cui):
        """
        Get semantic types for a UMLS concept ID, with caching
        
        Args:
            cui: UMLS Concept Unique Identifier
            
        Returns:
            List of semantic type codes
        """
        # Check cache first
        if cui in self.semantic_type_cache:
            return self.semantic_type_cache[cui]
        
        # Try API lookup with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import requests
                
                # UMLS REST API URL
                url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
                
                # Try to get API key from environment
                import os
                api_key = os.environ.get("UMLS_API_KEY", "dd107a9b-1977-4faf-b847-dcb8a8783e29")
                
                # Make API request
                response = requests.get(url, params={"apiKey": api_key}, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    semantic_types = []
                    
                    # Extract semantic types
                    if "result" in data and "semanticTypes" in data["result"]:
                        for st in data["result"]["semanticTypes"]:
                            if "type" in st:
                                semantic_types.append(st["type"])
                            elif "name" in st:
                                semantic_types.append(st["name"])
                    
                    # Cache result
                    self.semantic_type_cache[cui] = semantic_types
                    return semantic_types
                else:
                    # Retry on server errors
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        continue
                    return []
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                print(f"Error retrieving semantic types for CUI {cui}: {e}")
                return []
        
        # Return empty list if all retries fail
        return []
    
    def _detect_recommendations(self, chunk):
        """
        Detect if a chunk contains clinical recommendations
        
        Args:
            chunk: Document chunk to analyze
        """
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
            
            # Also initialize other categories for consistency
            chunk.metadata["procedures"] = []
            chunk.metadata["procedures_simple"] = []
            chunk.metadata["lab_tests"] = []
            chunk.metadata["lab_tests_simple"] = []
            
            # Check for recommendations
            is_recommendation = bool(self.recommendation_regex.search(text))
            chunk.metadata["content_type"] = "recommendation" if is_recommendation else "information"
            if is_recommendation:
                chunk.metadata["recommendation_strength"] = "unknown"
    
    def generate_embeddings(self, chunks, batch_size=None):
        """Generate embeddings in batches to reduce memory usage"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
            batch = chunks[i:i+batch_size]
            texts = [chunk.page_content for chunk in batch]
            
            # Generate embeddings for batch
            batch_embeddings = self.embedding_model.encode(texts)
            all_embeddings.append(batch_embeddings)
            
            # Force garbage collection to free memory
            del texts
            gc.collect()
        
        # Combine all batches
        combined = np.vstack(all_embeddings)
        
        # Clean up to free memory
        del all_embeddings
        gc.collect()
        
        return combined
    
    def build_index(self, embeddings):
        """Build optimized FAISS index based on dataset size"""
        faiss = import_faiss()
        dimension = embeddings.shape[1]
        
        # Choose index type based on dataset size
        if embeddings.shape[0] < 10000:
            # For small datasets, use exact search (flat index)
            index = faiss.IndexFlatL2(dimension)
        else:
            # For larger datasets, use IVF index with clustering for faster search
            nlist = min(4096, int(4 * np.sqrt(embeddings.shape[0])))
            quantizer = faiss.IndexFlatL2(dimension) 
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        
        return index

# Optimized retriever with better search strategies
class OptimizedRetriever:
    def __init__(self, index, chunks, embeddings, embedding_model):
        self.index = index
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
        # Encode query efficiently
        query_embedding = self.embedding_model.encode([query])
        
        if mode == "random":
            return self._random_mode(k)
        elif mode == "top":
            return self._top_mode(query_embedding, k)
        elif mode == "diversity":
            return self._diversity_mode(query_embedding, k, diversity_factor)
        elif mode == "class" or mode == "cluster":
            return self._cluster_mode(query_embedding, k)
        elif mode == "hybrid":
            return self._hybrid_mode(query_embedding, k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
    
    def _random_mode(self, k):
        """Random selection with unique results"""
        import random
        indices = random.sample(range(len(self.chunks)), min(k, len(self.chunks)))
        return [self.chunks[i] for i in indices]
    
    def _top_mode(self, query_embedding, k):
        """Basic similarity search"""
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]
    
    def _diversity_mode(self, query_embedding, k, diversity_factor):
        """Maximum Marginal Relevance for diverse results"""
        # Get a larger pool of candidates
        top_k = min(k * diversity_factor, len(self.chunks))
        distances, indices = self.index.search(query_embedding, top_k)
        indices = indices[0]
        
        # Apply MMR algorithm
        selected_indices = []
        selected_embeddings = []
        
        # Add the most relevant result first
        selected_indices.append(indices[0])
        selected_embeddings.append(self.embeddings[indices[0]])
        
        # Select remaining documents
        for _ in range(1, k):
            if len(selected_indices) >= len(indices):
                break
                
            max_diversity = -float('inf')
            most_diverse_idx = -1
            
            for idx in indices:
                if idx in selected_indices:
                    continue
                
                # Calculate similarity to query
                query_sim = 1.0 - distances[0][list(indices).index(idx)]
                
                # Calculate similarity to already selected documents
                doc_embedding = self.embeddings[idx]
                doc_sims = [np.dot(doc_embedding, sel_emb) / 
                           (np.linalg.norm(doc_embedding) * np.linalg.norm(sel_emb)) 
                           for sel_emb in selected_embeddings]
                
                max_doc_sim = max(doc_sims) if doc_sims else 0
                
                # Balance relevance and diversity
                diversity_score = 0.7 * query_sim - 0.3 * max_doc_sim
                
                if diversity_score > max_diversity:
                    max_diversity = diversity_score
                    most_diverse_idx = idx
            
            if most_diverse_idx != -1:
                selected_indices.append(most_diverse_idx)
                selected_embeddings.append(self.embeddings[most_diverse_idx])
        
        return [self.chunks[i] for i in selected_indices]
    
    def _cluster_mode(self, query_embedding, k):
        """Retrieve from different clusters for better coverage"""
        try:
            from sklearn.cluster import KMeans
            
            # Create clusters if not already done
            if self._kmeans is None or self._cluster_labels is None:
                n_clusters = min(int(np.sqrt(len(self.chunks))), 100)
                self._kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self._cluster_labels = self._kmeans.fit_predict(self.embeddings)
            
            # Get a larger pool of candidates
            top_k = min(k * 3, len(self.chunks))
            distances, indices = self.index.search(query_embedding, top_k)
            indices = indices[0]
            
            # Get clusters of retrieved documents
            retrieved_clusters = [self._cluster_labels[i] for i in indices]
            unique_clusters = list(set(retrieved_clusters))
            
            # Select top document from each cluster
            selected_chunks = []
            for cluster in unique_clusters:
                cluster_indices = [idx for i, idx in enumerate(indices) if retrieved_clusters[i] == cluster]
                if cluster_indices:
                    selected_chunks.append(self.chunks[cluster_indices[0]])
                    if len(selected_chunks) >= k:
                        break
            
            # If we need more, add from top results
            if len(selected_chunks) < k:
                remaining_indices = [i for i in indices if self.chunks[i] not in selected_chunks]
                selected_chunks.extend([self.chunks[i] for i in remaining_indices[:k-len(selected_chunks)]])
            
            return selected_chunks
        except ImportError:
            # Fallback if sklearn is not available
            print("Clustering not available, falling back to diversity mode")
            return self._diversity_mode(query_embedding, k, 3)
    
    def _hybrid_mode(self, query_embedding, k):
        """Combine top and diversity modes for balanced results"""
        # Use half for top results, half for diverse results
        half_k = max(1, k // 2)
        top_chunks = self._top_mode(query_embedding, half_k)
        
        diversity_chunks = self._diversity_mode(query_embedding, k - half_k, 3)
        
        # Combine results, removing duplicates
        seen = set()
        combined_chunks = []
        
        for chunk in top_chunks + diversity_chunks:
            chunk_id = id(chunk)
            if chunk_id not in seen:
                seen.add(chunk_id)
                combined_chunks.append(chunk)
                if len(combined_chunks) >= k:
                    break
        
        return combined_chunks[:k]

# Optimized text generation
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
    
    def generate_recommendation(self, query, retrieved_chunks, max_length=512):
        """Generate clinical recommendation with improved prompt and error handling"""
        # Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)
        
        # Create prompt with better structure for improved responses
        prompt = self._create_prompt(query, context)
        
        # Generate response with error handling
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to appropriate device
            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generation parameters optimized for clinical text
            generation_config = {
                "max_length": max_length,
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
                    if "CUDA out of memory" in str(e) or "MPS backend" in str(e):
                        # Handle out-of-memory by reducing generation parameters
                        print(f"Memory error: {e}. Retrying with reduced parameters...")
                        
                        # Reduce parameters to save memory
                        generation_config["max_length"] = min(256, max_length)
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
        
        for i, chunk in enumerate(chunks):
            # Extract metadata with fallbacks
            source = chunk.metadata.get("source", "Unknown")
            doc_type = chunk.metadata.get("document_type", "Unknown")
            
            # Include entity information if available
            conditions = ", ".join(chunk.metadata.get("conditions_simple", []))
            medications = ", ".join(chunk.metadata.get("medications_simple", []))
            
            # Create a header with context
            header = f"[{i+1}] Source: {source} (Type: {doc_type})"
            if conditions:
                header += f" | Conditions: {conditions}"
            if medications:
                header += f" | Medications: {medications}"
            
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

# Optimized caching system
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
        # Save in compressed npz format instead of raw npy
        cache_path = os.path.join(self.embeddings_dir, f"{cache_key}.npz")
        
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
    
    def save_index(self, index, cache_key):
        """Save FAISS index"""
        faiss = import_faiss()
        cache_path = os.path.join(self.index_dir, f"{cache_key}.index")
        
        # Save index
        faiss.write_index(index, cache_path)
        print(f"Cached FAISS index with key {cache_key}")
    
    def load_index(self, cache_key):
        """Load FAISS index"""
        faiss = import_faiss()
        cache_path = os.path.join(self.index_dir, f"{cache_key}.index")
        
        if os.path.exists(cache_path):
            try:
                index = faiss.read_index(cache_path)
                print(f"Loaded FAISS index from cache (key: {cache_key})")
                return index, True
            except Exception as e:
                print(f"Error loading cached index: {e}")
        
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

# Main system class
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
        
        # Generate or load embeddings
        if self.config.use_cache and self.cache and cache_key:
            # Try to load embeddings
            self.embeddings, emb_cache_hit = self.cache.load_embeddings(cache_key)
            
            if not emb_cache_hit:
                # Generate and cache embeddings
                print("Generating embeddings...")
                self.embeddings = self.document_processor.generate_embeddings(
                    self.chunks, 
                    batch_size=self.config.batch_size
                )
                self.cache.save_embeddings(self.embeddings, cache_key)
        else:
            # Generate embeddings
            print("Generating embeddings...")
            self.embeddings = self.document_processor.generate_embeddings(
                self.chunks, 
                batch_size=self.config.batch_size
            )
            if self.config.use_cache and self.cache and cache_key:
                self.cache.save_embeddings(self.embeddings, cache_key)
        
        # Build or load index
        if self.config.use_cache and self.cache and cache_key:
            self.index, index_cache_hit = self.cache.load_index(cache_key)
            
            if not index_cache_hit:
                # Build and cache index
                print("Building index...")
                self.index = self.document_processor.build_index(self.embeddings)
                self.cache.save_index(self.index, cache_key)
        else:
            # Build index
            print("Building index...")
            self.index = self.document_processor.build_index(self.embeddings)
            if self.config.use_cache and self.cache and cache_key:
                self.cache.save_index(self.index, cache_key)
        
        # Initialize retriever
        self.retriever = OptimizedRetriever(
            index=self.index,
            chunks=self.chunks,
            embeddings=self.embeddings,
            embedding_model=self.document_processor.embedding_model
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
        
        # Clear embedding model
        if hasattr(self.document_processor, '_embedding_model') and self.document_processor._embedding_model is not None:
            del self.document_processor._embedding_model
            self.document_processor._embedding_model = None
        
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
    parser = argparse.ArgumentParser(description="Optimized MC-RAG Clinical Decision Support System")
    
    # Setup subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the system with data")
    setup_parser.add_argument("--data-dir", default="./data", help="Data directory")
    setup_parser.add_argument("--output-dir", default="./results", help="Results directory")
    setup_parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    setup_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
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
            use_cache=not args.no_cache
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
            cache_dir=args.cache_dir
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
    main()