import os
import requests
import json
import time
import zipfile
import tarfile
import re
import shutil
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader
from io import BytesIO
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
import ftplib

# Constants
BASE_DATA_DIR = "./data"
GUIDELINES_DIR = os.path.join(BASE_DATA_DIR, "guidelines")
LITERATURE_DIR = os.path.join(BASE_DATA_DIR, "literature")
EHR_DATA_DIR = os.path.join(BASE_DATA_DIR, "ehr_data")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")

# Create necessary directories
def create_data_directories():
    """Create all necessary data directories"""
    directories = [
        BASE_DATA_DIR,
        GUIDELINES_DIR,
        os.path.join(GUIDELINES_DIR, "nice"),
        os.path.join(GUIDELINES_DIR, "acp"),
        os.path.join(GUIDELINES_DIR, "aha"),
        os.path.join(GUIDELINES_DIR, "nccn"),
        LITERATURE_DIR,
        os.path.join(LITERATURE_DIR, "pubmed"),
        os.path.join(LITERATURE_DIR, "pmc"),
        EHR_DATA_DIR,
        os.path.join(EHR_DATA_DIR, "mimic"),
        os.path.join(EHR_DATA_DIR, "i2b2"),
        IMAGES_DIR,
        os.path.join(IMAGES_DIR, "mimic_cxr"),
        os.path.join(IMAGES_DIR, "other")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def extract_evidence_quality(pdf_path_or_content):
    """
    Extract evidence quality information from a PDF
    
    Args:
        pdf_path_or_content: Either a file path to a PDF or PDF content as bytes
    """
    try:
        # Handle either file path or content
        if isinstance(pdf_path_or_content, str) and os.path.exists(pdf_path_or_content):
            # It's a file path
            pdf = PdfReader(pdf_path_or_content)
        else:
            # It's content data
            pdf = PdfReader(BytesIO(pdf_path_or_content))
        
        text = ""
        # Only process up to 10 pages to avoid long processing times
        max_pages = min(10, len(pdf.pages))
        
        for page in pdf.pages[:max_pages]:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"Error extracting text from page: {e}")
        
        # Look for common evidence grading systems
        evidence_levels = []
        
        # GRADE system
        if re.search(r'(high|moderate|low|very low)\s+quality\s+of\s+evidence', text, re.IGNORECASE):
            evidence_levels.append("GRADE")
        
        # Level of Evidence (I, II, III)
        if re.search(r'level\s+(of\s+)?evidence\s+(I{1,3}|[ABC])', text, re.IGNORECASE):
            evidence_levels.append("LOE")
            
        # Strength of Recommendation (A, B, C)
        if re.search(r'strength\s+(of\s+)?recommendation\s+[ABC]', text, re.IGNORECASE):
            evidence_levels.append("SOR")
        
        # NICE specific 
        if re.search(r'NICE\s+guidance', text, re.IGNORECASE) or re.search(r'NICE\s+guideline', text, re.IGNORECASE):
            evidence_levels.append("NICE")
            
        return ",".join(evidence_levels) if evidence_levels else "Unknown"
    except Exception as e:
        print(f"Error extracting evidence quality: {e}")
        return "Unknown"
    
# 1. Clinical Guidelines Functions
def download_nice_guidelines(output_dir, limit=None):
    """
    Download clinical practice guidelines from NICE website
    
    Args:
        output_dir: Directory to save the guidelines
        limit: Maximum number of guidelines to download (None for unlimited)
    
    Returns:
        List of successfully downloaded guidelines
    """
    base_url = "https://www.nice.org.uk"
    search_url = f"{base_url}/guidance/published?ngt=Clinical%20guidelines&ndt=Guidance&ps=9999"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add a User-Agent header to simulate a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Accessing NICE guidelines at: {search_url}")
    
    try:
        # Get all guidelines in a single request
        response = requests.get(search_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        print(f"Successfully connected to NICE website (Status: {response.status_code})")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the guidelines table using the exact class from the provided HTML
        table = soup.select_one('table.ProductListPage_table__6QiDc') or soup.select_one('table#results')
        
        if not table:
            print("Could not find guidelines table on the page")
            return []
        
        # Process the table rows
        rows = table.select('tbody tr')
        if not rows:
            print("No guideline rows found in table")
            return []
        
        print(f"Found {len(rows)} guidelines in the table")
        
        # Apply limit if specified
        if limit:
            rows = rows[:limit]
            print(f"Limiting to {limit} guidelines as requested")
        
        # We'll collect all guidelines here
        all_guidelines = []
        
        # Process each row in the table
        for row in rows:
            # Extract guideline information from table cells
            title_cell = row.select_one('td:first-child')
            if not title_cell or not title_cell.select_one('a'):
                continue
            
            link = title_cell.select_one('a')
            title = link.text.strip()
            url = link['href']
            if not url.startswith('http'):
                url = base_url + url
            
            # Extract reference number from second column
            ref_cell = row.select_one('td:nth-child(2)')
            ref_number = ref_cell.text.strip() if ref_cell else ""
            
            # Extract publication date from third column
            pub_date = ""
            pub_date_cell = row.select_one('td:nth-child(3)')
            if pub_date_cell:
                time_elem = pub_date_cell.select_one('time')
                if time_elem:
                    pub_date = time_elem.text.strip()
            
            # Extract last updated date from fourth column
            update_date = ""
            update_cell = row.select_one('td:nth-child(4)')
            if update_cell:
                time_elem = update_cell.select_one('time')
                if time_elem:
                    update_date = time_elem.text.strip()
            
            all_guidelines.append({
                'title': title,
                'url': url,
                'reference': ref_number,
                'publication_date': pub_date,
                'update_date': update_date,
                'source': 'NICE'
            })
        
        print(f"Found a total of {len(all_guidelines)} guidelines")
            
    except Exception as e:
        print(f"Error accessing guidelines list: {e}")
        return []
    
    # Download PDF versions where available
    successful_downloads = []
    
    for i, guideline in enumerate(all_guidelines):
        try:
            print(f"Processing guideline {i+1}/{len(all_guidelines)}: {guideline['title']} ({guideline['reference']})")
            
            # Visit the guideline page
            response = requests.get(guideline['url'], headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"Failed to access guideline page: {response.status_code}")
                guideline['downloaded'] = False
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the specific download button based on the provided HTML
            pdf_link = None
            
            # Method 1: Look for the specific button class and text
            download_buttons = soup.select('a.btn.btn--cta[data-track="guidancedownload"]')
            if download_buttons:
                for button in download_buttons:
                    if 'download guidance' in button.text.strip().lower():
                        pdf_link = button['href']
                        print(f"Found download button: {pdf_link}")
                        break
            
            # Method 2: Look for any link with data-track="guidancedownload"
            if not pdf_link:
                download_links = soup.select('a[data-track="guidancedownload"]')
                if download_links:
                    pdf_link = download_links[0]['href']
                    print(f"Found download link by data-track: {pdf_link}")
            
            # Method 3: Fallback to any link containing PDF
            if not pdf_link:
                pdf_links = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))
                if pdf_links:
                    pdf_link = pdf_links[0]['href']
                    print(f"Found PDF link by extension: {pdf_link}")
            
            if pdf_link:
                # Make sure the URL is absolute
                if not pdf_link.startswith('http'):
                    pdf_link = base_url + pdf_link
                
                print(f"Downloading PDF from: {pdf_link}")
                
                # Download PDF with timeout and headers
                pdf_response = requests.get(pdf_link, headers=headers, timeout=60, stream=True)
                
                if pdf_response.status_code == 200:
                    # Create a filename with the reference number for better organization
                    ref = guideline.get('reference', '').upper()
                    clean_title = re.sub(r'[\\/*?:"<>|]', "", guideline['title'])
                    clean_title = ''.join(c if c.isalnum() or c in ' -_.,()[]{}' else '_' for c in clean_title)
                    # Truncate filename if too long
                    if len(clean_title) > 80:
                        clean_title = clean_title[:77] + "..."
                    
                    filename = os.path.join(output_dir, f"NICE_{ref}_{clean_title}.pdf")
                    
                    # Stream the download to handle large files
                    with open(filename, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Successfully downloaded: {guideline['title']} ({ref})")
                    
                    # Add metadata
                    guideline['filename'] = filename
                    guideline['downloaded'] = True
                    
                    successful_downloads.append(guideline)
                else:
                    print(f"Failed to download PDF, status code: {pdf_response.status_code}")
                    guideline['downloaded'] = False
            else:
                print(f"No PDF link found for: {guideline['title']}")
                guideline['downloaded'] = False
                
        except Exception as e:
            print(f"Error processing {guideline['title']}: {e}")
            guideline['downloaded'] = False
        
        # Add a slight delay between requests to be polite to the server
        time.sleep(2)
    
    # Save complete metadata including both successful and failed downloads
    metadata_df = pd.DataFrame(all_guidelines)
    metadata_df.to_csv(os.path.join(output_dir, 'nice_metadata_all.csv'), index=False)
    
    # Save metadata for successful downloads only
    success_df = pd.DataFrame(successful_downloads)
    if not success_df.empty:
        success_df.to_csv(os.path.join(output_dir, 'nice_metadata.csv'), index=False)
    else:
        # Create an empty file with proper structure
        empty_df = pd.DataFrame(columns=['title', 'url', 'reference', 'source', 'filename', 'downloaded'])
        empty_df.to_csv(os.path.join(output_dir, 'nice_metadata.csv'), index=False)
    
    print(f"NICE guidelines download summary:")
    print(f"  - Total guidelines found: {len(all_guidelines)}")
    print(f"  - Successfully downloaded: {len(successful_downloads)}")
    
    # Return only the successfully downloaded guidelines
    return successful_downloads
    

def download_acp_guidelines(output_dir, limit=5):
    """Download clinical practice guidelines from American College of Physicians"""
    base_url = "https://www.acponline.org"
    search_url = f"{base_url}/clinical-information/guidelines"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        guidelines = []
        
        # Find guideline links (simplified approach)
        for link in soup.select('a[href*="guideline"]')[:limit]:
            title = link.text.strip()
            if not title or len(title) < 10:  # Skip short/empty titles
                continue
                
            url = link.get('href')
            if not url.startswith('http'):
                url = base_url + url
                
            guideline = {
                'title': title,
                'url': url,
                'source': 'ACP'
            }
            
            guidelines.append(guideline)
            
            # Try to find and download PDF
            try:
                guide_response = requests.get(url)
                guide_soup = BeautifulSoup(guide_response.content, 'html.parser')
                
                pdf_link = None
                for pdf in guide_soup.select('a[href$=".pdf"]'):
                    pdf_link = pdf.get('href')
                    if pdf_link:
                        if not pdf_link.startswith('http'):
                            pdf_link = base_url + pdf_link
                        break
                
                if pdf_link:
                    pdf_response = requests.get(pdf_link)
                    if pdf_response.status_code == 200:
                        clean_title = ''.join(c if c.isalnum() else '_' for c in title)
                        filename = os.path.join(output_dir, f"ACP_{clean_title}.pdf")
                        
                        with open(filename, 'wb') as f:
                            f.write(pdf_response.content)
                        
                        guideline['filename'] = filename
                        guideline['downloaded'] = True
                        guideline['evidence_quality'] = extract_evidence_quality(pdf_response.content)
                        
                        print(f"Downloaded ACP guideline: {title}")
                    else:
                        guideline['downloaded'] = False
                else:
                    guideline['downloaded'] = False
            except Exception as e:
                print(f"Error downloading ACP guideline PDF {title}: {e}")
                guideline['downloaded'] = False
        
        # Save metadata
        metadata_df = pd.DataFrame(guidelines)
        metadata_df.to_csv(os.path.join(output_dir, 'acp_metadata.csv'), index=False)
        
        return guidelines
    
    except Exception as e:
        print(f"Error downloading ACP guidelines: {e}")
        return []

def download_aha_guidelines(output_dir, limit=5):
    """Download clinical practice guidelines from American Heart Association"""
    base_url = "https://professional.heart.org"
    search_url = f"{base_url}/science-news/statements-guidelines"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        guidelines = []
        
        # Find guideline links (simplified approach)
        for article in soup.select('article.teaser')[:limit]:
            title_elem = article.select_one('h2')
            if not title_elem:
                continue
                
            title = title_elem.text.strip()
            link_elem = article.select_one('a')
            url = link_elem.get('href') if link_elem else None
            
            if url and not url.startswith('http'):
                url = base_url + url
            
            if url:
                guideline = {
                    'title': title,
                    'url': url,
                    'source': 'AHA'
                }
                
                guidelines.append(guideline)
                
                # Try to find and download PDF
                try:
                    guide_response = requests.get(url)
                    guide_soup = BeautifulSoup(guide_response.content, 'html.parser')
                    
                    pdf_link = None
                    for pdf in guide_soup.select('a[href$=".pdf"]'):
                        pdf_link = pdf.get('href')
                        if pdf_link:
                            if not pdf_link.startswith('http'):
                                pdf_link = base_url + pdf_link
                            break
                    
                    if pdf_link:
                        pdf_response = requests.get(pdf_link)
                        if pdf_response.status_code == 200:
                            clean_title = ''.join(c if c.isalnum() else '_' for c in title)
                            filename = os.path.join(output_dir, f"AHA_{clean_title}.pdf")
                            
                            with open(filename, 'wb') as f:
                                f.write(pdf_response.content)
                            
                            guideline['filename'] = filename
                            guideline['downloaded'] = True
                            guideline['evidence_quality'] = extract_evidence_quality(pdf_response.content)
                            
                            print(f"Downloaded AHA guideline: {title}")
                        else:
                            guideline['downloaded'] = False
                    else:
                        guideline['downloaded'] = False
                except Exception as e:
                    print(f"Error downloading AHA guideline PDF {title}: {e}")
                    guideline['downloaded'] = False
        
        # Save metadata
        metadata_df = pd.DataFrame(guidelines)
        metadata_df.to_csv(os.path.join(output_dir, 'aha_metadata.csv'), index=False)
        
        return guidelines
    
    except Exception as e:
        print(f"Error downloading AHA guidelines: {e}")
        return []

def download_nccn_guidelines(output_dir, limit=5):
    """Download clinical practice guidelines from National Comprehensive Cancer Network
    
    Note: NCCN requires registration and login for direct downloads.
    This function will use publicly available summaries instead.
    """
    base_url = "https://www.nccn.org"
    search_url = f"{base_url}/guidelines/category_1"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a session to handle cookies
        session = requests.Session()
        response = session.get(search_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        guidelines = []
        
        # Find guideline links (simplified approach)
        for link in soup.select('a[href*="guidelines"]')[:limit]:
            title = link.text.strip()
            if not title or len(title) < 10:  # Skip short/empty titles
                continue
                
            url = link.get('href')
            if not url.startswith('http'):
                url = base_url + url
                
            # Create simulated guideline with available information
            guideline = {
                'title': title,
                'url': url,
                'source': 'NCCN',
                'downloaded': False,  # Since we can't directly download PDFs
                'summary': f"NCCN Guideline for {title}. Full text requires registration at nccn.org."
            }
            
            guidelines.append(guideline)
            
            # For NCCN, we'll create text summaries since PDFs require login
            clean_title = ''.join(c if c.isalnum() else '_' for c in title)
            filename = os.path.join(output_dir, f"NCCN_{clean_title}.txt")
            
            with open(filename, 'w') as f:
                f.write(f"NCCN Guideline: {title}\n")
                f.write(f"Source URL: {url}\n")
                f.write("Note: Full NCCN guidelines require registration.\n")
                f.write("\nSummary:\n")
                f.write(f"This is a placeholder for NCCN {title} guideline summary.\n")
                f.write("In a production environment, you would need to implement proper authentication\n")
                f.write("or use NCCN API access with proper credentials to download actual guidelines.\n")
            
            guideline['filename'] = filename
            print(f"Created NCCN guideline placeholder: {title}")
        
        # Save metadata
        metadata_df = pd.DataFrame(guidelines)
        metadata_df.to_csv(os.path.join(output_dir, 'nccn_metadata.csv'), index=False)
        
        return guidelines
    
    except Exception as e:
        print(f"Error processing NCCN guidelines: {e}")
        return []

# 2. Biomedical Literature Functions

def download_pubmed_articles(output_dir, query="clinical trial hypertension", max_results=20, api_key=None):
    """
    Download PubMed abstracts using NCBI E-utilities
    
    Args:
        output_dir: Directory to save articles
        query: PubMed search query
        max_results: Maximum number of articles to download
        api_key: NCBI API key (optional, allows higher request rate)
    """
    # NCBI E-utilities base URLs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Search for articles
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance'
        }
        
        search_response = requests.get(esearch_url, params=search_params)
        search_data = search_response.json()
        
        if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
            print(f"No results found for query: {query}")
            return []
        
        article_ids = search_data['esearchresult']['idlist']
        print(f"Found {len(article_ids)} PubMed articles for query: {query}")
        
        # Create a list to store metadata
        articles = []
        
        # Fetch articles in batches to avoid server limitations
        batch_size = 5
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i+batch_size]
            
            # Construct fetch request
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(batch_ids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            fetch_response = requests.get(efetch_url, params=fetch_params)
            
            if fetch_response.status_code != 200:
                print(f"Error fetching articles: {fetch_response.status_code}")
                continue
            
            # Process the XML
            root = ET.fromstring(fetch_response.content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    # Extract article ID
                    pmid = article_elem.find('.//PMID').text
                    
                    # Extract title
                    title_elem = article_elem.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    # Extract abstract
                    abstract_text = ""
                    abstract_elems = article_elem.findall('.//AbstractText')
                    for abstract_elem in abstract_elems:
                        # Check if the abstract has labeled sections
                        label = abstract_elem.get('Label')
                        if label:
                            abstract_text += f"{label}: "
                        abstract_text += abstract_elem.text if abstract_elem.text else ""
                        abstract_text += "\n"
                    
                    # Extract publication date
                    pub_date = "Unknown"
                    pub_date_elem = article_elem.find('.//PubDate')
                    if pub_date_elem is not None:
                        year = pub_date_elem.find('Year')
                        month = pub_date_elem.find('Month')
                        day = pub_date_elem.find('Day')
                        
                        if year is not None:
                            pub_date = year.text
                            if month is not None:
                                pub_date = f"{month.text} {pub_date}"
                                if day is not None:
                                    pub_date = f"{pub_date} {day.text}"
                    
                    # Extract journal info
                    journal = "Unknown Journal"
                    journal_elem = article_elem.find('.//Journal/Title')
                    if journal_elem is not None:
                        journal = journal_elem.text
                    
                    # Extract authors
                    authors = []
                    author_elems = article_elem.findall('.//Author')
                    for author_elem in author_elems:
                        last_name = author_elem.find('LastName')
                        fore_name = author_elem.find('ForeName')
                        if last_name is not None:
                            author_name = last_name.text
                            if fore_name is not None:
                                author_name = f"{fore_name.text} {author_name}"
                            authors.append(author_name)
                    
                    # Extract article type/publication type
                    pub_types = []
                    pub_type_elems = article_elem.findall('.//PublicationType')
                    for pub_type_elem in pub_type_elems:
                        if pub_type_elem.text:
                            pub_types.append(pub_type_elem.text)
                    
                    # Create article metadata
                    article = {
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract_text,
                        'journal': journal,
                        'publication_date': pub_date,
                        'authors': ', '.join(authors),
                        'publication_types': ', '.join(pub_types),
                        'query': query
                    }
                    
                    # Save to file
                    filename = os.path.join(output_dir, f"pubmed_{pmid}.txt")
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {title}\n\n")
                        f.write(f"Authors: {article['authors']}\n")
                        f.write(f"Journal: {journal}\n")
                        f.write(f"Publication Date: {pub_date}\n")
                        f.write(f"Publication Types: {article['publication_types']}\n")
                        f.write(f"PMID: {pmid}\n\n")
                        f.write("Abstract:\n")
                        f.write(abstract_text)
                    
                    article['filename'] = filename
                    articles.append(article)
                    
                    print(f"Saved PubMed article: {pmid} - {title[:50]}...")
                    
                except Exception as e:
                    print(f"Error processing article: {e}")
            
            # Be kind to the NCBI server
            time.sleep(1)
        
        # Save metadata
        metadata_df = pd.DataFrame(articles)
        metadata_df.to_csv(os.path.join(output_dir, 'pubmed_metadata.csv'), index=False)
        
        return articles
    
    except Exception as e:
        print(f"Error downloading PubMed articles: {e}")
        return []

def download_pmc_articles(output_dir, query="systematic review diabetes", max_results=10):
    """Download PubMed Central full-text articles"""
    # NCBI E-utilities base URLs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Search for articles
        search_params = {
            'db': 'pmc',
            'term': query + " AND free full text[filter]",  # Only free full-text articles
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance'
        }
        
        search_response = requests.get(esearch_url, params=search_params)
        search_data = search_response.json()
        
        if 'esearchresult' not in search_data or 'idlist' not in search_data['esearchresult']:
            print(f"No PMC results found for query: {query}")
            return []
        
        article_ids = search_data['esearchresult']['idlist']
        print(f"Found {len(article_ids)} PMC articles for query: {query}")
        
        # Create a list to store metadata
        articles = []
        
        # Fetch articles in batches
        batch_size = 2  # Smaller batch size for full-text articles
        for i in range(0, len(article_ids), batch_size):
            batch_ids = article_ids[i:i+batch_size]
            
            # Construct fetch request for full-text
            fetch_params = {
                'db': 'pmc',
                'id': ','.join(batch_ids),
                'retmode': 'xml',
                'rettype': 'full'
            }
            
            fetch_response = requests.get(efetch_url, params=fetch_params)
            
            if fetch_response.status_code != 200:
                print(f"Error fetching PMC articles: {fetch_response.status_code}")
                continue
            
            # Save full XML for processing
            for pmcid in batch_ids:
                filename = os.path.join(output_dir, f"pmc_{pmcid}.xml")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(fetch_response.text)
                
                # Extract basic metadata for the CSV
                try:
                    root = ET.fromstring(fetch_response.content)
                    article_elem = root.find('.//article')
                    
                    if article_elem is None:
                        print(f"No article element found for PMC ID {pmcid}")
                        continue
                    
                    # Extract title
                    title = "Unknown Title"
                    title_elem = article_elem.find('.//article-title')
                    if title_elem is not None and title_elem.text:
                        title = ''.join(title_elem.itertext())
                    
                    # Extract journal
                    journal = "Unknown Journal"
                    journal_elem = article_elem.find('.//journal-title')
                    if journal_elem is not None and journal_elem.text:
                        journal = journal_elem.text
                    
                    # Create article metadata
                    article = {
                        'pmcid': pmcid,
                        'title': title,
                        'journal': journal,
                        'filename': filename,
                        'query': query
                    }
                    
                    articles.append(article)
                    print(f"Saved PMC article: {pmcid} - {title[:50]}...")
                    
                except Exception as e:
                    print(f"Error processing PMC article {pmcid}: {e}")
            
            # Be kind to the NCBI server
            time.sleep(2)
        
        # Save metadata
        metadata_df = pd.DataFrame(articles)
        metadata_df.to_csv(os.path.join(output_dir, 'pmc_metadata.csv'), index=False)
        
        return articles
    
    except Exception as e:
        print(f"Error downloading PMC articles: {e}")
        return []

# 3. Electronic Health Record Data Functions

def download_mimic_demo_data(output_dir, physionet_user=None, physionet_pass=None):
    """
    Download MIMIC demo data from PhysioNet
    
    Args:
        output_dir: Directory to save the data
        physionet_user: PhysioNet username (optional)
        physionet_pass: PhysioNet password (optional)
    
    Note: Full MIMIC dataset requires credentialing and approval.
    This function will download the official demo dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # MIMIC demo data URL - this is the actual demo dataset
    demo_url = "https://physionet.org/files/mimiciii-demo/1.4/"
    
    print("Attempting to download official MIMIC-III demo dataset...")
    
    # Check if we have credentials
    if physionet_user and physionet_pass:
        try:
            # Create a session with authentication
            session = requests.Session()
            login_url = "https://physionet.org/login/"
            
            # First get the CSRF token
            login_page = session.get(login_url)
            soup = BeautifulSoup(login_page.content, 'html.parser')
            csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'}).get('value')
            
            # Login
            login_data = {
                'username': physionet_user,
                'password': physionet_pass,
                'csrfmiddlewaretoken': csrf_token,
                'next': '/'
            }
            
            login_response = session.post(login_url, data=login_data, 
                                         headers={'Referer': login_url})
            
            if login_response.url != "https://physionet.org/":
                print("PhysioNet login failed. Please check your credentials.")
                raise Exception("Authentication failed")
                
            print("Successfully authenticated with PhysioNet")
            
            # Download the metadata file first to get file list
            files_url = demo_url + "files.csv"
            files_response = session.get(files_url)
            
            if files_response.status_code != 200:
                raise Exception(f"Failed to download files list: {files_response.status_code}")
            
            # Save the files list
            files_path = os.path.join(output_dir, "files.csv")
            with open(files_path, 'wb') as f:
                f.write(files_response.content)
            
            # Parse the files list
            with open(files_path, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header
                files_to_download = [row[0] for row in csv_reader if row[0].endswith('.csv.gz')]
            
            print(f"Found {len(files_to_download)} files to download")
            
            # Download each file
            for file_name in files_to_download:
                file_url = demo_url + file_name
                print(f"Downloading {file_name}...")
                
                file_response = session.get(file_url)
                if file_response.status_code != 200:
                    print(f"Failed to download {file_name}: {file_response.status_code}")
                    continue
                
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                
                # If it's a gzipped file, extract it
                if file_name.endswith('.gz'):
                    try:
                        with gzip.open(file_path, 'rb') as f_in:
                            with open(file_path[:-3], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        print(f"Extracted {file_name}")
                    except Exception as e:
                        print(f"Failed to extract {file_name}: {e}")
            
            print("MIMIC-III demo dataset download complete")
            return {
                'source': 'PhysioNet official',
                'files_downloaded': len(files_to_download)
            }
            
        except Exception as e:
            print(f"Error downloading MIMIC data: {e}")
            print("Falling back to direct download links...")
    
    # If we don't have credentials or the authenticated download failed,
    # try downloading specific CSV files that might be publicly accessible
    try:
        # Direct links to some demo files (these might work without authentication)
        demo_files = [
            'ADMISSIONS.csv.gz',
            'PATIENTS.csv.gz',
            'NOTEEVENTS.csv.gz'
        ]
        
        for file_name in demo_files:
            file_url = demo_url + file_name
            print(f"Trying direct download for {file_name}...")
            
            file_response = requests.get(file_url)
            if file_response.status_code != 200:
                print(f"Direct download failed for {file_name}: {file_response.status_code}")
                continue
            
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
            
            # Extract if it's a gzipped file
            if file_name.endswith('.gz'):
                try:
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(file_path[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Extracted {file_name}")
                except Exception as e:
                    print(f"Failed to extract {file_name}: {e}")
        
        print("Direct download attempt completed.")
        return {
            'source': 'Direct download',
            'files_downloaded': len(demo_files)
        }
        
    except Exception as e:
        print(f"Direct download failed: {e}")
        print("Falling back to creating placeholder files for development...")

    
    # Create ADMISSIONS table
    admissions_data = []
    for i in range(100):
        admission = {
            'SUBJECT_ID': i + 1000,
            'HADM_ID': i + 5000,
            'ADMITTIME': f"2150-{(i % 12) + 1:02d}-{(i % 28) + 1:02d} 10:00:00",
            'DISCHTIME': f"2150-{(i % 12) + 1:02d}-{(i % 28) + 3:02d} 15:30:00",
            'ADMISSION_TYPE': random.choice(['EMERGENCY', 'ELECTIVE', 'URGENT']),
            'DIAGNOSIS': random.choice(['Pneumonia', 'Sepsis', 'Heart Failure', 'COPD', 'Diabetes']),
        }
        admissions_data.append(admission)
    
    admissions_df = pd.DataFrame(admissions_data)
    admissions_df.to_csv(os.path.join(output_dir, 'ADMISSIONS.csv'), index=False)
    
    # Create PATIENTS table
    patients_data = []
    for i in range(100):
        patient = {
            'SUBJECT_ID': i + 1000,
            'GENDER': random.choice(['M', 'F']),
            'DOB': f"2080-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            'DOD': None if i % 5 else f"2151-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
        }
        patients_data.append(patient)
    
    patients_df = pd.DataFrame(patients_data)
    patients_df.to_csv(os.path.join(output_dir, 'PATIENTS.csv'), index=False)
    
    # Create NOTEEVENTS table with synthetic clinical notes
    noteevents_data = []
    note_types = ['Discharge Summary', 'Physician ', 'Nursing', 'ECG', 'Radiology']
    
    for i in range(200):
        subject_id = (i % 100) + 1000
        hadm_id = (i % 100) + 5000
        
        note_type = note_types[i % len(note_types)]
        
        if note_type == 'Discharge Summary':
            text = f"""Admission Date: [**Date**]
Discharge Date: [**Date**]

HISTORY OF PRESENT ILLNESS:
This is a {random.randint(50, 90)} year old patient with a history of {random.choice(['diabetes', 'hypertension', 'coronary artery disease'])} who presented with {random.choice(['shortness of breath', 'chest pain', 'fever', 'abdominal pain'])}.

HOSPITAL COURSE:
Patient was treated with {random.choice(['antibiotics', 'insulin', 'antihypertensives', 'pain management'])}.

MEDICATIONS ON ADMISSION:
1. {random.choice(['Metformin', 'Lisinopril', 'Aspirin', 'Atorvastatin'])} {random.choice(['500mg', '10mg', '81mg', '20mg'])} {random.choice(['daily', 'twice daily', 'as needed'])}

DISCHARGE MEDICATIONS:
1. {random.choice(['Metformin', 'Lisinopril', 'Aspirin', 'Atorvastatin'])} {random.choice(['500mg', '10mg', '81mg', '20mg'])} {random.choice(['daily', 'twice daily', 'as needed'])}

DISCHARGE DIAGNOSIS:
1. {random.choice(['Pneumonia', 'Sepsis', 'Heart Failure', 'COPD exacerbation', 'Diabetic ketoacidosis'])}
"""
        else:
            text = f"Clinical note for patient ID {subject_id}. This is a synthetic note for demonstration purposes."
        
        note = {
            'SUBJECT_ID': subject_id,
            'HADM_ID': hadm_id,
            'CHARTTIME': f"2150-{(i % 12) + 1:02d}-{(i % 28) + 1:02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
            'CATEGORY': 'Discharge',
            'DESCRIPTION': note_type,
            'TEXT': text
        }
        noteevents_data.append(note)
    
    noteevents_df = pd.DataFrame(noteevents_data)
    noteevents_df.to_csv(os.path.join(output_dir, 'NOTEEVENTS.csv'), index=False)
    
    # Create README file
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write("MIMIC-III Demo Dataset\n")
        f.write("=====================\n\n")
        f.write("This is a simulated demo version of the MIMIC-III dataset for development purposes.\n")
        f.write("The actual MIMIC-III dataset requires credentialing through PhysioNet.\n\n")
        f.write("Tables included:\n")
        f.write("- ADMISSIONS: 100 records\n")
        f.write("- PATIENTS: 100 records\n")
        f.write("- NOTEEVENTS: 200 records\n\n")
        f.write("For more information about MIMIC-III, visit: https://physionet.org/content/mimiciii/\n")
    
    print(f"Created simulated MIMIC demo data in {output_dir}")
    print("ADMISSIONS: 100 records")
    print("PATIENTS: 100 records")
    print("NOTEEVENTS: 200 records")
    
    return {
        'admissions': len(admissions_data),
        'patients': len(patients_data),
        'noteevents': len(noteevents_data)
    }

def download_i2b2_sample_data(output_dir, i2b2_credentials=None):
    """
    Download i2b2/n2c2 sample data from official sources
    
    Args:
        output_dir: Directory to save the data
        i2b2_credentials: Dictionary with 'username' and 'password' for i2b2 (optional)
    
    Note: Actual i2b2/n2c2 data requires application and approval.
    This function attempts to download public sample data where available.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to download public sample data from n2c2
    print("Attempting to download public n2c2/i2b2 data...")
    
    # Some n2c2 challenges make sample data publicly available
    # 2018 cohort selection challenge sample: https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t1/
    sample_url = "https://portal.dbmi.hms.harvard.edu/data/n2c2-2018-track1-download/n2c2-t1_gold_standard_test_data.tar.gz"
    
    try:
        print(f"Downloading sample data from {sample_url}...")
        response = requests.get(sample_url)
        
        if response.status_code == 200:
            # Save the file
            tar_path = os.path.join(output_dir, "n2c2-sample.tar.gz")
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the files
            try:
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=output_dir)
                print("Successfully downloaded and extracted n2c2 sample data")
                
                # Count files to report statistics
                xml_count = len(glob.glob(os.path.join(output_dir, "**/*.xml"), recursive=True))
                txt_count = len(glob.glob(os.path.join(output_dir, "**/*.txt"), recursive=True))
                
                return {
                    'source': 'n2c2 public sample',
                    'xml_files': xml_count,
                    'txt_files': txt_count
                }
            except Exception as e:
                print(f"Failed to extract tar file: {e}")
        else:
            print(f"Failed to download sample data: {response.status_code}")
    
    except Exception as e:
        print(f"Error accessing n2c2 sample data: {e}")
    
    # If we have credentials, try to use them
    if i2b2_credentials and 'username' in i2b2_credentials and 'password' in i2b2_credentials:
        try:
            print("Attempting to use i2b2 credentials to download data...")
            # This would require implementing a specific login flow for the i2b2 portal
            # Since each i2b2/n2c2 challenge has its own access mechanism, this is complex
            
            print("Note: Automated download with credentials requires challenge-specific implementation.")
            print("Please download the data manually from the n2c2/i2b2 portal.")
        except Exception as e:
            print(f"Error using i2b2 credentials: {e}")
    
    print("Note: Full i2b2/n2c2 datasets require application and approval.")
    print("Some datasets can be accessed at: https://portal.dbmi.hms.harvard.edu/")
    print("Creating i2b2/n2c2 format sample data for development...")
    
    # Create sample files in i2b2 format
    sample_size = 20
    
    # Sample clinical entities for the synthetic data
    medications = [
        "Aspirin", "Lisinopril", "Metoprolol", "Furosemide", "Atorvastatin",
        "Metformin", "Insulin", "Albuterol", "Prednisone", "Warfarin"
    ]
    
    dosages = ["10mg", "20mg", "40mg", "50mg", "100mg", "500mg", "1000mg"]
    frequencies = ["daily", "BID", "TID", "QID", "weekly", "as needed", "every morning"]
    
    conditions = [
        "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease", "Congestive Heart Failure",
        "Asthma", "COPD", "Atrial Fibrillation", "Depression", "Hyperlipidemia", "Chronic Kidney Disease"
    ]
    
    # Create synthetic clinical notes in i2b2 format
    for i in range(sample_size):
        patient_id = f"P{i+1:03d}"
        record_id = f"R{i+1:03d}"
        
        # Create directory for patient
        patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Select random medications and conditions for this patient
        patient_meds = random.sample(medications, random.randint(1, 4))
        patient_conditions = random.sample(conditions, random.randint(1, 3))
        
        # Create clinical note
        note_text = f"""Patient Record: {record_id}
Date: 2022-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}

HISTORY OF PRESENT ILLNESS:
The patient is a {random.randint(30, 85)} year-old {random.choice(['male', 'female'])} with a history of {', '.join(patient_conditions)} 
who presents with {random.choice(['shortness of breath', 'chest pain', 'fatigue', 'dizziness', 'fever'])}.

MEDICATIONS:
"""
        
        for med in patient_meds:
            dosage = random.choice(dosages)
            frequency = random.choice(frequencies)
            note_text += f"- {med} {dosage} {frequency}\n"
        
        note_text += f"""
ASSESSMENT AND PLAN:
1. {patient_conditions[0]}: {random.choice(['Continue current management', 'Increase medication dosage', 'Add new medication', 'Monitor closely'])}
2. Follow up in {random.randint(2, 8)} weeks
"""
        
        # Save the note
        with open(os.path.join(patient_dir, f"{record_id}.txt"), 'w') as f:
            f.write(note_text)
        
        # Create XML annotation file (simulating i2b2 annotations)
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<root>
  <patient id="{patient_id}">
    <record id="{record_id}">
      <annotations>
"""
        
        # Add medication annotations
        for med in patient_meds:
            xml_content += f"""        <annotation type="MEDICATION">
          <text>{med}</text>
          <properties>
            <dosage>{random.choice(dosages)}</dosage>
            <frequency>{random.choice(frequencies)}</frequency>
          </properties>
        </annotation>
"""
        
        # Add problem annotations
        for condition in patient_conditions:
            xml_content += f"""        <annotation type="PROBLEM">
          <text>{condition}</text>
        </annotation>
"""
        
        xml_content += """      </annotations>
    </record>
  </patient>
</root>
"""
        
        # Save the XML
        with open(os.path.join(patient_dir, f"{record_id}.xml"), 'w') as f:
            f.write(xml_content)
    
    # Create README
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write("i2b2/n2c2 Sample Dataset\n")
        f.write("=====================\n\n")
        f.write(f"This is a simulated sample of {sample_size} patient records in i2b2/n2c2 format.\n")
        f.write("Each patient directory contains a clinical note (.txt) and annotations (.xml).\n\n")
        f.write("The actual i2b2/n2c2 datasets require application and approval.\n")
        f.write("For more information, visit: https://n2c2.dbmi.hms.harvard.edu/\n")
    
    print(f"Created simulated i2b2/n2c2 sample data: {sample_size} patient records")
    
    return {
        'patient_count': sample_size,
        'record_count': sample_size,
        'format': 'i2b2'
    }

# 4. Medical Images Functions

def download_mimic_cxr_sample(output_dir, physionet_user=None, physionet_pass=None):
    """
    Download MIMIC-CXR sample data from PhysioNet
    
    Args:
        output_dir: Directory to save the data
        physionet_user: PhysioNet username (for authentication)
        physionet_pass: PhysioNet password (for authentication)
    
    Note: Full MIMIC-CXR dataset requires credentialing and approval.
    This function will attempt to download the small sample dataset available to credentialed users.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    
    # MIMIC-CXR URL
    mimic_cxr_url = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
    
    print("Attempting to download MIMIC-CXR sample data...")
    
    # Check if we have credentials
    if physionet_user and physionet_pass:
        try:
            # Create a session with authentication
            session = requests.Session()
            login_url = "https://physionet.org/login/"
            
            # First get the CSRF token
            login_page = session.get(login_url)
            soup = BeautifulSoup(login_page.content, 'html.parser')
            csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'}).get('value')
            
            # Login
            login_data = {
                'username': physionet_user,
                'password': physionet_pass,
                'csrfmiddlewaretoken': csrf_token,
                'next': '/'
            }
            
            login_response = session.post(login_url, data=login_data, 
                                         headers={'Referer': login_url})
            
            if login_response.url != "https://physionet.org/":
                print("PhysioNet login failed. Please check your credentials.")
                raise Exception("Authentication failed")
                
            print("Successfully authenticated with PhysioNet")
            
            # Download metadata file first
            metadata_url = mimic_cxr_url + "mimic-cxr-2.0.0-metadata.csv.gz"
            print(f"Downloading metadata file...")
            
            metadata_response = session.get(metadata_url)
            if metadata_response.status_code != 200:
                raise Exception(f"Failed to download metadata: {metadata_response.status_code}")
            
            metadata_path = os.path.join(output_dir, "mimic-cxr-metadata.csv.gz")
            with open(metadata_path, 'wb') as f:
                f.write(metadata_response.content)
            
            # Extract metadata
            try:
                with gzip.open(metadata_path, 'rb') as f_in:
                    with open(metadata_path[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("Extracted metadata file")
            except Exception as e:
                print(f"Failed to extract metadata file: {e}")
            
            # Download a small sample of images (first 20 files from the first patient)
            # This is just a demonstration - real usage would download more systematically
            sample_dir = "files/p10/p10000032/s50414267/"
            sample_url = mimic_cxr_url + sample_dir
            
            # First get the directory listing
            dir_response = session.get(sample_url)
            if dir_response.status_code != 200:
                raise Exception(f"Failed to access sample directory: {dir_response.status_code}")
            
            # Parse the HTML to find image files
            soup = BeautifulSoup(dir_response.content, 'html.parser')
            image_links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.jpg')]
            
            print(f"Found {len(image_links)} images to download")
            
            # Download each image (limit to 5 for demonstration)
            image_count = 0
            for image_link in image_links[:5]:
                image_url = sample_url + image_link
                print(f"Downloading {image_link}...")
                
                image_response = session.get(image_url)
                if image_response.status_code != 200:
                    print(f"Failed to download {image_link}: {image_response.status_code}")
                    continue
                
                # Create the directory structure
                patient_dir = os.path.join(output_dir, 'images', 'p10', 'p10000032', 's50414267')
                os.makedirs(patient_dir, exist_ok=True)
                
                image_path = os.path.join(patient_dir, image_link)
                with open(image_path, 'wb') as f:
                    f.write(image_response.content)
                
                image_count += 1
            
            # Download reports
            reports_url = mimic_cxr_url + "files/mimic-cxr-reports.zip"
            print("Downloading reports file...")
            
            reports_response = session.get(reports_url)
            if reports_response.status_code != 200:
                print(f"Failed to download reports: {reports_response.status_code}")
            else:
                reports_path = os.path.join(output_dir, "mimic-cxr-reports.zip")
                with open(reports_path, 'wb') as f:
                    f.write(reports_response.content)
                
                # Extract reports
                try:
                    with zipfile.ZipFile(reports_path, 'r') as zip_ref:
                        # Extract only files for our sample patient
                        for file in zip_ref.namelist():
                            if 'p10000032' in file:
                                zip_ref.extract(file, os.path.join(output_dir, 'reports'))
                    print("Extracted sample reports")
                except Exception as e:
                    print(f"Failed to extract reports: {e}")
            
            print("MIMIC-CXR sample download complete")
            return {
                'source': 'PhysioNet official',
                'images_downloaded': image_count,
                'has_reports': True
            }
            
        except Exception as e:
            print(f"Error downloading MIMIC-CXR data: {e}")
            print("Unable to download MIMIC-CXR files, authentication required.")
    else:
        print("PhysioNet credentials required to download MIMIC-CXR data.")
        print("Please provide PhysioNet username and password.")
    
    # Create sample findings for reports
    findings = [
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly without overt heart failure.",
        "Clear lungs bilaterally. No pleural effusion or pneumothorax.",
        "Bibasilar atelectasis. No pneumothorax or large pleural effusion.",
        "Hyperinflated lungs suggesting COPD. No acute infiltrate.",
        "Right middle lobe opacity consistent with pneumonia.",
        "Mild pulmonary edema with small bilateral pleural effusions.",
        "Left lower lobe consolidation concerning for pneumonia.",
        "Stable bilateral pulmonary nodules. No pleural effusion.",
        "Elevated right hemidiaphragm. No acute infiltrate."
    ]
    
    impressions = [
        "Normal chest radiograph.",
        "Stable cardiomegaly.",
        "No acute cardiopulmonary process.",
        "Bibasilar atelectasis, otherwise clear.",
        "COPD without acute disease.",
        "Right middle lobe pneumonia.",
        "Mild congestive heart failure.",
        "Left lower lobe pneumonia.",
        "Stable pulmonary nodules. Recommend follow-up CT.",
        "Elevated right hemidiaphragm, possibly due to phrenic nerve paralysis."
    ]
    
    # Create metadata CSV
    metadata = []
    
    for i in range(sample_size):
        subject_id = f"S{i+1000:04d}"
        study_id = f"STD{i+5000:05d}"
        
        finding_idx = i % len(findings)
        finding = findings[finding_idx]
        impression = impressions[finding_idx]
        
        # Create report
        report_text = f"""CHEST RADIOGRAPH

CLINICAL INFORMATION:
{random.choice(['Shortness of breath', 'Chest pain', 'Fever', 'Follow-up', 'Post-operative check'])}

TECHNIQUE:
PA and lateral chest radiograph.

FINDINGS:
{finding}

IMPRESSION:
{impression}
"""
        
        report_path = os.path.join(output_dir, 'reports', f"{study_id}.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Create a simulated image file (empty file with proper extension)
        # In a real scenario, you would download actual DICOM or PNG files
        image_path = os.path.join(output_dir, 'images', f"{study_id}.dcm")
        with open(image_path, 'w') as f:
            f.write("THIS IS A PLACEHOLDER FOR A CHEST X-RAY DICOM FILE")
        
        # Add to metadata
        metadata.append({
            'subject_id': subject_id,
            'study_id': study_id,
            'finding': finding,
            'impression': impression,
            'report_path': report_path,
            'image_path': image_path
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'mimic_cxr_metadata.csv'), index=False)
    
    # Create README
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write("MIMIC-CXR Sample Dataset\n")
        f.write("=====================\n\n")
        f.write(f"This is a simulated sample of {sample_size} chest X-ray records based on MIMIC-CXR format.\n")
        f.write("Each record includes a report and a placeholder for an image file.\n\n")
        f.write("The actual MIMIC-CXR dataset requires credentialing through PhysioNet.\n")
        f.write("For more information, visit: https://physionet.org/content/mimic-cxr/\n")
    
    print(f"Created simulated MIMIC-CXR sample data: {sample_size} records")
    
    return {
        'record_count': sample_size,
        'format': 'MIMIC-CXR'
    }

def download_other_medical_images(output_dir):
    """
    Download sample medical images from public repositories
    
    Args:
        output_dir: Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    
    print("Attempting to download medical images from public repositories...")
    
    # List of public medical image datasets with direct download links
    public_image_sources = [
        {
            'name': 'MedMNIST',
            'url': 'https://github.com/MedMNIST/MedMNIST/raw/main/examples/medmnist.zip',
            'type': 'zip'
        },
        {
            'name': 'COVID-19 Radiography Database',
            'url': 'https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/download?datasetVersionNumber=5',
            'type': 'kaggle'
        },
        {
            'name': 'OASIS Brain MRI',
            'url': 'https://www.oasis-brains.org/files/oasis_cross-sectional_disc1.tar.gz',
            'type': 'tar.gz'
        }
    ]
    
    downloaded_datasets = []
    
    # Try to download each dataset
    for source in public_image_sources:
        try:
            print(f"Attempting to download {source['name']}...")
            
            # Create directory for this dataset
            dataset_dir = os.path.join(output_dir, source['name'].lower().replace(' ', '_'))
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download based on type
            if source['type'] == 'zip' or source['type'] == 'tar.gz':
                response = requests.get(source['url'], stream=True)
                
                if response.status_code == 200:
                    # Get the file path
                    if source['type'] == 'zip':
                        file_path = os.path.join(dataset_dir, "dataset.zip")
                    else:
                        file_path = os.path.join(dataset_dir, "dataset.tar.gz")
                    
                    # Download the file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"Downloaded {source['name']} dataset")
                    
                    # Extract the file
                    try:
                        if source['type'] == 'zip':
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(dataset_dir)
                        else:
                            with tarfile.open(file_path, 'r:gz') as tar:
                                tar.extractall(path=dataset_dir)
                        
                        print(f"Extracted {source['name']} dataset")
                        
                        # Count files
                        image_count = len(glob.glob(os.path.join(dataset_dir, "**/*.jpg"), recursive=True))
                        image_count += len(glob.glob(os.path.join(dataset_dir, "**/*.png"), recursive=True))
                        image_count += len(glob.glob(os.path.join(dataset_dir, "**/*.dcm"), recursive=True))
                        
                        downloaded_datasets.append({
                            'name': source['name'],
                            'image_count': image_count,
                            'path': dataset_dir
                        })
                    except Exception as e:
                        print(f"Failed to extract {source['name']} dataset: {e}")
                else:
                    print(f"Failed to download {source['name']}: {response.status_code}")
            
            elif source['type'] == 'kaggle':
                print(f"To download {source['name']} from Kaggle, please use the Kaggle API or download manually")
                print(f"Instructions: https://www.kaggle.com/docs/api")
        
        except Exception as e:
            print(f"Error downloading {source['name']}: {e}")
    
    # If we couldn't download any datasets, try fetching from another public source
    if not downloaded_datasets:
        try:
            print("Attempting to download a few sample medical images from public repositories...")
            
            # Try OpenNeuro dataset (contains neuroimaging data)
            openneuro_url = "https://openneuro.org/api/datasets/ds000113/snapshots/1.0.0/files/sub-01:anat:sub-01_T1w.nii.gz"
            
            response = requests.get(openneuro_url)
            if response.status_code == 200:
                openneuro_dir = os.path.join(output_dir, 'openneuro')
                os.makedirs(openneuro_dir, exist_ok=True)
                
                file_path = os.path.join(openneuro_dir, "brain_mri.nii.gz")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_datasets.append({
                    'name': 'OpenNeuro Sample',
                    'image_count': 1,
                    'path': openneuro_dir
                })
            else:
                print(f"Failed to download from OpenNeuro: {response.status_code}")
        except Exception as e:
            print(f"Error downloading from OpenNeuro: {e}")
    
    if downloaded_datasets:
        print(f"Successfully downloaded {len(downloaded_datasets)} medical image datasets")
        for dataset in downloaded_datasets:
            print(f"  - {dataset['name']}: {dataset['image_count']} images")
        
        return {
            'source': 'Public repositories',
            'datasets': downloaded_datasets,
            'total_images': sum(d['image_count'] for d in downloaded_datasets)
        }
    else:
        print("Failed to download any medical image datasets from public repositories")
        print("Creating minimal sample data for development...")
        
        # Define modalities and body parts as fallback
        modalities = ['MRI', 'CT', 'Ultrasound', 'X-ray']
        body_parts = ['Brain', 'Chest', 'Abdomen', 'Knee', 'Liver', 'Heart']
    
    # Create metadata
    metadata = []
    
    for i in range(sample_size):
        modality = random.choice(modalities)
        body_part = random.choice(body_parts)
        
        image_id = f"IMG{i+1:03d}"
        
        # Create a simulated image file
        image_path = os.path.join(output_dir, 'images', f"{image_id}_{modality}_{body_part}.dcm")
        with open(image_path, 'w') as f:
            f.write(f"THIS IS A PLACEHOLDER FOR A {modality} IMAGE OF THE {body_part}")
        
        # Create a metadata file
        metadata_text = f"""IMAGE METADATA

Image ID: {image_id}
Modality: {modality}
Body Part: {body_part}
Resolution: {random.choice(['512x512', '1024x1024', '256x256'])}
Bits: {random.choice(['8-bit', '12-bit', '16-bit'])}
Source: Simulated data
"""
        
        metadata_path = os.path.join(output_dir, 'metadata', f"{image_id}.txt")
        with open(metadata_path, 'w') as f:
            f.write(metadata_text)
        
        # Add to metadata list
        metadata.append({
            'image_id': image_id,
            'modality': modality,
            'body_part': body_part,
            'image_path': image_path,
            'metadata_path': metadata_path
        })
    
    # Save combined metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'other_images_metadata.csv'), index=False)
    
    # Create README
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write("Medical Images Sample Dataset\n")
        f.write("===========================\n\n")
        f.write(f"This is a simulated sample of {sample_size} medical images of various modalities.\n")
        f.write("Each record includes a placeholder for an image file and metadata.\n\n")
        f.write("In a real implementation, you would download from open repositories like:\n")
        f.write("- The Cancer Imaging Archive (TCIA)\n")
        f.write("- MedPix\n")
        f.write("- NIH Chest X-ray Dataset\n")
        f.write("- OASIS Brain MRI Dataset\n")
    
    print(f"Created simulated medical images sample data: {sample_size} records")
    
    return {
        'record_count': sample_size,
        'modalities': modalities,
        'body_parts': body_parts
    }

# 5. Combined Setup Function

def setup_all_data(base_dir=BASE_DATA_DIR, guideline_limit=None, pubmed_max=20, pmc_max=10,
                  credentials=None):
    """
    Set up all required datasets for the MC-RAG system
    
    Args:
        base_dir: Base directory for all data
        guideline_limit: Maximum number of guidelines to download per source (None for unlimited)
        pubmed_max: Maximum number of PubMed articles to download
        pmc_max: Maximum number of PMC articles to download
        credentials: Dictionary with credentials for various sources:
            {
                'physionet': {'username': '...', 'password': '...'},
                'i2b2': {'username': '...', 'password': '...'},
                'ncbi_api_key': '...'
            }
    """
    # Extract credentials
    physionet_user = None
    physionet_pass = None
    i2b2_creds = None
    ncbi_api_key = None
    
    if credentials:
        if 'physionet' in credentials:
            physionet_user = credentials['physionet'].get('username')
            physionet_pass = credentials['physionet'].get('password')
        if 'i2b2' in credentials:
            i2b2_creds = credentials['i2b2']
        if 'ncbi_api_key' in credentials:
            ncbi_api_key = credentials['ncbi_api_key']
    # Create directory structure
    create_data_directories()
    
    results = {
        'guidelines': {},
        'literature': {},
        'ehr_data': {},
        'images': {}
    }
    
    # 1. Download clinical guidelines
    print("\n=== Downloading Clinical Guidelines ===\n")
    
    nice_dir = os.path.join(GUIDELINES_DIR, "nice")
    results['guidelines']['nice'] = download_nice_guidelines(nice_dir, limit=guideline_limit)
    
    acp_dir = os.path.join(GUIDELINES_DIR, "acp")
    results['guidelines']['acp'] = download_acp_guidelines(acp_dir, limit=guideline_limit)
    
    aha_dir = os.path.join(GUIDELINES_DIR, "aha")
    results['guidelines']['aha'] = download_aha_guidelines(aha_dir, limit=guideline_limit)
    
    nccn_dir = os.path.join(GUIDELINES_DIR, "nccn")
    results['guidelines']['nccn'] = download_nccn_guidelines(nccn_dir, limit=guideline_limit)
    
    # 2. Download biomedical literature
    print("\n=== Downloading Biomedical Literature ===\n")
    
    pubmed_dir = os.path.join(LITERATURE_DIR, "pubmed")
    results['literature']['pubmed'] = download_pubmed_articles(
        pubmed_dir, 
        query="clinical trial hypertension", 
        max_results=pubmed_max,
        api_key=ncbi_api_key
    )
    
    # Add more PubMed queries for diversity
    results['literature']['pubmed_diabetes'] = download_pubmed_articles(
        pubmed_dir, 
        query="systematic review diabetes treatment", 
        max_results=pubmed_max // 2,
        api_key=ncbi_api_key
    )
    
    pmc_dir = os.path.join(LITERATURE_DIR, "pmc")
    results['literature']['pmc'] = download_pmc_articles(
        pmc_dir, 
        query="meta-analysis cardiovascular disease", 
        max_results=pmc_max,
        api_key=ncbi_api_key
    )
    
    # 3. Set up EHR data
    print("\n=== Setting Up Electronic Health Record Data ===\n")
    
    mimic_dir = os.path.join(EHR_DATA_DIR, "mimic")
    results['ehr_data']['mimic'] = download_mimic_demo_data(mimic_dir, physionet_user, physionet_pass)
    
    i2b2_dir = os.path.join(EHR_DATA_DIR, "i2b2")
    results['ehr_data']['i2b2'] = download_i2b2_sample_data(i2b2_dir, i2b2_creds)
    
    # 4. Set up medical images
    print("\n=== Setting Up Medical Images ===\n")
    
    mimic_cxr_dir = os.path.join(IMAGES_DIR, "mimic_cxr")
    results['images']['mimic_cxr'] = download_mimic_cxr_sample(mimic_cxr_dir, physionet_user, physionet_pass)
    
    # Try to download NIH Chest X-ray Dataset (publicly available)
    nih_cxr_dir = os.path.join(IMAGES_DIR, "nih_cxr")
    os.makedirs(nih_cxr_dir, exist_ok=True)
    
    try:
        print("Attempting to download sample from NIH Chest X-ray Dataset...")
        # Sample URL for NIH Chest X-ray Dataset (this is a public dataset)
        nih_sample_url = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
        
        response = requests.get(nih_sample_url, stream=True)
        if response.status_code == 200:
            # Get the total file size
            file_size = int(response.headers.get('content-length', 0))
            
            # Download with progress indication
            downloaded = 0
            sample_path = os.path.join(nih_cxr_dir, "nih_sample.tar.gz")
            
            with open(sample_path, 'wb') as f:
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    print(f"\rDownloading NIH sample: {downloaded}/{file_size} bytes", end="")
            
            print("\nExtracting NIH sample...")
            with tarfile.open(sample_path) as tar:
                tar.extractall(path=nih_cxr_dir)
            
            image_count = len(glob.glob(os.path.join(nih_cxr_dir, "**/*.png"), recursive=True))
            results['images']['nih_cxr'] = {
                'source': 'NIH Chest X-ray Dataset',
                'image_count': image_count
            }
            print(f"Downloaded {image_count} images from NIH Chest X-ray Dataset")
        else:
            print(f"Failed to download NIH dataset: {response.status_code}")
    except Exception as e:
        print(f"Error downloading NIH Chest X-ray Dataset: {e}")
    
    # Try to download a few samples from public medical image repositories
    other_images_dir = os.path.join(IMAGES_DIR, "other")
    results['images']['other'] = download_other_medical_images(other_images_dir)
    
    # Create overall summary
    summary = {
        'guidelines': {
            'total_count': sum(1 for source in results['guidelines'].values() for item in source),
            'sources': list(results['guidelines'].keys())
        },
        'literature': {
            'pubmed_count': len(results['literature'].get('pubmed', [])) + len(results['literature'].get('pubmed_diabetes', [])),
            'pmc_count': len(results['literature'].get('pmc', []))
        },
        'ehr_data': {
            'mimic': results['ehr_data'].get('mimic', {}),
            'i2b2': results['ehr_data'].get('i2b2', {})
        },
        'images': {
            'mimic_cxr_count': results['images'].get('mimic_cxr', {}).get('record_count', 0),
            'other_count': results['images'].get('other', {}).get('record_count', 0)
        }
    }
    
    # Save summary
    with open(os.path.join(BASE_DATA_DIR, 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Data Setup Complete ===\n")
    print(f"Data directory: {BASE_DATA_DIR}")
    print(f"Guidelines: {summary['guidelines']['total_count']} from {', '.join(summary['guidelines']['sources'])}")
    print(f"Literature: {summary['literature']['pubmed_count']} PubMed, {summary['literature']['pmc_count']} PMC")
    print(f"EHR data: MIMIC ({results['ehr_data']['mimic'].get('patients', 0)} patients), i2b2 ({results['ehr_data']['i2b2'].get('patient_count', 0)} patients)")
    print(f"Images: {summary['images']['mimic_cxr_count']} MIMIC-CXR, {summary['images']['other_count']} other")
    
    return summary

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download medical datasets for MC-RAG")
    parser.add_argument("--physionet-user", help="PhysioNet username")
    parser.add_argument("--physionet-pass", help="PhysioNet password")
    parser.add_argument("--i2b2-user", help="i2b2/n2c2 username")
    parser.add_argument("--i2b2-pass", help="i2b2/n2c2 password")
    parser.add_argument("--ncbi-api-key", help="NCBI API key")
    parser.add_argument("--guideline-limit", type=int, help="Number of guidelines to download (omit for unlimited)")
    parser.add_argument("--pubmed-max", type=int, default=50, help="Maximum PubMed articles")
    parser.add_argument("--pmc-max", type=int, default=20, help="Maximum PMC articles")
    
    args = parser.parse_args()
    
    # Set up credentials dictionary
    credentials = {}
    
    if args.physionet_user and args.physionet_pass:
        credentials['physionet'] = {
            'username': args.physionet_user,
            'password': args.physionet_pass
        }
    
    if args.i2b2_user and args.i2b2_pass:
        credentials['i2b2'] = {
            'username': args.i2b2_user,
            'password': args.i2b2_pass
        }
    
    if args.ncbi_api_key:
        credentials['ncbi_api_key'] = args.ncbi_api_key
    
    print("Starting data download with the following settings:")
    print(f"  Guidelines per source: {'Unlimited' if args.guideline_limit is None else args.guideline_limit}")
    print(f"  PubMed articles: {args.pubmed_max}")
    print(f"  PMC articles: {args.pmc_max}")
    print(f"  PhysioNet credentials: {'Provided' if 'physionet' in credentials else 'Not provided'}")
    print(f"  i2b2 credentials: {'Provided' if 'i2b2' in credentials else 'Not provided'}")
    print(f"  NCBI API key: {'Provided' if 'ncbi_api_key' in credentials else 'Not provided'}")
    
    setup_all_data(
        guideline_limit=args.guideline_limit,
        pubmed_max=args.pubmed_max,
        pmc_max=args.pmc_max,
        credentials=credentials
    )