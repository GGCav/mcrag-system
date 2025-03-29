# MCRAG: Medical Context Retrieval-Augmented Generation System

An optimized system for retrieving and generating clinical recommendations based on medical literature and guidelines.

## Features

- Advanced NLP processing using spaCy and biomedical models
- UMLS entity linking for medical concepts
- Multiple retrieval strategies for diverse results
- Memory-efficient processing with caching
- Support for various document types

## Installation

See the [installation instructions](nlp-dependencies-installer.py) for setting up the required dependencies.

## Usage

\`\`\`bash
# Process a clinical query
python mcrag.py query --query "What is the recommended treatment for type 2 diabetes?"

# Run a batch of queries
python mcrag.py batch --file queries.txt
\`\`\`

## Documentation

See [MCRAG System Usage Guide](mcrag-usage-guide.md) for detailed usage instructions.