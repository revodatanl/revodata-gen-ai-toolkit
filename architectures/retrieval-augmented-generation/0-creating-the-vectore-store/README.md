# Creating the Vector Store

In this module, we'll create a vector store using Databricks Vector Search. This will serve as the foundation for our RAG applications in the upcoming modules. We'll work with a PDF document, parse it into chunks, embed those chunks, and store them in a vector index.

## Overview

The process involves several key steps:

1. Setting up the required Databricks resources
2. Loading and parsing a PDF document
3. Chunking the document into manageable pieces
4. Embedding the chunks using a transformer model
5. Creating a vector search endpoint and index
6. Uploading the embedded chunks to the vector index


## Step 1: Setup

First, we need to set up the required Databricks resources. Run the `0-setup.ipynb` notebook to:

- Create a catalog for our generative AI toolkit
- Create a schema for our use cases
- Create an external volume to store models (optional)

## Step 2: Parse and Process the PDF

Next, we'll use the `1-pdf-parser.ipynb` notebook to:

1. Load and parse the PDF document using docling
2. Chunk the document using a hybrid chunking approach
3. Embed the chunks using the Jina embeddings model
4. Structure the chunks with their embeddings for vector search

## Step 3: Create and Populate the Vector Index

In the same notebook, we'll:

1. Create a vector search endpoint if it doesn't exist
2. Create a vector index with the appropriate schema
3. Upload our embedded chunks to the vector index
