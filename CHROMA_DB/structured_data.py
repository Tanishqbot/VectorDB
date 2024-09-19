# Databricks notebook source
#%pip install openpyxl


# COMMAND ----------

# DBTITLE 1,EXTRACTING
import pandas as pd
import os

def process_recent_structured_file():
    """
    Automatically select and process the most recently uploaded structured file (CSV, XLS, XLSX)
    in the DBFS directory. Also display all available files and their contents.
    
    Returns:
    - data (DataFrame or None): Extracted data as a DataFrame if supported, else None.
    """
    # Define the DBFS path for structured data
    structured_data_path = "/FileStore/Group-6_Data/Structured-data/"  # DBFS volume path

    # List files in the directory
    files = dbutils.fs.ls(structured_data_path)
    
    # Display available files
    print("Available files in the structured data directory:")
    for file_info in files:
        print(f"File Name: {file_info.name}, Size: {file_info.size} bytes")

    # Filter for supported file types (CSV, XLS, XLSX)
    supported_files = [file_info for file_info in files if file_info.name.endswith(('.csv', '.xls', '.xlsx'))]
    
    if not supported_files:
        print("No supported files (CSV, XLS, XLSX) found in the structured data directory.")
        return None
    
    # Select the most recently modified file
    most_recent_file = max(supported_files, key=lambda f: f.modificationTime)
    
    # Get the full file path in DBFS
    file_path = most_recent_file.path  # Use `path` for the full DBFS path

    print(f"\nProcessing the most recent file: {most_recent_file.name}")

    # Temporary directory for file operations
    temp_dir = '/tmp/structured_data/'
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, most_recent_file.name)

    # Copy the file from DBFS to local temporary directory
    dbutils.fs.cp(file_path, f"file:{local_file_path}")

    # Helper functions to extract data from different file formats
    def extract_data_from_csv(csv_path):
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"An error occurred while reading CSV: {e}")
            return None
        return data

    def extract_data_from_excel(excel_path):
        try:
            data = pd.read_excel(excel_path)
        except Exception as e:
            print(f"An error occurred while reading Excel file: {e}")
            return None
        return data

    # Extract data based on file type
    if most_recent_file.name.endswith('.csv'):
        data = extract_data_from_csv(local_file_path)
        if data is not None:
            print(f"\nExtracted Data from CSV (First 5 rows):")
            print(data.head())  # Display first 5 rows
        return data
    elif most_recent_file.name.endswith(('.xls', '.xlsx')):
        data = extract_data_from_excel(local_file_path)
        if data is not None:
            print(f"\nExtracted Data from Excel (First 5 rows):")
            print(data.head())  # Display first 5 rows
        return data
    else:
        print("Unsupported structured file format. Only CSV, XLS, and XLSX files are supported.")
        return None

# Call the function
process_recent_structured_file()

# COMMAND ----------

#%pip install openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,EMBEDDINGS
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os

# Install the necessary package
# %pip install openpyxl

# Load BGE-Large model and tokenizer
model_name = "BAAI/bge-large-en"  # Update with the correct model name if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def process_recent_structured_file():
    """
    Automatically select and process the most recently uploaded structured file (CSV, XLS, XLSX)
    in the DBFS directory. Also display all available files and their contents.
    
    Returns:
    - data (DataFrame or None): Extracted data as a DataFrame if supported, else None.
    """
    # Define the DBFS path for structured data
    structured_data_path = "/FileStore/Group-6_Data/Structured-data/"  # DBFS volume path

    # List files in the directory
    files = dbutils.fs.ls(structured_data_path)
    
    # Display available files
    print("Available files in the structured data directory:")
    for file_info in files:
        print(f"File Name: {file_info.name}, Size: {file_info.size} bytes")

    # Filter for supported file types (CSV, XLS, XLSX)
    supported_files = [file_info for file_info in files if file_info.name.endswith(('.csv', '.xls', '.xlsx'))]
    
    if not supported_files:
        print("No supported files (CSV, XLS, XLSX) found in the structured data directory.")
        return None
    
    # Select the most recently modified file
    most_recent_file = max(supported_files, key=lambda f: f.modificationTime)
    
    # Get the full file path in DBFS
    file_path = most_recent_file.path  # Use `path` for the full DBFS path

    print(f"\nProcessing the most recent file: {most_recent_file.name}")

    # Temporary directory for file operations
    temp_dir = '/tmp/structured_data/'
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, most_recent_file.name)

    # Copy the file from DBFS to local temporary directory
    dbutils.fs.cp(file_path, f"file:{local_file_path}")

    # Helper functions to extract data from different file formats
    def extract_data_from_csv(csv_path):
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"An error occurred while reading CSV: {e}")
            return None
        return data

    def extract_data_from_excel(excel_path):
        try:
            data = pd.read_excel(excel_path)
        except Exception as e:
            print(f"An error occurred while reading Excel file: {e}")
            return None
        return data

    # Extract data based on file type
    if most_recent_file.name.endswith('.csv'):
        data = extract_data_from_csv(local_file_path)
        if data is not None:
            print(f"\nExtracted Data from CSV (First 5 rows):")
            print(data.head())  # Display first 5 rows
        return data
    elif most_recent_file.name.endswith(('.xls', '.xlsx')):
        data = extract_data_from_excel(local_file_path)
        if data is not None:
            print(f"\nExtracted Data from Excel (First 5 rows):")
            print(data.head())  # Display first 5 rows
        return data
    else:
        print("Unsupported structured file format. Only CSV, XLS, and XLSX files are supported.")
        return None

def chunk_text(text, chunk_size=100, overlap=50):
    """
    Split text into chunks of `chunk_size` with an `overlap` between chunks.
    Returns a list of chunks with their corresponding chunk numbers.
    """
    chunks = []
    chunk_number = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((text[start:end], chunk_number))
        chunk_number += 1
        start += chunk_size - overlap  # Move to the next chunk with overlap
    return chunks

def generate_embeddings(text_chunks):
    """
    Generate embeddings for a list of text chunks using the BGE-Large model.
    """
    embeddings = []
    for chunk_text, chunk_number in text_chunks:
        inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
        embeddings.append((chunk_number, chunk_text, embedding.tolist()))
    return embeddings

# Call the function to get data
data = process_recent_structured_file()

if data is not None:
    # Assuming the data is a DataFrame, convert it to a single string for chunking
    text_data = data.to_string()

    # Create chunks with size 200 and overlap 50
    text_chunks = chunk_text(text_data, chunk_size=200, overlap=50)

    # Generate embeddings for each chunk
    chunk_embeddings = generate_embeddings(text_chunks)

    # Display chunk number, chunk text, and the corresponding embedding
    for chunk_number, chunk_text, embedding in chunk_embeddings:
        print(f"Chunk Number: {chunk_number}")
        print(f"Chunk Text (First 100 chars): {chunk_text[:100]}...")
        print(f"Embedding (First 5 values): {embedding[:5]}...\n")


# COMMAND ----------

###UIDs###

# COMMAND ----------

# DBTITLE 1,UNIQUE IDs
import uuid

def assign_unique_ids_structured(chunks, embeddings):
    """
    Assign unique IDs to each chunk of structured data and store the chunk, chunk number, and its embedding in a list of dictionaries.
    
    Parameters:
    - chunks (list of tuples): List of text chunks along with their chunk numbers.
    - embeddings (list of tuples): List of embeddings for each chunk along with chunk numbers.
    
    Returns:
    - result (list of dict): List of dictionaries containing unique ID, chunk number, text, and embedding.
    """
    result = []
    for i, (chunk_text, chunk_number) in enumerate(chunks):
        result.append({
            'id': str(uuid.uuid4()),  # Generate a unique ID for each chunk
            'chunk_number': chunk_number + 1,  # Add chunk number (1-based indexing)
            'text': chunk_text,
            'embedding': embeddings[i][2]  # Get the embedding (already converted to a list in previous step)
        })
    return result

# Assign unique IDs
structured_output = assign_unique_ids_structured(text_chunks, chunk_embeddings)

# Print the output
for item in structured_output:
    print(f"Chunk Number: {item['chunk_number']}")
    print(f"ID: {item['id']}")
    print(f"Text: {item['text']}")
    print(f"Embedding: {item['embedding']}\n")

# COMMAND ----------

#%pip install chromadb==0.5.3


# COMMAND ----------

#%pip install opentelemetry-api opentelemetry-sdk


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#!pip install chromadb==0.5.3


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import chromadb

# Initialize the Chroma client
client = chromadb.Client()

collection_name = "my_collection"

# Function to create a new collection, deleting the old one if it exists
def create_or_replace_collection(client, collection_name):
    # List existing collections
    collections = client.list_collections()
    
    # Check if the collection already exists
    existing_collection_names = [col.name for col in collections]
    
    if collection_name in existing_collection_names:
        # If the collection exists, delete it
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted.")
    
    # Create a new collection
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created.")
    return collection

# Call the function to create or replace the collection
collection = create_or_replace_collection(client, collection_name)


# COMMAND ----------

# DBTITLE 1,CREATING DYNAMIC COLLECTION
import chromadb
import uuid
import torch

# Initialize the Chroma client
client = chromadb.Client()

collection_name = "my_collection"

# Function to create a new collection, deleting the old one if it exists
def create_or_replace_collection(client, collection_name):
    # List existing collections
    collections = client.list_collections()
    
    # Check if the collection already exists
    existing_collection_names = [col.name for col in collections]
    
    if collection_name in existing_collection_names:
        # If the collection exists, delete it
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted.")
    
    # Create a new collection
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created.")
    return collection

# Create or replace the collection
collection = create_or_replace_collection(client, collection_name)

def upsert_data_to_collection(collection, structured_data, embeddings):
    """
    Upsert structured data and embeddings into a collection.

    - collection: The collection to upsert data into.
    - structured_data (list of dict): List containing dictionaries with 'chunk_number' and 'text'.
    - embeddings (list of torch.Tensor): List containing embeddings as tensors.
    """
    # Convert embeddings to lists
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    
    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in range(len(structured_data))]
    
    # Prepare metadata
    metadata = [{'chunk_number': item['chunk_number'], 'text': item['text']} for item in structured_data]
    
    # Upsert data into the collection
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadata
    )
    
    print("Structured data has been upserted into the collection.")
    print(f"Number of items upserted: {len(ids)}")

# Assuming `text_chunks` and `chunk_embeddings` are defined from previous code
structured_data = [{'chunk_number': chunk_number, 'text': chunk_text} for chunk_text, chunk_number in text_chunks]
embeddings = [torch.tensor(embedding) for _, _, embedding in chunk_embeddings]

# Upsert data into the collection
upsert_data_to_collection(collection, structured_data, embeddings)


# COMMAND ----------

# DBTITLE 1,QUERYING AND VALIDATION
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

def query_collection(collection, query_text, tokenizer, model, top_k=5):
    """
    Query the collection with a given text and return the top_k most similar results.

    - collection: The collection to query.
    - query_text: The text to use for querying.
    - tokenizer: The tokenizer to use for encoding the query text.
    - model: The model to use for generating embeddings for the query text.
    - top_k: The number of top results to return.
    
    Returns:
    - List of tuples containing (chunk_number, text, distance, similarity).
    """
    # Encode the query text
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    
    # Fetch all items (embeddings and metadata) from the collection
    all_items = collection.get(include=['embeddings', 'metadatas'])
    all_embeddings = np.array(all_items['embeddings'])

    # Check if there are embeddings in the collection
    if len(all_embeddings) == 0:
        print("No embeddings found in the collection.")
        return []

    # Compute distances and similarities
    query_embedding_array = np.array(query_embedding).reshape(1, -1)
    cosine_similarities = cosine_similarity(query_embedding_array, all_embeddings).flatten()
    euclidean_distances_array = euclidean_distances(query_embedding_array, all_embeddings).flatten()

    # Get top k indices for similarity
    top_k_indices_similarity = np.argsort(-cosine_similarities)[:top_k]

    # Normalize Euclidean distances to percentages
    max_distance = euclidean_distances_array.max()
    if max_distance > 0:
        euclidean_distances_array = (1 - (euclidean_distances_array / max_distance)) * 100

    # Prepare results
    results = []
    for idx in top_k_indices_similarity:
        chunk_number = all_items['metadatas'][idx]['chunk_number']
        text = all_items['metadatas'][idx]['text']
        distance = euclidean_distances_array[idx]  # Already converted to percentage
        similarity = cosine_similarities[idx] * 100  # Convert cosine similarity to percentage
        results.append((chunk_number, text, distance, similarity))

    return results

# Example usage
query_text = input("Enter the query text: ")
results = query_collection(collection, query_text, tokenizer, model)

# Print results
for result in results:
    print(f"Chunk Number: {result[0]}")
    print(f"Text: {result[1]}")
    print(f"Euclidean Distance (as percentage): {result[2]:.2f}%")
    print(f"Cosine Similarity: {result[3]:.2f}%")
    print()

