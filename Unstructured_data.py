# Databricks notebook source
#!pip install PyPDF2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC READING THE DOCUMENTS

# COMMAND ----------

import PyPDF2
from bs4 import BeautifulSoup
import json
import os
import shutil

def process_recent_unstructured_file():
    """
    Automatically select and process the most recently uploaded unstructured file (PDF, HTML, JSON)
    in the DBFS directory. Also display all available files and their contents.
    
    Returns:
    - data (str or dict): Extracted data or text if supported, else None.
    """
    # Define the DBFS path for unstructured data
    unstructured_data_path = "/FileStore/Group-6_Data/Unstructured-data/"  # DBFS volume path

    # List files in the directory
    files = dbutils.fs.ls(unstructured_data_path)
    
    # Display available files
    print("Available files in the unstructured data directory:")
    for file_info in files:
        print(f"File Name: {file_info.name}, Size: {file_info.size} bytes")

    # Filter for supported file types (PDF, HTML, JSON)
    supported_files = [file_info for file_info in files if file_info.name.endswith(('.pdf', '.html', '.json'))]
    
    if not supported_files:
        print("No supported files (PDF, HTML, JSON) found in the unstructured data directory.")
        return None
    
    # Select the most recently modified file
    most_recent_file = max(supported_files, key=lambda f: f.modificationTime)
    
    # Get the full file path in DBFS
    file_path = most_recent_file.path  # Use `path` for the full DBFS path

    print(f"\nProcessing the most recent file: {most_recent_file.name}")

    # Temporary directory for file operations
    temp_dir = '/tmp/unstructured_data/'
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, most_recent_file.name)

    # Copy the file from DBFS to local temporary directory
    dbutils.fs.cp(file_path, f"file:{local_file_path}")

    # Helper functions to extract data from different file formats
    def extract_text_from_pdf(pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"An error occurred while reading PDF: {e}")
        return text

    def extract_text_from_html(html_path):
        text = ""
        try:
            with open(html_path, 'r') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text()
        except Exception as e:
            print(f"An error occurred while reading HTML: {e}")
        return text

    if most_recent_file.name.endswith('.pdf'):
        pdf_text = extract_text_from_pdf(local_file_path)
        print(f"\nExtracted Text from PDF (First 1000 characters):")
        print(pdf_text[:1000])  # Display first 1000 characters
        return pdf_text
    elif most_recent_file.name.endswith('.html'):
        html_text = extract_text_from_html(local_file_path)
        print(f"\nExtracted Text from HTML (First 1000 characters):")
        print(html_text[:1000])  # Display first 1000 characters
        return html_text
    elif most_recent_file.name.endswith('.json'):
        try:
            with open(local_file_path, 'r') as file:
                json_data = json.load(file)
                print(f"\nExtracted Data from JSON:")
                print(json.dumps(json_data, indent=2))  # Pretty-print JSON data
                return json_data
        except Exception as e:
            print(f"An error occurred while processing JSON file: {e}")
        return None
    else:
        print("Unsupported unstructured file format. Only PDF, HTML, and JSON files are supported.")
        return None

# Call the function
process_recent_unstructured_file()


# COMMAND ----------

# MAGIC %md
# MAGIC CHUNKING AND EMBEDDING

# COMMAND ----------

import nltk
from sentence_transformers import SentenceTransformer
import json

# Ensure NLTK data is downloaded
#nltk.download('punkt')

def chunk_sentences_with_overlap(sentences, chunk_size=200, overlap_size=50):
    """Combine sentences into chunks of a specific size with overlap."""
    chunks = []
    i = 0
    while i < len(sentences):
        # Create a chunk starting from the current sentence
        chunk = " ".join(sentences[i:i + chunk_size])  # Join 'chunk_size' number of sentences together
        chunks.append(chunk)
        # Move forward but with overlap
        i += (chunk_size - overlap_size)
    return chunks

def convert_text_to_embeddings(text, chunk_size=200, overlap_size=50):
    """Convert text into embeddings with custom chunk and overlap size."""
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)

    # Chunk sentences into larger pieces with overlap
    chunks = chunk_sentences_with_overlap(sentences, chunk_size, overlap_size)

    # Load pre-trained BGE model
    model = SentenceTransformer('BAAI/bge-large-en')  # BGE model variant

    # Generate embeddings for each chunk of sentences
    embeddings = model.encode(chunks, convert_to_tensor=True)

    return chunks, embeddings

# Call the function to process the most recent unstructured file
extracted_data = process_recent_unstructured_file()

if extracted_data:
    if isinstance(extracted_data, str):  # For PDF and HTML
        # Convert extracted text into embeddings
        chunks, embeddings = convert_text_to_embeddings(extracted_data, chunk_size=100, overlap_size=50)

        # Print some of the chunks and their corresponding embeddings
        for i in range(min(200, len(chunks))):  # Display the first 200 chunks and embeddings
            print(f"Chunk {i+1}: {chunks[i]}")
            print(f"Embedding Vector: {embeddings[i]}\n")
    elif isinstance(extracted_data, dict):  # For JSON
        # If needed, handle JSON data for embeddings (custom logic required here)
        print("JSON data extraction and processing not implemented yet.")


# COMMAND ----------



'''def assign_unique_ids(chunks, embeddings):
    """
    Assign unique IDs to each chunk and its corresponding embedding.
    
    Args:
    - chunks (list of str): List of text chunks.
    - embeddings (list of torch.Tensor): List of embeddings corresponding to the chunks.
    
    Returns:
    - List of tuples: Each tuple contains (unique_id, chunk, embedding).
    """
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Create a list of tuples containing (ID, chunk, embedding)
    ided_embeddings = [(id, chunk, embedding) for id, chunk, embedding in zip(ids, chunks, embeddings)]

    return ided_embeddings

# Example usage
chunks, embeddings = convert_text_to_embeddings(extracted_data, chunk_size=200, overlap_size=50)
ided_embeddings = assign_unique_ids(chunks, embeddings)

# Print some of the results
for i in range(min(5, len(ided_embeddings))):  # Display the first 5 tuples
    unique_id, chunk, embedding = ided_embeddings[i]
    print(f"ID: {unique_id}\nChunk: {chunk}\nEmbedding Vector: {embedding}\n")'''


# COMMAND ----------

'''import chromadb

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
collection = create_or_replace_collection(client, collection_name)'''


# COMMAND ----------

'''import uuid
import chromadb
from chromadb.config import Settings
import torch  # Ensure torch is imported for tensor operations

# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Function to create or replace a collection
def create_or_replace_collection(client, collection_name):
    # Check if the collection already exists
    collections = client.list_collections()
    existing_collections = [col.name for col in collections]
    
    if collection_name in existing_collections:
        # Collection exists, delete it and create a new one
        print(f"Collection '{collection_name}' already exists. Replacing it.")
        client.delete_collection(name=collection_name)
    
    # Create a new collection
    collection = client.create_collection(name=collection_name)
    return collection

# Create or replace the collection
collection_name = "my_collection"
collection = create_or_replace_collection(client, collection_name)

# Function to upsert data to the collection
def upsert_data_to_collection(collection, chunks, embeddings):
    # Convert tensor embeddings to lists
    embeddings_list = [embedding.tolist() for embedding in embeddings]

    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Prepare metadata
    metadata = [{'text': chunk} for chunk in chunks]

    # Upsert the data into the collection
    collection.upsert(
        ids=ids,                   # List of unique IDs
        embeddings=embeddings_list, # List of embeddings converted from tensors
            
    )
    print("Data has been upserted to the collection.")

# Example usage
chunks, embeddings = convert_text_to_embeddings(extracted_data, chunk_size=200, overlap_size=50)
upsert_data_to_collection(collection, chunks, embeddings)'''

# COMMAND ----------

# MAGIC %md
# MAGIC UNIQUE UIDs AND DYNAMIC COLLECTION

# COMMAND ----------

import chromadb
from chromadb.config import Settings
import torch
import uuid  # For generating unique IDs

# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Function to create or replace a collection dynamically
def create_or_replace_collection(client, collection_name):
    """
    Create a new collection or replace an existing one with the same name.
    
    Args:
    - client (chromadb.Client): The ChromaDB client.
    - collection_name (str): The name of the collection.
    
    Returns:
    - collection (chromadb.Collection): The newly created or replaced collection.
    """
    # Get list of existing collections
    collections = client.list_collections()
    existing_collections = [col.name for col in collections]
    
    # Delete the existing collection if it exists
    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' already exists. Replacing it.")
        client.delete_collection(name=collection_name)
    
    # Create a new collection
    collection = client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")
    return collection

# Function to upsert data (chunks and embeddings) into the collection
def upsert_data_to_collection(collection, chunks, embeddings):
    """
    Upserts the provided chunks and their corresponding embeddings into the collection.
    
    Args:
    - collection (chromadb.Collection): The ChromaDB collection to upsert data into.
    - chunks (list of str): The text chunks.
    - embeddings (list of torch.Tensor): The embeddings corresponding to the chunks.
    """
    # Convert tensor embeddings to lists
    embeddings_list = [embedding.tolist() for embedding in embeddings]

    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Prepare metadata for each chunk (in this case, just the text of the chunk)
    metadata = [{'text': chunk} for chunk in chunks]

    # Upsert the data into the collection
    collection.upsert(
        ids=ids,                   # List of unique IDs
        embeddings=embeddings_list, # List of embeddings converted from tensors
        metadatas=metadata          # Metadata containing the text chunk
    )
    print("Data has been upserted into the collection.")

# Example usage
# Ensure extracted_data is available, and convert it into chunks and embeddings
chunks, embeddings = convert_text_to_embeddings(extracted_data, chunk_size=200, overlap_size=50)

# Create or replace the collection dynamically
collection_name = "my_collection"
collection = create_or_replace_collection(client, collection_name)

# Upsert the chunks and embeddings into the collection
upsert_data_to_collection(collection, chunks, embeddings)


# COMMAND ----------

'''import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoModel, AutoTokenizer

# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Load the existing collection
collection_name = "my_collection"  # Replace with your actual collection name
collection = client.get_collection(name=collection_name)

# Initialize the model and tokenizer
model_name = "BAAI/bge-large-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_query_embedding(query_text):
    """
    Convert the user's query text into an embedding using the BAAI/bge-large-en model.

    Args:
    - query_text (str): The text of the user's query.

    Returns:
    - query_embedding (torch.Tensor): The embedding for the query text.
    """
    inputs = tokenizer(query_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the last hidden state as the embedding
    query_embedding = outputs.last_hidden_state.mean(dim=1)
    return query_embedding

def query_vector_database(query_text):
    """
    Query the vector database using the user's query text.

    Args:
    - query_text (str): The text of the user's query.

    Returns:
    - results (dict): Results from the vector database query.
    """
    # Get query embedding from the input text
    query_embedding = get_query_embedding(query_text)
    
    # Query the collection in the vector database
    results = collection.query(
        query_embeddings=query_embedding.tolist()  # Convert the query embedding to a list
    )
    
    return results

def ask_for_query():
    """
    Prompt the user to input a query and process it until they choose to exit.
    """
    while True:
        # Get user input for the query
        user_query_text = input("Enter your query (or type 'exit' to stop): ")
        
        if user_query_text.lower() == 'exit':
            print("Exiting the query process.")
            break
        
        # Query the vector database
        results = query_vector_database(user_query_text)
        
        # Display the results
        print(f"Results for query '{user_query_text}':")
        print("Metadata: ", results.get('metadatas'))
        
        # Ask if the user wants to continue querying
        continue_query = input("Would you like to make another query? (yes/no): ").lower()
        
        if continue_query not in ['yes', 'y']:
            print("Thank you.")
            break

if __name__ == "__main__":
    # Start the query process
    ask_for_query()'''


# COMMAND ----------

# MAGIC %md
# MAGIC VALIDATION AND PROMPT FOR USER

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import torch

def convert_query_to_embedding(query_text, model):
    """
    Convert a query text into an embedding using the provided model.
    
    Args:
    - query_text (str): The query text.
    - model (SentenceTransformer): The model used for embedding.
    
    Returns:
    - query_embedding (torch.Tensor): The embedding of the query text.
    """
    # Convert query text to embedding
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    return query_embedding

def validate_query_with_comparison(query_text, model, stored_embeddings, stored_chunks):
    """
    Validate the query by comparing it with stored embeddings using cosine similarity and Euclidean distance.
    
    Args:
    - query_text (str): The query text.
    - model (SentenceTransformer): The model used for embedding.
    - stored_embeddings (list of torch.Tensor): The embeddings of the stored chunks.
    - stored_chunks (list of str): The chunks of text.
    
    Returns:
    - best_metric (str): The metric that gave the best match ("Cosine Similarity" or "Euclidean Distance").
    - best_chunk (str): The chunk of text that is the best match.
    - best_score (float): The best score (percentage) based on the best metric.
    - cosine_scores (list of float): List of cosine similarity scores.
    - euclidean_scores (list of float): List of inverted Euclidean distance scores.
    """
    # Convert query to embedding
    query_embedding = convert_query_to_embedding(query_text, model).cpu().numpy()

    # Convert stored embeddings to NumPy arrays
    stored_embeddings_np = [embedding.cpu().numpy() for embedding in stored_embeddings]

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity([query_embedding], stored_embeddings_np)[0]

    # Calculate Euclidean distance (inverted for similarity comparison)
    euclidean_similarities = [1 / (1 + euclidean(query_embedding, emb)) for emb in stored_embeddings_np]  # Inverted Euclidean distance

    # Get the best match for cosine similarity
    best_cosine_idx = cosine_similarities.argmax()
    best_cosine_chunk = stored_chunks[best_cosine_idx]
    best_cosine_score = cosine_similarities[best_cosine_idx] * 100  # Convert to percentage

    # Get the best match for Euclidean similarity
    best_euclidean_idx = euclidean_similarities.index(max(euclidean_similarities))
    best_euclidean_chunk = stored_chunks[best_euclidean_idx]
    best_euclidean_score = euclidean_similarities[best_euclidean_idx] * 100  # Convert to percentage

    # Determine which metric gives the best match
    if best_cosine_score >= best_euclidean_score:
        best_metric = "Cosine Similarity"
        best_chunk = best_cosine_chunk
        best_score = best_cosine_score
    else:
        best_metric = "Euclidean Distance"
        best_chunk = best_euclidean_chunk
        best_score = best_euclidean_score

    return best_metric, best_chunk, best_score, cosine_similarities, euclidean_similarities

# Example usage with your existing code
query_text = input("Enter your query text:")  # Replace with the actual query text
model = SentenceTransformer('BAAI/bge-large-en')

# Assuming `embeddings` and `chunks` are obtained from your code
best_metric, best_chunk, best_score, cosine_scores, euclidean_scores = validate_query_with_comparison(query_text, model, embeddings, chunks)

# Display the results
print(f"Best metric: {best_metric}")
print(f"Best matching chunk: {best_chunk}")
print(f"Best score: {best_score}%")
print(f"Cosine similarity scores: {cosine_scores}")
print(f"Euclidean similarity scores: {euclidean_scores}")


# COMMAND ----------

#!pip install chromadb==0.5.3

# COMMAND ----------

#%pip install --upgrade opentelemetry-api opentelemetry-sdk

# COMMAND ----------

dbutils.library.restartPython()
