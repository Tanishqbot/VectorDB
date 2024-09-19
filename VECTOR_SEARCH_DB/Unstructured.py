# Databricks notebook source
# MAGIC %md
# MAGIC EXTRACTING THE TEXT

# COMMAND ----------

# DBTITLE 1,Extracting the Text
import PyPDF2
from bs4 import BeautifulSoup
import json
import shutil
import docx

def process_recent_unstructured_file():
    """
    Automatically select and process the most recently uploaded unstructured file (PDF, HTML, JSON, DOCX)
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

    # Filter for supported file types (PDF, HTML, JSON, DOCX)
    supported_files = [file_info for file_info in files if file_info.name.endswith(('.pdf', '.html', '.json', '.docx'))]
    
    if not supported_files:
        print("No supported files (PDF, HTML, JSON, DOCX) found in the unstructured data directory.")
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

    def extract_text_from_docx(docx_path):
        text = ""
        try:
            doc = docx.Document(docx_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"An error occurred while reading DOCX: {e}")
        return text

    # Extract text based on file type
    if most_recent_file.name.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(local_file_path)
    elif most_recent_file.name.endswith('.html'):
        extracted_text = extract_text_from_html(local_file_path)
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
    elif most_recent_file.name.endswith('.docx'):
        extracted_text = extract_text_from_docx(local_file_path)
    else:
        print("Unsupported unstructured file format. Only PDF, HTML, JSON, and DOCX files are supported.")
        return None

    # Return the extracted text for further processing
    return extracted_text


# COMMAND ----------

# MAGIC %md
# MAGIC CHUNKING

# COMMAND ----------

# DBTITLE 1,CHUNKING
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import OpenAIGPTTokenizer

def initialize_text_splitter():
    max_chunk_size = 100
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, 
        chunk_size=max_chunk_size, 
        chunk_overlap=50
    )
    return text_splitter, tokenizer

def split_plain_text(text, min_chunk_size=20, max_chunk_size=100):
    if not text:
        return []

    text_splitter, tokenizer = initialize_text_splitter()
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if len(tokenizer.encode(chunk)) > min_chunk_size]

def process_text(text):
    chunks = split_plain_text(text)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 80)

# Main code to process the most recent file and display text chunks
extracted_data = process_recent_unstructured_file()
if isinstance(extracted_data, dict):
    extracted_text = extracted_data.get('text', '')
else:
    extracted_text = extracted_data

if extracted_text:
    process_text(extracted_text)


# COMMAND ----------

# MAGIC %md
# MAGIC EMBEDDINGS

# COMMAND ----------

# DBTITLE 1,EMBEDDINGS
from sentence_transformers import SentenceTransformer
import torch

# Initialize the embedding model
def initialize_embedding_model(model_name="BAAI/bge-large-en"):
    model = SentenceTransformer(model_name)
    return model

# Function to generate embeddings for a list of text chunks
def generate_embeddings(text_chunks):
    model = initialize_embedding_model()
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings

# Updated process_text to include embedding generation
def process_text(text):
    chunks = split_plain_text(text)
    
    # Display the text chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 80)
    
    # Generate embeddings for the chunks
    print("Generating embeddings for the chunks...")
    embeddings = generate_embeddings(chunks)
    
    # Display some embeddings
    for i in range(min(3, len(embeddings))):  # Display first 3 embeddings
        print(f"Embedding for Chunk {i+1}:")
        print(embeddings[i])
        print("-" * 80)
    
    return chunks, embeddings

# Main code to process the most recent file and generate embeddings
extracted_data = process_recent_unstructured_file()
if isinstance(extracted_data, dict):
    extracted_text = extracted_data.get('text', '')
else:
    extracted_text = extracted_data

if extracted_text:
    chunks, embeddings = process_text(extracted_text)
    # You can now store the embeddings in a Delta table or use them for further processing.


# COMMAND ----------

# MAGIC %md
# MAGIC DELTA TABLE

# COMMAND ----------

# DBTITLE 1,DELTA TABLE
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import numpy as np

# Initialize Spark session (this is automatically available in Databricks notebooks)
spark = SparkSession.builder.getOrCreate()

# Define schema for the DataFrame, including embeddings
schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("content", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True)  # Adding embedding field
])

# Define the path to the Delta table
table_path = "main.default.group_six"

# Get extracted text (assuming process_recent_unstructured_file() is defined)
extracted_data = process_recent_unstructured_file()
if isinstance(extracted_data, dict):
    extracted_text = extracted_data.get('text', '')
else:
    extracted_text = extracted_data

# Use the process_text function to split the text into chunks
chunks = split_plain_text(extracted_text)

# Generate embeddings for the chunks
embeddings = generate_embeddings(chunks)

# Convert embeddings from tensor to list of floats
embeddings_list = [embedding.tolist() for embedding in embeddings]

# Prepare data for DataFrame including embeddings
data = [(f"chunk_{i}", chunk, embedding) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list))]

# Create DataFrame with chunks data and embeddings
df = spark.createDataFrame(data, schema=schema)

# Overwrite the existing Delta table schema and contents
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_path)

print(f"Data inserted into Delta table '{table_path}' successfully.")


# COMMAND ----------

#!pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ENDPOINT FOR VSI

# COMMAND ----------

# MAGIC %md
# MAGIC VALIDATION
# MAGIC

# COMMAND ----------

'''import os
import numpy as np
from databricks.vector_search.client import VectorSearchClient
# dbutils.widgets.text("query_text", "")

# Initialize environment variables
workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

# Initialize VectorSearchClient
vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret
)

# Get the index
index = vsc.get_index(endpoint_name="validatrix", index_name="main.default.group_six_vector_index")

# Prompt user for query
query_text = input("Enter your query: ")
# query_text = dbutils.widgets.get("query_text")

# Perform similarity search
results = index.similarity_search(num_results=3, columns=["content"], query_text=query_text)

# Function to calculate Euclidean distance
def calculate_euclidean_distance(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors.
    """
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

# Normalize cosine similarity to percentage
def cosine_similarity_to_percentage(cosine_similarity):
    """
    Converts cosine similarity to percentage.
    Cosine similarity ranges from -1 to 1.
    This function maps it to 0% to 100%.
    """
    return ((cosine_similarity + 1) / 2) * 100

# Normalize Euclidean distance to percentage (for interpretation)
def euclidean_distance_to_percentage(distance, max_distance=2):
    """
    Normalizes Euclidean distance to a percentage.
    We assume max_distance = 2 (which would be a large distance in high-dimensional space).
    This is arbitrary and can be adjusted based on the dataset's scale.
    """
    return max(0, (1 - distance / max_distance) * 100)

def process_results(results, query_text):
    """
    Process and format the vector search results.

    Args:
    - results (dict): The vector search results in JSON format.
    - query_text (str): The query text used for the search.

    Returns:
    - str: A formatted string of the extracted text content with similarity metrics.
    """
    if isinstance(results, dict):
        # Access the 'data_array' from the JSON response
        data_array = results.get('result', {}).get('data_array', [])
        
        if isinstance(data_array, list):
            result_texts = []
            for idx, entry in enumerate(data_array):
                if isinstance(entry, list) and len(entry) == 2:
                    content, score = entry
                    # Convert cosine similarity to percentage
                    cosine_similarity_pct = cosine_similarity_to_percentage(score)

                    # Dummy vector comparison for Euclidean distance (you would use actual embeddings here)
                    # For demonstration purposes, let's use random vectors or retrieved vectors for Euclidean.
                    query_embedding = np.random.rand(512)  # Simulate query embedding (replace with actual vector)
                    result_embedding = np.random.rand(512)  # Simulate result embedding (replace with actual vector)
                    
                    euclidean_distance = calculate_euclidean_distance(query_embedding, result_embedding)
                    euclidean_distance_pct = euclidean_distance_to_percentage(euclidean_distance)

                    result_text = (
                        f"Result {idx + 1}:\n"
                        f"Content: {content}\n"
                        f"Cosine Similarity Score: {cosine_similarity_pct:.2f}%\n"
                        f"Euclidean Distance Score: {euclidean_distance_pct:.2f}%\n"
                        f"{'-' * 80}"
                    )
                    result_texts.append(result_text)
            return "\n".join(result_texts) if result_texts else "No valid results found in the 'data_array'."
        else:
            return "Unexpected format for 'data_array'."
    else:
        return "Unexpected format of results. Please check the structure."

# Store the formatted results in a variable
formatted_results = process_results(results, query_text)

# Output the results
print(f"Type of results: {type(results)}")
print(f"Contents of results: {results}")
print(f"Formatted Results:\n{formatted_results}")'''

# You can use the `formatted_results` variable for further processing or storage as needed


# COMMAND ----------

#!pip install databricks-vectorsearch

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,SEARCHING THE QUERY AND VALIDATION
import os
import numpy as np
from databricks.vector_search.client import VectorSearchClient

# Initialize environment variables
workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

# Initialize VectorSearchClient
vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret
)

# Get the index
index = vsc.get_index(endpoint_name="validatrix", index_name="main.default.group_six_vector_index")

# Create a text widget to take user input
dbutils.widgets.text("query_text", "Enter your query here")

# Retrieve user input from the widget
query_text = dbutils.widgets.get("query_text")

# Perform similarity search
results = index.similarity_search(num_results=3, columns=["content"], query_text=query_text)

# Function to calculate Euclidean distance
def calculate_euclidean_distance(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors.
    """
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

# Normalize cosine similarity to percentage
def cosine_similarity_to_percentage(cosine_similarity):
    """
    Converts cosine similarity to percentage.
    Cosine similarity ranges from -1 to 1.
    This function maps it to 0% to 100%.
    """
    return ((cosine_similarity + 1) / 2) * 100

# Normalize Euclidean distance to percentage (for interpretation)
def euclidean_distance_to_percentage(distance, max_distance=2):
    """
    Normalizes Euclidean distance to a percentage.
    We assume max_distance = 2 (which would be a large distance in high-dimensional space).
    This is arbitrary and can be adjusted based on the dataset's scale.
    """
    return max(0, (1 - distance / max_distance) * 100)

def process_results(results, query_text):
    """
    Process and format the vector search results.

    Args:
    - results (dict): The vector search results in JSON format.
    - query_text (str): The query text used for the search.

    Returns:
    - str: A formatted string of the extracted text content with similarity metrics.
    """
    if isinstance(results, dict):
        # Access the 'data_array' from the JSON response
        data_array = results.get('result', {}).get('data_array', [])
        
        if isinstance(data_array, list):
            result_texts = []
            for idx, entry in enumerate(data_array):
                if isinstance(entry, list) and len(entry) == 2:
                    content, score = entry
                    # Convert cosine similarity to percentage
                    cosine_similarity_pct = cosine_similarity_to_percentage(score)

                    # Dummy vector comparison for Euclidean distance (you would use actual embeddings here)
                    # For demonstration purposes, let's use random vectors or retrieved vectors for Euclidean.
                    query_embedding = np.random.rand(512)  # Simulate query embedding (replace with actual vector)
                    result_embedding = np.random.rand(512)  # Simulate result embedding (replace with actual vector)
                    
                    euclidean_distance = calculate_euclidean_distance(query_embedding, result_embedding)
                    euclidean_distance_pct = euclidean_distance_to_percentage(euclidean_distance)

                    result_text = (
                        f"Result {idx + 1}:\n"
                        f"Content: {content}\n"
                        f"Cosine Similarity Score: {cosine_similarity_pct:.2f}%\n"
                        f"Euclidean Distance Score: {euclidean_distance_pct:.2f}%\n"
                        f"{'-' * 80}"
                    )
                    result_texts.append(result_text)
            return "\n".join(result_texts) if result_texts else "No valid results found in the 'data_array'."
        else:
            return "Unexpected format for 'data_array'."
    else:
        return "Unexpected format of results. Please check the structure."

# Store the formatted results in a variable
formatted_results = process_results(results, query_text)

# Output the results
print(f"Type of results: {type(results)}")
print(f"Contents of results: {results}")
print(f"Formatted Results:\n{formatted_results}")


# COMMAND ----------



# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,REFINING THE RESULTS
'''import os
import numpy as np
from databricks.vector_search.client import VectorSearchClient
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize environment variables
workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

# Initialize VectorSearchClient
vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret
)

# Get the index
index = vsc.get_index(endpoint_name="validatrix", index_name="main.default.group_six_vector_index")

# Create a text widget to take user input
dbutils.widgets.text("query_text", "Enter your query here")

# Retrieve user input from the widget
query_text = dbutils.widgets.get("query_text")

# Perform similarity search
results = index.similarity_search(num_results=5, columns=["content"], query_text=query_text)

# Initialize GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def refine_text_with_gpt2(text):
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate responses with tuned parameters
    outputs = model.generate(
        inputs,
        max_length=1000,
        max_new_tokens=500,
        num_return_sequences=1,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=60
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clean_text(text):
    # Remove scores and extra details
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if "Score:" in line:
            continue  # Skip lines with scores
        if line.strip() and not line.startswith("Content:"):
            filtered_lines.append(line.strip())  # Add cleaned lines
    return "\n".join(filtered_lines)

def process_results(results, query_text):
    if isinstance(results, dict):
        # Access the 'data_array' from the JSON response
        data_array = results.get('result', {}).get('data_array', [])
        
        if isinstance(data_array, list):
            result_texts = []
            contexts = []
            for idx, entry in enumerate(data_array):
                if isinstance(entry, list) and len(entry) == 2:
                    content, score = entry
                    contexts.append(f"Content:\n{content}\nScore: {score}\n{'-' * 80}")
                    
                    # Add result details
                    result_text = (
                        f"Result {idx + 1}:\n"
                        f"Content: {content}\n"
                        f"Score: {score}\n"
                        f"{'-' * 80}"
                    )
                    result_texts.append(result_text)

            # Aggregate contexts for refined generation
            combined_contexts = "\n".join(contexts)
            refined_response = refine_text_with_gpt2(combined_contexts)
            
            # Clean the refined response to remove unnecessary content
            refined_response_cleaned = clean_text(refined_response)
            
            # Return only the cleaned refined response
            return f"Cleaned Refined Response:\n{refined_response_cleaned}" if refined_response_cleaned else "No valid results found in the 'data_array'."
        else:
            return "Unexpected format for 'data_array'."
    else:
        return "Unexpected format of results. Please check the structure."

# Store the formatted results in a variable
formatted_results = process_results(results, query_text)

# Output the cleaned refined results
print(f"Formatted Results:\n{formatted_results}")'''


# COMMAND ----------


