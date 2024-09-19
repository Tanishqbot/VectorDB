# Databricks notebook source
!pip install openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `main`.`default`.`group_six_structured` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `main`.`default`.`group_six_structured` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

#!pip install databricks-vectorsearch

# COMMAND ----------

'''files = dbutils.fs.ls('/FileStore/Group-6_Data/Structured-data/')
for file in files:
    print(file.path)'''

# COMMAND ----------

#!pip install cohere

# COMMAND ----------

# DBTITLE 1,EXTRACTING AND CHUNKING
import pandas as pd
import os

def process_recent_structured_file():
    """
    Automatically select and process the most recently uploaded structured file (CSV, XLS, XLSX)
    in the DBFS directory. Also display all available files and their contents.
    
    Returns:
    - data (DataFrame or None): Extracted data as a DataFrame if supported, else None.
    """
    structured_data_path = "/FileStore/Group-6_Data/Structured-data/"  # DBFS volume path

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
    elif most_recent_file.name.endswith(('.xls', '.xlsx')):
        data = extract_data_from_excel(local_file_path)
    else:
        print("Unsupported structured file format. Only CSV, XLS, and XLSX files are supported.")
        return None

    if data is not None:
        print(f"\nExtracted Data (First 5 rows):")
        print(data.head(10))  # Display first 5 rows
        return data
    return None

def chunk_data_with_overlap(data, chunk_size=100, overlap_size=50):
    """
    Chunk the DataFrame into chunks of size `chunk_size` with an overlap of `overlap_size`.
    
    Parameters:
    - data (DataFrame): The input DataFrame to be chunked.
    - chunk_size (int): Number of records per chunk.
    - overlap_size (int): Number of overlapping records between consecutive chunks.
    
    Returns:
    - List of DataFrames: Chunked DataFrames with overlapping rows.
    """
    chunks = []
    num_rows = len(data)

    # Ensure chunk_size is greater than overlap_size
    if chunk_size <= overlap_size:
        raise ValueError("Chunk size must be greater than overlap size.")
    
    # Generate chunks with overlap
    for i in range(0, num_rows, chunk_size - overlap_size):
        end = min(i + chunk_size, num_rows)  # Ensure we don't go beyond the data
        chunk = data.iloc[i:end]
        chunks.append(chunk)
    
    return chunks

# Main function to process file and chunk data
def main():
    # Process the most recent structured file and get the data
    data = process_recent_structured_file()

    if data is not None:
        # Chunk the data with chunk size of 100 and overlap size of 50
        chunks = chunk_data_with_overlap(data, chunk_size=100, overlap_size=50)
        
        # Display all chunks
        for idx, chunk in enumerate(chunks):
            print(f"\nChunk {idx + 1}:")
            print(chunk.head())  # Display the first few rows of each chunk to make output manageable
            print(f"\nNumber of records in this chunk: {len(chunk)}")

# Run the main function
main()


# COMMAND ----------

'''import os
import numpy as np
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import euclidean

# Load BERT model and tokenizer for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def generate_embedding(text):
    """
    Generate embeddings for the given text using a pre-trained BERT model.
    
    Args:
    - text (str): The input text.

    Returns:
    - np.ndarray: The flattened embedding for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    # Flatten the embedding to 1D
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding

# Initialize environment variables
workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

# Initialize VectorSearchClient with disable_notice to suppress the warning
vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret,
    disable_notice=True  # Suppress the authentication notice
)

index = vsc.get_index(endpoint_name="group-six", index_name="main.default.group_six_struct_vsi")

# query_text = input("Enter your query: ")

# Perform similarity search
results = index.similarity_search(num_results=3, columns=["text"], query_text=query_text)

# Generate query embedding and flatten it to 1D
query_embedding = generate_embedding(query_text)

def calculate_euclidean_distance(vector1, vector2):
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
    - vector1 (np.ndarray): First vector.
    - vector2 (np.ndarray): Second vector.

    Returns:
    - float: Euclidean distance between vector1 and vector2.
    """
    return euclidean(vector1, vector2)

def process_results(results, query_embedding):
    """
    Process and format the vector search results, and calculate Euclidean distances.
    
    Args:
    - results (dict): The vector search results in JSON format.
    - query_embedding (np.ndarray): The embedding of the query text.

    Returns:
    - str: A formatted string of the extracted text content with similarity and Euclidean distance.
    """
    if isinstance(results, dict):
        # Access the 'data_array' from the JSON response
        data_array = results.get('result', {}).get('data_array', [])
        
        if isinstance(data_array, list):
            result_texts = []
            for idx, entry in enumerate(data_array):
                if isinstance(entry, list) and len(entry) == 2:
                    content, score = entry
                    # Generate embedding for the result content and flatten it to 1D
                    result_embedding = generate_embedding(content)
                    # Calculate Euclidean distance between query and result embedding
                    euclidean_dist = calculate_euclidean_distance(query_embedding, result_embedding)
                    
                    result_text = (f"Result {idx + 1}:\n"
                                   f"Content: {content}\n"
                                   f"Cosine Similarity Score: {score:.4f}\n"
                                   f"Euclidean Distance: {euclidean_dist:.4f}\n"
                                   f"{'-' * 115}")
                    result_texts.append(result_text)
            return "\n".join(result_texts) if result_texts else "No valid results found in the 'data_array'."
        else:
            return "Unexpected format for 'data_array'."
    else:
        return "Unexpected format of results. Please check the structure."

# Process the results and calculate Euclidean distance
formatted_results = process_results(results, query_embedding)

# Output the formatted results
print(f"Formatted Results:\n{formatted_results}")'''

# COMMAND ----------

# DBTITLE 1,UPSERTING AND QUERYING
import os
import numpy as np
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import euclidean

# Load BERT model and tokenizer for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def generate_embedding(text):
    """
    Generate embeddings for the given text using a pre-trained BERT model.
    
    Args:
    - text (str): The input text.

    Returns:
    - np.ndarray: The flattened embedding for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    # Flatten the embedding to 1D
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embedding

# Initialize environment variables
workspace_url = os.environ.get("WORKSPACE_URL")
sp_client_id = os.environ.get("SP_CLIENT_ID")
sp_client_secret = os.environ.get("SP_CLIENT_SECRET")

# Initialize VectorSearchClient with disable_notice to suppress the warning
vsc = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=sp_client_id,
    service_principal_client_secret=sp_client_secret,
    disable_notice=True  # Suppress the authentication notice
)

index = vsc.get_index(endpoint_name="group-six", index_name="main.default.group_six_struct_vsi")

query_text = "Enter your query: "

# Perform similarity search
results = index.similarity_search(num_results=3, columns=["text"], query_text=query_text)

# Generate query embedding and flatten it to 1D
query_embedding = generate_embedding(query_text)

def calculate_euclidean_distance(vector1, vector2):
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
    - vector1 (np.ndarray): First vector.
    - vector2 (np.ndarray): Second vector.

    Returns:
    - float: Euclidean distance between vector1 and vector2.
    """
    return euclidean(vector1, vector2)

def cosine_similarity_to_percentage(cosine_similarity):
    """
    Converts cosine similarity to percentage.
    Cosine similarity ranges from -1 to 1.
    This function maps it to 0% to 100%.
    """
    return ((cosine_similarity + 1) / 2) * 100

def euclidean_distance_to_percentage(distance, max_distance=2):
    """
    Normalizes Euclidean distance to a percentage.
    We assume max_distance = 2 (which would be a large distance in high-dimensional space).
    This is arbitrary and can be adjusted based on the dataset's scale.
    """
    return max(0, (1 - distance / max_distance) * 100)

def process_results(results, query_embedding):
    """
    Process and format the vector search results, and calculate Euclidean distances.
    
    Args:
    - results (dict): The vector search results in JSON format.
    - query_embedding (np.ndarray): The embedding of the query text.

    Returns:
    - str: A formatted string of the extracted text content with similarity and Euclidean distance.
    """
    if isinstance(results, dict):
        # Access the 'data_array' from the JSON response
        data_array = results.get('result', {}).get('data_array', [])
        
        if isinstance(data_array, list):
            result_texts = []
            for idx, entry in enumerate(data_array):
                if isinstance(entry, list) and len(entry) == 2:
                    content, cosine_sim = entry
                    # Generate embedding for the result content and flatten it to 1D
                    result_embedding = generate_embedding(content)
                    # Calculate Euclidean distance between query and result embedding
                    euclidean_dist = calculate_euclidean_distance(query_embedding, result_embedding)
                    
                    # Convert cosine similarity to percentage
                    cosine_sim_pct = cosine_similarity_to_percentage(cosine_sim)
                    # Normalize Euclidean distance to percentage
                    euclidean_dist_pct = euclidean_distance_to_percentage(euclidean_dist)

                    result_text = (f"Result {idx + 1}:\n"
                                   f"Content: {content}\n"
                                   f"Cosine Similarity: {cosine_sim_pct:.2f}%\n"
                                   f"Euclidean Distance: {euclidean_dist_pct:.2f}%\n"
                                   f"{'-' * 115}")
                    result_texts.append(result_text)
            return "\n".join(result_texts) if result_texts else "No valid results found in the 'data_array'."
        else:
            return "Unexpected format for 'data_array'."
    else:
        return "Unexpected format of results. Please check the structure."

# Process the results and calculate Euclidean distance
formatted_results = process_results(results, query_embedding)

# Output the formatted results
print(f"Formatted Results:\n{formatted_results}")


# COMMAND ----------

# import cohere

# # Initialize Cohere client with your API key
# co = cohere.Client("yKsx1iJsGbkajUpZyHygjNPyZkQ0s4nro1wuROGo")

# def generate_response_from_cohere(query, context):
#     """
#     Use Cohere's API to generate a refined response based on the query and retrieved context.
    
#     Args:
#     - query (str): The user's query.
#     - context (str): The retrieved text content for augmentation.

#     Returns:
#     - str: The generated refined response from Cohere.
#     """
#     prompt = (f"Given the following query: '{query}', "
#               f"and considering this context: '{context}', "
#               "please provide a refined, relevant answer that addresses the query.")
    
#     response = co.generate(
#         model='command-xlarge-nightly',  # Choose an appropriate Cohere model
#         prompt=prompt,
#         max_tokens=200,
#         temperature=0.7
#     )
    
#     return response.generations[0].text.strip()

# # Example integration into your result processing function
# def process_and_refine_results_with_cohere(results, query_embedding, query_text):
#     """
#     Process, refine, and format the vector search results using RAG (retrieval-augmented generation).
    
#     Args:
#     - results (dict): The vector search results.
#     - query_embedding (np.ndarray): Embedding of the query text.
#     - query_text (str): The original user query.

#     Returns:
#     - str: A refined, generated answer combining retrieval and generation.
#     """
#     if isinstance(results, dict):
#         data_array = results.get('result', {}).get('data_array', [])
        
#         if isinstance(data_array, list):
#             best_result_content = None
#             best_cosine_sim = -1
#             result_texts = []
            
#             for idx, entry in enumerate(data_array):
#                 if isinstance(entry, list) and len(entry) == 2:
#                     content, cosine_sim = entry
#                     result_embedding = generate_embedding(content)
#                     euclidean_dist = calculate_euclidean_distance(query_embedding, result_embedding)

#                     cosine_sim_pct = cosine_similarity_to_percentage(cosine_sim)
#                     euclidean_dist_pct = euclidean_distance_to_percentage(euclidean_dist)

#                     result_text = (f"Result {idx + 1}:\n"
#                                    f"Content: {content}\n"
#                                    f"Cosine Similarity: {cosine_sim_pct:.2f}%\n"
#                                    f"Euclidean Distance: {euclidean_dist_pct:.2f}%\n"
#                                    f"{'-' * 115}")
#                     result_texts.append(result_text)

#                     if cosine_sim > best_cosine_sim:
#                         best_cosine_sim = cosine_sim
#                         best_result_content = content

#             if best_result_content:
#                 # Use Cohere to generate a refined response
#                 refined_answer = generate_response_from_cohere(query_text, best_result_content)
#                 return f"Refined Answer:\n{refined_answer}\n\n" + "\n".join(result_texts)
#             else:
#                 return "No valid results found in the 'data_array'."
#         else:
#             return "Unexpected format for 'data_array'."
#     else:
#         return "Unexpected format of results."

# # Example usage:
# formatted_results = process_and_refine_results_with_cohere(results, query_embedding, query_text)
# print(f"Formatted and Refined Results:\n{formatted_results}")


# COMMAND ----------


