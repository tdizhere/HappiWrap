# import os
# import pandas as pd
# import torch
# from transformers import DistilBertTokenizer, DistilBertModel
# from torch.nn.functional import cosine_similarity
# from flask import Flask, request, jsonify
# import pickle

# # Initialize Flask app
# app = Flask(__name__)

# # Set paths for data and pickle files
# DATA_FOLDER = './content'
# MAIN_PICKLE_FILE = 'main_embeddings.pkl'
# BACKUP_PICKLE_FILE = 'backup.pkl'

# # Load your dataset (CSV files from the 'data' folder)
# def load_data(folder_path):
#     all_data = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(folder_path, filename)
#             data = pd.read_csv(file_path)
#             all_data.append(data)
#     if not all_data:
#         raise ValueError("No CSV files found in the specified folder.")
#     return pd.concat(all_data, ignore_index=True)

# # Load dataset from 'data' folder
# dataset = load_data(DATA_FOLDER)

# # Ensure columns like 'name' and 'ratings' have no NaN values
# dataset['name'] = dataset['name'].fillna('')
# dataset['ratings'] = dataset['ratings'].fillna(0)

# # Set environment variable to disable SSL verification
# os.environ['TRANSFORMERS_VERIF_SSL'] = 'false'

# # Load the DistilBERT tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# embeddings_ready = False

# # Function to encode text using DistilBERT
# def encode_text_batch(text_list, batch_size=16):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embeddings = []

#     for i in range(0, len(text_list), batch_size):
#         batch = text_list[i:i + batch_size]
#         inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = model(**inputs)
#         batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu()
#         embeddings.append(batch_embeddings)

#     return torch.cat(embeddings, dim=0)

# # Prepare embeddings for all gifts in batches
# def prepare_gift_embeddings():
#     global embeddings_ready

#     # Load embeddings from the main pickle file if it exists
#     if os.path.exists(MAIN_PICKLE_FILE):
#         print("Loading gift embeddings from the main pickle file...")
#         with open(MAIN_PICKLE_FILE, 'rb') as f:
#             gift_embeddings = pickle.load(f)
#         dataset['Gift_Embedding'] = list(gift_embeddings)
#         embeddings_ready = True
#     elif os.path.exists(BACKUP_PICKLE_FILE):
#         # Load backup embeddings if main is not available
#         print("Loading gift embeddings from the backup pickle file...")
#         with open(BACKUP_PICKLE_FILE, 'rb') as f:
#             gift_embeddings = pickle.load(f)
#         dataset['Gift_Embedding'] = list(gift_embeddings)
#         embeddings_ready = True
#     else:
#         # Generate embeddings if neither file is available
#         print("Generating and saving gift embeddings...")
#         gift_texts = dataset['name'].tolist()
#         gift_embeddings = encode_text_batch(gift_texts, batch_size=16)
#         dataset['Gift_Embedding'] = list(gift_embeddings)
#         print("Loading gift embeddings from the backup pickle file...")
#         # with open(BACKUP_PICKLE_FILE, 'rb') as f:
#         #     gift_embeddings = pickle.load(f)
#         # dataset['Gift_Embedding'] = list(gift_embeddings)
#         # embeddings_ready = True

#         # Save embeddings to the main and backup pickle files
#         with open(MAIN_PICKLE_FILE, 'wb') as f:
#             pickle.dump(gift_embeddings, f)
#         with open(BACKUP_PICKLE_FILE, 'wb') as f:
#             pickle.dump(gift_embeddings, f)

#         embeddings_ready = True
#         print("Gift embeddings saved to both main and backup pickle files.")

# # Function to recommend gifts based on user input
# @app.route('/recommend', methods=['GET'])
# def recommend_gifts():
#     global embeddings_ready
#     if not embeddings_ready:
#         prepare_gift_embeddings()  # Ensure embeddings are ready before processing

#     if not embeddings_ready:
#         return jsonify({"error": "Embeddings are not ready yet. Please try again later."}), 503

#     age = request.args.get('age')
#     gender = request.args.get('gender')
#     relationship = request.args.get('relationship')
#     occasion = request.args.get('occasion')

#     if not all([age, gender, relationship, occasion]):
#         return jsonify({"error": "All parameters (age, gender, relationship, occasion) are required."}), 400

#     # User input embedding
#     user_input = f"{age} {gender} {relationship} {occasion}"
#     user_embedding = encode_text_batch([user_input])[0].unsqueeze(0)  # Single input

#     # Calculate similarity
#     similarities = []
#     for index, row in dataset.iterrows():
#         gift_embedding = row['Gift_Embedding'].unsqueeze(0)  # Ensure correct shape
#         sim = cosine_similarity(user_embedding, gift_embedding).item()
#         similarities.append(sim)

#     # Add similarity scores to dataset
#     dataset['Similarity'] = similarities

#     # Recommend top 5 gifts based on similarity
#     recommended_gifts = dataset.nlargest(5, 'Similarity')[['name', 'ratings', 'link', 'actual_price']]
#     return jsonify(recommended_gifts.to_dict(orient='records'))

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.functional import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load your dataset (CSV files from the 'data' folder)
def load_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)
    if not all_data:
        raise ValueError("No CSV files found in the specified folder.")
    return pd.concat(all_data, ignore_index=True)

# Load dataset from the 'data' folder
dataset = load_data('data')  # Adjust the path as necessary

# Ensure columns like 'Gift' have no NaN values
dataset['name'].fillna('', inplace=True)
dataset['ratings'].fillna(0, inplace=True)

# Function to encode text using DistilBERT
def encode_text_batch(text_list, batch_size=16):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu()
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)

# Prepare embeddings for all gifts in batches
def prepare_gift_embeddings():
    gift_texts = dataset['name'].tolist()
    gift_embeddings = encode_text_batch(gift_texts, batch_size=16)
    dataset['Gift_Embedding'] = list(gift_embeddings)

# Prepare embeddings for gifts once
prepare_gift_embeddings()

# FastAPI application instance
app = FastAPI()

# Request model for gift recommendations
class GiftRequest(BaseModel):
    age: str
    gender: str
    relationship: str
    occasion: str

@app.post("/recommend_gifts/")
async def recommend_gifts(request: GiftRequest):
    # User input embedding
    user_input = f"{request.age} {request.gender} {request.relationship} {request.occasion}"
    user_embedding = encode_text_batch([user_input])[0].unsqueeze(0)  # Single input

    # Calculate similarity
    similarities = []
    for index, row in dataset.iterrows():
        gift_embedding = row['Gift_Embedding'].unsqueeze(0)  # Ensure correct shape
        sim = cosine_similarity(user_embedding, gift_embedding).item()
        similarities.append(sim)

    # Add similarity scores to dataset
    dataset['Similarity'] = similarities

    # Recommend top 5 gifts based on similarity
    recommended_gifts = dataset.nlargest(5, 'Similarity')[['name', 'ratings', 'link', 'actual_price']]
    return recommended_gifts.to_dict(orient='records')

# Entry point to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
