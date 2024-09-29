
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn.functional import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load your dataset (CSV files from the 'content' folder)
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

# Load dataset from the 'content' folder
dataset = load_data('./content')  # Adjust the path as necessary

# Ensure columns like 'name' have no NaN values
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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

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

    # Convert Gift Embeddings to a tensor
    gift_embeddings_tensor = torch.stack(dataset['Gift_Embedding'].tolist())

    # Calculate similarity
    similarities = cosine_similarity(user_embedding, gift_embeddings_tensor)

    # Add similarity scores to dataset
    dataset['Similarity'] = similarities.tolist()

    # Recommend top 5 gifts based on similarity
    recommended_gifts = dataset.nlargest(5, 'Similarity')[['name', 'ratings', 'link', 'actual_price']]
    return recommended_gifts.to_dict(orient='records')

# Entry point to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8069)
# import pandas as pd
# import torch
# from transformers import DistilBertTokenizer, DistilBertModel
# from torch.nn.functional import cosine_similarity
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os

# # Load the DistilBERT tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # Load your dataset (CSV files from the 'content' folder)
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

# # Load dataset from the 'content' folder
# dataset = load_data('./content')  # Adjust the path as necessary

# # Ensure columns like 'name' have no NaN values
# dataset['name'].fillna('', inplace=True)
# dataset['ratings'].fillna(0, inplace=True)

# # Function to encode text using DistilBERT
# def encode_text_batch(text_list, batch_size=16):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embeddings = []

#     for i in range(0, len(text_list), batch_size):
#         batch = text_list[i:i+batch_size]
#         inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = model(**inputs)
#         batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu()
#         embeddings.append(batch_embeddings)

#     return torch.cat(embeddings, dim=0)

# # Prepare embeddings for all gifts in batches
# def prepare_gift_embeddings():
#     gift_texts = dataset['name'].tolist()
#     gift_embeddings = encode_text_batch(gift_texts, batch_size=16)
#     dataset['Gift_Embedding'] = list(gift_embeddings)

# # Prepare embeddings for gifts once
# prepare_gift_embeddings()

# # FastAPI application instance
# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust this for production
#     allow_credentials=True,
#     allow_methods=["GET","POST"],
#     allow_headers=["*"],
# )

# # Request model for gift recommendations
# class GiftRequest(BaseModel):
#     age: str
#     gender: str
#     relationship: str
#     occasion: str
# @app.get("/recommend_gifts/")
# async def recommend_gifts_get(age: str, gender: str, relationship: str, occasion: str):
#     # New GET method implementation
#     user_input = f"{age} {gender} {relationship} {occasion}"
#     user_embedding = encode_text_batch([user_input])[0].unsqueeze(0)  # Single input

#     # Convert Gift Embeddings to a tensor
#     gift_embeddings_tensor = torch.stack(dataset['Gift_Embedding'].tolist())

#     # Calculate similarity
#     similarities = cosine_similarity(user_embedding, gift_embeddings_tensor)

#     # Add similarity scores to dataset
#     dataset['Similarity'] = similarities.tolist()

#     # Recommend top 5 gifts based on similarity
#     recommended_gifts = dataset.nlargest(5, 'Similarity')[['name', 'ratings', 'link', 'actual_price']]
#     return recommended_gifts.to_dict(orient='records')
# @app.post("/recommend_gifts/")
# async def recommend_gifts(request: GiftRequest):
#     # User input embedding
#     user_input = f"{request.age} {request.gender} {request.relationship} {request.occasion}"
#     user_embedding = encode_text_batch([user_input])[0].unsqueeze(0)  # Single input

#     # Convert Gift Embeddings to a tensor
#     gift_embeddings_tensor = torch.stack(dataset['Gift_Embedding'].tolist())

#     # Calculate similarity
#     similarities = cosine_similarity(user_embedding, gift_embeddings_tensor)

#     # Add similarity scores to dataset
#     dataset['Similarity'] = similarities.tolist()

#     # Recommend top 5 gifts based on similarity
#     recommended_gifts = dataset.nlargest(5, 'Similarity')[['name', 'ratings', 'link', 'actual_price']]
#     return recommended_gifts.to_dict(orient='records')

# # Entry point to run the app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)