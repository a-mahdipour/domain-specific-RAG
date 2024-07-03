"""
Impelmenting domain-specific RAG system within an organization in a high-level overview.
"""

import numpy as np
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load models and tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to get embeddings
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Preprocess and embed documents
documents = ["Document 1 content", "Document 2 content", "Document 3 content"]
doc_embeddings = [get_embedding(doc, bert_tokenizer, bert_model) for doc in documents]

# Function to retrieve relevant documents
def retrieve_documents(query, doc_embeddings, top_k=3):
    query_embedding = get_embedding(query, bert_tokenizer, bert_model)
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Function to generate response
def generate_response(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=150)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example query
query = "Explain document 1"
retrieved_docs = retrieve_documents(query, doc_embeddings)
response = generate_response(query, retrieved_docs)

print(response)
