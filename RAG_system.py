"""
Here are teh steps:
- RAGSystem class initializes the tokenizers and models for BERT and GPT-2.
- FAISS index is used for efficient nearest-neighbor search.
- add_documents method converts documents to embeddings and adds them to the FAISS index.
Embedding Generation:
- get_embedding method uses BERT to convert text into embeddings.
- retrieve_documents method uses FAISS to find the top-k similar documents based on cosine similarity.
Response Generation:
- generate_response method concatenates the query with the retrieved documents and generates a response using GPT-2.
- answer_query method integrates retrieval and generation to provide a final answer.
"""

# Setup and Imports

# pip install transformers sklearn torch faiss-cpu

#Implementing RAG

import numpy as np
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import faiss

class RAGSystem:
    def __init__(self, bert_model_name='bert-base-uncased', gpt2_model_name='gpt2'):
        # Load tokenizers and models
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(768)  # 768 is the dimension of BERT embeddings
        self.documents = []

    def add_documents(self, documents):
        """Add documents to the RAG system."""
        self.documents = documents
        embeddings = [self.get_embedding(doc) for doc in documents]
        embeddings = np.array(embeddings)
        self.index.add(embeddings)

    def get_embedding(self, text):
        """Convert text to BERT embedding."""
        inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

    def retrieve_documents(self, query, top_k=3):
        """Retrieve top-k relevant documents for a given query."""
        query_embedding = self.get_embedding(query)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_response(self, query, retrieved_docs):
        """Generate a response using GPT-2 with retrieved documents as context."""
        context = " ".join(retrieved_docs)
        input_text = f"Query: {query}\nContext: {context}\nAnswer:"
        inputs = self.gpt2_tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)
        return self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_query(self, query):
        """Main method to process a query and return an answer."""
        retrieved_docs = self.retrieve_documents(query)
        response = self.generate_response(query, retrieved_docs)
        return response

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Document 1 content about AI.",
        "Document 2 content about Machine Learning.",
        "Document 3 content about Deep Learning."
    ]

    # Initialize RAG System
    rag_system = RAGSystem()

    # Add documents to the system
    rag_system.add_documents(documents)

    # Example query
    query = "Tell me about AI"
    response = rag_system.answer_query(query)

    print(f"Query: {query}")
    print(f"Response: {response}")
