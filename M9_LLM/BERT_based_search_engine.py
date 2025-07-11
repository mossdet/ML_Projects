import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
            
class BERT_TextSearch:

    def __init__(self, text_fields,type="bert-base-uncased", n_batches=8):
        self.text_fields = text_fields
        self.type = type
        self.n_batches = n_batches
        self.matrices = {}

        self.tokenizer = BertTokenizer.from_pretrained(self.type)
        self.model = BertModel.from_pretrained(self.type)
        self.model.eval()  # Set the model to evaluation mode if not training
        self.model_params = None
        pass

    def make_batches(self, seq, n):
        result = []
        for i in range(0, len(seq), n):
            batch = seq[i:i+n]
            result.append(batch)
        return result

    def compute_embeddings(self, records):
        text_batches = self.make_batches(records, self.n_batches)
        all_embeddings = []
        
        for batch in tqdm(text_batches):
            encoded_input = self.tokenizer(batch, **self.model_params)
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                hidden_states = outputs.last_hidden_state
                
                batch_embeddings = hidden_states.mean(dim=1)
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings_np)
        
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings

    def fit(self, records, model_params={}):
        self.model_params = model_params
        self.df = pd.DataFrame(records)
        for f in self.text_fields:
            X_emb = self.compute_embeddings(self.df[f].to_list())
            self.matrices[f] = X_emb

    def search(self, query, n_results=10):
        score = np.zeros(len(self.df))

        # Get the embedding for the query
        encoded_input = self.tokenizer([query], **self.model_params)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            q_emb = hidden_states.mean(dim=1)
            q_emb = q_emb.cpu().numpy()

        for f in self.text_fields:
            # Compute cosine similarity between query embedding and documents' embeddings
            s = cosine_similarity(self.matrices[f], q_emb).flatten()
            score += s

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')


if __name__ == "__main__":

    # Example usage

    # Read Shakeaspeare text file
    play_df = pd.read_csv('Week_6_DeepLearning/Data/Shakespeare_data.csv', encoding='utf-8')
    play_df = play_df.dropna()
    play_df = play_df[['PlayerLine']].reset_index(drop=True)
    documents = play_df.to_dict(orient='records')

    # Limit to first 1000 records for testing
    documents = documents[:1000]
    
    # Define the text fields to be indexed
    text_fields = ['PlayerLine']

    index = BERT_TextSearch(
    text_fields=text_fields, type="bert-base-uncased", n_batches=8,
    )
    model_params = {
        'padding':True,
        'truncation':True,
        'return_tensors':'pt',
    }
    index.fit(documents, model_params=model_params)

    query = 'We are very shaken' # Example query from the dataset
    # Searching the index
    print("Searching for query:", query)
    results = index.search(
        query=query,
        n_results=5,
    )

    # Predict the sentence with the closest meaning to the query
    print(f"Query: {query}")
    print(f"Result: {results[0]['PlayerLine']}\n\n")
