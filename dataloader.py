from typing import List, Tuple
from schemas import IndexContent
from typing import List
from tqdm import tqdm
from Qdrant import QdrantDB
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch import Tensor
import pandas as pd


class SpladeEmbedding:
    def __init__(self) -> None:
        doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
        self.doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id)

        query_model_id = "naver/efficient-splade-VI-BT-large-query"
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
        self.query_model = AutoModelForMaskedLM.from_pretrained(query_model_id)

    def compute_vector(self, text: str, tokenizer, sparse_model) -> tuple[Tensor]:
        tokens = tokenizer(text, return_tensors="pt")
        output = sparse_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()

        return vec, tokens

    def doc_compute_vectors(self, text: str) -> tuple[list, list]:

        query_vec, tokens = self.compute_vector(
            text, self.doc_tokenizer, self.doc_model
        )
        query_indices = query_vec.nonzero().numpy().flatten()
        query_values = query_vec.detach().numpy()[query_indices]
        return query_indices, query_values

    def query_compute_vectors(self, text: str) -> tuple[list, list]:

        query_vec, tokens = self.compute_vector(
            text, self.query_tokenizer, self.query_model
        )
        query_indices = query_vec.nonzero().numpy().flatten()
        query_values = query_vec.detach().numpy()[query_indices]
        return query_indices, query_values

class CSVIndex:
    def __init__(
        self,
    ) -> None:
        self.sparse_embedding_model = SpladeEmbedding()
        self.vector_db = QdrantDB()


    def csv_reader(self, file_name : str) -> List[str]:
        try:
            data = pd.read_csv(file_name)
            
            # Display the first few rows of the DataFrame
            data_records = data.to_dict(orient = "records")
            extracted_data = []
            for i in data_records:
                data = {}
                data["text"] = f"""
                question : {i["question"]}
                answer : {i["answer"]}
                """
                data["source"] = i["doc_link"]
                data["evidence_text"] = i["evidence_text"]
                extracted_data.append(data)

            return extracted_data

        except Exception as e:
            print(f"An error occurred: {e}")


    def index_csv(self, file_name : str, collection_name : str) :
        text_data = self.csv_reader(file_name)

        collection_name = self.vector_db.create_collections(
            collection_name,
        )

        embedded_data: List[IndexContent] = []

        for id in tqdm(range(0, len(text_data))):
            index_data = IndexContent()
            index_data.id = id
            index_data.payload = {
                "page_content": text_data[id]["text"],
                "source": text_data[id]["source"],
                "evidenced_text" : text_data[id]["evidence_text"]
            }

            query_indices, query_values = (
                self.sparse_embedding_model.doc_compute_vectors(text_data[id]["text"])
            )

            index_data.sparse_indices = query_indices
            index_data.sparse_values = query_values
            embedded_data.append(index_data)

        for data in tqdm(embedded_data):
            self.vector_db.update_collection(collection_name, data)

    
    def find_data(
        self, query: str, collection_name : str
    ) -> Tuple[List[dict]]:

        index_data = IndexContent()


        query_indices, query_values = self.sparse_embedding_model.query_compute_vectors(
            query
        )

        index_data.sparse_indices = query_indices
        index_data.sparse_values = query_values

        context_data = self.vector_db.search_data(
            collection_name, index_data
        )
        
        return context_data
