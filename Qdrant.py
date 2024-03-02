from typing import List
from qdrant_client.http import models
from qdrant_client.conversions import common_types as types
from schemas import IndexContent
from qdrant_client import QdrantClient
import os


class QdrantDB:
    def __init__(self) -> None:
        self.client = QdrantClient(api_key = os.environ["Qdrant_api_key"], url = os.environ["Qdrant_url"])

    def get_collections(self) -> List[str]:
        collection_list = []
        try:
            collections = self.client.get_collections()
            for i in collections.collections:
                collection_list.append(i.name)
        except:
            pass

        return collection_list

    def delete_collections(self, collection_name: str) -> bool:
        response = self.client.delete_collection(collection_name)
        return response

    def create_collections(
        self, collection_name: str
    ) -> str:
        if collection_name == None or collection_name == "":
            raise Exception("Collection name could not be None")

        collection_list = self.get_collections()

        if collection_name in collection_list:
            raise Exception("Collection already exists")

        response = self.client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

        if response == True:
            return collection_name
        else:
            raise Exception("Failed in creating collection due to some Internal Error")

    def update_collection(self, collection_name: str, index_data: IndexContent) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=index_data.id,
                    payload=index_data.payload,  # Add any additional payload if necessary
                    vector={
                        "text-sparse": models.SparseVector(
                            indices=index_data.sparse_indices.tolist(),
                            values=index_data.sparse_values.tolist(),
                        )
                    },
                )
            ],
        )

    def search_data(
        self, collection_name, input_data: IndexContent
    ) -> List[dict]:
        collection_list = self.get_collections()

        data : List[dict] = []

        if not collection_name in collection_list:
            raise Exception("Given references are not available")

        fetched_data : List[List[types.ScoredPoint]] = self.client.search_batch(
                collection_name=collection_name,
                requests=[
                    models.SearchRequest(
                        vector=models.NamedSparseVector(
                            name="text-sparse",
                            vector=models.SparseVector(
                                indices=input_data.sparse_indices.tolist(),
                                values=input_data.sparse_values.tolist(),
                            ),
                        ),
                        limit=input_data.number_of_context,
                        with_payload=True,
                    ),
                ],
            )
        for search_context in fetched_data[0]:
            data.append(search_context.payload)


        return data
