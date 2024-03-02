from fastapi import Query
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field, validator
# from src.utils.Constant import *

class IndexContent(BaseModel):
    id: Optional[str] = None
    payload: Optional[dict] = None
    sparse_indices: Optional[str] = None
    sparse_values: Optional[str] = None
    number_of_context: Optional[int] = 3



class LLMInput(BaseModel):
    query : str 
    system_msg: Optional[str] = (
        "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown"
    )
    temperature: float = Query(default=0.7, ge=0, le=1)
    collection_name : str = "test_data"
