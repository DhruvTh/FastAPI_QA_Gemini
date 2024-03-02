from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Iterator, Union
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from schemas import LLMInput
from typing import Iterator, Any
import json
from langchain_core.prompts import ChatPromptTemplate
from dataloader import CSVIndex
import traceback

prompt_modification_string = """You're a researcher who responds to user queries by extracting relevant information from the context and useful documents. Please include sources with your answers if available. Avoid duplicating sources. You do not need to rely completely on sources for answer. If the answer isn't found in the context, you may provide an answer based on your knowledge and interaction with user or indicate that the answer is unknown. 
    Make sure to add sources at the end of answer and make sure that each source is unique.  
    Do not include sources like this. If multiple context has same source then write only one source.
    Ex :  
    * http://tgzow4.pdf
    * http://tgzow4.pdf
    
    Context: {data_string}

    Query: {query}"""

class GeminiAILLM():
    def __init__(self) -> None:
        self.api_key = os.environ["GeminiAI_API_KEY"]
        self.tool = CSVIndex()

    
    def modify_prompt(self, input_data: LLMInput) -> str:

        query = input_data.query

        data = self.tool.find_data(query, input_data.collection_name)

        data_string = "\n\n".join(json.dumps(item) for item in data)

        modified_query = ChatPromptTemplate.from_template(
                prompt_modification_string
            )

        query = modified_query.format_messages(
                data_string=str(data_string), query=query
            )
        print(query[0].content)
        return query[0].content

    def prepare_prompt(
        self, input_data: LLMInput
    ) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        
        messages: List[Union[AIMessage, HumanMessage, SystemMessage]] = []
        messages.append(SystemMessage(content=input_data.system_msg))
        messages.append(HumanMessage(content=input_data.query))

        messages[-1].content = self.modify_prompt(input_data)

        return messages

    def get_llm(self, input_data: LLMInput) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=input_data.temperature,
                google_api_key=self.api_key,
                convert_system_message_to_human=True,
            )

    def get_stream_data(
        self,
        llm: ChatGoogleGenerativeAI,
        input_data: LLMInput,
        messages: List[Union[HumanMessage, AIMessage]] | HumanMessage,
    ) -> Iterator:
        return llm.stream(messages)

    def get_chunk_content(self, chunk) -> str:
        return chunk.content

    def generate_stream_response(self, input_data: LLMInput) -> Iterator:
        try:

            messages = self.prepare_prompt(input_data)

            llm = self.get_llm(input_data)

            response = self.get_stream_data(llm, input_data, messages)
            
            for chunk in response:
                if self.get_chunk_content(chunk) != None:
                    yield self.get_chunk_content(chunk)

        except Exception as error:

            traceback_str = "".join(
                traceback.format_exception(None, error, error.__traceback__)
            )

            yield traceback_str