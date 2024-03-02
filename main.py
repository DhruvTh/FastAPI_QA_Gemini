from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from schemas import LLMInput
from dataloader import CSVIndex
from llm import GeminiAILLM
import uvicorn

app = FastAPI()

load_dotenv()

llm_model = GeminiAILLM()

collection_name = "test_data"

@app.post("/stream_chat")
def stream_chat(
    input_data: LLMInput
):
    stream_data = llm_model.generate_stream_response(input_data)
    return StreamingResponse(stream_data, media_type="text/plain")

@app.get("/index_data")
def stream_chat():

    csv_loader = CSVIndex()

    csv_loader.index_csv("Financebench.csv", collection_name)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        timeout_keep_alive=9000
    )
