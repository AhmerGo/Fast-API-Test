import sys
import logging
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database_helper import Connection
from bson import json_util
from asyncio import Lock
sys.path.insert(1, '../LLM/')
from HSU import HSU

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    user_id: str
    chat_history: list = []

class SaveChatRequest(BaseModel):
    user_id: str
    user_inputs: list
    bot_inputs: list

model_path = "../LLM/Models/wizardlm-13b-v1.2.Q4_0.gguf"
index_path = "../LLM/HSU_index"

lock = Lock()

@app.get("/")
def index():
    return {"message": "API is running"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

async def process_chat_request(user_input, user_id):
    try:
        result = HSU.rag(user_input, user_id)
        if result and 'answer' in result:
            logging.info(f"Generated response: {result['answer']}")
            return {"reply": result['answer']}
        else:
            logging.error("Failed to generate a response.")
            return None
    except Exception as e:
        logging.error(f"An error occurred during chat processing: {e}", exc_info=True)
        return None
    
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    user_id = request.user_id
    chat_history = request.chat_history

    if not user_input:
        logging.warning("User input is missing")
        raise HTTPException(
            status_code=400,
            detail="No user_input provided"
        )

    result = await process_chat_request(user_input, user_id)
    if result:
        return result
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat"
        )

@app.post("/save_chat")
async def save_chat(request: SaveChatRequest):
    user_id = request.user_id
    user_inputs = request.user_inputs
    bot_inputs = request.bot_inputs

    if not user_inputs or not bot_inputs:
        return JSONResponse(content={"error": "Missing required parameters"}, status_code=400)

    async def save_chat_history(user_id, user_inputs, bot_inputs):
        try:
            db_connection = Connection()
            db_connection.connect("admin", "password", "admin")  # Placeholder password
            # Retrieve the existing chat history for the user from MongoDB
            chat_history = db_connection.get_chat_history(user_id)
            # Append the new user inputs and bot inputs to the chat history
            chat_history.extend(zip(user_inputs, bot_inputs))
            # Update the chat history in MongoDB
            db_connection.update_chat_history(user_id, chat_history)
            db_connection.close()
        except Exception as e:
            logging.error(f"An error occurred while saving chat history: {e}", exc_info=True)

    try:
        await asyncio.gather(save_chat_history(user_id, user_inputs, bot_inputs))
        return JSONResponse(content={"message": "Chat history updated successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, workers=2)
