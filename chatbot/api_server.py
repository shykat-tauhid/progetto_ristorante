import os
from threading import Lock
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError
from pydantic import BaseModel, Field

from openai.types.chat import ChatCompletionMessageParam

from chatbot.restaurant_bot import (
    REQUEST_TIMEOUT,
    chat_turn,
    create_conversation,
    create_model_client,
)


app = FastAPI(
    title="Chat ristorante API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
model_client = create_model_client()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class SessionResponse(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    reply: str


class ChatSession:
    def __init__(self) -> None:
        self.messages: list[ChatCompletionMessageParam] = create_conversation()
        self.lock = Lock()


sessions: dict[str, ChatSession] = {}
sessions_lock = Lock()


@app.post("/api/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    session_id = str(uuid4())
    with sessions_lock:
        sessions[session_id] = ChatSession()
    return SessionResponse(session_id=session_id)


@app.post("/api/sessions/{session_id}/reset", response_model=SessionResponse)
def reset_session(session_id: str) -> SessionResponse:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    with session.lock:
        session.messages = create_conversation()
    return SessionResponse(session_id=session_id)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_input = request.message.strip()
    if not user_input:
        raise HTTPException(status_code=422, detail="Il messaggio non può essere vuoto")
    with sessions_lock:
        session = sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata; ricarica la pagina")

    try:
        with session.lock:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as http:
                reply = chat_turn(user_input, session.messages, model_client, http)
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Errore contattando il modello: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Risposta del modello non supportata: {exc}",
        ) from exc
    return ChatResponse(reply=reply)
