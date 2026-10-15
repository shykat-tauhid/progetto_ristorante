import json
import os
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


load_dotenv()

API_BASE_URL = os.getenv("RESTAURANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "8"))

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "create_reservation",
            "description": "Crea una prenotazione. Comunica all'utente l'ID restituito, necessario per modificare o cancellare la prenotazione.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Nome e cognome"},
                    "phone": {"type": "string", "description": "Numero di telefono"},
                    "reservation_at": {
                        "type": "string",
                        "description": "Data e ora ISO 8601 locale, per esempio 2026-10-20T20:00:00",
                    },
                    "party_size": {"type": "integer", "minimum": 1, "description": "Numero di persone"},
                    "notes": {"type": "string", "description": "Note facoltative"},
                },
                "required": ["customer_name", "phone", "reservation_at", "party_size"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_reservation",
            "description": "Modifica una prenotazione esistente usando il suo ID. Invia solo i campi che devono cambiare.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "integer", "minimum": 1},
                    "customer_name": {"type": "string"},
                    "phone": {"type": "string"},
                    "reservation_at": {"type": "string", "description": "Data e ora ISO 8601 locale"},
                    "party_size": {"type": "integer", "minimum": 1},
                    "notes": {"type": "string"},
                },
                "required": ["reservation_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reservation",
            "description": "Cancella una prenotazione esistente usando il suo ID.",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "integer", "minimum": 1}},
                "required": ["reservation_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": "Restituisce i piatti disponibili nel menu e i relativi prezzi.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_info",
            "description": "Restituisce nome, indirizzo, contatti e orari del ristorante.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def _request(
    http: httpx.Client,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    response = http.request(method, f"{API_BASE_URL}{path}", json=payload)
    if response.status_code == 204:
        return {"success": True, "message": "Prenotazione cancellata"}
    response.raise_for_status()
    return response.json()


def execute_tool(
    http: httpx.Client,
    name: str,
    arguments: dict[str, Any],
) -> Any:
    if name == "create_reservation":
        return _request(http, "POST", "/reservations", payload=arguments)
    if name == "update_reservation":
        reservation_id = arguments.get("reservation_id")
        if not isinstance(reservation_id, int) or isinstance(reservation_id, bool):
            raise ValueError("reservation_id deve essere un intero")
        updates = {key: value for key, value in arguments.items() if key != "reservation_id"}
        if not updates:
            raise ValueError("Specificare almeno un campo da modificare")
        return _request(http, "PATCH", f"/reservations/{reservation_id}", payload=updates)
    if name == "delete_reservation":
        reservation_id = arguments.get("reservation_id")
        if not isinstance(reservation_id, int) or isinstance(reservation_id, bool):
            raise ValueError("reservation_id deve essere un intero")
        return _request(http, "DELETE", f"/reservations/{reservation_id}")
    if name == "get_menu":
        return _request(http, "GET", "/menu")
    if name == "get_restaurant_info":
        return _request(http, "GET", "/restaurant-info")
    raise ValueError(f"Tool non riconosciuto: {name}")


def _tool_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        return {"error": f"API HTTP {exc.response.status_code}: {detail}"}
    if isinstance(exc, httpx.RequestError):
        return {
            "error": (
                f"Non riesco a raggiungere l'API del PC1 ({API_BASE_URL}). "
                "Verifica indirizzo, server e rete."
            )
        }
    if isinstance(exc, (ValueError, TypeError)):
        return {"error": str(exc)}
    raise exc


def create_model_client() -> OpenAI:
    return OpenAI(
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )


def create_conversation() -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                "Sei l'assistente di un ristorante. Rispondi in italiano, in modo cortese e conciso. "
                "Usa i tool per menu, informazioni e operazioni sulle prenotazioni; non inventare "
                "disponibilità o dati. Prima di creare una prenotazione raccogli nome, telefono, "
                "data/ora e numero di persone; chiedi chiarimenti se mancano. Per modifiche e "
                "cancellazioni chiedi l'ID della prenotazione se non è già noto nella conversazione. "
                "Dopo aver creato una prenotazione comunica sempre il suo ID."
            ),
        }
    ]


def chat_turn(
    user_input: str,
    messages: list[ChatCompletionMessageParam],
    client: OpenAI,
    http: httpx.Client,
) -> str:
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_input})
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            assistant_message = completion.choices[0].message
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    assistant_message.model_dump(exclude_none=True),
                )
            )

            if not assistant_message.tool_calls:
                return assistant_message.content or ""

            for tool_call in assistant_message.tool_calls:
                if tool_call.type != "function":
                    raise RuntimeError("Il modello ha restituito un tool call non supportato")
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Gli argomenti del tool devono essere un oggetto JSON")
                    result = execute_tool(http, tool_call.function.name, arguments)
                except (json.JSONDecodeError, httpx.HTTPError, ValueError, TypeError) as exc:
                    result = _tool_error(exc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return "Ho raggiunto il limite di passaggi per questa richiesta. Riprova con una domanda più semplice."
    except (APIError, RuntimeError):
        del messages[turn_start:]
        raise


def run_chat() -> None:
    client = create_model_client()
    messages = create_conversation()

    print("Bot ristorante pronto. Scrivi /help per i comandi, /exit per terminare.")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as http:
        while True:
            try:
                user_input = input("Tu: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBot: Arrivederci!")
                return

            if not user_input:
                continue
            if user_input.lower() in {"/exit", "/quit"}:
                print("Bot: Arrivederci!")
                return
            if user_input.lower() == "/help":
                print("Bot: Puoi chiedermi di prenotare, modificare o cancellare un tavolo, consultare il menu o avere informazioni. Usa /reset per azzerare la conversazione.")
                continue
            if user_input.lower() == "/reset":
                messages = create_conversation()
                print("Bot: Conversazione azzerata.")
                continue

            try:
                print(f"Bot: {chat_turn(user_input, messages, client, http)}")
            except APIError as exc:
                print(f"Errore contattando il modello ({MODEL_BASE_URL}, modello {MODEL_NAME}): {exc}")
            except RuntimeError as exc:
                print(f"Errore nella risposta del modello: {exc}")


if __name__ == "__main__":
    run_chat()
