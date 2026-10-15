import json, os, sys, time, uuid
from datetime import datetime
import threading
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import requests
import sounddevice as sd
import pyttsx3
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from progetto_ristorante.frontend.pii import Anonymizer

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
API = os.environ["RESTAURANT_API_URL"].rstrip("/")
LLM_URL = os.environ["MODEL_BASE_URL"].rstrip("/")
LLM_KEY = os.getenv("MODEL_API_KEY", "local")
LLM_MODEL = os.getenv("MODEL_NAME", "local")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
MAX_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "8"))
STORE_REAL_PII = os.getenv("STORE_REAL_PII", "true").lower() == "true"
LANG = os.getenv("LANGUAGE", "it")

SR, BLOCK = 16000, 480  # 30 ms
MAX_SEC, SILENCE_SEC = 15, 1.0
MIN_VOICE_RMS = 0.003

_stt: WhisperModel | None = None

# ---------------------------------------------------------------- audio
PIPER_MODEL = os.getenv("PIPER_MODEL", "models/it_IT-paola-medium.onnx")
_voice = None
try:
    from piper import PiperVoice
    _voice = PiperVoice.load(PIPER_MODEL)
    print(f"✅ Piper caricato: {PIPER_MODEL}")
except Exception as e:
    print(f"⚠️  Piper non disponibile ({e}); uso la voce di Windows (pyttsx3)")


def _speak_piper(text: str):
    if hasattr(_voice, "synthesize_stream_raw"):      # piper-tts < 1.3
        pcm = b"".join(_voice.synthesize_stream_raw(text))
        sd.play(np.frombuffer(pcm, dtype=np.int16), _voice.config.sample_rate)
        sd.wait()
    else:                                              # piper-tts >= 1.3
        for chunk in _voice.synthesize(text):
            sd.play(chunk.audio_int16_array, chunk.sample_rate)
            sd.wait()


def _speak_sapi(text: str):
    eng = pyttsx3.init()
    for v in eng.getProperty("voices"):
        if "it" in str(v.languages).lower() or "ital" in v.name.lower() or "elsa" in v.name.lower():
            eng.setProperty("voice", v.id)
            break
    eng.setProperty("rate", 175)
    eng.say(text)
    eng.runAndWait()
    eng.stop()


def speak(text: str):
    print(f"🤖 {text}")
    text = text.replace("*", "").replace("#", "").strip()
    if not text:
        return
    try:
        _speak_piper(text) if _voice else _speak_sapi(text)
    except Exception as e:
        print("Errore TTS:", e)
        _speak_sapi(text)


def listen(stop_event=None):
    frames, started, silent = [], False, 0
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK) as s:
        noise = np.mean([np.sqrt(np.mean(s.read(BLOCK)[0] ** 2)) for _ in range(10)])
        thr = max(noise * 4, MIN_VOICE_RMS)
        t0 = time.time()
        while time.time() - t0 < MAX_SEC + 10 and not (stop_event and stop_event.is_set()):
            block, _ = s.read(BLOCK)
            loud = np.sqrt(np.mean(block ** 2)) > thr
            if loud:
                started, silent = True, 0
            elif started:
                silent += 1
            if started:
                frames.append(block[:, 0])
                if silent * BLOCK / SR > SILENCE_SEC or len(frames) * BLOCK / SR > MAX_SEC:
                    break
    return np.concatenate(frames) if frames else None


def transcribe(audio) -> str:
    global _stt
    if _stt is None:
        _stt = WhisperModel(
            os.getenv("WHISPER_MODEL", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
        )
    segs, _ = _stt.transcribe(audio, language=LANG, vad_filter=True, beam_size=5)
    return " ".join(s.text.strip() for s in segs).strip()

# ---------------------------------------------------------------- tool -> REST API
def _call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kw)
        if r.status_code == 204:
            return {"ok": True}
        if r.status_code >= 400:
            return {"error": r.status_code, "detail": r.text[:500]}
        return r.json()
    except requests.RequestException as e:
        return {"error": "network", "detail": str(e)}


def run_tool(name: str, args: dict, anon: Anonymizer):
    # L'LLM vede solo segnaposto: per scrivere nel DB (opzionale) li ripristino qui, in locale.
    if STORE_REAL_PII:
        args = {k: anon.restore(v) if isinstance(v, str) else v for k, v in args.items()}
    rid = args.pop("reservation_id", None)
    if name == "get_menu":
        return _call("GET", "/menu")
    if name == "get_restaurant_info":
        return _call("GET", "/restaurant-info")
    if name == "create_reservation":
        return _call("POST", "/reservations", json=args)
    if name == "update_reservation":
        return _call("PATCH", f"/reservations/{rid}", json=args)
    if name == "delete_reservation":
        return _call("DELETE", f"/reservations/{rid}")
    return {"error": f"tool sconosciuto: {name}"}


def _tool(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}

_rid = {"reservation_id": {"type": "integer", "description": "Codice numerico della prenotazione"}}
_fields = {
    "customer_name": {"type": "string", "description": "Nome del cliente (usa il segnaposto ricevuto, es. [NOME_1])"},
    "phone": {"type": "string", "description": "Telefono del cliente (usa il segnaposto ricevuto, es. [TELEFONO_2])"},
    "reservation_at": {"type": "string", "description": "Data e ora ISO 8601, es. 2026-10-20T20:30:00"},
    "party_size": {"type": "integer", "minimum": 1},
    "notes": {"type": "string"},
}
TOOLS = [
    _tool("get_menu", "Restituisce il menu (prezzi in centesimi di euro)."),
    _tool("get_restaurant_info", "Indirizzo, telefono, email e orari di apertura del ristorante."),
    _tool("create_reservation", "Crea una prenotazione. Servono nome, telefono, data/ora e numero di persone.",
          _fields, ["customer_name", "phone", "reservation_at", "party_size"]),
    _tool("update_reservation", "Modifica una prenotazione esistente. Passa solo i campi da cambiare.",
          {**_rid, **_fields}, ["reservation_id"]),
    _tool("delete_reservation", "Cancella una prenotazione esistente.", _rid, ["reservation_id"]),
]

GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]

def system_prompt() -> str:
    now = datetime.now()
    return (
        "Sei l'assistente VOCALE di un ristorante e parli al telefono in italiano.\n"
        f"Oggi è {GIORNI[now.weekday()]} {now:%d/%m/%Y}, sono le {now:%H:%M}. Risolvi 'domani', 'sabato' ecc. in date reali.\n"
        "Regole:\n"
        "- Risposte brevi (1-2 frasi), niente elenchi, niente markdown, niente emoji: verrà letto ad alta voce.\n"
        "- Per prenotare chiedi, una cosa alla volta, ciò che manca: nome, telefono, data e ora, numero di persone.\n"
        "- I dati personali ti arrivano come segnaposto tipo [NOME_1] o [TELEFONO_2]: usali identici negli strumenti "
        "e NON inventare mai valori al loro posto.\n"
        "- Dopo una prenotazione riassumi data, ora, persone e comunica il CODICE prenotazione: serve per modifiche o cancellazioni.\n"
        "- Per modificare o cancellare chiedi il codice prenotazione se non lo hai.\n"
        "- I prezzi del menu sono in centesimi: pronunciali in euro. Non inventare piatti o orari: usa gli strumenti.\n"
        "- Non rispondere a domande estranee al ristorante."
    )

# ---------------------------------------------------------------- LLM loop
def llm(messages):
    r = requests.post(f"{LLM_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {LLM_KEY}"},
                      json={"model": LLM_MODEL, "messages": messages, "tools": TOOLS,
                            "tool_choice": "auto", "temperature": 0.2},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def answer(messages, anon) -> str:
    for _ in range(MAX_ROUNDS):
        msg = llm(messages)
        calls = msg.get("tool_calls")
        messages.append({k: v for k, v in msg.items() if v is not None})
        if not calls:
            return (msg.get("content") or "").strip()
        for c in calls:
            fn = c["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"🔧 {fn['name']}({'{...}' if STORE_REAL_PII else args})")
            res = run_tool(fn["name"], args, anon)
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(res, ensure_ascii=False)})
    return "Mi scusi, non sono riuscito a completare la richiesta. Può ripetere?"


def check_services():
    for url, label in ((f"{API}/health", "BE"), (f"{LLM_URL}/models", "LLM")):
        try:
            requests.get(url, timeout=5).raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"{label} non raggiungibile ({url}): {e}") from e


def new_conversation():
    return Anonymizer(), [{"role": "system", "content": system_prompt()}]


def process_text(text, messages, anon):
    if any(w in text.lower() for w in ("arrivederci", "esci", "chiudi")):
        return "Grazie e arrivederci!", True

    safe = anon.clean(text)
    print(f"➡️  (visto dall'LLM)     {safe}")
    messages.append({"role": "user", "content": safe})
    try:
        reply = answer(messages, anon)
    except Exception as e:
        print("Errore:", e)
        reply = "Mi scusi, c'è stato un problema di collegamento. Può ripetere?"
    return anon.restore(reply), False


class AssistantGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Assistente del ristorante")
        self.root.geometry("900x680")
        self.root.minsize(650, 480)
        self.root.configure(bg="#f4f6f8")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.anon, self.messages = new_conversation()
        self.stop_event = None
        self.listening = False
        self.busy = True
        self.ended = False
        self.service_warning = None

        header = tk.Frame(self.root, bg="#173b35", padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(
            header, text="Assistente del ristorante", bg="#173b35", fg="white",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        self.status = tk.Label(
            header, text="Connessione ai servizi in corso...", bg="#173b35",
            fg="#d4e6df", font=("Segoe UI", 10),
        )
        self.status.pack(anchor="w", pady=(5, 0))

        self.chat = scrolledtext.ScrolledText(
            self.root, wrap="word", state="disabled", padx=18, pady=16,
            bg="white", fg="#202a28", relief="flat", font=("Segoe UI", 11),
        )
        self.chat.pack(fill="both", expand=True, padx=18, pady=(18, 10))
        self.chat.tag_configure("user", foreground="#176b55", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("assistant", foreground="#263238", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("body", foreground="#202a28", font=("Segoe UI", 11))

        controls = tk.Frame(self.root, bg="#f4f6f8", padx=18, pady=12)
        controls.pack(fill="x")
        self.entry = tk.Entry(
            controls, font=("Segoe UI", 11), relief="solid", bd=1,
            state="disabled",
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=9, padx=(0, 8))
        self.entry.bind("<Return>", self.send_text)
        self.send_button = tk.Button(
            controls, text="Invia", command=self.send_text, state="disabled",
            bg="#176b55", fg="white", activebackground="#125642",
            activeforeground="white", relief="flat", padx=18, pady=9,
        )
        self.send_button.pack(side="left", padx=(0, 8))
        self.voice_button = tk.Button(
            controls, text="Avvia microfono", command=self.toggle_voice,
            state="disabled", bg="#e1ebe7", fg="#173b35",
            activebackground="#d3e2dc", relief="flat", padx=14, pady=9,
        )
        self.voice_button.pack(side="left", padx=(0, 8))
        self.new_button = tk.Button(
            controls, text="Nuova chat", command=self.reset_conversation,
            state="disabled", bg="#e8ebee", fg="#30383b",
            activebackground="#dce1e4", relief="flat", padx=14, pady=9,
        )
        self.new_button.pack(side="left")

        threading.Thread(target=self.initialize, daemon=True).start()

    def initialize(self):
        warning = None
        try:
            check_services()
        except RuntimeError as e:
            warning = (
                f"Servizi non raggiungibili: {e}. "
                "Microfono e trascrizione restano disponibili; le risposte AI potrebbero non funzionare."
            )
            print(warning)
        greeting = "Buongiorno, sono l'assistente del ristorante. Come posso aiutarla?"
        self.root.after(0, self.add_message, "assistant", greeting)
        try:
            speak(greeting)
        except Exception as e:
            print("Errore TTS:", e)
            warning = f"{warning or 'Pronto.'} Voce non disponibile: {e}"
            self.root.after(0, self.enable_controls, warning)
            return
        self.root.after(0, self.enable_controls, warning)

    def enable_controls(self, warning=None):
        self.busy = False
        self.service_warning = warning
        self.status.configure(
            text=warning or "Pronto. Scriva un messaggio o avvii il microfono.",
            fg="#ffc9c9" if warning else "#d4e6df",
        )
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.voice_button.configure(state="normal")
        self.new_button.configure(state="normal")
        self.entry.focus_set()

    def add_message(self, role, text):
        self.chat.configure(state="normal")
        label = "Tu" if role == "user" else "Assistente"
        tag = "user" if role == "user" else "assistant"
        self.chat.insert("end", f"{label}\n", tag)
        self.chat.insert("end", f"{text}\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def set_busy(self, busy, status=None):
        self.busy = busy
        state = "disabled" if busy or self.ended else "normal"
        self.entry.configure(state=state)
        self.send_button.configure(state=state)
        self.new_button.configure(state="normal" if not self.listening else "disabled")
        if status:
            self.status.configure(text=status)

    def send_text(self, _event=None):
        if self.busy or self.listening or self.ended:
            return "break"
        text = self.entry.get().strip()
        if not text:
            return "break"
        self.entry.delete(0, "end")
        self.add_message("user", text)
        self.set_busy(True, "Sto preparando la risposta...")
        threading.Thread(target=self.reply_to_text, args=(text,), daemon=True).start()
        return "break"

    def reply_to_text(self, text):
        try:
            reply, finished = process_text(text, self.messages, self.anon)
        except Exception as e:
            print("Errore:", e)
            self.root.after(0, self.text_failed, str(e))
            return
        self.root.after(0, self.show_reply, reply, finished)
        audio_error = None
        try:
            speak(reply)
        except Exception as e:
            print("Errore TTS:", e)
            audio_error = str(e)
        self.root.after(0, self.finish_reply, finished, audio_error)

    def text_failed(self, message):
        self.set_busy(False, f"Errore: {message}")

    def show_reply(self, reply, finished):
        self.add_message("assistant", reply)
        if finished:
            self.status.configure(text="Conversazione terminata. Avvia una nuova chat per continuare.")

    def finish_reply(self, finished, audio_error=None):
        self.ended = finished
        self.set_busy(False)
        if audio_error:
            self.status.configure(text=f"Errore audio: {audio_error}")
        elif not finished:
            self.status.configure(
                text=self.service_warning or "Pronto. Scriva un messaggio o avvii il microfono.",
                fg="#ffc9c9" if self.service_warning else "#d4e6df",
            )
        if finished:
            self.entry.configure(state="disabled")
            self.send_button.configure(state="disabled")
            self.voice_button.configure(state="disabled")

    def toggle_voice(self):
        if self.listening:
            if self.stop_event:
                self.stop_event.set()
            self.voice_button.configure(state="disabled", text="Arresto...")
            self.status.configure(text="Interruzione dell'ascolto...")
            return
        if self.busy or self.ended:
            return

        self.stop_event = threading.Event()
        self.listening = True
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.new_button.configure(state="disabled")
        self.voice_button.configure(text="Ferma ascolto")
        self.status.configure(text="Microfono pronto: parli chiaramente.")
        threading.Thread(target=self.voice_loop, args=(self.stop_event,), daemon=True).start()

    def voice_loop(self, stop_event):
        finished = False
        error_message = None
        try:
            while not stop_event.is_set():
                self.root.after(0, self.status.configure, {"text": "In ascolto..."})
                audio = listen(stop_event)
                if stop_event.is_set():
                    break
                if audio is None:
                    continue
                self.root.after(
                    0, self.status.configure,
                    {"text": "Audio rilevato. Trascrizione in corso (il primo avvio può richiedere tempo)..."},
                )
                text = transcribe(audio)
                if not text:
                    continue
                self.root.after(0, self.status.configure, {"text": "Voce riconosciuta. Elaborazione della richiesta..."})
                print(f"👤 (trascrizione locale) {text}")
                self.root.after(0, self.add_message, "user", text)
                reply, finished = process_text(text, self.messages, self.anon)
                self.root.after(0, self.add_message, "assistant", reply)
                speak(reply)
                if finished:
                    break
        except Exception as e:
            print("Errore:", e)
            error_message = str(e)
        finally:
            self.root.after(0, self.voice_stopped, finished, error_message)

    def voice_stopped(self, finished, error_message):
        self.listening = False
        self.ended = finished
        self.voice_button.configure(text="Avvia microfono", state="disabled" if finished else "normal")
        if error_message:
            self.set_busy(False, f"Errore: {error_message}")
        elif finished:
            self.set_busy(False)
            self.status.configure(text="Conversazione terminata. Avvia una nuova chat per continuare.")
            self.entry.configure(state="disabled")
            self.send_button.configure(state="disabled")
        else:
            self.set_busy(False)
            self.status.configure(
                text=self.service_warning or "Pronto. Scriva un messaggio o avvii il microfono.",
                fg="#ffc9c9" if self.service_warning else "#d4e6df",
            )

    def reset_conversation(self):
        if self.busy or self.listening:
            return
        self.anon, self.messages = new_conversation()
        self.ended = False
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.add_message("assistant", "Buongiorno, sono l'assistente del ristorante. Come posso aiutarla?")
        self.status.configure(text="Nuova conversazione avviata.")
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.voice_button.configure(state="normal")
        threading.Thread(
            target=speak,
            args=("Buongiorno, sono l'assistente del ristorante. Come posso aiutarla?",),
            daemon=True,
        ).start()

    def close(self):
        if self.stop_event:
            self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------- main
def main_terminal():
    anon, messages = new_conversation()
    try:
        check_services()
    except RuntimeError as e:
        sys.exit(str(e))

    speak("Buongiorno, sono l'assistente del ristorante. Come posso aiutarla?")
    while True:
        print("🎤 In ascolto...")
        audio = listen()
        if audio is None:
            continue
        text = transcribe(audio)
        if not text:
            continue
        print(f"👤 (trascrizione locale) {text}")
        reply, finished = process_text(text, messages, anon)
        speak(reply)
        if finished:
            break


if __name__ == "__main__":
    if "--terminal" in sys.argv[1:]:
        main_terminal()
    else:
        AssistantGUI().run()