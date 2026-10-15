"""Rimozione informazioni personali PRIMA dell'invio al BE (vincolo 3)."""
import re

PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CF", re.compile(r"\b[A-Za-z]{6}\d{2}[A-Za-z]\d{2}[A-Za-z]\d{3}[A-Za-z]\b")),
    ("TELEFONO", re.compile(r"(?<!\d)(?:\+?39[\s.-]?)?(?:3\d{2}|0\d{1,3})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)")),
    # il parlato: "mi chiamo Mario Rossi", "a nome di Anna", "sono Luca"
    ("NOME", re.compile(
        r"\b(?:mi chiamo|a nome(?: di)?|il mio nome è|sono)\s+([A-ZÀ-Ý][\wà-ÿ']+(?:\s+[A-ZÀ-Ý][\wà-ÿ']+)?)",
        re.IGNORECASE)),
]

try:
    import spacy
    _nlp = spacy.load("it_core_news_sm")
except Exception:
    _nlp = None


class Anonymizer:
    """Sostituisce i dati personali con segnaposto e tiene la mappa SOLO in locale."""

    def __init__(self):
        self.map = {}      # segnaposto -> valore reale
        self._n = 0

    def _ph(self, kind, value):
        for k, v in self.map.items():
            if v == value:
                return k
        self._n += 1
        key = f"[{kind}_{self._n}]"
        self.map[key] = value
        return key

    def clean(self, text: str) -> str:
        for kind, pat in PATTERNS:
            if kind == "NOME":
                text = pat.sub(lambda m: m.group(0).replace(m.group(1), self._ph("NOME", m.group(1))), text)
            else:
                text = pat.sub(lambda m, k=kind: self._ph(k, m.group(0)), text)
        if _nlp:  # NER per nomi di persona non introdotti da formule
            for ent in reversed(_nlp(text).ents):
                if ent.label_ == "PER":
                    text = text[:ent.start_char] + self._ph("NOME", ent.text) + text[ent.end_char:]
        return text

    def restore(self, text: str) -> str:
        for k, v in self.map.items():
            text = text.replace(k, v)
        return text
