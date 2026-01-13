import re
import json
import os
from urllib.request import urlopen, Request
from urllib.parse import quote

RULES_URL = "https://mtg.fandom.com/es/wiki/Reglas_completas"
SCRYFALL_NAMED = "https://api.scryfall.com/cards/named?fuzzy="

__version__ = "2.2"


def _fetch(url: str, timeout=25) -> str:
    req = Request(url, headers={"User-Agent": "TretaMagicJudge/2.2"})
    with urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return raw.decode("utf-8", errors="ignore")


def _remove_blocks(html: str) -> str:
    html = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", html, flags=re.I)
    html = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", html, flags=re.I)
    html = re.sub(r"<noscript\b[^<]*(?:(?!</noscript>)<[^<]*)*</noscript>", " ", html, flags=re.I)
    return html


def _strip_html(html: str) -> str:
    html = _remove_blocks(html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = " ".join(text.split())

    junk_patterns = [
        r"\bdocument\.[A-Za-z0-9_\.]+\b",
        r"\bwindow\.[A-Za-z0-9_\.]+\b",
        r"\bwg[A-Za-z0-9_]+\b",
        r"\bRLCONF\b",
        r"\bclient-js\b",
    ]
    for jp in junk_patterns:
        text = re.sub(jp, " ", text)

    return " ".join(text.split())


class RulesCache:
    def __init__(self):
        self._text = None
        self._last_error = None

    def load(self) -> str:
        if self._text:
            return self._text
        try:
            html = _fetch(RULES_URL, timeout=30)
            self._text = _strip_html(html)
            self._last_error = None
            return self._text
        except Exception as e:
            self._last_error = str(e)
            self._text = None
            return ""


def detect_format(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["modern", "estándar", "standard", "draft", "sellado", "pauper", "legacy", "vintage", "pioneer"]):
        if "modern" in q: return "modern"
        if "pioneer" in q: return "pioneer"
        if "pauper" in q: return "pauper"
        if "legacy" in q: return "legacy"
        if "vintage" in q: return "vintage"
        if "draft" in q: return "draft"
        if "sellado" in q or "sealed" in q: return "sealed"
        if "standard" in q or "estándar" in q: return "standard"
    return "commander"


def _snippets(text: str, question: str, max_snips=4) -> list[str]:
    q = question.lower()

    kws = []
    for w in re.findall(r"[a-záéíóúñü]{4,}", q):
        if w not in (
            "como", "cuando", "puedo", "puede", "hacer", "tengo", "sobre", "porque", "donde", "para",
            "este", "esta", "esto", "esas", "esos", "carta", "cartas", "dice", "significa", "quiere", "decir"
        ):
            kws.append(w)

    kws = kws[:6] if kws else q.split()[:4]
    low = text.lower()
    hits = []

    for w in kws:
        pos = low.find(w)
        if pos != -1:
            start = max(0, pos - 520)
            end = min(len(text), pos + 950)
            hits.append(text[start:end][:1400])

    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    return uniq[:max_snips]


def scryfall_lookup(card_name: str) -> tuple[dict | None, str | None]:
    """
    Devuelve (data, error). error None si ok.
    """
    try:
        url = SCRYFALL_NAMED + quote(card_name)
        raw = _fetch(url, timeout=20)
        data = json.loads(raw)
        if data.get("object") == "error":
            return None, data.get("details") or "Scryfall no encontró la carta."
        return data, None
    except Exception as e:
        return None, str(e)


def build_answer_openai(question: str, rules_context: str, fmt: str, card_context: dict | None):
    from openai import OpenAI
    client = OpenAI()

    sys = (
        "Eres Treta (juez de Magic). Responde en español, claro y útil.\n"
        "Usa SOLO el contexto proporcionado.\n"
        "Por defecto aplica Commander salvo que el usuario indique otro formato.\n"
        "No muestres fuentes, links ni citas.\n"
        "Si no hay suficiente contexto para estar segura, di exactamente:\n"
        "No lo puedo confirmar con seguridad con el texto que tengo.\n"
        "Formato:\n"
        "1) Respuesta directa.\n"
        "2) Explicación en 2-6 pasos.\n"
    )

    user = {
        "question": question,
        "format": fmt,
        "rules_context": rules_context,
        "card_context": card_context or {}
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def answer_question(question: str, rules_cache: RulesCache, card_name: str | None = None) -> str:
    fmt = detect_format(question)

    rules_text = rules_cache.load()
    snips = _snippets(rules_text, question, max_snips=4) if rules_text else []
    rules_context = "\n\n---\n\n".join(snips) if snips else ""

    card_ctx = None
    scry_err = None

    if card_name:
        data, err = scryfall_lookup(card_name)
        scry_err = err
        if data:
            card_ctx = {
                "name": data.get("name", card_name),
                "mana_cost": data.get("mana_cost", ""),
                "type_line": data.get("type_line", ""),
                "oracle_text": data.get("oracle_text", ""),
                "colors": data.get("colors", []),
                "power": data.get("power", ""),
                "toughness": data.get("toughness", ""),
            }

    # ✅ Si el usuario pidió carta pero Scryfall no pudo → decirlo claro
    if card_name and not card_ctx and scry_err:
        return f"No pude consultar la carta '{card_name}'. Detalle: {scry_err}"

    if not rules_context and not card_ctx:
        return "No lo puedo confirmar con seguridad con el texto que tengo."

    try:
        return build_answer_openai(question, rules_context, fmt, card_ctx)
    except Exception as e:
        return f"El Juez IA no está disponible (¿OPENAI_API_KEY?). Detalle: {e}"
