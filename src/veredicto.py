"""
veredicto.py — MINDALAE verdict module (3/4).

Turns the raw search results produced by buscar.py into a single
natural-language recommendation for the user:

    resultados — top-K reviews enriched with business metadata.
    query      — the original free-text request from the user.

On import, environment variables are loaded and an OpenAI-compatible
client pointing at Groq is created once into a module-level variable and
reused across every verdict. Each call assembles a text block describing
the candidate businesses, combines it with the system prompt and the
query, and delegates the final wording to Groq's LLM.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
MODELO_LLM: str = "llama-3.3-70b-versatile"

PROMPT_SISTEMA = "TODO: definir con el equipo de prompt engineering."


# ---------------------------------------------------------------------------
# Module-level state (populated once by _inicializar at import time)
# ---------------------------------------------------------------------------

_cliente: OpenAI | None = None


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _verificar_api_key() -> None:
    """Checks that GROQ_API_KEY is defined; exits with error if missing."""
    if not os.getenv("GROQ_API_KEY"):
        print(
            f"[ERROR] Required environment variable not found: 'GROQ_API_KEY'\n"
            f"        Define it in a .env file at the project root or in your\n"
            f"        environment before importing this module."
        )
        sys.exit(1)


def _inicializar() -> None:
    """Loads environment variables and creates the Groq client once."""
    global _cliente

    print(f"[INIT] Loading environment variables...")
    load_dotenv()

    print(f"[INIT] Verifying GROQ_API_KEY...")
    _verificar_api_key()

    print(f"[INIT] Creating Groq client (base_url={GROQ_BASE_URL})...")
    _cliente = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)

    print(f"[INIT] Ready: client configured for model '{MODELO_LLM}'.")


def _construir_bloque_negocios(resultados: pd.DataFrame) -> str:
    """Builds a text block describing each business in the results DataFrame."""
    bloques: list[str] = []
    for posicion, (_, fila) in enumerate(resultados.iterrows(), start=1):
        bloques.append(
            f"Negocio {posicion}\n"
            f"  Nombre:     {fila['name']}\n"
            f"  Dirección:  {fila['address']}\n"
            f"  Categorías: {fila['categories']}\n"
            f"  Estrellas:  {fila['stars']}\n"
            f"  Reseña:     {fila['text']}"
        )
    return "\n\n".join(bloques)


def generar_veredicto(resultados: pd.DataFrame, query: str) -> str:
    """Returns a natural-language verdict for the query from the search results.

    Assembles a text block summarizing each business returned by search
    (name, address, categories, stars, review), combines it with the system
    prompt and the user's query, asks Groq's LLM to produce the final
    recommendation, and returns its text as a plain string.
    """
    bloque_negocios = _construir_bloque_negocios(resultados)

    contenido_usuario = (
        f"Consulta del usuario:\n{query}\n\n"
        f"Negocios candidatos:\n{bloque_negocios}"
    )

    respuesta = _cliente.chat.completions.create(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": contenido_usuario},
        ],
    )
    return respuesta.choices[0].message.content


# ---------------------------------------------------------------------------
# Initialization at import time
# ---------------------------------------------------------------------------

_inicializar()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from buscar import buscar

    query_prueba = "quiet place for breakfast with garden view"

    print(f"\n[TEST] Query: '{query_prueba}'")
    resultados = buscar(query_prueba, k=5)

    print(f"[TEST] Generating verdict with {MODELO_LLM}...")
    veredicto = generar_veredicto(resultados, query_prueba)

    print(f"[TEST] Verdict:\n")
    print(veredicto)
