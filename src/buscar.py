"""
buscar.py — MINDALAE search module (2/4).

Provides real-time semantic search over the artifacts produced by
preparar_datos.py:

    vectores.npy     — (N, 384) matrix of normalized SBERT embeddings.
    review_clean.csv — filtered reviews enriched with business metadata.

On import, the SBERT model, the embedding matrix, and the reviews
DataFrame are loaded once into module-level variables and reused across
every search. Each query is encoded with the same model, compared against
the full matrix via a single dot product (valid cosine similarity because
all vectors are L2-normalized), and the top-K most similar reviews are
returned together with their scores.
"""

# =============================================================================
# CRITICAL POSITIONAL INVARIANT — READ BEFORE TOUCHING THESE ARTIFACTS
#
# Row i of vectores.npy corresponds EXACTLY to row i of review_clean.csv.
# This positional alignment is the mechanism by which search retrieves the
# correct text given a vector index.
#
# DO NOT manually modify review_clean.csv (reorder, delete, add, or filter
# rows) without regenerating vectores.npy by running preparar_datos.py from
# scratch. If alignment breaks, the system will return recommendations with
# seemingly correct text but scores from a completely different place.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATH_CSV = PROJECT_ROOT / "data" / "processed" / "review_clean.csv"
PATH_NPY = PROJECT_ROOT / "data" / "processed" / "vectores.npy"

MODELO_SBERT: str = "all-MiniLM-L6-v2"
K_POR_DEFECTO: int = 10


# ---------------------------------------------------------------------------
# Module-level state (populated once by _inicializar at import time)
# ---------------------------------------------------------------------------

_modelo: SentenceTransformer | None = None
_vectores: np.ndarray | None = None
_df_reviews: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _verificar_artefactos() -> None:
    """Checks that the required artifacts exist; exits with error if missing."""
    artefactos_requeridos = {
        "review_clean.csv": PATH_CSV,
        "vectores.npy": PATH_NPY,
    }
    for nombre, ruta in artefactos_requeridos.items():
        if not ruta.exists():
            print(
                f"[ERROR] Required artifact not found: '{nombre}'\n"
                f"        Expected path: {ruta}\n"
                f"        Run preparar_datos.py first to generate it."
            )
            sys.exit(1)


def _verificar_invariante(df: pd.DataFrame, vectores: np.ndarray) -> None:
    """Verifies that the number of CSV rows matches the number of vectors."""
    assert len(df) == vectores.shape[0], (
        f"[BROKEN INVARIANT] CSV has {len(df)} rows but .npy contains "
        f"{vectores.shape[0]} vectors. Artifacts are misaligned. "
        f"Re-run preparar_datos.py from scratch to regenerate both files."
    )


def _inicializar() -> None:
    """Loads the SBERT model and the artifacts into module-level variables."""
    global _modelo, _vectores, _df_reviews

    print(f"[INIT] Checking artifacts in {PATH_CSV.parent}...")
    _verificar_artefactos()

    print(f"[INIT] Loading SBERT model '{MODELO_SBERT}'...")
    _modelo = SentenceTransformer(MODELO_SBERT)

    print(f"[INIT] Loading embedding matrix {PATH_NPY.name}...")
    _vectores = np.load(PATH_NPY)

    print(f"[INIT] Loading reviews {PATH_CSV.name}...")
    _df_reviews = pd.read_csv(PATH_CSV)

    print(f"[INIT] Verifying positional invariant...")
    _verificar_invariante(_df_reviews, _vectores)

    print(
        f"[INIT] Ready: {len(_df_reviews):,} reviews, "
        f"matrix shape={_vectores.shape}, dtype={_vectores.dtype}."
    )


def _seleccionar_top_k(similitudes: np.ndarray, k: int) -> np.ndarray:
    """Returns the indices of the k highest similarities, sorted descending."""
    k = max(1, min(k, similitudes.shape[0]))
    indices_parciales = np.argpartition(-similitudes, k - 1)[:k]
    return indices_parciales[np.argsort(-similitudes[indices_parciales])]


def buscar(query: str, k: int = K_POR_DEFECTO) -> pd.DataFrame:
    """Returns the top-K reviews most semantically similar to the query.

    Encodes the query with the same normalized SBERT model used at setup,
    computes cosine similarity as a dot product against the full embedding
    matrix, and returns the corresponding DataFrame rows with a 'score'
    column, ordered from most to least similar.
    """
    vector_consulta = _modelo.encode(query, normalize_embeddings=True)
    similitudes = _vectores @ vector_consulta
    indices_top = _seleccionar_top_k(similitudes, k)

    resultado = _df_reviews.iloc[indices_top].copy()
    resultado["score"] = similitudes[indices_top]
    return resultado


# ---------------------------------------------------------------------------
# Initialization at import time
# ---------------------------------------------------------------------------

_inicializar()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query_prueba = "quiet place for breakfast with garden view"

    print(f"\n[TEST] Query: '{query_prueba}'")
    resultados = buscar(query_prueba, k=5)

    print(f"[TEST] Top 5 results:\n")
    for posicion, (_, fila) in enumerate(resultados.iterrows(), start=1):
        preview = str(fila["text"])[:100].replace("\n", " ")
        print(f"  {posicion}. {fila['name']}  (score={fila['score']:.4f})")
        print(f"     {preview}...\n")
