"""
preparar_datos.py — MINDALAE setup module (1/4).

Transforms raw Yelp Open Dataset files (review.json, business.json)
into two persistent artifacts ready for real-time consumption:

    review_clean.csv — filtered reviews enriched with business metadata.
    vectores.npy     — (N, 384) matrix of normalized SBERT embeddings.

Run once per target city, offline. The cost of vectorizing thousands of
reviews with SBERT is paid here so that semantic search at runtime takes
milliseconds. Re-running overwrites previous artifacts.
"""

# =============================================================================
# CRITICAL POSITIONAL INVARIANT — READ BEFORE TOUCHING THESE ARTIFACTS
#
# Row i of vectores.npy corresponds EXACTLY to row i of review_clean.csv.
# This positional alignment is the mechanism by which the search module
# retrieves the correct text given a vector index.
#
# DO NOT manually modify review_clean.csv (reorder, delete, add, or filter
# rows) without regenerating vectores.npy by running this script from
# scratch. If alignment breaks, the system will return recommendations
# with seemingly correct text but scores from a completely different place.
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

CIUDAD_OBJETIVO: str = "Tucson"

PATH_REVIEW_JSON = PROJECT_ROOT / "data" / "raw" / "review.json"
PATH_BUSINESS_JSON = PROJECT_ROOT / "data" / "raw" / "business.json"
PATH_CSV_OUTPUT = PROJECT_ROOT / "data" / "processed" / "review_clean.csv"
PATH_NPY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vectores.npy"

MODELO_SBERT: str = "all-MiniLM-L6-v2"
CHUNKSIZE: int = 100_000
BATCH_SIZE: int = 64
LONGITUD_MINIMA_RESENA: int = 50


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def verificar_archivos_de_entrada() -> None:
    """Checks that input files exist; exits with error if any is missing."""
    archivos_requeridos = {
        "review.json": PATH_REVIEW_JSON,
        "business.json": PATH_BUSINESS_JSON,
    }
    for nombre, ruta in archivos_requeridos.items():
        if not Path(ruta).exists():
            print(
                f"[ERROR] Required file not found: '{nombre}'\n"
                f"        Expected path: {Path(ruta).resolve()}"
            )
            sys.exit(1)


def cargar_businesses_de_ciudad(path: Path, ciudad: str) -> pd.DataFrame:
    """Loads business.json and filters businesses located in the target city."""
    columnas = ["business_id", "name", "address", "city", "stars", "categories"]
    df = pd.read_json(path, lines=True)[columnas]
    return df[df["city"] == ciudad].copy()


def construir_set_de_business_ids(df_businesses: pd.DataFrame) -> set[str]:
    """Extracts business_ids as a set for O(1) lookup inside isin()."""
    return set(df_businesses["business_id"])


def cargar_reviews_por_chunks(
    path: Path, business_ids_validos: set[str]
) -> pd.DataFrame:
    """Reads review.json in chunks, keeping only reviews from the target city."""
    columnas_utiles = ["review_id", "business_id", "text"]
    chunks_filtrados: list[pd.DataFrame] = []

    iterador = pd.read_json(path, lines=True, chunksize=CHUNKSIZE)
    for chunk in iterador:
        chunk_filtrado = chunk.loc[
            chunk["business_id"].isin(business_ids_validos), columnas_utiles
        ]
        if not chunk_filtrado.empty:
            chunks_filtrados.append(chunk_filtrado)

    return pd.concat(chunks_filtrados, ignore_index=True)


def hacer_merge(
    df_reviews: pd.DataFrame, df_businesses: pd.DataFrame
) -> pd.DataFrame:
    """Joins reviews with their business metadata via inner merge on business_id."""
    return df_reviews.merge(df_businesses, on="business_id", how="inner")


def limpiar_textos(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace and discards reviews shorter than LONGITUD_MINIMA_RESENA characters."""
    df = df.copy()
    df["text"] = df["text"].str.strip()
    return df[df["text"].str.len() >= LONGITUD_MINIMA_RESENA]


def vectorizar_reviews(
    df: pd.DataFrame, modelo: SentenceTransformer
) -> np.ndarray:
    """Generates normalized SBERT embeddings for all reviews in the DataFrame."""
    return modelo.encode(
        df["text"].tolist(),
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )


def guardar_artefactos(
    df: pd.DataFrame,
    vectores: np.ndarray,
    path_csv: Path,
    path_npy: Path,
) -> None:
    """Persists the cleaned DataFrame and the embedding matrix to disk."""
    df.to_csv(path_csv, index=False)
    np.save(path_npy, vectores)


def verificar_invariante(df: pd.DataFrame, vectores: np.ndarray) -> None:
    """Verifies that the number of CSV rows matches the number of vectors."""
    assert len(df) == vectores.shape[0], (
        f"[BROKEN INVARIANT] CSV has {len(df)} rows but .npy contains "
        f"{vectores.shape[0]} vectors. Artifacts are misaligned. "
        f"Run the full script from scratch to regenerate both files."
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs the complete data preparation pipeline for MINDALAE."""
    print(f"[1/9] Checking input files...")
    verificar_archivos_de_entrada()
    print(f"      Files found: {PATH_REVIEW_JSON.name}, {PATH_BUSINESS_JSON.name}")

    print(f"[2/9] Loading businesses from '{CIUDAD_OBJETIVO}' in {PATH_BUSINESS_JSON.name}...")
    df_businesses = cargar_businesses_de_ciudad(PATH_BUSINESS_JSON, CIUDAD_OBJETIVO)
    print(f"      {len(df_businesses):,} businesses found in {CIUDAD_OBJETIVO}.")

    print(f"[3/9] Building set of valid business_ids...")
    business_ids_validos = construir_set_de_business_ids(df_businesses)
    print(f"      {len(business_ids_validos):,} unique IDs in set.")

    print(f"[4/9] Reading reviews in chunks (size {CHUNKSIZE:,}) from {PATH_REVIEW_JSON.name}...")
    df_reviews = cargar_reviews_por_chunks(PATH_REVIEW_JSON, business_ids_validos)
    print(f"      {len(df_reviews):,} relevant reviews accumulated.")

    print(f"[5/9] Merging reviews with business metadata...")
    df = hacer_merge(df_reviews, df_businesses)
    print(f"      {len(df):,} rows after merge.")

    print(f"[6/9] Cleaning texts and resetting index...")
    df = limpiar_textos(df)
    df = df.reset_index(drop=True)
    print(f"      {len(df):,} reviews after discarding short texts (< {LONGITUD_MINIMA_RESENA} characters).")

    print(f"[7/9] Loading SBERT model '{MODELO_SBERT}'...")
    modelo = SentenceTransformer(MODELO_SBERT)
    print(f"      Model loaded into memory.")

    print(f"[8/9] Vectorizing {len(df):,} reviews (batch_size={BATCH_SIZE})...")
    vectores = vectorizar_reviews(df, modelo)
    print(f"      Matrix generated: shape={vectores.shape}, dtype={vectores.dtype}.")

    print(f"[9/9] Saving artifacts and verifying positional invariant...")
    guardar_artefactos(df, vectores, PATH_CSV_OUTPUT, PATH_NPY_OUTPUT)
    verificar_invariante(df, vectores)
    print(f"      {PATH_CSV_OUTPUT.name} → {PATH_CSV_OUTPUT}")
    print(f"      {PATH_NPY_OUTPUT.name}    → {PATH_NPY_OUTPUT}")

    print(f"\n[OK] Setup complete. {len(df):,} reviews from {CIUDAD_OBJETIVO} ready for MINDALAE.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()