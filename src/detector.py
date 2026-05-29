"""
detector.py — MINDALAE controversy detector module (4/4).

Scans the candidate businesses returned by buscar.py for reviews that
contradict the user query — same topic, opposite direction — so the
recommender can warn about inconsistent experiences:

    query        — the original free-text request from the user.
    business_ids — the business_ids of the candidates to scan.

On import, the SBERT model, the precomputed embedding matrix, and the
reviews DataFrame already loaded by buscar.py are reused; no second copy
of any of them is held here, and no review is re-encoded at call time.
Each call positionally slices the matrix and the DataFrame with the same
boolean mask (relying on buscar.py's positional invariant: row i of the
matrix is row i of the DataFrame), normalizes the candidate vectors,
computes their cosine similarity against the query, and keeps the
reviews whose similarity is at or below UMBRAL_CONTROVERSIA.
"""

import numpy as np
import pandas as pd

import buscar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UMBRAL_CONTROVERSIA: float = 0.1
# Tentative starting value, to be calibrated empirically. Natural-language
# reviews rarely reach a perfect -1, so this value will likely be tuned later.


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def detectar_controversia(query: str, business_ids: list[str]) -> pd.DataFrame:
    """Returns the reviews that contradict the query within each candidate business.

    Builds one boolean mask over the shared reviews DataFrame and applies
    it positionally to both the DataFrame and the precomputed embedding
    matrix; buscar.py's positional invariant guarantees row i of the
    matrix is row i of the DataFrame, so a single mask keeps them
    aligned. The candidate vectors are L2-normalized explicitly so the
    dot product against the query equals cosine similarity regardless of
    whether the stored matrix happens to already be normalized. Returns
    the rows whose similarity is at or below UMBRAL_CONTROVERSIA, each
    enriched with a 'score' column — one row per offending review.
    """
    mascara_negocios = (
        buscar._df_reviews["business_id"].isin(business_ids).to_numpy()
    )
    candidatas = buscar._df_reviews.loc[mascara_negocios].reset_index(drop=True)
    vectores_candidatas = buscar._vectores[mascara_negocios]
    vectores_candidatas = vectores_candidatas / np.linalg.norm(
        vectores_candidatas, axis=1, keepdims=True
    )

    vector_consulta = buscar._modelo.encode(query, normalize_embeddings=True)
    similitudes = vectores_candidatas @ vector_consulta

    mascara_controversia = similitudes <= UMBRAL_CONTROVERSIA
    resultado = candidatas.loc[mascara_controversia].copy()
    resultado["score"] = similitudes[mascara_controversia]
    return resultado


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query_prueba = "quiet place for breakfast with garden view"

    print(f"\n[TEST] Query: '{query_prueba}'")
    resultados_busqueda = buscar.buscar(query_prueba, k=10)
    ids_candidatos = resultados_busqueda["business_id"].unique().tolist()
    print(f"[TEST] {len(ids_candidatos)} candidate businesses to scan.")

    print(f"[TEST] Scanning for controversies (UMBRAL_CONTROVERSIA={UMBRAL_CONTROVERSIA})...")
    alarmas = detectar_controversia(query_prueba, ids_candidatos)

    if alarmas.empty:
        print(f"[TEST] No business triggered the alarm.")
    else:
        n_negocios = alarmas["business_id"].nunique()
        print(
            f"[TEST] {n_negocios} business(es) triggered the alarm "
            f"with {len(alarmas)} contradictory review(s):\n"
        )
        for posicion, (_, fila) in enumerate(alarmas.iterrows(), start=1):
            preview = str(fila["text"])[:100].replace("\n", " ")
            print(f"  {posicion}. {fila['name']}  (score={fila['score']:.4f})")
            print(f"     {preview}...\n")
