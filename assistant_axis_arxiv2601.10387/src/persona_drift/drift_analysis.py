"""
Drift analysis: ridge regression + k-means on user message embeddings (Section 4.2).

Findings to reproduce:
  - R² 0.53–0.77 predicting axis projection from user message embeddings
  - R² ≈ 0.10 predicting delta (change in projection)
  - K-means clusters reveal which message types maintain vs. cause drift (Table 5)
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize

from config import DRIFT_N_CLUSTERS, EMBEDDING_MODEL, N_EMBEDDING_SAMPLES
from persona_drift.conversation_sim import Conversation


def embed_messages(
    messages: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Embed user messages using Qwen 3 0.6B Embedding model (Section 4.2).
    Returns L2-normalized embeddings [n_messages, d_embed].
    """
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(model_name, device=device)
    embeddings = embed_model.encode(
        messages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize (Section 4.2)
        convert_to_numpy=True,
    )
    return embeddings


def extract_user_messages_and_targets(
    conversations: list[Conversation],
    n_samples: int = N_EMBEDDING_SAMPLES,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Extract (user_message, next_turn_projection, delta) triples from conversations.

    For each turn in each conversation:
      - user_message = the user's message at turn t
      - projection = axis projection of NEXT assistant response (turn t)
      - delta = projection[t] - projection[t-1] (change in projection)

    Returns: (messages, projections, deltas) where projections and deltas are [n_samples].
    """
    messages = []
    projections = []
    deltas = []

    for conv in conversations:
        for t, turn in enumerate(conv.turns):
            messages.append(turn.user_message)
            projections.append(turn.axis_projection)
            if t > 0:
                delta = turn.axis_projection - conv.turns[t - 1].axis_projection
            else:
                delta = 0.0
            deltas.append(delta)

            if len(messages) >= n_samples:
                break
        if len(messages) >= n_samples:
            break

    return (
        messages[:n_samples],
        np.array(projections[:n_samples]),
        np.array(deltas[:n_samples]),
    )


def run_ridge_regression(
    embeddings: np.ndarray,    # [n, d_embed], L2-normalized
    targets: np.ndarray,       # [n], projection or delta values
    alphas: list[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    cv: int = 5,
) -> dict:
    """
    Fit ridge regression predicting targets from embeddings (Section 4.2).
    Returns: {r2, model, predictions}
    Expected R²: 0.53–0.77 for projection, ~0.10 for delta.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, targets, test_size=0.2, random_state=42
    )
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return {"r2": r2, "model": model, "y_test": y_test, "y_pred": y_pred}


def run_drift_regression_analysis(
    conversations: list[Conversation],
    embed_model_name: str = EMBEDDING_MODEL,
) -> dict:
    """
    Full drift analysis pipeline (Section 4.2).

    Steps:
    1. Extract user messages + targets from conversations
    2. Embed user messages with Qwen 3 0.6B Embedding (L2 normalized)
    3. Fit ridge regression for projection (R² expected 0.53–0.77)
    4. Fit ridge regression for delta (R² expected ~0.10)
    5. K-means clustering to characterize drift-inducing vs. drift-preventing messages

    Returns analysis results dict.
    """
    messages, projections, deltas = extract_user_messages_and_targets(conversations)
    print(f"  Embedding {len(messages)} user messages...")
    embeddings = embed_messages(messages, model_name=embed_model_name)

    print("  Running ridge regression (projection)...")
    proj_result = run_ridge_regression(embeddings, projections)
    print(f"  R² (projection): {proj_result['r2']:.3f}")

    print("  Running ridge regression (delta)...")
    delta_result = run_ridge_regression(embeddings, deltas)
    print(f"  R² (delta): {delta_result['r2']:.3f}")

    print(f"  Running k-means clustering (k={DRIFT_N_CLUSTERS})...")
    cluster_result = cluster_messages(embeddings, projections, messages, k=DRIFT_N_CLUSTERS)

    return {
        "n_samples": len(messages),
        "r2_projection": proj_result["r2"],
        "r2_delta": delta_result["r2"],
        "projection_regression": proj_result,
        "delta_regression": delta_result,
        "clusters": cluster_result,
        "validation": {
            "r2_projection_passes": proj_result["r2"] >= 0.53,
            "r2_delta_passes": delta_result["r2"] <= 0.20,
        },
    }


def cluster_messages(
    embeddings: np.ndarray,     # [n, d]
    projections: np.ndarray,    # [n], axis projection values
    messages: list[str],
    k: int = DRIFT_N_CLUSTERS,
) -> dict:
    """
    K-means cluster user message embeddings (Section 4.2).
    Characterize which clusters lead to high (Assistant maintained) vs.
    low (drift induced) projection values (Table 5).
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_summaries = []
    for c in range(k):
        mask = labels == c
        cluster_projs = projections[mask]
        cluster_msgs = [messages[i] for i, m in enumerate(mask) if m]
        cluster_summaries.append({
            "cluster_id": c,
            "size": int(mask.sum()),
            "mean_projection": float(cluster_projs.mean()),
            "std_projection": float(cluster_projs.std()),
            "example_messages": cluster_msgs[:3],
        })

    # Sort by mean projection (high = assistant maintained, low = drift)
    cluster_summaries.sort(key=lambda x: x["mean_projection"], reverse=True)

    return {
        "k": k,
        "labels": labels.tolist(),
        "summaries": cluster_summaries,
        "high_projection_clusters": cluster_summaries[:3],   # messages that maintain assistant
        "low_projection_clusters": cluster_summaries[-3:],   # messages that cause drift
    }
