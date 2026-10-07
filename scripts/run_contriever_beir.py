"""
BEIR Contriever PCA + Retrieval
===============================

Milestone 4:
- Contriever dense retrieval
- BEIR-format datasets
- PCA fitted on the corpus
- Query vectors transformed using the same corpus-fitted PCA
- NDCG@10 evaluation
- Supports:
    LIMIT-50k
    SciDocs

Budget dimensions:
    768
    384
    192
    96
    48

Important:
- PCA is fitted ONLY on corpus embeddings.
- Query embeddings are never used to fit PCA.
- Intermediate embeddings/projections are not intended for GitHub.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from beir.datasets.data_loader import GenericDataLoader
from safetensors.torch import load_file
from transformers import AutoModel, BertConfig, BertTokenizer

from truncate import PCAProjector


# ======================================================================
# SETTINGS
# ======================================================================

DATASETS = {
    "LIMIT-50k": r"E:\ir_project\data\limit-50k",
    "SciDocs": r"E:\ir_project\data\scidocs",
}

OUTPUT_DIR = r"E:\ir_project\data\contriever_beir"

MODEL_DIR = (
    r"E:\ir_project\hf_cache\hub\models--facebook--contriever"
    r"\snapshots\4bce1e1123ec62a1cbc1f9d27dc9c8e6e1e51ce0"
)

DIMENSIONS = [
    768,
    384,
    192,
    96,
    48,
]

BATCH_SIZE = 8
MAX_LENGTH = 512

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# NORMALIZATION
# ======================================================================

def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return vectors / norms


# ======================================================================
# CONTRIEVER MODEL
# ======================================================================

def load_contriever():
    print("\nLoading Contriever tokenizer...")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )

    print("Tokenizer loaded successfully.")

    print("\nBuilding Contriever configuration...")

    # IMPORTANT:
    # The tokenizer reports an incorrect vocab_size in this local
    # snapshot. The actual checkpoint has a 30522 x 768 word embedding.
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
    )

    print("Hidden size:", config.hidden_size)
    print("Hidden layers:", config.num_hidden_layers)
    print("Attention heads:", config.num_attention_heads)

    print("\nCreating Contriever model...")

    model = AutoModel.from_config(config)

    safetensors_file = os.path.join(
        MODEL_DIR,
        "model.safetensors",
    )

    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(
            f"Contriever checkpoint not found:\n{safetensors_file}"
        )

    print("Loading model.safetensors...")

    state_dict = load_file(
        safetensors_file,
        device="cpu",
    )

    print(
        "Weight tensors loaded:",
        len(state_dict),
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        print("\nWARNING: Missing keys:")
        for key in missing_keys:
            print(" ", key)

    if unexpected_keys:
        print("\nWARNING: Unexpected keys:")
        for key in unexpected_keys:
            print(" ", key)

    model.to(DEVICE)
    model.eval()

    print("\nContriever model loaded successfully.")
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB",
        )

    return tokenizer, model


# ======================================================================
# MEAN POOLING
# ======================================================================

def mean_pooling(
    last_hidden_state,
    attention_mask,
):
    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1,
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


# ======================================================================
# ENCODING
# ======================================================================

def encode_texts(
    texts,
    tokenizer,
    model,
    label,
):
    """
    Encode texts using exactly the same procedure as the validated
    CSFCube Contriever pipeline.
    """

    print(
        f"\nGenerating {label} embeddings..."
    )

    embeddings = []

    total = len(texts)

    with torch.no_grad():

        for start in range(
            0,
            total,
            BATCH_SIZE,
        ):

            batch_texts = texts[
                start:start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            pooled = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

            embeddings.append(
                pooled.cpu().numpy().astype(
                    np.float32
                )
            )

            done = min(
                start + BATCH_SIZE,
                total,
            )

            if done % 1000 == 0 or done == total:
                print(
                    f"  {done}/{total}"
                )

    result = np.concatenate(
        embeddings,
        axis=0,
    )

    print(
        f"{label} embedding shape:",
        result.shape,
    )

    return result


# ======================================================================
# LOAD BEIR DATA
# ======================================================================

def load_dataset(dataset_name, data_folder):

    print(
        "\n" + "=" * 70
    )

    print(
        f"Loading {dataset_name}"
    )

    print(
        "=" * 70
    )

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_folder
    ).load("test")

    print(
        "Corpus:",
        len(corpus),
    )

    print(
        "Queries:",
        len(queries),
    )

    print(
        "Queries with qrels:",
        len(qrels),
    )

    # --------------------------------------------------------------
    # BEIR corpus IDs
    # --------------------------------------------------------------

    corpus_ids = [
        str(doc_id)
        for doc_id in corpus.keys()
    ]

    # --------------------------------------------------------------
    # Build corpus text.
    #
    # This keeps title and text together while avoiding a leading
    # empty title for LIMIT.
    # --------------------------------------------------------------

    corpus_texts = []

    for doc_id in corpus_ids:

        title = str(
            corpus[doc_id].get(
                "title",
                "",
            )
            or ""
        ).strip()

        text = str(
            corpus[doc_id].get(
                "text",
                "",
            )
            or ""
        ).strip()

        if title and text:
            combined = title + " " + text

        elif title:
            combined = title

        else:
            combined = text

        corpus_texts.append(
            combined
        )

    # --------------------------------------------------------------
    # Query IDs/texts
    # --------------------------------------------------------------

    query_ids = [
        str(query_id)
        for query_id in queries.keys()
    ]

    query_texts = [
        str(
            queries[query_id]
        )
        for query_id in query_ids
    ]

    # --------------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------------

    if not corpus_ids:
        raise ValueError(
            f"{dataset_name}: corpus is empty."
        )

    if not query_ids:
        raise ValueError(
            f"{dataset_name}: queries are empty."
        )

    if not qrels:
        raise ValueError(
            f"{dataset_name}: qrels are empty."
        )

    print(
        "\nFirst corpus document:"
    )

    print(
        corpus_ids[0],
        "->",
        corpus_texts[0][:200],
    )

    print(
        "\nFirst query:"
    )

    print(
        query_ids[0],
        "->",
        query_texts[0],
    )

    if query_ids[0] in qrels:
        print(
            "First query qrels:",
            qrels[query_ids[0]],
        )

    return (
        corpus_ids,
        corpus_texts,
        query_ids,
        query_texts,
        qrels,
    )


# ======================================================================
# RETRIEVAL
# ======================================================================

def generate_rankings(
    dataset_name,
    corpus_ids,
    corpus_embeddings,
    query_ids,
    query_embeddings,
    qrels,
):
    print(
        "\n" + "=" * 70
    )

    print(
        f"Generating {dataset_name} rankings"
    )

    print(
        "=" * 70
    )

    corpus_embeddings = np.asarray(
        corpus_embeddings,
        dtype=np.float32,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    if corpus_embeddings.shape[1] != 768:
        raise ValueError(
            "Expected 768-dimensional corpus embeddings."
        )

    if query_embeddings.shape[1] != 768:
        raise ValueError(
            "Expected 768-dimensional query embeddings."
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # PCA is fitted on the ENTIRE corpus.
    # Queries are never used for fitting.
    # --------------------------------------------------------------

    for dimension in DIMENSIONS:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Processing dimension: {dimension}"
        )

        print(
            "-" * 70
        )

        # ==========================================================
        # Full 768 dimensions
        # ==========================================================

        if dimension == 768:

            projected_corpus = (
                corpus_embeddings.copy()
            )

            projected_queries = (
                query_embeddings.copy()
            )

        # ==========================================================
        # PCA
        # ==========================================================

        else:

            print(
                f"Fitting PCA on "
                f"{len(corpus_ids)} corpus documents..."
            )

            projector = PCAProjector.fit(
                corpus_embeddings,
                n_components=dimension,
            )

            projected_corpus = projector.transform(
                corpus_embeddings
            ).astype(
                np.float32
            )

            projected_queries = projector.transform(
                query_embeddings
            ).astype(
                np.float32
            )

        # ----------------------------------------------------------
        # Normalize AFTER projection.
        # This exactly follows the working CSFCube pipeline.
        # ----------------------------------------------------------

        projected_corpus = normalize_rows(
            projected_corpus
        )

        projected_queries = normalize_rows(
            projected_queries
        )

        print(
            "Corpus shape:",
            projected_corpus.shape,
        )

        print(
            "Query shape:",
            projected_queries.shape,
        )

        # ----------------------------------------------------------
        # Corpus ID -> row index
        # ----------------------------------------------------------

        corpus_to_index = {
            doc_id: index
            for index, doc_id in enumerate(
                corpus_ids
            )
        }

        # ----------------------------------------------------------
        # Generate rankings
        # ----------------------------------------------------------

        rankings = {}

        for query_index, query_id in enumerate(
            query_ids
        ):

            query_vector = (
                projected_queries[
                    query_index
                ]
            )

            scores = (
                projected_corpus
                @ query_vector
            )

            order = np.argsort(
                -scores,
                kind="stable",
            )

            ranked_ids = [
                corpus_ids[index]
                for index in order
            ]

            rankings[query_id] = ranked_ids

        print(
            "Rankings generated:",
            len(rankings),
        )

        # ----------------------------------------------------------
        # IMPORTANT diagnostic:
        # Show the top results for the first query.
        # ----------------------------------------------------------

        first_query_id = query_ids[0]

        print(
            "\nTop-10 diagnostic for:",
            first_query_id,
        )

        print(
            "Query:",
            first_query_id,
        )

        first_ranking = rankings[
            first_query_id
        ]

        relevant = {
            str(doc_id)
            for doc_id in qrels.get(
                first_query_id,
                {},
            ).keys()
        }

        for rank, doc_id in enumerate(
            first_ranking[:10],
            start=1,
        ):

            marker = (
                " <-- RELEVANT"
                if doc_id in relevant
                else ""
            )

            print(
                f"  {rank:2d}. "
                f"{doc_id}{marker}"
            )

        # ----------------------------------------------------------
        # Save ranking
        # ----------------------------------------------------------

        dataset_slug = (
            dataset_name
            .lower()
            .replace("-", "_")
        )

        ranking_file = os.path.join(
            OUTPUT_DIR,
            f"{dataset_slug}_contriever_rankings_{dimension}d.json",
        )

        with open(
            ranking_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                rankings,
                f,
                indent=2,
            )

        print(
            "Saved:",
            ranking_file,
        )

        # ----------------------------------------------------------
        # Save projected embeddings as intermediate artifact.
        # These will NOT be committed to GitHub.
        # ----------------------------------------------------------

        projection_file = os.path.join(
            OUTPUT_DIR,
            f"{dataset_slug}_contriever_projected_{dimension}d.npz",
        )

        np.savez_compressed(
            projection_file,
            corpus_ids=np.asarray(
                corpus_ids
            ),
            embeddings=projected_corpus,
        )

        print(
            "Saved projection:",
            projection_file,
        )


# ======================================================================
# NDCG
# ======================================================================

def dcg_at_10(relevances):

    total = 0.0

    for rank, relevance in enumerate(
        relevances[:10],
        start=1,
    ):

        total += (
            (2.0 ** float(relevance) - 1.0)
            / np.log2(rank + 1)
        )

    return float(total)


def ndcg_for_query(
    ranked_ids,
    query_qrels,
):

    actual_relevances = [
        float(
            query_qrels.get(
                str(doc_id),
                0.0,
            )
        )
        for doc_id in ranked_ids[:10]
    ]

    actual_dcg = dcg_at_10(
        actual_relevances
    )

    ideal_relevances = sorted(
        (
            float(value)
            for value in query_qrels.values()
        ),
        reverse=True,
    )[:10]

    ideal_dcg = dcg_at_10(
        ideal_relevances
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_rankings(
    rankings,
    qrels,
):

    scores = []

    for query_id, ranked_ids in rankings.items():

        query_id = str(query_id)

        if query_id not in qrels:
            continue

        score = ndcg_for_query(
            ranked_ids,
            qrels[query_id],
        )

        scores.append(
            score
        )

    if not scores:
        raise ValueError(
            "No queries with both rankings and qrels."
        )

    return float(
        sum(scores) / len(scores)
    )


# ======================================================================
# MAIN DATASET RUN
# ======================================================================

def run_dataset(
    dataset_name,
    data_folder,
    tokenizer,
    model,
):

    (
        corpus_ids,
        corpus_texts,
        query_ids,
        query_texts,
        qrels,
    ) = load_dataset(
        dataset_name,
        data_folder,
    )

    # --------------------------------------------------------------
    # Encode corpus
    # --------------------------------------------------------------

    corpus_embeddings = encode_texts(
        corpus_texts,
        tokenizer,
        model,
        f"{dataset_name} corpus",
    )

    # --------------------------------------------------------------
    # Encode queries
    # --------------------------------------------------------------

    query_embeddings = encode_texts(
        query_texts,
        tokenizer,
        model,
        f"{dataset_name} queries",
    )

    # --------------------------------------------------------------
    # Save raw embeddings temporarily.
    # Useful for debugging/reproducibility.
    # --------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    dataset_slug = (
        dataset_name
        .lower()
        .replace("-", "_")
    )

    raw_embedding_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_slug}_contriever_embeddings.npz",
    )

    np.savez_compressed(
        raw_embedding_file,
        corpus_ids=np.asarray(
            corpus_ids
        ),
        corpus_embeddings=corpus_embeddings,
        query_ids=np.asarray(
            query_ids
        ),
        query_embeddings=query_embeddings,
    )

    print(
        "\nSaved raw embeddings:",
        raw_embedding_file,
    )

    # --------------------------------------------------------------
    # Generate rankings
    # --------------------------------------------------------------

    generate_rankings(
        dataset_name=dataset_name,
        corpus_ids=corpus_ids,
        corpus_embeddings=corpus_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        qrels=qrels,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BEIR Contriever PCA + Retrieval"
    )

    print(
        "=" * 70
    )

    print(
        "\nModel:",
        MODEL_DIR,
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    print(
        "Device:",
        DEVICE,
    )

    tokenizer, model = load_contriever()

    for dataset_name, data_folder in DATASETS.items():

        run_dataset(
            dataset_name=dataset_name,
            data_folder=data_folder,
            tokenizer=tokenizer,
            model=model,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEIR Contriever retrieval complete."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()