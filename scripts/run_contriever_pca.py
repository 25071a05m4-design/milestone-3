"""
CSFCube Contriever PCA + Retrieval
==================================

Milestone 4:
- Contriever dense retrieval
- PCA fitted only on candidate corpus documents
- Query vectors transformed using the same corpus-fitted PCA
- Retrieval over the official CSFCube candidate pools
- Dimensions:
    768
    384
    192
    96
    48

Important:
The generated .json and .npz files are intermediate artifacts.
Do not commit the large embedding/projection files to GitHub.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from safetensors.torch import load_file
from transformers import AutoModel, BertConfig, BertTokenizer

from truncate import PCAProjector


# ======================================================================
# PATHS
# ======================================================================

DATA_ROOT = r"E:\ir_project\data\CSFCube"

MODEL_DIR = (
    r"E:\ir_project\hf_cache\hub\models--facebook--contriever"
    r"\snapshots\4bce1e1123ec62a1cbc1f9d27dc9c8e6e1e51ce0"
)

EMBEDDINGS_FILE = os.path.join(
    DATA_ROOT,
    "contriever_embeddings.npz",
)

POOL_FILE = os.path.join(
    DATA_ROOT,
    "test-pid2pool-csfcube.json",
)

QUERY_FILE = os.path.join(
    DATA_ROOT,
    "queries-release.csv",
)

OUTPUT_DIR = os.path.join(
    DATA_ROOT,
    "contriever_rankings",
)


# ======================================================================
# EXPERIMENT SETTINGS
# ======================================================================

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
# UTILITY
# ======================================================================

def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a matrix.
    """

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
# LOAD EMBEDDINGS
# ======================================================================

def load_embeddings():
    """
    Load the previously generated Contriever embeddings.
    """

    print("Loading embeddings...")

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            "Contriever embeddings were not found:\n"
            f"{EMBEDDINGS_FILE}\n\n"
            "Run:\n"
            "python scripts\\build_contriever_embeddings.py"
        )

    data = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    paper_ids = [
        str(x)
        for x in data["paper_ids"]
    ]

    embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    print(
        "Paper IDs:",
        len(paper_ids),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    if len(paper_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of paper IDs does not match "
            "number of embedding rows."
        )

    if embeddings.shape[1] != 768:
        raise ValueError(
            "Expected Contriever embeddings "
            "with 768 dimensions, got "
            f"{embeddings.shape[1]}."
        )

    return paper_ids, embeddings


# ======================================================================
# LOAD CSFCUBE CANDIDATE POOLS
# ======================================================================

def load_pools():
    """
    Load the official CSFCube candidate pools.

    The CSFCube release does NOT contain exactly 240 candidates
    for every query. Candidate counts vary by query.

    Correct structure:

        {
            "query_id": {
                "cands": [...],
                "cand_source": [...]
            }
        }

    We use only the "cands" field.
    """

    print("\nLoading CSFCube candidate pools...")

    if not os.path.exists(POOL_FILE):
        raise FileNotFoundError(
            f"Candidate pool file not found:\n{POOL_FILE}"
        )

    with open(
        POOL_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        raw_pools = json.load(f)

    if not isinstance(raw_pools, dict):
        raise ValueError(
            "Unexpected CSFCube pool format. "
            "Expected a JSON object."
        )

    pools = {}

    for query_id, value in raw_pools.items():

        query_id = str(query_id)

        if not isinstance(value, dict):
            raise ValueError(
                f"Unexpected pool format "
                f"for query {query_id}."
            )

        if "cands" not in value:
            raise ValueError(
                f"No 'cands' field found "
                f"for query {query_id}."
            )

        candidates = [
            str(doc_id)
            for doc_id in value["cands"]
        ]

        if not candidates:
            raise ValueError(
                f"Query {query_id} has no candidates."
            )

        pools[query_id] = candidates

    if not pools:
        raise ValueError(
            "No CSFCube candidate pools were loaded."
        )

    print(
        "Queries:",
        len(pools),
    )

    candidate_counts = [
        len(candidates)
        for candidates in pools.values()
    ]

    print(
        "Candidate counts:",
        f"min={min(candidate_counts)},",
        f"max={max(candidate_counts)},",
        f"mean={sum(candidate_counts) / len(candidate_counts):.1f}",
    )

    for query_id in list(pools.keys())[:5]:
        print(
            f"  Query {query_id}: "
            f"{len(pools[query_id])} candidates"
        )

    return pools


# ======================================================================
# LOAD CSFCUBE QUERIES
# ======================================================================

def load_queries():
    """
    Load CSFCube query texts.

    Official queries-release.csv columns:

        pid
        facet
        year
        title
        paper type

    For retrieval:
        pid   -> query ID
        title -> query text

    The CSFCube candidate-pool query IDs are the paper IDs.
    """

    print("\nLoading CSFCube queries...")

    if not os.path.exists(QUERY_FILE):
        raise FileNotFoundError(
            f"Query file not found:\n{QUERY_FILE}"
        )

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "queries-release.csv has no header."
            )

        print(
            "Query CSV columns:",
            reader.fieldnames,
        )

        if "pid" not in reader.fieldnames:
            raise ValueError(
                "Expected 'pid' column in "
                "CSFCube queries-release.csv."
            )

        if "title" not in reader.fieldnames:
            raise ValueError(
                "Expected 'title' column in "
                "CSFCube queries-release.csv."
            )

        queries = {}

        for row in reader:

            query_id = str(
                row.get("pid", "")
            ).strip()

            query_text = str(
                row.get("title", "")
            ).strip()

            if not query_id:
                continue

            queries[query_id] = query_text

    print(
        "Queries loaded:",
        len(queries),
    )

    if not queries:
        raise ValueError(
            "No CSFCube queries were loaded."
        )

    return queries


# ======================================================================
# LOAD CONTRIEVER
# ======================================================================

def load_contriever():
    """
    Construct the Contriever BERT architecture and load the local
    safetensors checkpoint.

    This avoids torch.load() on pytorch_model.bin because the installed
    PyTorch version is below the minimum required by recent Transformers
    security checks for pickle-based weights.
    """

    print("\nLoading Contriever tokenizer...")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        "\nBuilding Contriever configuration..."
    )

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

    print(
        "Hidden size:",
        config.hidden_size,
    )

    print(
        "Hidden layers:",
        config.num_hidden_layers,
    )

    print(
        "Attention heads:",
        config.num_attention_heads,
    )

    print(
        "\nCreating Contriever model..."
    )

    model = AutoModel.from_config(
        config
    )

    print(
        "Model architecture created."
    )

    safetensors_file = os.path.join(
        MODEL_DIR,
        "model.safetensors",
    )

    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(
            "Contriever safetensors checkpoint "
            "was not found:\n"
            f"{safetensors_file}"
        )

    print(
        "\nLoading model.safetensors..."
    )

    state_dict = load_file(
        safetensors_file,
        device="cpu",
    )

    print(
        "Weight tensors loaded:",
        len(state_dict),
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing_keys:
        print(
            "\nWARNING: Missing keys:"
        )

        for key in missing_keys:
            print(
                " ",
                key,
            )

    if unexpected_keys:
        print(
            "\nWARNING: Unexpected keys:"
        )

        for key in unexpected_keys:
            print(
                " ",
                key,
            )

    model.to(DEVICE)
    model.eval()

    print(
        "\nContriever model loaded successfully."
    )

    print(
        "Device:",
        DEVICE,
    )

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
    """
    Mean-pool token embeddings using the attention mask.
    """

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
# ENCODE QUERIES
# ======================================================================

def encode_queries(
    queries,
    tokenizer,
    model,
):
    """
    Encode all CSFCube queries using Contriever.

    Returns:
        query_ids
        query_embeddings
    """

    print(
        "\nGenerating query embeddings..."
    )

    query_ids = list(
        queries.keys()
    )

    query_texts = [
        queries[query_id]
        for query_id in query_ids
    ]

    embeddings = []

    with torch.no_grad():

        for start in range(
            0,
            len(query_ids),
            BATCH_SIZE,
        ):

            batch_ids = query_ids[
                start:start + BATCH_SIZE
            ]

            batch_texts = [
                queries[query_id]
                for query_id in batch_ids
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
                len(query_ids),
            )

            print(
                f"  {done}/{len(query_ids)}"
            )

    query_embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    print(
        "\nQuery embedding shape:",
        query_embeddings.shape,
    )

    return query_ids, query_embeddings


# ======================================================================
# GENERATE RANKINGS
# ======================================================================

def generate_rankings(
    paper_ids,
    embeddings,
    pools,
    query_ids,
    query_embeddings,
):
    """
    Generate rankings for every PCA dimension.

    PCA is fitted ONLY on unique candidate corpus documents.

    For each dimension:
        corpus -> PCA
        queries -> SAME PCA
        normalize both
        cosine similarity
        rank candidates
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "Generating Contriever PCA rankings"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------------
    # Paper ID -> embedding row
    # --------------------------------------------------------------

    paper_ids = [
        str(x)
        for x in paper_ids
    ]

    paper_to_index = {
        paper_id: index
        for index, paper_id in enumerate(
            paper_ids
        )
    }

    # --------------------------------------------------------------
    # Collect unique candidate documents
    # --------------------------------------------------------------

    unique_candidate_ids = []
    seen = set()

    for query_id, candidates in pools.items():

        for doc_id in candidates:

            doc_id = str(doc_id)

            if doc_id not in seen:

                seen.add(doc_id)
                unique_candidate_ids.append(
                    doc_id
                )

    print(
        "\nPCA training corpus:"
    )

    print(
        "Unique candidate documents:",
        len(unique_candidate_ids),
    )

    # --------------------------------------------------------------
    # Verify candidate IDs
    # --------------------------------------------------------------

    missing_candidates = [
        doc_id
        for doc_id in unique_candidate_ids
        if doc_id not in paper_to_index
    ]

    if missing_candidates:

        print(
            "\nMissing candidate IDs:",
            len(missing_candidates),
        )

        print(
            "Examples:",
            missing_candidates[:20],
        )

        raise ValueError(
            "Some CSFCube candidate documents "
            "are missing from Contriever embeddings."
        )

    # --------------------------------------------------------------
    # PCA fitting matrix
    # --------------------------------------------------------------

    candidate_indices = [
        paper_to_index[doc_id]
        for doc_id in unique_candidate_ids
    ]

    candidate_matrix = embeddings[
        candidate_indices
    ].astype(np.float32)

    print(
        "Candidate matrix:",
        candidate_matrix.shape,
    )

    # --------------------------------------------------------------
    # Query ID -> embedding
    # --------------------------------------------------------------

    query_ids = [
        str(x)
        for x in query_ids
    ]

    if len(query_ids) != len(
        query_embeddings
    ):
        raise ValueError(
            "Number of query IDs does not match "
            "number of query embeddings."
        )

    query_id_to_index = {
        query_id: index
        for index, query_id in enumerate(
            query_ids
        )
    }

    # --------------------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Process each dimensionality
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
        # 768d: original embeddings
        # ==========================================================

        if dimension == embeddings.shape[1]:

            print(
                "Full dimension: "
                "using original normalized embeddings."
            )

            projected_embeddings = (
                embeddings.astype(np.float32)
            )

            projected_queries = (
                query_embeddings.astype(
                    np.float32
                )
            )

        # ==========================================================
        # PCA dimensions
        # ==========================================================

        else:

            if dimension >= embeddings.shape[1]:

                raise ValueError(
                    f"Invalid PCA dimension: "
                    f"{dimension}"
                )

            print(
                f"Fitting PCA on "
                f"{candidate_matrix.shape[0]} "
                f"candidate documents..."
            )

            projector = PCAProjector.fit(
                candidate_matrix,
                n_components=dimension,
            )

            # ------------------------------------------------------
            # Transform ALL corpus embeddings.
            # ------------------------------------------------------

            projected_embeddings = (
                projector.transform(
                    embeddings
                ).astype(np.float32)
            )

            # ------------------------------------------------------
            # IMPORTANT:
            #
            # Transform queries with the SAME PCA projector.
            #
            # PCA was fitted only on corpus candidates.
            # Query vectors are NOT used to fit PCA.
            # ------------------------------------------------------

            projected_queries = (
                projector.transform(
                    query_embeddings
                ).astype(np.float32)
            )

            print(
                "Projected corpus shape:",
                projected_embeddings.shape,
            )

            print(
                "Projected query shape:",
                projected_queries.shape,
            )

        # ==========================================================
        # Normalize after projection
        # ==========================================================

        projected_embeddings = normalize_rows(
            projected_embeddings
        )

        projected_queries = normalize_rows(
            projected_queries
        )

        # ==========================================================
        # Sanity check
        # ==========================================================

        if (
            projected_embeddings.shape[1]
            != projected_queries.shape[1]
        ):

            raise ValueError(
                "Corpus/query dimensionality mismatch "
                "after projection: "
                f"corpus={projected_embeddings.shape[1]}, "
                f"queries={projected_queries.shape[1]}"
            )

        # ==========================================================
        # Generate rankings
        # ==========================================================

        rankings = {}

        for query_id in query_ids:

            query_id = str(query_id)

            if query_id not in pools:

                continue

            query_index = (
                query_id_to_index[query_id]
            )

            query_vector = (
                projected_queries[
                    query_index
                ]
            )

            candidate_ids = [
                str(doc_id)
                for doc_id in pools[query_id]
            ]

            candidate_indices = [
                paper_to_index[doc_id]
                for doc_id in candidate_ids
            ]

            candidate_vectors = (
                projected_embeddings[
                    candidate_indices
                ]
            )

            # ------------------------------------------------------
            # Candidate vectors:
            #     (num_candidates, dimension)
            #
            # Query vector:
            #     (dimension,)
            #
            # Result:
            #     (num_candidates,)
            # ------------------------------------------------------

            scores = (
                candidate_vectors
                @ query_vector
            )

            # Stable descending sort.
            order = np.argsort(
                -scores,
                kind="stable",
            )

            ranked_ids = [
                candidate_ids[index]
                for index in order
            ]

            rankings[query_id] = ranked_ids

        print(
            "Rankings generated:",
            len(rankings),
        )

        if len(rankings) != len(pools):

            missing_queries = [
                query_id
                for query_id in pools
                if query_id not in rankings
            ]

            raise ValueError(
                "Not all CSFCube queries received "
                "rankings.\n"
                f"Missing: {missing_queries}"
            )

        # ==========================================================
        # Save ranking JSON
        # ==========================================================

        ranking_file = os.path.join(
            OUTPUT_DIR,
            f"contriever_rankings_{dimension}d.json",
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

        # ==========================================================
        # Save projected embeddings
        #
        # These are intermediate files.
        # Do NOT commit them to GitHub.
        # ==========================================================

        projection_file = os.path.join(
            OUTPUT_DIR,
            f"contriever_projected_{dimension}d.npz",
        )

        np.savez_compressed(
            projection_file,
            paper_ids=np.asarray(
                paper_ids
            ),
            embeddings=projected_embeddings,
        )

        print(
            "Saved projection:",
            projection_file,
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "CSFCube Contriever PCA + Retrieval"
    )

    print(
        "=" * 70
    )

    print(
        "\nData root:"
    )

    print(
        DATA_ROOT
    )

    print(
        "\nEmbeddings:"
    )

    print(
        EMBEDDINGS_FILE
    )

    print(
        "\nModel:"
    )

    print(
        MODEL_DIR
    )

    print(
        "\nCandidate pools:"
    )

    print(
        POOL_FILE
    )

    print(
        "\nQueries:"
    )

    print(
        QUERY_FILE
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------------
    # Load corpus embeddings
    # --------------------------------------------------------------

    paper_ids, embeddings = (
        load_embeddings()
    )

    # --------------------------------------------------------------
    # Load candidate pools
    # --------------------------------------------------------------

    pools = load_pools()

    # --------------------------------------------------------------
    # Load query texts
    # --------------------------------------------------------------

    queries = load_queries()

    # --------------------------------------------------------------
    # Verify query/pool alignment
    # --------------------------------------------------------------

    pool_query_ids = set(
        pools.keys()
    )

    query_ids = set(
        queries.keys()
    )

    missing_query_texts = sorted(
        pool_query_ids - query_ids
    )

    extra_query_texts = sorted(
        query_ids - pool_query_ids
    )

    if missing_query_texts:

        raise ValueError(
            "CSFCube candidate pools contain "
            "query IDs that are missing from "
            "queries-release.csv:\n"
            f"{missing_query_texts}"
        )

    if extra_query_texts:

        print(
            "\nWarning: queries-release.csv "
            "contains additional rows not present "
            "in candidate pools:"
        )

        print(
            extra_query_texts
        )

    # --------------------------------------------------------------
    # Load Contriever
    # --------------------------------------------------------------

    tokenizer, model = (
        load_contriever()
    )

    # --------------------------------------------------------------
    # Encode queries
    # --------------------------------------------------------------

    ordered_queries = {
        query_id: queries[query_id]
        for query_id in pools.keys()
    }

    query_ids, query_embeddings = (
        encode_queries(
            ordered_queries,
            tokenizer,
            model,
        )
    )

    # --------------------------------------------------------------
    # Generate PCA rankings
    # --------------------------------------------------------------

    generate_rankings(
        paper_ids=paper_ids,
        embeddings=embeddings,
        pools=pools,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Contriever PCA retrieval complete."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()