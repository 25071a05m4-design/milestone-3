from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from beir.datasets.data_loader import GenericDataLoader


# ============================================================
# CONFIG
# ============================================================

DATASETS = {
    "LIMIT-50k": Path(r"E:\ir_project\data\limit-50k"),
    "SciDocs": Path(r"E:\ir_project\data\scidocs"),
}

CSFCUBE_DIR = Path(r"E:\ir_project\data\CSFCube")

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

OUT_DIR = Path(r"E:\ir_project\data\medcpt_beir")
RESULTS_DIR = Path("results")

DIMS = [768, 384, 192, 96, 48]

BATCH_SIZE_QUERY = 32
BATCH_SIZE_ARTICLE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================

def load_model(name):
    print(f"Loading {name} ...")

    tokenizer = AutoTokenizer.from_pretrained(name)

    model = AutoModel.from_pretrained(
        name,
        torch_dtype=torch.float32,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Loaded {name} on {DEVICE}")

    return tokenizer, model


# ============================================================
# ENCODING
# ============================================================

@torch.no_grad()
def encode_texts(texts, tokenizer, model, batch_size, max_length):
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        hidden = model(**encoded).last_hidden_state

        # Official MedCPT representation:
        # [CLS] token embedding
        embeddings = hidden[:, 0, :]

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        outputs.append(
            embeddings.cpu().numpy().astype(np.float32)
        )

        if start % (batch_size * 20) == 0:
            print(
                f"  encoded {min(start + batch_size, len(texts))}/"
                f"{len(texts)}"
            )

    return np.concatenate(outputs, axis=0)


# ============================================================
# PCA
# ============================================================

def fit_pca(X, dim):
    X = X.astype(np.float64)

    mean = X.mean(axis=0)
    Xc = X - mean

    # covariance-free PCA using SVD
    _, _, Vt = np.linalg.svd(
        Xc,
        full_matrices=False,
    )

    components = Vt[:dim]

    return mean.astype(np.float32), components.astype(np.float32)


def project(X, mean, components):
    Y = (X - mean) @ components.T

    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Y = Y / np.maximum(norms, 1e-12)

    return Y.astype(np.float32)


# ============================================================
# NDCG
# ============================================================

def dcg(rels):
    total = 0.0

    for rank, rel in enumerate(rels, start=1):
        total += (
            (2.0 ** float(rel) - 1.0)
            / math.log2(rank + 1)
        )

    return total


def mean_ndcg(rankings, qrels, k=10):
    scores = []

    for qid, ranked_docs in rankings.items():

        if qid not in qrels:
            continue

        qr = qrels[qid]

        actual = [
            float(qr.get(doc_id, 0))
            for doc_id in ranked_docs[:k]
        ]

        ideal = sorted(
            [float(v) for v in qr.values()],
            reverse=True,
        )[:k]

        ideal_dcg = dcg(ideal)

        if ideal_dcg == 0:
            continue

        scores.append(
            dcg(actual) / ideal_dcg
        )

    return float(np.mean(scores))


# ============================================================
# BEIR RETRIEVAL
# ============================================================

def run_beir_dataset(name, data_dir, query_tokenizer, query_model,
                     article_tokenizer, article_model):

    print()
    print("=" * 70)
    print(f"MEDCPT: {name}")
    print("=" * 70)

    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(data_dir)
    ).load("test")

    print("Corpus:", len(corpus))
    print("Queries:", len(queries))
    print("Qrels:", sum(len(v) for v in qrels.values()))

    doc_ids = list(corpus.keys())

    doc_texts = []

    for doc_id in doc_ids:
        doc = corpus[doc_id]

        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""

        if title and text:
            combined = title + " " + text
        else:
            combined = title or text

        doc_texts.append(combined)

    query_ids = list(queries.keys())

    query_texts = [
    queries[qid]
    for qid in query_ids
]

    # --------------------------------------------------------
    # Encode corpus ONCE
    # --------------------------------------------------------

    doc_embeddings = encode_texts(
        doc_texts,
        article_tokenizer,
        article_model,
        BATCH_SIZE_ARTICLE,
        512,
    )

    # --------------------------------------------------------
    # Encode queries ONCE
    # --------------------------------------------------------

    query_embeddings = encode_texts(
        query_texts,
        query_tokenizer,
        query_model,
        BATCH_SIZE_QUERY,
        64,
    )

    print("Document embeddings:", doc_embeddings.shape)
    print("Query embeddings:", query_embeddings.shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUT_DIR / f"{name.lower().replace('-', '_')}_docs.npy",
        doc_embeddings,
    )

    np.save(
        OUT_DIR / f"{name.lower().replace('-', '_')}_queries.npy",
        query_embeddings,
    )

    results = {}

    # --------------------------------------------------------
    # Five PCA budgets
    # --------------------------------------------------------

    for dim in DIMS:

        print()
        print(f"Running MedCPT PCA dimension = {dim}")

        mean, components = fit_pca(
            doc_embeddings,
            dim,
        )

        docs_pca = project(
            doc_embeddings,
            mean,
            components,
        )

        queries_pca = project(
            query_embeddings,
            mean,
            components,
        )

        rankings = {}

        for qi, qid in enumerate(query_ids):

            scores = docs_pca @ queries_pca[qi]

            top_k = min(100, len(scores))

            idx = np.argpartition(
                -scores,
                top_k - 1
            )[:top_k]

            idx = idx[
                np.argsort(
                    -scores[idx]
                )
            ]

            rankings[qid] = [
                doc_ids[i]
                for i in idx
            ]

        score = mean_ndcg(
            rankings,
            qrels,
            k=10,
        )

        budget = dim / 768.0

        results[str(budget)] = {
            "dimension": dim,
            "ndcg@10": score,
        }

        print(
            f"  budget={budget} "
            f"dim={dim} "
            f"NDCG@10={score:.9f}"
        )

    output = {
        "method": "MedCPT",
        "dataset": name,
        "metric": "NDCG@10",
        "budgets": results,
    }

    output_path = RESULTS_DIR / (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_medcpt.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("Saved:", output_path)

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDCPT FULL-CORPUS RETRIEVAL")
    print("=" * 70)

    print("Device:", DEVICE)

    query_tokenizer, query_model = load_model(
        QUERY_MODEL
    )

    article_tokenizer, article_model = load_model(
        ARTICLE_MODEL
    )

    all_results = {}

    for name, data_dir in DATASETS.items():

        all_results[name] = run_beir_dataset(
            name,
            data_dir,
            query_tokenizer,
            query_model,
            article_tokenizer,
            article_model,
        )

    print()
    print("=" * 70)
    print("MEDCPT BEIR RETRIEVAL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()