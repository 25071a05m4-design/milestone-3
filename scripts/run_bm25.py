"""Run BM25 baselines and evaluate NDCG@10."""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pytrec_eval
from rank_bm25 import BM25Okapi

from evaluation import mean_ndcg_at_k


DATASETS = {
    "LIMIT-50k": r"E:\ir_project\data\limit-50k",
    "SciDocs": r"E:\ir_project\data\scidocs",
    "CSFCube": r"E:\ir_project\data\CSFCube",
}

OUTPUT_DIR = "results"

CSFCUBE_FACETS = {
    "background": "test-pid2anns-csfcube-background.json",
    "method": "test-pid2anns-csfcube-method.json",
    "result": "test-pid2anns-csfcube-result.json",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------
# BEIR DATASETS
# ---------------------------------------------------------

def load_beir_corpus(data_dir):
    path = os.path.join(data_dir, "corpus.jsonl")

    corpus = {}
    tokenized = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            doc_id = str(row["_id"])

            text = (
                f"{row.get('title', '')} "
                f"{row.get('text', '')}"
            )

            corpus[doc_id] = text
            tokenized.append(tokenize(text))

    return corpus, tokenized


def load_beir_queries(data_dir):
    path = os.path.join(data_dir, "queries.jsonl")

    queries = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            queries[str(row["_id"])] = row.get(
                "text",
                "",
            )

    return queries


def load_beir_qrels(data_dir):
    path = os.path.join(
        data_dir,
        "qrels",
        "test.tsv",
    )

    qrels = {}

    with open(path, encoding="utf-8") as f:
        next(f)

        for line in f:
            query_id, doc_id, score = (
                line.rstrip("\n").split("\t")
            )

            qrels.setdefault(
                str(query_id),
                {},
            )[str(doc_id)] = int(float(score))

    return qrels


# ---------------------------------------------------------
# CSFCUBE
# ---------------------------------------------------------

def load_csfcube_documents(data_dir):
    path = os.path.join(
        data_dir,
        "abstracts-csfcube-preds.jsonl",
    )

    documents = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            paper_id = str(row["paper_id"])

            title = row.get("title", "")

            abstract = row.get(
                "abstract",
                [],
            )

            if isinstance(abstract, list):
                abstract = " ".join(abstract)

            documents[paper_id] = (
                title
                + "\n"
                + abstract
            )

    return documents


def load_csfcube_pool(data_dir):
    path = os.path.join(
        data_dir,
        "test-pid2pool-csfcube.json",
    )

    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_csfcube_annotations(
    data_dir,
    filename,
):
    path = os.path.join(
        data_dir,
        filename,
    )

    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def annotation_to_qrels(annotation):
    """Convert CSFCube annotation to doc -> relevance."""

    candidate_ids = [
        str(x)
        for x in annotation["cands"]
    ]

    relevance_values = annotation["relevance_max"]

    return {
        str(doc_id): int(relevance)
        for doc_id, relevance in zip(
            candidate_ids,
            relevance_values,
        )
    }


# ---------------------------------------------------------
# BM25 RANKING
# ---------------------------------------------------------

def build_bm25_rankings(
    corpus,
    tokenized_corpus,
    queries,
    candidate_pools=None,
):
    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    doc_ids = list(corpus.keys())

    doc_index = {
        doc_id: index
        for index, doc_id in enumerate(doc_ids)
    }

    rankings = {}

    for count, (
        query_id,
        query_text,
    ) in enumerate(
        queries.items(),
        start=1,
    ):
        scores = bm25.get_scores(
            tokenize(query_text)
        )

        if candidate_pools is None:

            candidate_ids = doc_ids

        else:

            candidate_ids = [
                str(x)
                for x in candidate_pools[
                    str(query_id)
                ]["cands"]
            ]

        candidate_indices = [
            doc_index[doc_id]
            for doc_id in candidate_ids
        ]

        candidate_scores = [
            scores[index]
            for index in candidate_indices
        ]

        order = np.argsort(
            -np.asarray(candidate_scores),
            kind="stable",
        )

        rankings[str(query_id)] = [
            candidate_ids[i]
            for i in order
        ]

        if count % 10 == 0 or count == len(queries):
            print(
                f"  Ranked {count}/{len(queries)} queries"
            )

    return rankings


# ---------------------------------------------------------
# PYTREC EVALUATION
# ---------------------------------------------------------

def evaluate_pytrec(
    rankings,
    qrels,
):
    run = {}

    for query_id, doc_ids in rankings.items():

        run[str(query_id)] = {
            str(doc_id): float(
                len(doc_ids) - rank
            )
            for rank, doc_id in enumerate(doc_ids)
        }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {"ndcg_cut.10"},
    )

    scores = evaluator.evaluate(run)

    values = [
        float(result["ndcg_cut_10"])
        for result in scores.values()
    ]

    return sum(values) / len(values)


# ---------------------------------------------------------
# CSFCUBE EVALUATION
# ---------------------------------------------------------

def evaluate_csfcube_facet(
    rankings,
    annotations,
):
    qrels = {}

    for query_id, annotation in annotations.items():

        qrels[str(query_id)] = annotation_to_qrels(
            annotation
        )

    return mean_ndcg_at_k(
        rankings,
        qrels,
        k=10,
    )


# ---------------------------------------------------------
# BEIR DATASET RUNNER
# ---------------------------------------------------------

def run_beir_dataset(dataset_name):

    data_dir = DATASETS[dataset_name]

    print("=" * 70)
    print("DATASET:", dataset_name)
    print("=" * 70)

    print("Loading corpus...")

    corpus, tokenized_corpus = load_beir_corpus(
        data_dir
    )

    print(
        "Corpus documents:",
        len(corpus),
    )

    print("Loading queries...")

    queries = load_beir_queries(
        data_dir
    )

    print(
        "Queries:",
        len(queries),
    )

    print("Loading qrels...")

    qrels = load_beir_qrels(
        data_dir
    )

    print(
        "Qrel queries:",
        len(qrels),
    )

    rankings = build_bm25_rankings(
        corpus,
        tokenized_corpus,
        queries,
    )

    custom_ndcg = mean_ndcg_at_k(
        rankings,
        qrels,
        k=10,
    )

    pytrec_ndcg = evaluate_pytrec(
        rankings,
        qrels,
    )

    difference = abs(
        custom_ndcg - pytrec_ndcg
    )

    print()
    print(
        "Custom NDCG@10:",
        custom_ndcg,
    )

    print(
        "pytrec_eval NDCG@10:",
        pytrec_ndcg,
    )

    print(
        "Absolute difference:",
        difference,
    )

    if difference > 1e-9:
        raise RuntimeError(
            "NDCG mismatch between custom evaluator "
            "and pytrec_eval."
        )

    print("VERIFIED: scores match.")

    output_filename = (
        dataset_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_bm25.json"
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        output_filename,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result = {
        "dataset": dataset_name,
        "method": "BM25",
        "budget": "1.0",
        "metric": "NDCG@10",
        "score": pytrec_ndcg,
        "verified_with": "pytrec_eval",
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "Saved:",
        output_file,
    )


# ---------------------------------------------------------
# CSFCUBE RUNNER
# ---------------------------------------------------------

def run_csfcube():

    data_dir = DATASETS["CSFCube"]

    print("=" * 70)
    print("DATASET: CSFCube")
    print("=" * 70)

    print("Loading documents...")

    documents = load_csfcube_documents(
        data_dir
    )

    print(
        "Documents:",
        len(documents),
    )

    print("Loading candidate pools...")

    pools = load_csfcube_pool(
        data_dir
    )

    print(
        "Queries:",
        len(pools),
    )

    # -----------------------------------------------------
    # CSFCube queries
    #
    # Query papers are included in the same JSONL file.
    # Use their title + abstract as the BM25 query text.
    # -----------------------------------------------------

    queries = {}

    for query_id in pools:

        query_id = str(query_id)

        if query_id not in documents:
            raise KeyError(
                f"Query paper {query_id} "
                "not found in CSFCube documents."
            )

        queries[query_id] = documents[query_id]

    # -----------------------------------------------------
    # BM25 ranking restricted to candidate pools
    # -----------------------------------------------------

    tokenized_documents = [
        tokenize(text)
        for text in documents.values()
    ]

    rankings = build_bm25_rankings(
        documents,
        tokenized_documents,
        queries,
        candidate_pools=pools,
    )

    # -----------------------------------------------------
    # Evaluate all three CSFCube facets
    # -----------------------------------------------------

    facet_scores = {}

    for facet, filename in CSFCUBE_FACETS.items():

        print()
        print("Facet:", facet)

        annotations = load_csfcube_annotations(
            data_dir,
            filename,
        )

        score = evaluate_csfcube_facet(
            rankings,
            annotations,
        )

        facet_scores[facet] = score

        print(
            "NDCG@10:",
            score,
        )

    # -----------------------------------------------------
    # Overall CSFCube score
    # -----------------------------------------------------

    overall = (
        sum(facet_scores.values())
        / len(facet_scores)
    )

    print()
    print(
        "CSFCube mean NDCG@10:",
        overall,
    )

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "csfcube_bm25.json",
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result = {
        "dataset": "CSFCube",
        "method": "BM25",
        "budget": "1.0",
        "metric": "NDCG@10",
        "facet_scores": facet_scores,
        "score": overall,
        "verified_with": "project_csfcube_protocol",
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "Saved:",
        output_file,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
    )

    args = parser.parse_args()

    if args.dataset == "CSFCube":

        run_csfcube()

    else:

        run_beir_dataset(
            args.dataset
        )


if __name__ == "__main__":
    main()