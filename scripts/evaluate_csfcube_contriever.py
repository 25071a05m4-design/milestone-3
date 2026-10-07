"""
Evaluate Contriever PCA rankings on CSFCube.

For each dimensionality:
    768
    384
    192
    96
    48

we compute NDCG@10 for the three official CSFCube facets:

    background
    method
    result

The final Contriever score for each dimension is the mean of the
three facet NDCG@10 values.

Output:
    results/csfcube_contriever.json
"""

from __future__ import annotations

import json
import math
import os
from typing import Any


# ======================================================================
# PATHS
# ======================================================================

DATA_ROOT = r"E:\ir_project\data\CSFCube"

RANKINGS_DIR = os.path.join(
    DATA_ROOT,
    "contriever_rankings",
)

BACKGROUND_FILE = os.path.join(
    DATA_ROOT,
    "test-pid2anns-csfcube-background.json",
)

METHOD_FILE = os.path.join(
    DATA_ROOT,
    "test-pid2anns-csfcube-method.json",
)

RESULT_FILE = os.path.join(
    DATA_ROOT,
    "test-pid2anns-csfcube-result.json",
)

OUTPUT_FILE = os.path.join(
    "results",
    "csfcube_contriever.json",
)

DIMENSIONS = [
    768,
    384,
    192,
    96,
    48,
]


# ======================================================================
# NDCG
# ======================================================================

def dcg_at_10(relevances):
    """
    Compute DCG@10 using the same gain formulation used by the
    existing CSFCube evaluation:

        gain = 2^relevance - 1
    """

    total = 0.0

    for rank, relevance in enumerate(
        relevances[:10],
        start=1,
    ):

        relevance = float(relevance)

        total += (
            (2.0 ** relevance - 1.0)
            / math.log2(rank + 1)
        )

    return total


def ndcg_at_10(ranked_ids, qrels):
    """
    Compute NDCG@10 for one query.
    """

    ranked_relevances = [
        float(
            qrels.get(
                str(doc_id),
                0.0,
            )
        )
        for doc_id in ranked_ids[:10]
    ]

    actual_dcg = dcg_at_10(
        ranked_relevances
    )

    ideal_relevances = sorted(
        (
            float(value)
            for value in qrels.values()
        ),
        reverse=True,
    )[:10]

    ideal_dcg = dcg_at_10(
        ideal_relevances
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


# ======================================================================
# LOAD JSON
# ======================================================================

def load_json(path):
    print(
        f"Loading:\n{path}"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ======================================================================
# EXTRACT RELEVANCE
# ======================================================================

def extract_relevance_mapping(
    annotation_data: Any,
):
    """
    Convert the CSFCube annotation JSON into:

        {
            query_id: {
                document_id: relevance,
                ...
            },
            ...
        }

    The CSFCube release has appeared in slightly different JSON
    layouts, so this function handles the known annotation patterns.
    """

    if not isinstance(
        annotation_data,
        dict,
    ):
        raise ValueError(
            "Expected annotation JSON "
            "to be a dictionary."
        )

    output = {}

    for query_id, value in annotation_data.items():

        query_id = str(query_id)

        # --------------------------------------------------------------
        # Pattern 1:
        #
        # {
        #   query_id: {
        #       "cands": [...],
        #       "relevance_max": [...]
        #   }
        # }
        # --------------------------------------------------------------

        if isinstance(value, dict):

            candidates = value.get(
                "cands"
            )

            relevance = value.get(
                "relevance_max"
            )

            if (
                isinstance(candidates, list)
                and isinstance(relevance, list)
            ):

                if len(candidates) != len(
                    relevance
                ):

                    raise ValueError(
                        f"Query {query_id}: "
                        f"{len(candidates)} candidates "
                        f"but {len(relevance)} relevance "
                        "values."
                    )

                output[query_id] = {
                    str(doc_id): float(score)
                    for doc_id, score in zip(
                        candidates,
                        relevance,
                    )
                }

                continue

            # ----------------------------------------------------------
            # Some releases may use relevance_ann1/ann2/adju but not
            # relevance_max. Prefer relevance_max when available.
            # ----------------------------------------------------------

            if (
                isinstance(candidates, list)
                and "relevance_ann1" in value
            ):

                ann1 = value.get(
                    "relevance_ann1",
                    [],
                )

                ann2 = value.get(
                    "relevance_ann2",
                    [],
                )

                if (
                    isinstance(ann1, list)
                    and isinstance(ann2, list)
                    and len(ann1) == len(candidates)
                    and len(ann2) == len(candidates)
                ):

                    max_relevance = [
                        max(
                            float(a),
                            float(b),
                        )
                        for a, b in zip(
                            ann1,
                            ann2,
                        )
                    ]

                    output[query_id] = {
                        str(doc_id): score
                        for doc_id, score in zip(
                            candidates,
                            max_relevance,
                        )
                    }

                    continue

        # --------------------------------------------------------------
        # Pattern 2:
        #
        # query_id -> list of annotation records
        # --------------------------------------------------------------

        if isinstance(value, list):

            mapping = {}

            for item in value:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                doc_id = (
                    item.get("pid")
                    or item.get("paper_id")
                    or item.get("doc_id")
                    or item.get("corpus_id")
                    or item.get("id")
                )

                score = (
                    item.get("relevance_max")
                    if "relevance_max" in item
                    else item.get("relevance")
                )

                if (
                    doc_id is not None
                    and score is not None
                ):

                    mapping[
                        str(doc_id)
                    ] = float(score)

            if mapping:

                output[query_id] = mapping

                continue

    if not output:
        raise ValueError(
            "Could not extract relevance judgments "
            "from the CSFCube annotation file."
        )

    return output


# ======================================================================
# EVALUATE ONE RANKING
# ======================================================================

def evaluate_ranking(
    rankings,
    qrels,
):
    """
    Compute mean NDCG@10 across queries.
    """

    scores = []

    for query_id, ranked_ids in rankings.items():

        query_id = str(query_id)

        if query_id not in qrels:
            continue

        score = ndcg_at_10(
            ranked_ids,
            qrels[query_id],
        )

        scores.append(score)

    if not scores:
        raise ValueError(
            "No ranking queries matched "
            "the annotation queries."
        )

    return (
        sum(scores) / len(scores),
        len(scores),
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "CSFCube Contriever NDCG@10 Evaluation"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------------

    print(
        "\nLoading annotations..."
    )

    background_raw = load_json(
        BACKGROUND_FILE
    )

    method_raw = load_json(
        METHOD_FILE
    )

    result_raw = load_json(
        RESULT_FILE
    )

    background_qrels = (
        extract_relevance_mapping(
            background_raw
        )
    )

    method_qrels = (
        extract_relevance_mapping(
            method_raw
        )
    )

    result_qrels = (
        extract_relevance_mapping(
            result_raw
        )
    )

    print(
        "\nAnnotation query counts:"
    )

    print(
        "  Background:",
        len(background_qrels),
    )

    print(
        "  Method:",
        len(method_qrels),
    )

    print(
        "  Result:",
        len(result_qrels),
    )

    # --------------------------------------------------------------
    # Evaluate all five dimensions
    # --------------------------------------------------------------

    results = {}

    for dimension in DIMENSIONS:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Contriever {dimension}d"
        )

        print(
            "-" * 70
        )

        ranking_file = os.path.join(
            RANKINGS_DIR,
            f"contriever_rankings_{dimension}d.json",
        )

        rankings = load_json(
            ranking_file
        )

        background_score, background_n = (
            evaluate_ranking(
                rankings,
                background_qrels,
            )
        )

        method_score, method_n = (
            evaluate_ranking(
                rankings,
                method_qrels,
            )
        )

        result_score, result_n = (
            evaluate_ranking(
                rankings,
                result_qrels,
            )
        )

        mean_score = (
            background_score
            + method_score
            + result_score
        ) / 3.0

        print(
            f"Background NDCG@10: "
            f"{background_score:.9f}"
        )

        print(
            f"Method NDCG@10:     "
            f"{method_score:.9f}"
        )

        print(
            f"Result NDCG@10:     "
            f"{result_score:.9f}"
        )

        print(
            f"Mean NDCG@10:       "
            f"{mean_score:.9f}"
        )

        print(
            "Queries evaluated:",
            background_n,
            method_n,
            result_n,
        )

        results[str(dimension)] = {
            "background": background_score,
            "method": method_score,
            "result": result_score,
            "mean_ndcg_at_10": mean_score,
            "queries": {
                "background": background_n,
                "method": method_n,
                "result": result_n,
            },
        }

    # --------------------------------------------------------------
    # Save compact result
    # --------------------------------------------------------------

    output_dir = os.path.dirname(
        OUTPUT_FILE
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    output = {
        "dataset": "CSFCube",
        "method": "Contriever",
        "metric": "NDCG@10",
        "dimensions": DIMENSIONS,
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "Evaluation complete."
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nFinal Contriever scores:"
    )

    for dimension in DIMENSIONS:

        score = results[
            str(dimension)
        ]["mean_ndcg_at_10"]

        print(
            f"  {dimension:>3}d: "
            f"{score:.9f}"
        )


if __name__ == "__main__":
    main()