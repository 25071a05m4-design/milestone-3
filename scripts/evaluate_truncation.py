import json
import os
import numpy as np


BASE = "CSFCube"

PRED_FILE = os.path.join(
    BASE,
    "test-pid2pool-csfcube.json"
)

SPLIT_FILE = os.path.join(
    BASE,
    "evaluation_splits.json"
)

FACETS = [
    "background",
    "method",
    "result"
]

CUTS = [
    10,
    20,
    50,
    100,
    150,
    200,
    240
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(rels, k):
    return sum(rels[:k]) / k if k > 0 else 0.0


def recall_at_k(rels, k):
    total_relevant = sum(rels)

    if total_relevant == 0:
        return 0.0

    return sum(rels[:k]) / total_relevant


def average_precision(rels):
    total_relevant = sum(rels)

    if total_relevant == 0:
        return 0.0

    score = 0.0
    seen = 0

    for i, rel in enumerate(rels, start=1):
        if rel == 1:
            seen += 1
            score += seen / i

    return score / total_relevant


def reciprocal_rank(rels):
    for i, rel in enumerate(rels, start=1):
        if rel == 1:
            return 1.0 / i

    return 0.0


def r_precision(rels):
    total_relevant = sum(rels)

    if total_relevant == 0:
        return 0.0

    return precision_at_k(rels, total_relevant)


def ndcg_at_k(rels, k):
    rels = rels[:k]

    dcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(rels)
    )

    ideal = sorted(rels, reverse=True)

    idcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ideal)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_metrics(ranked_judgements):
    graded = ranked_judgements

    # Same threshold used by official CSFCube evaluator.
    binary = [
        1 if rel >= 2 else 0
        for rel in graded
    ]

    ndcg = ndcg_at_k(
        graded,
        len(graded)
    )

    ndcg_pr = ndcg_at_k(
        graded,
        int(0.20 * len(graded))
    )

    ndcg_20 = ndcg_at_k(
        graded,
        20
    )

    precision = precision_at_k(
        binary,
        20
    )

    recall = recall_at_k(
        binary,
        20
    )

    rprec = r_precision(binary)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    ap = average_precision(binary)

    rr = reciprocal_rank(binary)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_precision": rprec,
        "av_precision": ap,
        "reciprocal_rank": rr,
        "ndcg": ndcg,
        "ndcg@20": ndcg_20,
        "ndcg%20": ndcg_pr
    }


def evaluate_queries(
    predictions,
    gold,
    query_ids,
    cutoff
):
    metrics = []

    for split_id in query_ids:

        qid = split_id.split("_")[0]

        if qid not in predictions:
            continue

        if qid not in gold:
            continue

        pred_cands = predictions[qid]["cands"]

        gold_cands = gold[qid]["cands"]
        gold_rels = gold[qid]["relevance_adju"]

        relevance = dict(
            zip(gold_cands, gold_rels)
        )

        ranked_ids = pred_cands[:cutoff]

        ranked_rels = [
            relevance.get(pid, 0)
            for pid in ranked_ids
        ]

        metrics.append(
            compute_metrics(ranked_rels)
        )

    if not metrics:
        return None

    result = {}

    for key in metrics[0]:
        result[key] = float(
            np.mean([m[key] for m in metrics])
        )

    result["queries"] = len(metrics)

    return result


def print_metrics(label, metrics):
    print(
        f"{label:12s} | "
        f"NDCG {metrics['ndcg']:.4f} | "
        f"NDCG@20 {metrics['ndcg@20']:.4f} | "
        f"NDCG%20 {metrics['ndcg%20']:.4f} | "
        f"P@20 {metrics['precision']:.4f} | "
        f"R@20 {metrics['recall']:.4f} | "
        f"R-Prec {metrics['r_precision']:.4f}"
    )


def main():

    predictions = load_json(PRED_FILE)
    splits = load_json(SPLIT_FILE)

    print("Predictions:", len(predictions))
    print()

    for facet in FACETS:

        gold_file = os.path.join(
            BASE,
            f"test-pid2anns-csfcube-{facet}.json"
        )

        gold = load_json(gold_file)

        fold1 = splits[facet]["fold1_test"]
        fold2 = splits[facet]["fold2_test"]

        print("=" * 100)
        print("FACET:", facet)
        print(
            f"Fold 1 queries: {len(fold1)} | "
            f"Fold 2 queries: {len(fold2)}"
        )
        print("=" * 100)

        for cutoff in CUTS:

            m1 = evaluate_queries(
                predictions,
                gold,
                fold1,
                cutoff
            )

            m2 = evaluate_queries(
                predictions,
                gold,
                fold2,
                cutoff
            )

            print()
            print(f"TOP-{cutoff}")

            print_metrics(
                "Fold 1",
                m1
            )

            print_metrics(
                "Fold 2",
                m2
            )

            print(
                f"{'Average':12s} | "
                f"NDCG {(m1['ndcg'] + m2['ndcg']) / 2:.4f} | "
                f"NDCG@20 {(m1['ndcg@20'] + m2['ndcg@20']) / 2:.4f} | "
                f"NDCG%20 {(m1['ndcg%20'] + m2['ndcg%20']) / 2:.4f} | "
                f"P@20 {(m1['precision'] + m2['precision']) / 2:.4f} | "
                f"R@20 {(m1['recall'] + m2['recall']) / 2:.4f} | "
                f"R-Prec {(m1['r_precision'] + m2['r_precision']) / 2:.4f}"
            )

        print()


if __name__ == "__main__":
    main()