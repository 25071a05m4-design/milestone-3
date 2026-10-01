import json
import os
import numpy as np


ROOT = "CSFCube"
RANKING_DIR = os.path.join(ROOT, "contriever_rankings")

FACETS = {
    "background": "test-pid2anns-csfcube-background.json",
    "method": "test-pid2anns-csfcube-method.json",
    "result": "test-pid2anns-csfcube-result.json",
}

DIMS = [768, 384, 192, 96, 48]


def dcg(relevances):
    total = 0.0

    for rank, rel in enumerate(relevances, start=1):
        total += (2 ** rel - 1) / np.log2(rank + 1)

    return total


def ndcg(relevances):
    if not relevances:
        return 0.0

    actual = dcg(relevances)
    ideal = dcg(sorted(relevances, reverse=True))

    if ideal == 0:
        return 0.0

    return actual / ideal


def average_precision(binary_rels):
    hits = 0
    total = 0.0

    for rank, rel in enumerate(binary_rels, start=1):
        if rel:
            hits += 1
            total += hits / rank

    if hits == 0:
        return 0.0

    return total / hits


def evaluate_query(ranking, annotation):

    candidate_ids = [
        str(x)
        for x in annotation["cands"]
    ]

    relevance_values = annotation["relevance_max"]

    relevance = {
        str(pid): int(rel)
        for pid, rel in zip(
            candidate_ids,
            relevance_values
        )
    }

    rels = [
        relevance.get(str(pid), 0)
        for pid in ranking
    ]

    binary = [
        1 if x >= 2 else 0
        for x in rels
    ]

    top20 = rels[:20]
    binary20 = binary[:20]

    # NDCG over the returned ranking
    n = ndcg(rels)

    # NDCG@20
    n20 = ndcg(top20)

    # P@20
    p20 = sum(binary20) / 20.0

    # Recall@20
    total_relevant = sum(binary)

    if total_relevant > 0:
        recall20 = sum(binary20) / total_relevant
    else:
        recall20 = 0.0

    # R-Precision
    r = total_relevant

    if r > 0:
        r_binary = binary[:r]
        rprec = sum(r_binary) / r
    else:
        rprec = 0.0

    return {
        "ndcg": n,
        "ndcg20": n20,
        "p20": p20,
        "recall20": recall20,
        "rprec": rprec,
    }


def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("CONTRIEVER + PCA CSFCUBE EVALUATION")
    print("=" * 70)

    all_results = []

    for facet, filename in FACETS.items():

        annotation_path = os.path.join(
            ROOT,
            filename
        )

        annotations = load_annotations(
            annotation_path
        )

        print(
            f"\nFacet: {facet}"
        )

        print(
            "Queries:",
            len(annotations)
        )

        for dim in DIMS:

            ranking_path = os.path.join(
                RANKING_DIR,
                f"contriever-{dim}.json"
            )

            with open(
                ranking_path,
                "r",
                encoding="utf-8"
            ) as f:
                rankings = json.load(f)

            scores = []

            for query_id, annotation in annotations.items():

                query_id = str(query_id)

                if query_id not in rankings:
                    print(
                        "WARNING: missing query",
                        query_id
                    )
                    continue

                result = evaluate_query(
                    rankings[query_id],
                    annotation
                )

                scores.append(result)

            avg = {
                key: np.mean([
                    x[key]
                    for x in scores
                ])
                for key in scores[0]
            }

            print(
                f"{dim:>4} | "
                f"NDCG {avg['ndcg']:.4f} | "
                f"NDCG@20 {avg['ndcg20']:.4f} | "
                f"P@20 {avg['p20']:.4f} | "
                f"Recall@20 {avg['recall20']:.4f} | "
                f"R-Prec {avg['rprec']:.4f}"
            )

            all_results.append({
                "facet": facet,
                "dimension": dim,
                **avg,
            })

    output = os.path.join(
        ROOT,
        "contriever_results.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()