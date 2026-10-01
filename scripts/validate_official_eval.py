import json
import os
import sys

ROOT = "CSFCube"
RANKING_DIR = os.path.join(ROOT, "contriever_rankings")

sys.path.insert(
    0,
    os.path.join(ROOT, "eval_scripts")
)

from ranking_eval import graded_eval_pool_rerank


FACETS = {
    "background": "background",
    "method": "method",
    "result": "result",
}

DIMS = [768, 384, 192, 96, 48]


def main():

    print("=" * 70)
    print("VALIDATING AGAINST OFFICIAL CSFCUBE EVALUATOR")
    print("=" * 70)

    for facet_name, facet_arg in FACETS.items():

        print("\n" + "=" * 70)
        print("FACET:", facet_name)
        print("=" * 70)

        for dim in DIMS:

            ranking_file = os.path.join(
                RANKING_DIR,
                f"contriever-{dim}.json"
            )

            with open(
                ranking_file,
                "r",
                encoding="utf-8"
            ) as f:
                rankings = json.load(f)

            print(
                f"\nDimension: {dim}"
            )

            print(
                "Ranking file:",
                ranking_file
            )

            # Print ranking structure for inspection.
            first_query = next(iter(rankings))

            print(
                "Queries:",
                len(rankings)
            )

            print(
                "First query:",
                first_query
            )

            print(
                "Candidates:",
                len(rankings[first_query])
            )


if __name__ == "__main__":
    main()