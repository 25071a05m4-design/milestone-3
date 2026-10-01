import json
import os
import numpy as np

from truncate import PCAProjector


ROOT = "CSFCube"

EMBEDDINGS_FILE = os.path.join(
    ROOT,
    "contriever_embeddings.npz"
)

POOL_FILE = os.path.join(
    ROOT,
    "test-pid2pool-csfcube.json"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "contriever_rankings"
)

DIMENSIONS = [768, 384, 192, 96, 48]


def load_embeddings():
    data = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=True
    )

    paper_ids = [
        str(x)
        for x in data["paper_ids"]
    ]

    embeddings = data["embeddings"].astype(
        np.float32
    )

    mapping = {
        pid: embeddings[i]
        for i, pid in enumerate(paper_ids)
    }

    print("Embeddings:", embeddings.shape)

    return mapping


def load_pool():
    with open(
        POOL_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def l2_normalize(x):
    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / np.maximum(norms, 1e-12)


def main():

    print("Loading embeddings...")
    embeddings = load_embeddings()

    print("Loading candidate pools...")
    pools = load_pool()

    print("Queries:", len(pools))

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Collect the complete candidate corpus.
    # PCA is fitted on corpus documents only.
    # Query vectors are NOT used to fit PCA.
    # ---------------------------------------------------------

    candidate_ids = sorted({
        str(pid)
        for query_data in pools.values()
        for pid in query_data["cands"]
    })

    print(
        "Unique candidate documents:",
        len(candidate_ids)
    )

    corpus_matrix = np.vstack([
        embeddings[pid]
        for pid in candidate_ids
    ])

    print(
        "PCA corpus shape:",
        corpus_matrix.shape
    )

    # ---------------------------------------------------------
    # Fit PCA once at maximum truncated dimension.
    # Lower dimensions use the leading components.
    # ---------------------------------------------------------

    max_dim = max(
        d for d in DIMENSIONS
        if d < 768
    )

    print(
        "\nFitting PCA:",
        max_dim,
        "dimensions..."
    )

    projector = PCAProjector.fit(
        corpus_matrix,
        n_components=max_dim
    )

    print("PCA fitted.")

    # ---------------------------------------------------------
    # Run retrieval for every requested dimension.
    # ---------------------------------------------------------

    for dim in DIMENSIONS:

        print("\n" + "=" * 60)
        print("Dimension:", dim)
        print("=" * 60)

        rankings = {}

        if dim == 768:

            # Original Contriever embeddings.
            reduced_corpus = l2_normalize(
                corpus_matrix
            )

            corpus_lookup = {
                pid: reduced_corpus[i]
                for i, pid in enumerate(candidate_ids)
            }

        else:

            # Project corpus using frozen PCA.
            projected = projector.transform(
                corpus_matrix
            )

            projected = projected[:, :dim]

            projected = l2_normalize(
                projected
            )

            corpus_lookup = {
                pid: projected[i]
                for i, pid in enumerate(candidate_ids)
            }

        for count, (query_id, query_data) in enumerate(
            pools.items(),
            start=1
        ):

            query_id = str(query_id)

            candidate_ids_for_query = [
                str(x)
                for x in query_data["cands"]
            ]

            # -------------------------------------------------
            # Query representation
            # -------------------------------------------------

            query_vector = embeddings[query_id]

            if dim == 768:

                query_vector = query_vector.reshape(
                    1, -1
                )

            else:

                query_vector = projector.transform(
                    query_vector.reshape(1, -1)
                )

                query_vector = query_vector[:, :dim]

            query_vector = l2_normalize(
                query_vector
            )[0]

            # -------------------------------------------------
            # Candidate scoring
            # -------------------------------------------------

            candidate_matrix = np.vstack([
                corpus_lookup[pid]
                for pid in candidate_ids_for_query
            ])

            scores = candidate_matrix @ query_vector

            order = np.argsort(
                -scores,
                kind="stable"
            )

            ranked_ids = [
                candidate_ids_for_query[i]
                for i in order
            ]

            rankings[query_id] = ranked_ids

            if count % 10 == 0 or count == len(pools):
                print(
                    f"  Ranked {count}/{len(pools)} queries"
                )

        output_file = os.path.join(
            OUTPUT_DIR,
            f"contriever-{dim}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                rankings,
                f,
                indent=2
            )

        print(
            "Saved:",
            output_file
        )


if __name__ == "__main__":
    main()