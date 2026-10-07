"""
Build Contriever embeddings for CSFCube.

This version:
- Loads CSFCube from E:\\ir_project\\data\\CSFCube
- Uses the locally cached Contriever model
- Uses the slow BERT tokenizer
- Loads model.safetensors directly
- Does not use pytorch_model.bin
- Uses CUDA when available
- Produces normalized 768-dimensional embeddings
"""

import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, BertTokenizer


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = r"E:\ir_project\data\CSFCube"

MODEL_DIR = (
    r"E:\ir_project\hf_cache\hub"
    r"\models--facebook--contriever"
    r"\snapshots"
    r"\4bce1e1123ec62a1cbc1f9d27dc9c8e6e1e51ce0"
)

INPUT_FILE = os.path.join(
    DATA_ROOT,
    "abstracts-csfcube-preds.jsonl"
)

OUTPUT_FILE = os.path.join(
    DATA_ROOT,
    "contriever_embeddings.npz"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
MAX_LENGTH = 512


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    last_hidden_state,
    attention_mask
):
    mask = attention_mask.unsqueeze(
        -1
    ).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ============================================================
# LOAD CSFCUBE DOCUMENTS
# ============================================================

def load_documents():

    documents = {}

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"CSFCube input file was not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            paper_id = str(
                row["paper_id"]
            )

            title = row.get(
                "title",
                ""
            )

            abstract = row.get(
                "abstract",
                []
            )

            if isinstance(
                abstract,
                list
            ):
                abstract = " ".join(
                    str(x)
                    for x in abstract
                )

            elif abstract is None:
                abstract = ""

            else:
                abstract = str(
                    abstract
                )

            text = (
                str(title).strip()
                + "\n"
                + abstract.strip()
            ).strip()

            documents[paper_id] = text

    return documents


# ============================================================
# LOAD CONTRIEVER
# ============================================================

def load_contriever():

    safetensors_file = os.path.join(
        MODEL_DIR,
        "model.safetensors"
    )

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory was not found:\n{MODEL_DIR}"
        )

    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(
            f"model.safetensors was not found:\n"
            f"{safetensors_file}"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading Contriever tokenizer..."
    )

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False
    )

    print(
        "Tokenizer loaded successfully."
    )

    # --------------------------------------------------------
    # Contriever / BERT configuration
    # --------------------------------------------------------

    print(
        "\nBuilding Contriever configuration..."
    )

    # IMPORTANT:
    # "bert" is already supplied as the first argument.
    # Do NOT pass model_type="bert" again.

    config = AutoConfig.for_model(
        "bert",
        architectures=["BertModel"],
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        type_vocab_size=2,
        vocab_size=30522,
    )

    print(
        "Hidden size:",
        config.hidden_size
    )

    print(
        "Hidden layers:",
        config.num_hidden_layers
    )

    print(
        "Attention heads:",
        config.num_attention_heads
    )

    # --------------------------------------------------------
    # Create model architecture
    # --------------------------------------------------------

    print(
        "\nCreating Contriever model..."
    )

    model = AutoModel.from_config(
        config
    )

    print(
        "Model architecture created."
    )

    # --------------------------------------------------------
    # Load safetensors
    # --------------------------------------------------------

    print(
        "\nLoading model.safetensors..."
    )

    state_dict = load_file(
        safetensors_file
    )

    print(
        "Weight tensors loaded:",
        len(state_dict)
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    del state_dict

    if missing_keys:

        print(
            "\nWARNING: Missing keys:"
        )

        for key in missing_keys[:20]:
            print(
                " ",
                key
            )

    if unexpected_keys:

        print(
            "\nWARNING: Unexpected keys:"
        )

        for key in unexpected_keys[:20]:
            print(
                " ",
                key
            )

    if (
        not missing_keys
        and not unexpected_keys
    ):
        print(
            "All model weights loaded successfully."
        )

    print(
        "\nContriever model loaded successfully."
    )

    return tokenizer, model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CSFCube Contriever Embedding Builder"
    )
    print("=" * 70)

    print(
        "\nData root:"
    )
    print(DATA_ROOT)

    print(
        "\nInput:"
    )
    print(INPUT_FILE)

    print(
        "\nModel:"
    )
    print(MODEL_DIR)

    print(
        "\nOutput:"
    )
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    print(
        "\nLoading CSFCube documents..."
    )

    documents = load_documents()

    print(
        "Documents:",
        len(documents)
    )

    if len(documents) == 0:
        raise RuntimeError(
            "No CSFCube documents were loaded."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_contriever()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA memory:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024 ** 3),
                2
            ),
            "GB"
        )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Document IDs
    # --------------------------------------------------------

    paper_ids = list(
        documents.keys()
    )

    total = len(
        paper_ids
    )

    embeddings = []

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings..."
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Maximum sequence length:",
        MAX_LENGTH
    )

    with torch.no_grad():

        for start in range(
            0,
            total,
            BATCH_SIZE
        ):

            batch_ids = paper_ids[
                start:start + BATCH_SIZE
            ]

            batch_texts = [
                documents[paper_id]
                for paper_id in batch_ids
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            pooled = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            # L2 normalize Contriever embeddings.
            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1
            )

            embeddings.append(
                pooled.cpu().numpy().astype(
                    np.float32
                )
            )

            done = min(
                start + BATCH_SIZE,
                total
            )

            if (
                done % 100 == 0
                or done == total
            ):
                print(
                    f"  {done}/{total}"
                )

    # --------------------------------------------------------
    # Combine embeddings
    # --------------------------------------------------------

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    print(
        "\nEmbedding shape:",
        embeddings.shape
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if embeddings.shape[0] != total:

        raise RuntimeError(
            "Embedding count does not match "
            "document count."
        )

    if embeddings.shape[1] != 768:

        raise RuntimeError(
            "Expected 768-dimensional Contriever "
            f"embeddings, got {embeddings.shape[1]}."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "Embeddings contain NaN or infinite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        "Mean embedding norm:",
        float(norms.mean())
    )

    print(
        "Minimum embedding norm:",
        float(norms.min())
    )

    print(
        "Maximum embedding norm:",
        float(norms.max())
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving embeddings..."
    )

    np.savez_compressed(
        OUTPUT_FILE,
        paper_ids=np.array(
            paper_ids
        ),
        embeddings=embeddings
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "File size:",
        round(
            os.path.getsize(
                OUTPUT_FILE
            ) / (1024 ** 2),
            2
        ),
        "MB"
    )

    print(
        "\nDone."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()