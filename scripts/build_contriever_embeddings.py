import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


ROOT = "CSFCube"
MODEL_NAME = "facebook/contriever"
OUTPUT = os.path.join(ROOT, "contriever_embeddings.npz")

BATCH_SIZE = 8
MAX_LENGTH = 512


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_documents():
    documents = {}

    path = os.path.join(ROOT, "abstracts-csfcube-preds.jsonl")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            paper_id = str(row["paper_id"])
            title = row.get("title", "")

            abstract = row.get("abstract", [])
            if isinstance(abstract, list):
                abstract = " ".join(abstract)

            text = title + "\n" + abstract

            documents[paper_id] = text

    return documents


def main():
    print("Loading CSFCube documents...")

    documents = load_documents()

    print("Documents:", len(documents))

    print("\nLoading Contriever...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Device:", device)

    paper_ids = list(documents.keys())
    embeddings = []

    print("\nGenerating embeddings...")

    with torch.no_grad():

        for start in range(0, len(paper_ids), BATCH_SIZE):

            batch_ids = paper_ids[start:start + BATCH_SIZE]
            batch_texts = [documents[x] for x in batch_ids]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            outputs = model(**encoded)

            pooled = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            # Normalize because retrieval will use cosine similarity.
            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

            embeddings.append(
                pooled.cpu().numpy().astype(np.float32)
            )

            done = min(start + BATCH_SIZE, len(paper_ids))

            if done % 100 == 0 or done == len(paper_ids):
                print(f"  {done}/{len(paper_ids)}")

    embeddings = np.concatenate(embeddings, axis=0)

    print("\nEmbedding shape:", embeddings.shape)

    np.savez_compressed(
        OUTPUT,
        paper_ids=np.array(paper_ids),
        embeddings=embeddings,
    )

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()