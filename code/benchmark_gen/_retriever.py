"""ColPali retriever for the 5-DOT v3 index (1898 pages)."""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import os
import numpy as np
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

COLPALI_MODEL_ID = "vidore/colpali-v1.2"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"


class ColPaliRetriever:
    def __init__(self, index_path: str = V3_INDEX, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[retriever] loading ColPali ({COLPALI_MODEL_ID}) on {self.device}")
        self.model = ColPali.from_pretrained(
            COLPALI_MODEL_ID, torch_dtype=torch.bfloat16, device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_ID)

        print(f"[retriever] loading v3 index from {index_path}")
        records = torch.load(index_path, weights_only=False, map_location="cpu")
        self.db = []
        for rec in records:
            emb = rec["embedding"].to(self.device).to(torch.bfloat16)
            self.db.append({"embedding": emb, "metadata": rec["metadata"]})
        print(f"[retriever] index size: {len(self.db)} pages")

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        inputs = self.processor.process_queries([query]).to(self.device)
        q_emb = self.model(**inputs)
        scores = []
        for doc in self.db:
            interaction = torch.matmul(q_emb, doc["embedding"].T)
            score = interaction.max(dim=-1).values.sum(dim=-1).item()
            scores.append(score)
        top = np.argsort(scores)[::-1][:k]
        return [self.db[i]["metadata"] for i in top]

    def rank_of(self, query: str, target_image_path: str, k_check: int = 50) -> int | None:
        top = self.retrieve(query, k=k_check)
        for i, m in enumerate(top):
            if m["image_path"] == target_image_path:
                return i + 1
        return None
