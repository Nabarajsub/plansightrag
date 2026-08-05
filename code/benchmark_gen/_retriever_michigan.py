# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import os, numpy as np, torch
from colpali_engine.models import ColPali, ColPaliProcessor

MICHIGAN_INDEX = f"{PSR_ROOT}/qna_expansion/michigan_index.pt"
COLPALI = "vidore/colpali-v1.2"

class ColPaliRetriever:
    def __init__(self, index_path=MICHIGAN_INDEX, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[retriever] loading ColPali on {self.device}")
        self.model = ColPali.from_pretrained(COLPALI, torch_dtype=torch.bfloat16, device_map=self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(COLPALI)
        print(f"[retriever] loading Michigan index from {index_path}")
        records = torch.load(index_path, weights_only=False, map_location="cpu")
        self.db = [{"embedding": r["embedding"].to(self.device).to(torch.bfloat16), "metadata": r["metadata"]} for r in records]
        print(f"[retriever] index size: {len(self.db)} pages")

    @torch.no_grad()
    def retrieve(self, query, k=5):
        inputs = self.processor.process_queries([query]).to(self.device)
        q_emb = self.model(**inputs)
        scores = [torch.matmul(q_emb, d["embedding"].T).max(dim=-1).values.sum(dim=-1).item() for d in self.db]
        top = np.argsort(scores)[::-1][:k]
        return [self.db[i]["metadata"] for i in top]

    def rank_of(self, query, target_image_path, k_check=50):
        top = self.retrieve(query, k=k_check)
        for i, m in enumerate(top):
            if m["image_path"] == target_image_path: return i + 1
        return None
