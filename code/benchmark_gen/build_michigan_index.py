"""Embed Michigan DOT pages with ColPali + Qwen-VL metadata.
Output: michigan_index.pt (298 pages, kept SEPARATE from v3 for zero-shot eval).
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import glob, json, os, re, time
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMG_DIR = f"{PSR_ROOT}/data/michigan_2025/images"
OUT = f"{PSR_ROOT}/qna_expansion/michigan_index.pt"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

META_PROMPT = """Analyze this engineering Standard Plan drawing.

Return STRICTLY valid JSON:
{
  "plan_id": "the standard plan number visible on the sheet (e.g. 'A40A', 'III-7B', '799-1'); 'Unknown' if not visible",
  "sheet_title": "the descriptive title of the sheet; 'Unknown' if not visible",
  "category": "engineering discipline: Bridge, Roadway, Traffic, Drainage, Erosion, Pavement, Signing, Geotechnical, General",
  "keywords": ["3-8 short visual keywords"]
}

Agency = MDOT, year = 2025. Return ONLY the JSON."""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def main():
    print("[init] loading Qwen-VL-7B")
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    qp = AutoProcessor.from_pretrained(QWEN_ID, trust_remote_code=True)

    print("[init] loading ColPali")
    cp = ColPali.from_pretrained(
        "vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    pngs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    print(f"[main] {len(pngs)} pages to embed")
    recs = []
    t0 = time.time()
    for i, p in enumerate(pngs):
        img = Image.open(p).convert("RGB")
        # ColPali embed
        with torch.no_grad():
            batch = cpp.process_images([img]).to(cp.device)
            emb = cp(**batch)[0].to(torch.bfloat16).cpu()
        # Qwen metadata
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": META_PROMPT}]}]
        text = qp.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = qp(text=[text], images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(qm.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = qm.generate(**inputs, max_new_tokens=200, do_sample=False)
        raw = qp.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        meta = parse_json(raw) or {}
        recs.append({"embedding": emb, "metadata": {
            "image_path": p, "filename": os.path.basename(p),
            "agency": "MDOT", "publication_year": "2025",
            "plan_id":     str(meta.get("plan_id", "Unknown"))[:80],
            "sheet_title": str(meta.get("sheet_title", "Unknown"))[:200],
            "category":    str(meta.get("category", "Unknown"))[:60],
            "keywords":    meta.get("keywords", []) if isinstance(meta.get("keywords"), list) else [],
            "unique_id": f"MDOT_{meta.get('plan_id','unk')}_{os.path.splitext(os.path.basename(p))[0]}",
        }})
        if (i + 1) % 25 == 0:
            print(f"[progress] {i+1}/{len(pngs)}  elapsed={(time.time()-t0)/60:.1f} min")

    torch.save(recs, OUT)
    print(f"[done] wrote {OUT} ({len(recs)} pages)")


if __name__ == "__main__":
    main()
