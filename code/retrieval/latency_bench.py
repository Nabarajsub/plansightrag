"""Consolidated deployment-latency benchmark on a single GPU.

Measures, one query at a time (batch 1) on the stated hardware:
  - indexing throughput  : ColNomic encode of the 1,898-page index (one-time, offline)
  - retrieval latency     : query encode + MaxSim over the 1,898-page index (p50/p99)
  - re-ranking overhead   : per-candidate MonoQwen2-VL pass (Top-K -> +K passes)
Agentic per-audit latency is reported separately from the compliance runs.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, time, statistics, sys
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

ROOT = f"{PSR_ROOT}"
PAGES = f"{ROOT}/baselines_v2/pages.json"
TEST = f"{ROOT}/lora_finetune/split_test.jsonl"
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
MONO = "lightonai/MonoQwen2-VL-v0.1"
OUT = f"{ROOT}/baselines_v2/reports/latency_bench.json"
DEV = "cuda"


def main():
    pages = json.load(open(PAGES))
    test = [json.loads(l) for l in open(TEST)]
    ppaths = [p["image_path"] for p in pages]
    gpu = torch.cuda.get_device_name(0)
    print(f"[lat] GPU={gpu} | {len(ppaths)} index pages | {len(test)} queries")

    col = ColQwen2_5.from_pretrained(COLNOMIC, torch_dtype=torch.bfloat16, device_map=DEV).eval()
    proc = ColQwen2_5_Processor.from_pretrained(COLNOMIC)

    # ---- indexing throughput (encode all 1,898 pages, batch 4) ----
    pembs = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(0, len(ppaths), 4):
        imgs = [Image.open(p).convert("RGB") for p in ppaths[s:s + 4]]
        with torch.no_grad():
            o = col(**proc.process_images(imgs).to(DEV))
        pembs += [o[j].to(torch.float16).cpu() for j in range(o.shape[0])]
    torch.cuda.synchronize(); idx_t = time.perf_counter() - t0
    pages_per_s = len(ppaths) / idx_t

    # ---- retrieval latency (query encode + MaxSim over 1,898), batch 1 ----
    pembs_gpu = [pe.to(DEV).float() for pe in pembs]
    lat = []
    for r in test[:120]:
        torch.cuda.synchronize(); q0 = time.perf_counter()
        with torch.no_grad():
            qe = col(**proc.process_queries([r["query"]]).to(DEV))[0].float()
        _ = np.array([torch.matmul(qe, pe.T).max(-1).values.sum().item() for pe in pembs_gpu])
        torch.cuda.synchronize(); lat.append((time.perf_counter() - q0) * 1000.0)
    lat.sort()
    p = lambda q: round(lat[int(q * (len(lat) - 1))], 1)
    del pembs_gpu; torch.cuda.empty_cache()

    # ---- re-ranking per-pass overhead (MonoQwen2-VL pointwise) ----
    rerank = {}
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        mp = AutoProcessor.from_pretrained(MONO, trust_remote_code=True)
        mm = Qwen2VLForConditionalGeneration.from_pretrained(MONO, quantization_config=bnb,
                                                             device_map=DEV, torch_dtype=torch.bfloat16).eval()
        q = test[0]["query"]; cand = ppaths[:20]
        passes = []
        for cp in cand:
            img = Image.open(cp).convert("RGB")
            msg = [{"role": "user", "content": [{"type": "image", "image": img},
                    {"type": "text", "text": f"Does this engineering plan answer: {q}? Answer True or False."}]}]
            t = mp.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = mp(text=[t], images=[img], return_tensors="pt").to(DEV)
            torch.cuda.synchronize(); r0 = time.perf_counter()
            with torch.no_grad():
                mm.generate(**inp, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize(); passes.append((time.perf_counter() - r0) * 1000.0)
        rerank = {"per_pass_ms_p50": round(statistics.median(passes), 1),
                  "per_pass_ms_mean": round(statistics.mean(passes), 1),
                  "topK": 20,
                  "overhead_s_K20": round(20 * statistics.median(passes) / 1000.0, 2),
                  "overhead_s_earlystop2": round(2 * statistics.median(passes) / 1000.0, 2)}
    except Exception as e:
        rerank = {"error": str(e)[:200]}

    rep = {
        "gpu": gpu,
        "indexing": {"n_pages": len(ppaths), "total_s": round(idx_t, 1),
                     "pages_per_s": round(pages_per_s, 2), "min_for_1898": round(idx_t / 60.0, 1)},
        "retrieval_latency_ms": {"p50": p(0.50), "p90": p(0.90), "p99": p(0.99),
                                 "mean": round(statistics.mean(lat), 1), "n": len(lat)},
        "reranking": rerank,
        "agentic_audit_s_per_query_ref": 60.9,
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
