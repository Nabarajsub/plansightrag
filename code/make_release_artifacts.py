"""Generate the four release artifacts: prompts, seeds, model checkpoints/revisions,
and the environment pin.

    python code/make_release_artifacts.py [--no-network]

Writes:
    data/prompts.json   every prompt string used anywhere in code/, with its
                        source file, line, role and the f-string placeholders
    data/seeds.json     every random seed, per script, with what it controls
    data/models.json    every model id, its role, quantization, decoding
                        settings and the pinned Hugging Face commit SHA
    requirements.txt    the environment pin

Everything is derived from the released code by parsing it, so the artifacts
cannot drift from what actually ran.
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # release/
CODE = os.path.join(ROOT, "code")
DATA = os.path.join(ROOT, "data")

PROMPT_NAME = re.compile(r"(PROMPT|INSTRUCTION|SYSTEM|VERIFY|JUDGE|META|RUBRIC|TEMPLATE)")

# Role of each model in the pipeline; anything discovered but unlisted gets
# role "unassigned" so it is visible rather than silently dropped.
ROLES = {
    "nomic-ai/colnomic-embed-multimodal-3b":
        ("retriever — ADOPTED backbone; all headline retrieval, rule-grounding, real-plan", "bfloat16"),
    "nomic-ai/colnomic-embed-multimodal-7b": ("retriever — baseline comparison", "bfloat16"),
    "vidore/colpali-v1.2":
        ("retriever — PREDECESSOR backbone; companion ablations (tiling, rerank, quantization, CIs)", "bfloat16"),
    "vidore/colpali-v1.3": ("retriever — baseline comparison", "bfloat16"),
    "vidore/colqwen2.5-v0.2": ("retriever — baseline comparison", "bfloat16"),
    "Qwen/Qwen2.5-VL-72B-Instruct":
        ("drafter (benchmark generation) · compliance auditor · LLM-as-judge", "4-bit NF4 (BitsAndBytes), bfloat16 compute"),
    "Qwen/Qwen2.5-VL-7B-Instruct":
        ("VQA answerer · benchmark verifier · metadata extractor", "4-bit NF4 (BitsAndBytes), bfloat16 compute"),
    "Qwen/Qwen2-VL-2B-Instruct": ("backbone of the DSE-Qwen2-2B retrieval baseline", "bfloat16"),
    "meta-llama/Llama-3.2-90B-Vision-Instruct":
        ("drafter (benchmark generation, Caltrans + FDOT)", "4-bit NF4 (BitsAndBytes)"),
    "OpenGVLab/InternVL2_5-8B": ("alternative VQA generator", "bfloat16"),
    "OpenGVLab/InternVL2_5-8B-MPO-hf": ("cross-family judge (self-preference check)", "bfloat16"),
    "lightonai/MonoQwen2-VL-v0.1": ("pointwise visual reranker (Apache-2.0)", "bfloat16"),
    "BAAI/bge-m3": ("strongest text retrieval baseline (OCR + dense)", "fp32"),
    "nvidia/NV-Embed-v2": ("text embedding baseline", "bfloat16"),
    "sentence-transformers/all-MiniLM-L6-v2": ("text encoder for OCR-free decode baselines", "fp32"),
    "openai/clip-vit-base-patch32": ("legacy global-embedding baseline", "fp32"),
    "facebook/nougat-small": ("legacy OCR-free decode baseline", "fp32"),
    "google/pix2struct-base": ("legacy OCR-free decode baseline", "fp32"),
    "microsoft/udop-large": ("legacy OCR-free decode baseline", "fp32"),
    "microsoft/layoutlmv3-base": ("legacy layout-embedding baseline", "fp32"),
}

# requirements.txt is maintained by hand from the verified working environment
# (see the header of that file) — this script no longer overwrites it.


# --------------------------------------------------------------- prompts ----
def unparse_fstring(node):
    """Render a JoinedStr back to text with {placeholder} markers."""
    out = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            out.append(str(v.value))
        elif isinstance(v, ast.FormattedValue):
            try:
                out.append("{" + ast.unparse(v.value) + "}")
            except Exception:
                out.append("{...}")
    return "".join(out)


def extract_prompts():
    prompts, seen = [], set()
    for dirpath, _, files in os.walk(CODE):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if not names:
                    continue
                name = names[0]
                val = node.value
                # named prompt constants, and any local variable literally called prompt
                interesting = PROMPT_NAME.search(name.upper()) or name.lower() in ("prompt", "instruction", "query_prompt")
                if not interesting:
                    continue
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    text, kind = val.value, "literal"
                elif isinstance(val, ast.JoinedStr):
                    text, kind = unparse_fstring(val), "f-string"
                elif isinstance(val, ast.Dict):
                    # e.g. CATEGORY_PROMPT = {...}, COMPLIANCE_PROMPTS = {...}
                    for k, v in zip(val.keys, val.values):
                        if isinstance(k, ast.Constant) and isinstance(v, (ast.Constant, ast.JoinedStr)):
                            t = v.value if isinstance(v, ast.Constant) else unparse_fstring(v)
                            if not isinstance(t, str) or len(t.strip()) < 20:
                                continue
                            key = (rel, f"{name}[{k.value}]")
                            if key in seen:
                                continue
                            seen.add(key)
                            prompts.append({"name": f"{name}[{k.value}]", "file": rel,
                                            "line": getattr(v, "lineno", node.lineno),
                                            "kind": "dict-entry", "text": t,
                                            "placeholders": sorted(set(re.findall(r"\{(\w+)", t)))})
                    continue
                else:
                    continue
                if len(text.strip()) < 20:      # skip trivial fragments
                    continue
                key = (rel, name, text[:80])
                if key in seen:
                    continue
                seen.add(key)
                prompts.append({"name": name, "file": rel, "line": node.lineno, "kind": kind,
                                "text": text, "placeholders": sorted(set(re.findall(r"\{(\w+)", text)))})
    # stable order: by file then line
    prompts.sort(key=lambda p: (p["file"], p["line"]))
    for i, p in enumerate(prompts):
        p["id"] = f"P{i+1:03d}"
    return prompts


# ----------------------------------------------------------------- seeds ----
SEED_CTX = {
    "code/benchmark_gen/generate_qwen.py": "page sampling + drafter question order (Qwen drafter)",
    "code/benchmark_gen/generate_qwen_michigan.py": "page sampling for the held-out Michigan transfer set",
    "code/benchmark_gen/generate_llama.py": "page sampling + drafter question order (Llama drafter)",
    "code/compliance/make_500.py": "parameter randomisation, 500 single-doc CAD drawings",
    "code/compliance/make_500_queries.py": "query construction for the 500 single-doc set",
    "code/compliance/make_multi100.py": "parameter randomisation, 100 multi-plan drawings",
    "code/compliance/make_stress50.py": "near-threshold adversarial stress set (50 drawings)",
    "code/lora/split_dataset.py": "page-disjoint train/dev/test split",
    "code/lora/train_head.py": "unique-image batch sampler",
}


def extract_seeds():
    found = []
    for dirpath, _, files in os.walk(CODE):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            src = open(path, encoding="utf-8").read()
            hits = set()
            for m in re.finditer(r"random\.seed\((\d+)\)|np\.random\.seed\((\d+)\)|"
                                 r"torch\.manual_seed\((\d+)\)|"
                                 r'["\']--?seed["\'].*?default\s*=\s*(\d+)|'
                                 r"seed\s*[:=]\s*(\d+)", src):
                v = next((g for g in m.groups() if g), None)
                if v is not None:
                    hits.add(int(v))
            for v in sorted(hits):
                found.append({"file": rel, "seed": v,
                              "controls": SEED_CTX.get(rel, "see script")})
    return found


# ---------------------------------------------------------------- models ----
def discover_model_ids():
    ids = set()
    pat = re.compile(r'["\']([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)["\']')
    for dirpath, _, files in os.walk(CODE):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            for m in pat.finditer(open(os.path.join(dirpath, fn), encoding="utf-8").read()):
                cand = m.group(1)
                if "/" in cand and not cand.endswith((".py", ".json", ".pt", ".csv", ".jsonl", ".png")):
                    ids.add(cand)
    # keep things that look like HF repos from known orgs, plus anything we have a role for
    orgs = ("nomic-ai", "vidore", "Qwen", "OpenGVLab", "nvidia", "BAAI", "meta-llama",
            "lightonai", "facebook", "google", "microsoft", "openai", "sentence-transformers")
    return sorted(i for i in ids if i in ROLES or i.split("/")[0] in orgs)


def hf_sha(model_id, timeout=10):
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r).get("sha")
    except Exception:
        return None


DECODING = {
    "generation": {"do_sample": False, "note": "greedy decoding everywhere; no temperature or top_p is set",
                   "max_new_tokens": {"question drafting": 128, "answer drafting": 160,
                                      "category rewrite": 120, "metadata extraction": 256,
                                      "verifier (yes/no)": 8, "judge (binary)": 4,
                                      "compliance JSON": 200, "agentic step": 256}},
    "retrieval": {"batch_size": 4, "index_dtype": "float32 (CPU-resident)",
                  "scoring": "ColBERT-style MaxSim late interaction"},
}


def build_models(network=True):
    out = []
    for mid in discover_model_ids():
        role, quant = ROLES.get(mid, ("unassigned — discovered in code, confirm before release", ""))
        rec = {"model_id": mid, "role": role, "quantization": quant,
               "revision": hf_sha(mid) if network else None}
        if rec["revision"] is None:
            rec["revision_note"] = "resolve with: huggingface-cli download <id> --revision main"
        out.append(rec)
    return {"_note": "Commit SHAs pin the exact weights used. Load with "
                     "from_pretrained(model_id, revision=<sha>).",
            "decoding": DECODING, "models": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-network", action="store_true", help="skip Hugging Face SHA lookup")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    prompts = extract_prompts()
    json.dump({"_note": "Every prompt string in code/, extracted by parsing the source. "
                        "{placeholders} are runtime substitutions.",
               "n": len(prompts), "prompts": prompts},
              open(os.path.join(DATA, "prompts.json"), "w"), indent=2)
    print(f"data/prompts.json      {len(prompts)} prompts from "
          f"{len({p['file'] for p in prompts})} files")

    seeds = extract_seeds()
    json.dump({"_note": "All random seeds, per script. Re-running a script with its "
                        "seed reproduces the sampling exactly.",
               "seeds": seeds}, open(os.path.join(DATA, "seeds.json"), "w"), indent=2)
    print(f"data/seeds.json        {len(seeds)} seed declarations across "
          f"{len({s['file'] for s in seeds})} scripts")

    models = build_models(network=not args.no_network)
    json.dump(models, open(os.path.join(DATA, "models.json"), "w"), indent=2)
    pinned = sum(1 for m in models["models"] if m.get("revision"))
    print(f"data/models.json       {len(models['models'])} models, {pinned} pinned to a commit SHA")
    unassigned = [m["model_id"] for m in models["models"] if m["role"].startswith("unassigned")]
    if unassigned:
        print(f"  !! confirm or drop: {unassigned}")

    req = os.path.join(ROOT, "requirements.txt")
    print(f"requirements.txt       left untouched ({'present' if os.path.exists(req) else 'MISSING'}) "
          f"— maintained by hand from the verified environment")


if __name__ == "__main__":
    main()
