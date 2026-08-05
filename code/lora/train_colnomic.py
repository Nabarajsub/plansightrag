"""LoRA / head fine-tuning of ColNomic-3B (the adopted retriever) on the 5-DOT
engineering-plan train split -- the H3 ablation for manuscript_swap.

Mirrors the ColPali ablation (train_head.py / train_lora.py) but on ColQwen2_5 /
nomic-ai/colnomic-embed-multimodal-3b. Three configs (selected by flags):
  head-gentle   : --mode head --lr 5e-5   (train only custom_text_proj, gentle)
  head-standard : --mode head --lr 1e-4   (train only custom_text_proj)
  full-LM       : --mode full --lr 5e-5   (LoRA on Qwen2.5-VL attn+MLP + head)

Trains with ColbertPairwiseCELoss + UniqueImageBatchSampler (distinct page per
batch slot). Saves the adapter/head for eval_colnomic.py.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, json, os, random, time
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import get_linear_schedule_with_warmup
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
from peft import LoraConfig, get_peft_model

MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"
OUT_DIR = f"{PSR_ROOT}/lora_finetune"
# Qwen2.5-VL language-model attention + MLP projections (LoRA targets for full-LM)
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "custom_text_proj"]


class QnADataset(Dataset):
    def __init__(self, p):
        self.rows = [r for r in (json.loads(l) for l in open(p)) if os.path.exists(r["image_path"])]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return {"query": self.rows[i]["query"], "image_path": self.rows[i]["image_path"]}


class UniqueImageBatchSampler(Sampler):
    def __init__(self, ds, bs, seed=0):
        self.bs = bs; self.seed = seed; self.epoch = 0
        g = defaultdict(list)
        for i, r in enumerate(ds.rows): g[r["image_path"]].append(i)
        self.groups = list(g.values())
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1
        picked = [rng.choice(g) for g in self.groups]; rng.shuffle(picked)
        for i in range(0, len(picked) - self.bs + 1, self.bs):
            yield picked[i:i + self.bs]
    def __len__(self): return len(self.groups) // self.bs


def make_collate(proc):
    def collate(batch):
        q = proc.process_queries([b["query"] for b in batch])
        d = proc.process_images([Image.open(b["image_path"]).convert("RGB") for b in batch])
        return q, d
    return collate


@torch.no_grad()
def eval_dev(model, loader, device):
    model.eval(); c = t = 0
    for q_in, d_in in loader:
        q_in = {k: v.to(device) for k, v in q_in.items()}; d_in = {k: v.to(device) for k, v in d_in.items()}
        qe = model(**q_in); de = model(**d_in)
        raw = torch.einsum("bnd,csd->bcns", qe.float(), de.float())
        scores = raw.max(dim=3).values.sum(dim=2)
        c += (scores.argmax(dim=1) == torch.arange(scores.size(0), device=device)).sum().item(); t += scores.size(0)
    model.train(); return c / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "full"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lora_rank", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    out = f"{OUT_DIR}/colnomic_{args.tag}"
    os.makedirs(out, exist_ok=True)
    print(f"[init] {args.tag}: mode={args.mode} lr={args.lr} bs={args.batch_size}")

    model = ColQwen2_5.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev)
    proc = ColQwen2_5_Processor.from_pretrained(MODEL_ID)

    if args.mode == "head":
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.custom_text_proj.parameters(): p.requires_grad_(True)
    else:
        for p in model.parameters(): p.requires_grad_(False)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
            target_modules=QWEN_TARGETS, bias="none"))
        model.gradient_checkpointing_enable()  # full-LM is memory-heavy
        model.enable_input_require_grads()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable {n_tr:,}/{n_tot:,} ({100*n_tr/n_tot:.4f}%)")

    tr = QnADataset(f"{OUT_DIR}/split_train.jsonl"); dv = QnADataset(f"{OUT_DIR}/split_dev.jsonl")
    collate = make_collate(proc)
    trs = UniqueImageBatchSampler(tr, args.batch_size); dvs = UniqueImageBatchSampler(dv, args.batch_size)
    tl = DataLoader(tr, batch_sampler=trs, collate_fn=collate, num_workers=4)
    dl = DataLoader(dv, batch_sampler=dvs, collate_fn=collate, num_workers=2)
    print(f"[data] train={len(tr)} ({len(trs)} batches) dev={len(dv)}")

    loss_fn = ColbertPairwiseCELoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = len(tl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(steps * args.warmup_ratio), steps)

    base = eval_dev(model, dl, dev); print(f"[baseline] dev in-batch acc = {base:.4f}")
    best = base; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        run = 0.0
        for s, (q_in, d_in) in enumerate(tl):
            q_in = {k: v.to(dev) for k, v in q_in.items()}; d_in = {k: v.to(dev) for k, v in d_in.items()}
            loss = loss_fn(model(**q_in), model(**d_in))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step(); run += loss.item()
            if (s + 1) % 25 == 0:
                print(f"[ep{ep}] {s+1}/{len(tl)} loss={run/(s+1):.4f} lr={sch.get_last_lr()[0]:.2e} {(time.time()-t0)/60:.1f}m")
        acc = eval_dev(model, dl, dev)
        print(f"[ep{ep}] avg_loss={run/len(tl):.4f} dev_acc={acc:.4f} (base {base:.4f})")
        if acc > best:
            best = acc
            if args.mode == "head":
                torch.save(model.custom_text_proj.state_dict(), f"{out}/custom_text_proj.pt")
            else:
                model.save_pretrained(out)
            print(f"[ep{ep}] new best -> saved to {out}")
    print(f"[done] {args.tag}: best dev {best:.4f} (base {base:.4f}) {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
