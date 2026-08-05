"""Continue training ColPali's existing LoRA adapter on the 5-DOT data.

After diagnosing that ColPali ships with LIVE lora.Linear modules throughout
the Gemma language model (q/k/v/o + gate/up/down per layer + custom_text_proj),
the correct fine-tune is simply to unfreeze the existing lora_A / lora_B
parameters and continue training them with engineering-plan data. No PEFT
re-wrapping needed.

This is equivalent to "continue the official ColPali LoRA fine-tune on a new
domain". The base PaliGemma weights stay frozen; the (already-trained) LoRA
deltas get adapted.
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
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss

COLPALI_ID = "vidore/colpali-v1.2"
OUT_DIR = f"{PSR_ROOT}/lora_finetune"


class QnADataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        self.rows = [r for r in self.rows if os.path.exists(r["image_path"])]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return {"query": r["query"], "image_path": r["image_path"]}


class UniqueImageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0):
        self.batch_size = batch_size; self.seed = seed; self.epoch = 0
        groups = defaultdict(list)
        for i, r in enumerate(dataset.rows): groups[r["image_path"]].append(i)
        self.groups = list(groups.values())
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1
        picked = [rng.choice(g) for g in self.groups]; rng.shuffle(picked)
        for i in range(0, len(picked) - self.batch_size + 1, self.batch_size):
            yield picked[i : i + self.batch_size]
    def __len__(self): return len(self.groups) // self.batch_size


def make_collator(processor):
    def collate(batch):
        q = processor.process_queries([b["query"] for b in batch])
        d = processor.process_images([Image.open(b["image_path"]).convert("RGB") for b in batch])
        return q, d
    return collate


@torch.no_grad()
def evaluate_dev(model, loader, device):
    model.eval(); c = t = 0
    for q_in, d_in in loader:
        q_in = {k: v.to(device) for k, v in q_in.items()}
        d_in = {k: v.to(device) for k, v in d_in.items()}
        qe = model(**q_in); de = model(**d_in)
        raw = torch.einsum("bnd,csd->bcns", qe.float(), de.float())
        scores = raw.max(dim=3).values.sum(dim=2)
        preds = scores.argmax(dim=1)
        gold = torch.arange(scores.size(0), device=device)
        c += (preds == gold).sum().item(); t += scores.size(0)
    model.train()
    return c / max(t, 1)


def save_lora_deltas(model, path):
    """Save just the lora_A/lora_B weights -- they fully reconstruct the adapter."""
    sd = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
          if ("lora_A" in n or "lora_B" in n)}
    torch.save(sd, path)
    return len(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"{OUT_DIR}/split_train.jsonl")
    ap.add_argument("--dev", default=f"{OUT_DIR}/split_dev.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="LM-wide LoRA needs lower bs than head-only")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--out", default=f"{OUT_DIR}/full_lora_adapter/lora_deltas.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")

    print("[init] loading bare ColPali")
    model = ColPali.from_pretrained(COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device)
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)

    # Freeze everything, then unfreeze ONLY the existing lora_A / lora_B
    # parameters (and the projection head, which is itself a lora.Linear).
    for p in model.parameters(): p.requires_grad_(False)
    n_lora = 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True); n_lora += p.numel()
    print(f"[init] unfroze {n_lora:,} LoRA params "
          f"({100*n_lora/sum(p.numel() for p in model.parameters()):.3f}% of model)")

    train_ds = QnADataset(args.train); dev_ds = QnADataset(args.dev)
    collate = make_collator(processor)
    train_sampler = UniqueImageBatchSampler(train_ds, args.batch_size)
    dev_sampler = UniqueImageBatchSampler(dev_ds, args.batch_size)
    print(f"[data] train={len(train_ds)} pairs ({len(train_sampler.groups)} unique pages -> "
          f"{len(train_sampler)} batches/epoch)  dev={len(dev_ds)} pairs "
          f"({len(dev_sampler.groups)} unique pages -> {len(dev_sampler)} batches)")
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate, num_workers=4)
    dev_loader = DataLoader(dev_ds, batch_sampler=dev_sampler, collate_fn=collate, num_workers=2)

    loss_fn = ColbertPairwiseCELoss()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    base_dev = evaluate_dev(model, dev_loader, device)
    print(f"[baseline] pre-training dev in-batch acc = {base_dev:.4f}")
    if base_dev < 0.5:
        print("[ABORT] baseline below 0.5 -- eval pipeline is broken, not training.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_dev = base_dev
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, (q_in, d_in) in enumerate(train_loader):
            q_in = {k: v.to(device) for k, v in q_in.items()}
            d_in = {k: v.to(device) for k, v in d_in.items()}
            qe = model(**q_in); de = model(**d_in)
            loss = loss_fn(qe, de)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step(); sched.step()
            running += loss.item()
            if (step + 1) % 25 == 0:
                print(f"[ep {epoch}] step {step+1}/{len(train_loader)}  "
                      f"loss={running/(step+1):.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"elapsed={(time.time()-t0)/60:.1f} min")
        dev_acc = evaluate_dev(model, dev_loader, device)
        print(f"[ep {epoch}] DONE  avg_loss={running/len(train_loader):.4f}  "
              f"dev_inbatch_acc={dev_acc:.4f}  (baseline {base_dev:.4f})")
        if dev_acc > best_dev:
            best_dev = dev_acc
            n = save_lora_deltas(model, args.out)
            print(f"[ep {epoch}] new best -> saved {n} LoRA tensors to {args.out}")

    print(f"[done] best dev = {best_dev:.4f}  (baseline {base_dev:.4f})  "
          f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
