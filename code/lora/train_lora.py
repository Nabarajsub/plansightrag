"""LoRA fine-tune ColPali on the 3,211-pair train split (5-DOT engineering plans).

v3 — adds unique-image batch sampler on top of the v2 recipe.

v1 (failed, R@5 77 -> 35): LoRA was applied to all 208 projection layers
including the SigLIP vision tower; no LR warmup at lr=1e-4 blew up the visual
encoder in the first ~50 steps. Adapter preserved as lora_adapter_v1_broken/.

v2 (broken metric): correct target_modules, but baseline dev in-batch acc
was only 10% on a 16-way task. Cause: the dataset has 89% of training queries
sharing their page with at least one other query (3,211 pairs over 1,449
unique pages). In-batch contrastive loss with image duplicates inside the
batch gives contradictory signals -- "image X is positive for query A AND a
negative for query B" -- and corrodes the embedding geometry.

v3 fix: a batch sampler that guarantees every batch contains 16 *distinct*
images. Within each epoch we sample one query per image group (rotating
across epochs), so the loss only ever sees clean negatives.

Recipe (otherwise matches the official ColPali fine-tuning):
  - LoRA ONLY on language-model layers + the custom_text_proj embedding head;
    the SigLIP vision tower stays frozen.
  - r=32, lora_alpha=32 (scaling 1.0).
  - linear LR warmup (10% of steps) then linear decay; peak lr 5e-5.
  - gradient clipping (max_norm 1.0).
  - gradient checkpointing with enable_input_require_grads -> batch 16 fits.
  - a pre-training baseline dev eval, so collapse / metric breakage shows up
    at step 0 not after a wasted epoch.

- ColbertPairwiseCELoss: in-batch negatives over MaxSim late-interaction scores.
- Trains on hits AND misses (misses are the hardest contrastive signal).
- Page-disjoint train/dev/test (see split_dataset.py) -> no memorization.
- Saves the best-dev LoRA adapter to lora_adapter/.
"""
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

from __future__ import annotations

import argparse, json, os, random, time
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss

COLPALI_ID = "vidore/colpali-v1.2"
OUT_DIR = f"{PSR_ROOT}/lora_finetune"

# Official ColPali LoRA targets: language-model attn+MLP and the projection
# head -- explicitly NOT the vision tower.
TARGET_MODULES = (
    r"(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
    r"|.*(custom_text_proj).*$)"
)


class QnADataset(Dataset):
    """Each item: (query_text, positive_page_image_path)."""
    def __init__(self, jsonl_path):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        self.rows = [r for r in self.rows if os.path.exists(r["image_path"])]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"query": r["query"], "image_path": r["image_path"]}


class UniqueImageBatchSampler(Sampler):
    """Yields batches of dataset indices whose image_paths are all distinct.

    The training data has many queries per page (avg 2.1, max 6+). A vanilla
    sampler routinely produces batches with image collisions, which break the
    in-batch contrastive loss. This sampler:
      1. groups dataset indices by image_path,
      2. each epoch, randomly picks ONE query per image (rotating across
         epochs because of the shuffle), and
      3. yields batches of `batch_size` distinct-image indices.
    Across multiple epochs every query is seen (in expectation).
    """
    def __init__(self, dataset, batch_size, seed=0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        groups = defaultdict(list)
        for i, r in enumerate(dataset.rows):
            groups[r["image_path"]].append(i)
        self.groups = list(groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        picked = [rng.choice(g) for g in self.groups]  # one query per image
        rng.shuffle(picked)
        for i in range(0, len(picked) - self.batch_size + 1, self.batch_size):
            yield picked[i : i + self.batch_size]

    def __len__(self):
        return len(self.groups) // self.batch_size


def make_collator(processor):
    def collate(batch):
        queries = [b["query"] for b in batch]
        images = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        q_inputs = processor.process_queries(queries)
        d_inputs = processor.process_images(images)
        return q_inputs, d_inputs
    return collate


@torch.no_grad()
def evaluate_dev(model, dev_loader, device):
    """Fraction of in-batch queries whose positive page is the argmax."""
    model.eval()
    correct = total = 0
    for q_inputs, d_inputs in dev_loader:
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
        q_emb = model(**q_inputs)
        d_emb = model(**d_inputs)
        raw = torch.einsum("bnd,csd->bcns", q_emb.float(), d_emb.float())
        scores = raw.max(dim=3).values.sum(dim=2)  # [B, B]
        preds = scores.argmax(dim=1)
        gold = torch.arange(scores.size(0), device=device)
        correct += (preds == gold).sum().item()
        total += scores.size(0)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"{OUT_DIR}/split_train.jsonl")
    ap.add_argument("--dev", default=f"{OUT_DIR}/split_dev.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="forward batch = in-batch negative pool size")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--out_adapter", default=f"{OUT_DIR}/lora_adapter")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  epochs={args.epochs}  bs={args.batch_size}  "
          f"lr={args.lr}  warmup={args.warmup_ratio}  rank={args.lora_rank}")

    print("[init] loading ColPali base")
    model = ColPali.from_pretrained(COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device)
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.05,
        target_modules=TARGET_MODULES, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing -- requires input grads for the frozen base.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    train_ds = QnADataset(args.train)
    dev_ds = QnADataset(args.dev)
    collate = make_collator(processor)
    train_sampler = UniqueImageBatchSampler(train_ds, args.batch_size)
    dev_sampler = UniqueImageBatchSampler(dev_ds, args.batch_size)
    print(f"[data] train={len(train_ds)} pairs ({len(train_sampler.groups)} unique pages -> "
          f"{len(train_sampler)} batches/epoch)  dev={len(dev_ds)} pairs "
          f"({len(dev_sampler.groups)} unique pages -> {len(dev_sampler)} batches)")
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate, num_workers=4)
    dev_loader = DataLoader(dev_ds, batch_sampler=dev_sampler,
                            collate_fn=collate, num_workers=2)

    loss_fn = ColbertPairwiseCELoss()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps)

    # Sanity baseline: the un-tuned model should already separate 16-way batches well.
    base_dev = evaluate_dev(model, dev_loader, device)
    print(f"[baseline] pre-training dev in-batch acc = {base_dev:.4f}")

    best_dev = base_dev
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, (q_inputs, d_inputs) in enumerate(train_loader):
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
            q_emb = model(**q_inputs)
            d_emb = model(**d_inputs)
            loss = loss_fn(q_emb, d_emb)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step(); sched.step()
            running += loss.item()
            if (step + 1) % 25 == 0:
                print(f"[epoch {epoch}] step {step+1}/{len(train_loader)}  "
                      f"loss={running/(step+1):.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"elapsed={(time.time()-t0)/60:.1f} min")
        dev_acc = evaluate_dev(model, dev_loader, device)
        print(f"[epoch {epoch}] DONE  avg_loss={running/len(train_loader):.4f}  "
              f"dev_inbatch_acc={dev_acc:.4f}")
        if dev_acc > best_dev:
            best_dev = dev_acc
            model.save_pretrained(args.out_adapter)
            print(f"[epoch {epoch}] new best dev_acc={dev_acc:.4f} -> saved {args.out_adapter}")

    print(f"[done] best dev in-batch acc = {best_dev:.4f}  "
          f"(baseline {base_dev:.4f})  total {(time.time()-t0)/60:.1f} min")
    if best_dev <= base_dev:
        print("[WARN] training did not beat the baseline -- no adapter saved this run.")
    print(f"[done] LoRA adapter at {args.out_adapter}")


if __name__ == "__main__":
    main()
