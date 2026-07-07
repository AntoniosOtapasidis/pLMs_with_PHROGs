"""
This is the finetuning script

Fine-tunes biohub/ESMC-300M (via LoRA) as a classifier over broad PHROG
functional categories, evaluates on a family-wise train/val/test split,
and extracts fine-tuned embeddings for the labeled PHROGs for downstream
analysis.

Standalone script — run as:
    python embedding_work_finetuned.py
or in the background:
    nohup setsid python embedding_work_finetuned.py > finetune_run.log 2>&1 &


"""
import gc
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available when run headless/in background
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gc


# ----------------------------------------------------
# 1. Paths
# ----------------------------------------------------
base_dir = Path("/users/antonios/pLMs_with_PHROGs/data")
PHROG_SEQ_TAR = base_dir / "FAA_phrog_deduped.tar.gz"
PHROG_ANNOT_TSV = base_dir / "phrog_annot_v4_updated.tsv"

RESULTS_DIR = base_dir / "finetune_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:1"
TRAIN_BATCH_SIZE = 4
N_EPOCHS = 10


# ----------------------------------------------------
# 2. FASTA parsing + PHROG sequence loading
# ----------------------------------------------------
def parse_fasta_string(fasta_content: str):
    """Parses a raw FASTA text string cleanly from memory without needing local files."""
    sequences = []
    current_header = None
    current_seq = []

    for line in fasta_content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header:
                sequences.append((current_header, "".join(current_seq)))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)

    if current_header:
        sequences.append((current_header, "".join(current_seq)))

    return sequences


print("Loading annotations and PHROG sequences...")
annot_df = pd.read_csv(PHROG_ANNOT_TSV, sep="\t")
category_map = annot_df.set_index("phrog")["category"].to_dict()

PHROG_sequences = {}
with tarfile.open(PHROG_SEQ_TAR, "r:gz") as t:
    for member in t.getmembers():
        if not member.name.endswith(".faa"):
            continue
        name = member.name.split("/")[-1]
        phrog_id = int(name.replace("phrog_", "").replace(".faa", ""))
        f = t.extractfile(member)
        if f is None:
            continue
        content = f.read().decode("utf-8")
        seqs = [seq for _, seq in parse_fasta_string(content) if seq]
        PHROG_sequences[phrog_id] = seqs

print(f"PHROGs loaded: {len(PHROG_sequences)}")
print(f"Total sequences: {sum(len(v) for v in PHROG_sequences.values())}")


# ----------------------------------------------------
# 3. Category-label helpers
# ----------------------------------------------------
def get_category(pid):
    val = category_map.get(pid, "unknown function")
    if not isinstance(val, str) or val.lower() == "nan":
        return "unknown function"
    return val


known_categories = sorted(set(
    get_category(pid) for pid in PHROG_sequences
    if get_category(pid) != "unknown function"
))
category2idx = {cat: i for i, cat in enumerate(known_categories)}
idx2category = {i: cat for cat, i in category2idx.items()}
dark_matter_pids = [pid for pid in PHROG_sequences if get_category(pid) == "unknown function"]

print(f"{len(known_categories)} known categories:")
for i, cat in enumerate(known_categories):
    print(f"  {i}: {cat}")
print(f"{len(dark_matter_pids)} PHROGs with unknown function (dark matter)")
assert len(known_categories) > 0, "No categorized PHROGs found — check get_category"


# ----------------------------------------------------
# 4. Family-wise, category-stratified 70/15/15 split
# ----------------------------------------------------
known_pids = [pid for pid in PHROG_sequences if get_category(pid) != "unknown function"]
category_counts = Counter(get_category(pid) for pid in known_pids)

MIN_PER_CLASS = 7  # defensive floor; not triggered on current data (smallest class has 104 members)
splittable_pids = [pid for pid in known_pids if category_counts[get_category(pid)] >= MIN_PER_CLASS]
dropped_pids = [pid for pid in known_pids if category_counts[get_category(pid)] < MIN_PER_CLASS]
if dropped_pids:
    print(f"Dropped {len(dropped_pids)} PHROGs from categories below the {MIN_PER_CLASS}-member floor")

labels = [get_category(pid) for pid in splittable_pids]

train_val_pids, test_pids = train_test_split(
    splittable_pids, test_size=0.15, random_state=42, stratify=labels
)
train_val_labels = [get_category(pid) for pid in train_val_pids]
train_pids, val_pids = train_test_split(
    train_val_pids, test_size=0.15 / 0.85, random_state=42, stratify=train_val_labels
)

train_pairs = [(s, pid) for pid in train_pids for s in PHROG_sequences[pid]]
val_pairs   = [(s, pid) for pid in val_pids   for s in PHROG_sequences[pid]]
test_pairs  = [(s, pid) for pid in test_pids  for s in PHROG_sequences[pid]]

print(f"Train: {len(train_pairs)} seqs | {len(train_pids)} PHROGs")
print(f"Val:   {len(val_pairs)} seqs | {len(val_pids)} PHROGs")
print(f"Test:  {len(test_pairs)} seqs | {len(test_pids)} PHROGs")

# ── SANITY CHECKS ──────────────────────────────────────────────────────────
assert set(train_pids).isdisjoint(val_pids), "PHROG leakage: train/val"
assert set(train_pids).isdisjoint(test_pids), "PHROG leakage: train/test"
assert set(val_pids).isdisjoint(test_pids), "PHROG leakage: val/test"
all_split_pids = set(train_pids) | set(val_pids) | set(test_pids)
assert all_split_pids.isdisjoint(dark_matter_pids), "Dark-matter PHROG leaked into a split"
print("OK: no PHROG ID leakage across train/val/test, and dark matter fully excluded")

for name, pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
    present = set(get_category(pid) for pid in pids)
    missing = set(known_categories) - present
    if missing:
        print(f"WARNING: {name} missing categories: {missing}")
    else:
        print(f"OK: all {len(known_categories)} categories present in {name}")


# ----------------------------------------------------
# 5. Model + LoRA setup
# ----------------------------------------------------
print(f"Loading biohub/ESMC-300M onto {DEVICE}...")
model_ft = AutoModelForSequenceClassification.from_pretrained(
    "biohub/ESMC-300M",
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
    num_labels=len(known_categories),
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    lora_dropout=0.01,
    target_modules=["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"],
    # Without this, classifier.dense matches no target_modules pattern and stays
    # frozen at random init — only classifier.out_proj (via the "out_proj" name
    # collision) would get a LoRA delta on top of that untrained random layer.
    modules_to_save=["classifier"],
)
model_ft = get_peft_model(model_ft, lora_config)
model_ft.print_trainable_parameters()

tokenizer_ft = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
optimizer = AdamW(model_ft.parameters(), lr=1e-4)


# ----------------------------------------------------
# 6. Batched train/val/test loop
# ----------------------------------------------------
def make_batches(pairs, batch_size, sort_by_length=True, shuffle_batches=True, seed=None):
    """Sort by length once (minimizes padding waste), chunk into batches,
    then shuffle the ORDER of batches (not individual pairs) so each epoch
    still sees a different training order without breaking the length-bucketing."""
    if sort_by_length:
        pairs = sorted(pairs, key=lambda p: len(p[0]))
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    if shuffle_batches:
        random.Random(seed).shuffle(batches)
    return batches


def run_epoch(pairs, train: bool, batch_size=TRAIN_BATCH_SIZE, desc="", seed=None):
    model_ft.train() if train else model_ft.eval()
    total_loss, total_count = 0.0, 0
    all_labels, all_preds = [], []
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for batch in tqdm(make_batches(pairs, batch_size, shuffle_batches=train, seed=seed), desc=desc):
            seqs = [s for s, pid in batch]
            label_ids = torch.tensor(
                [category2idx[get_category(pid)] for s, pid in batch], dtype=torch.long
            ).to(model_ft.device)
            inputs = tokenizer_ft(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(model_ft.device) for k, v in inputs.items()}
            out = model_ft(**inputs, labels=label_ids)
            if train:
                out.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_loss += out.loss.item() * len(batch)
            total_count += len(batch)
            all_labels.extend(label_ids.detach().cpu().tolist())
            all_preds.extend(out.logits.detach().argmax(dim=-1).cpu().tolist())
    avg_loss = total_loss / total_count if total_count else float("nan")
    acc = accuracy_score(all_labels, all_preds) if total_count else float("nan")
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if total_count else float("nan")
    return avg_loss, total_count, acc, macro_f1, all_labels, all_preds


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

for epoch in range(N_EPOCHS):
    avg_train, train_count, train_acc, train_f1, _, _ = run_epoch(
        train_pairs, train=True, desc=f"Epoch {epoch + 1} train", seed=epoch
    )
    avg_val, val_count, val_acc, val_f1, _, _ = run_epoch(
        val_pairs, train=False, desc=f"Epoch {epoch + 1} val"
    )
    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    print(f"Epoch {epoch + 1} | train loss: {avg_train:.4f} acc: {train_acc:.4f} macro-F1: {train_f1:.4f} "
          f"| val loss: {avg_val:.4f} acc: {val_acc:.4f} macro-F1: {val_f1:.4f} "
          f"| train_n: {train_count} | val_n: {val_count}")

avg_test, test_count, test_acc, test_f1, test_labels, test_preds = run_epoch(
    test_pairs, train=False, desc="Test evaluation"
)
print(f"Final test loss: {avg_test:.4f} | acc: {test_acc:.4f} | macro-F1: {test_f1:.4f} | test_n: {test_count}")

test_report = classification_report(
    test_labels, test_preds,
    labels=list(range(len(known_categories))),
    target_names=known_categories,
    zero_division=0,
    output_dict=True,
)
print("\nPer-category test metrics:")
print(classification_report(
    test_labels, test_preds,
    labels=list(range(len(known_categories))),
    target_names=known_categories,
    zero_division=0,
))

test_confusion = confusion_matrix(test_labels, test_preds, labels=list(range(len(known_categories))))


# ----------------------------------------------------
# 7. Save results: trained adapter, loss history, loss curve
# ----------------------------------------------------
adapter_path = RESULTS_DIR / "esmc300m_lora_category_classifier"
model_ft.save_pretrained(adapter_path)
print(f"Saved LoRA adapter to {adapter_path}")

results = {
    "known_categories": known_categories,
    "history": history,
    "test_loss": avg_test,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "test_classification_report": test_report,
    "test_confusion_matrix": test_confusion.tolist(),
    "train_pids": train_pids,
    "val_pids": val_pids,
    "test_pids": test_pids,
}
results_path = RESULTS_DIR / "training_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved training/val/test results to {results_path}")

plt.figure()
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.axhline(avg_test, color="red", linestyle="--", label=f"test ({avg_test:.4f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
loss_curve_path = RESULTS_DIR / "loss_curve.png"
plt.savefig(loss_curve_path, dpi=200)
print(f"Saved loss curve to {loss_curve_path}")

plt.figure()
plt.plot(history["train_f1"], label="train")
plt.plot(history["val_f1"], label="val")
plt.axhline(test_f1, color="red", linestyle="--", label=f"test ({test_f1:.4f})")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.tight_layout()
f1_curve_path = RESULTS_DIR / "f1_curve.png"
plt.savefig(f1_curve_path, dpi=200)
print(f"Saved F1 curve to {f1_curve_path}")

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(test_confusion, display_labels=known_categories).plot(
    ax=ax, xticks_rotation=45, colorbar=False
)
plt.tight_layout()
confusion_path = RESULTS_DIR / "test_confusion_matrix.png"
plt.savefig(confusion_path, dpi=200)
print(f"Saved test confusion matrix to {confusion_path}")


# ----------------------------------------------------
# 8. Extract & save fine-tuned embeddings (train+val+test PHROGs only)
# ----------------------------------------------------
EMBED_BATCH_SIZE = 16

labeled_records = (
    [(s, pid, "train") for s, pid in train_pairs]
    + [(s, pid, "val") for s, pid in val_pairs]
    + [(s, pid, "test") for s, pid in test_pairs]
)
labeled_records.sort(key=lambda r: len(r[0]))  # length-bucket for less padding waste

embed_output_path = RESULTS_DIR / "esmc300m_finetuned_embeddings.npy"
meta_output_path = RESULTS_DIR / "esmc300m_finetuned_embeddings_meta.tsv"

print(f"Extracting fine-tuned embeddings for {len(labeled_records)} labeled sequences...")
model_ft.eval()
mm = None
meta_rows = []
with torch.inference_mode():
    for i in tqdm(range(0, len(labeled_records), EMBED_BATCH_SIZE), desc="Extracting embeddings"):
        chunk = labeled_records[i:i + EMBED_BATCH_SIZE]
        seqs = [c[0] for c in chunk]
        inputs = tokenizer_ft(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model_ft.device) for k, v in inputs.items()}

        output = model_ft(**inputs, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled_np = pooled.float().cpu().numpy()

        if mm is None:
            mm = np.lib.format.open_memmap(
                embed_output_path, mode="w+", dtype="float32",
                shape=(len(labeled_records), pooled_np.shape[1]),
            )

        mm[i:i + len(chunk)] = pooled_np
        meta_rows.extend([(pid, split, get_category(pid)) for _, pid, split in chunk])

mm.flush()
meta_df = pd.DataFrame(meta_rows, columns=["phrog_id", "split", "category"])
meta_df.to_csv(meta_output_path, sep="\t", index=False)

print(f"Saved fine-tuned embeddings {mm.shape} to {embed_output_path}")
print(f"Saved metadata to {meta_output_path}")

gc.collect()
torch.cuda.empty_cache()
print("Done.")
