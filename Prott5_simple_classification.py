# Fine-tune ProtT5 with LoRA as a classifier over broad PHROG functional categories.
# Same train/val/test split as the ESMC-300M run, same deduped archive.
#
# Three approaches live in this file, as functions:
#   1. linear_probe_pipeline()  — frozen ProtT5 + LogisticRegression on top (cheap, already run — 77.4% acc / 0.726 macro-F1)
#   2. lora_finetune_pipeline() — LoRA-adapt ProtT5 itself + a trainable classification head (more expensive, higher ceiling)
#   3. blast_baseline_pipeline() — classical BLAST nearest-neighbor baseline, no pLM at all
#
# Run as a script: the __main__ block at the bottom calls lora_finetune_pipeline().

import gc
import json
import re
import subprocess
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from transformers import T5Tokenizer, T5EncoderModel

BASE_DIR = Path("/users/antonios/pLMs_with_PHROGs/data")
TAR_PATH = BASE_DIR / "FAA_phrog_deduped.tar.gz"
RESULTS_DIR = BASE_DIR / "finetune_results"
CHECKPOINT = "Rostlab/prot_t5_xl_half_uniref50-enc"


# ----------------------------------------------------
# Shared helpers
# ----------------------------------------------------
def parse_fasta_string(fasta_content):
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


def load_known_category_records(base_dir=BASE_DIR, tar_path=TAR_PATH):
    """Reuses the EXACT train/val/test PHROG split from the ESMC fine-tuning run,
    so results stay directly comparable. Returns (records, known_categories,
    category_map) where records is a list of (seq, phrog_id, split)."""
    training_results = json.loads((base_dir / "finetune_results" / "training_results.json").read_text())
    train_pids = set(training_results["train_pids"])
    val_pids = set(training_results["val_pids"])
    test_pids = set(training_results["test_pids"])
    known_categories = training_results["known_categories"]
    wanted_pids = train_pids | val_pids | test_pids

    annot_df = pd.read_csv(base_dir / "phrog_annot_v4.tsv", sep="\t")
    category_map = annot_df.set_index("phrog")["category"].to_dict()

    def split_of(pid):
        if pid in train_pids: return "train"
        if pid in val_pids: return "val"
        return "test"

    PHROG_sequences = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            match = re.search(r'phrog_(\d+)\.(faa|fasta)$', member.name)
            if not match:
                continue
            phrog_id = int(match.group(1))
            if phrog_id not in wanted_pids:
                continue
            f_in = tar.extractfile(member)
            if f_in is None:
                continue
            content = f_in.read().decode("utf-8")
            seqs = [seq for _, seq in parse_fasta_string(content) if seq]
            PHROG_sequences[phrog_id] = seqs

    records = []  # (seq, phrog_id, split)
    for pid, seqs in PHROG_sequences.items():
        split = split_of(pid)
        for seq in seqs:
            records.append((seq, pid, split))
    records.sort(key=lambda r: len(r[0]))

    print(f"Loaded {len(PHROG_sequences):,} known-category PHROGs "
          f"(train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)})")
    print(f"{len(records):,} sequences total")
    return records, known_categories, category_map


def prott5_prep(seqs):
    """ProtT5 preprocessing: uppercase, rare-AA -> X, space-separated residues.
    No <AA2fold> prefix — that's ProstT5-specific, not ProtT5."""
    return [" ".join(list(re.sub(r"[UZOB]", "X", s.upper()))) for s in seqs]


def make_batches(pairs, batch_size, sort_by_length=True, shuffle_batches=True, seed=None):
    import random
    if sort_by_length:
        pairs = sorted(pairs, key=lambda p: len(p[0]))
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    if shuffle_batches:
        random.Random(seed).shuffle(batches)
    return batches


# ----------------------------------------------------
# Approach 1: SIMPLE CLASSIFIER — ProtT5 stays completely frozen, only a
# LogisticRegression is trained on top of its embeddings. Cheap (no GPU
# backprop, just embedding extraction + a quick sklearn fit). Already run —
# 77.4% test accuracy / 0.726 macro-F1. Not the pipeline __main__ currently
# calls; call this function directly if you want to regenerate its outputs
# (prott5_linear_probe_classifier.joblib, prott5_linear_probe_results.json).
# ----------------------------------------------------
def linear_probe_pipeline():
    from sklearn.linear_model import LogisticRegression
    import joblib

    records, known_categories, category_map = load_known_category_records()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(CHECKPOINT).to(device)
    model.full() if device.type == 'cpu' else model.half()
    model.eval()

    BATCH_SIZE = 8
    CHECKPOINT_EVERY = 50
    embed_output_path = BASE_DIR / "prott5_known_category_embeddings.npy"
    meta_output_path = BASE_DIR / "prott5_known_category_embeddings_meta.tsv"
    progress_path = BASE_DIR / "prott5_known_category_embeddings.progress.json"

    if progress_path.exists() and embed_output_path.exists():
        start_idx = json.loads(progress_path.read_text())["next_index"]
        mm = np.lib.format.open_memmap(embed_output_path, mode="r+")
        print(f"Resuming: {start_idx:,}/{len(records):,} already embedded.")
    else:
        start_idx = 0
        mm = None
        with open(meta_output_path, "w") as f:
            f.write("phrog_id\tsplit\n")

    with torch.no_grad():
        for batch_num, i in enumerate(range(start_idx, len(records), BATCH_SIZE)):
            batch = records[i:i + BATCH_SIZE]
            seqs = [r[0] for r in batch]
            phrog_ids = [r[1] for r in batch]
            splits = [r[2] for r in batch]

            prepped = prott5_prep(seqs)
            ids = tokenizer.batch_encode_plus(prepped, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
            output = model(ids.input_ids, attention_mask=ids.attention_mask)

            hidden = output.last_hidden_state
            mask = ids.attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled_np = pooled.float().cpu().numpy()

            if mm is None:
                mm = np.lib.format.open_memmap(
                    embed_output_path, mode="w+", dtype="float32",
                    shape=(len(records), pooled_np.shape[1]),
                )
            mm[i:i + len(batch)] = pooled_np
            with open(meta_output_path, "a") as f:
                for pid, sp in zip(phrog_ids, splits):
                    f.write(f"{pid}\t{sp}\n")

            if batch_num % CHECKPOINT_EVERY == 0:
                mm.flush()
                progress_path.write_text(json.dumps({"next_index": i + len(batch)}))
                print(f"  {i + len(batch):,}/{len(records):,} embedded")

    mm.flush()
    print(f"Saved ProtT5 known-category embeddings {mm.shape} to {embed_output_path}")

    embeddings = np.load(embed_output_path, mmap_mode="r")
    meta_df = pd.read_csv(meta_output_path, sep="\t")
    meta_df["category"] = meta_df["phrog_id"].map(category_map)

    train_mask = (meta_df["split"] == "train").to_numpy()
    test_mask = (meta_df["split"] == "test").to_numpy()
    X_train = np.asarray(embeddings[train_mask])
    y_train = meta_df.loc[train_mask, "category"].to_numpy()
    X_test = np.asarray(embeddings[test_mask])
    y_test = meta_df.loc[test_mask, "category"].to_numpy()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"Test accuracy: {acc:.4f} | Test macro-F1: {macro_f1:.4f}")

    cm = confusion_matrix(y_test, preds, labels=known_categories)
    joblib.dump(clf, RESULTS_DIR / "prott5_linear_probe_classifier.joblib")
    with open(RESULTS_DIR / "prott5_linear_probe_results.json", "w") as f:
        json.dump({
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "test_classification_report": classification_report(y_test, preds, labels=known_categories, zero_division=0, output_dict=True),
            "test_confusion_matrix": cm.tolist(),
            "known_categories": known_categories,
        }, f, indent=2)


# ----------------------------------------------------
# Approach 0 (baseline): BLAST — classical sequence-similarity nearest-
# neighbor classifier Same train/test split as the
# other two approaches (reuses load_known_category_records()), and the BLAST
# database is built from train-split sequences ONLY — a test sequence can
# never match itself or a same-family relative, so this is directly
# comparable to ProtT5's held-out test accuracy/macro-F1, not an inflated
# "found itself in the database" number. Not the pipeline __main__ currently
# calls; call this function directly if you want to (re)generate its outputs.
# ----------------------------------------------------
def blast_baseline_pipeline():
    records, known_categories, category_map = load_known_category_records()

    out_dir = RESULTS_DIR / "blast_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = [(pid, i, seq) for i, (seq, pid, split) in enumerate(records) if split == "train"]
    test_records = [(pid, i, seq) for i, (seq, pid, split) in enumerate(records) if split == "test"]
    print(f"Train: {len(train_records):,} sequences | Test: {len(test_records):,} sequences")

    def write_fasta(recs, path):
        with open(path, "w") as f:
            for pid, i, seq in recs:
                f.write(f">phrog_{pid}_{i}\n{seq}\n")

    train_fasta = out_dir / "train_sequences.fasta"
    test_fasta = out_dir / "test_sequences.fasta"
    write_fasta(train_records, train_fasta)
    write_fasta(test_records, test_fasta)

    db_path = out_dir / "traindb"
    print("Building BLAST database from train-split sequences only...")
    subprocess.run(
        ["makeblastdb", "-in", str(train_fasta), "-dbtype", "prot", "-out", str(db_path)],
        check=True,
    )

    blast_out = out_dir / "blast_results.tsv"
    print("Running blastp (all test sequences vs train-only database, one batch call)...")
    subprocess.run(
        [
            "blastp", "-query", str(test_fasta), "-db", str(db_path),
            "-outfmt", "6 qseqid sseqid pident length evalue bitscore",
            "-max_target_seqs", "5", "-evalue", "10",
            "-num_threads", "32",
            "-out", str(blast_out),
        ],
        check=True,
    )

    cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
    hits = pd.read_csv(blast_out, sep="\t", names=cols)
    # best hit per query = lowest e-value, tie-broken by highest bitscore
    best_hits = (
        hits.sort_values(["evalue", "bitscore"], ascending=[True, False])
        .groupby("qseqid", as_index=False)
        .first()
    )

    def phrog_id_from_seqid(seqid):
        return int(re.match(r"phrog_(\d+)_\d+", seqid).group(1))

    best_hits["query_phrog_id"] = best_hits["qseqid"].apply(phrog_id_from_seqid)
    best_hits["hit_phrog_id"] = best_hits["sseqid"].apply(phrog_id_from_seqid)
    best_hits["true_category"] = best_hits["query_phrog_id"].map(category_map)
    best_hits["predicted_category"] = best_hits["hit_phrog_id"].map(category_map)

    all_query_ids = {f"phrog_{pid}_{i}" for pid, i, seq in test_records}
    matched_ids = set(best_hits["qseqid"])
    n_no_hit = len(all_query_ids - matched_ids)
    print(f"{len(best_hits):,}/{len(all_query_ids):,} test sequences got a BLAST hit "
          f"(evalue<=10); {n_no_hit:,} had no significant hit at all")

    # FULL COVERAGE (the fair number): every test sequence is scored, and a
    # no-hit query counts as wrong -- ProtT5 always outputs some category for
    # every test sequence and never gets to abstain, so BLAST shouldn't
    # either. "NO_HIT" is a sentinel prediction that can never equal a real
    # category, so it always counts as a miss for whatever the true category was.
    NO_HIT = "NO_HIT"
    true_by_qid = {f"phrog_{pid}_{i}": category_map.get(pid) for pid, i, seq in test_records}
    pred_by_qid = dict(zip(best_hits["qseqid"], best_hits["predicted_category"]))

    y_true_full, y_pred_full = [], []
    for qid in sorted(all_query_ids):
        true_cat = true_by_qid[qid]
        pred_cat = pred_by_qid.get(qid)
        y_true_full.append(true_cat)
        y_pred_full.append(pred_cat if isinstance(pred_cat, str) else NO_HIT)

    acc_full = accuracy_score(y_true_full, y_pred_full)
    macro_f1_full = f1_score(y_true_full, y_pred_full, average="macro", labels=known_categories, zero_division=0)
    print(f"BLAST baseline (FULL COVERAGE, no-hit=wrong -- fair vs. ProtT5): "
          f"accuracy {acc_full:.4f} | macro-F1 {macro_f1_full:.4f} (n={len(y_true_full):,})")

    # HIT-ONLY (diagnostic, not the fair comparison number): how good is
    # BLAST's opinion when it actually has one.
    evaluated = best_hits.dropna(subset=["predicted_category"])
    y_true_hit_only = evaluated["true_category"].tolist()
    y_pred_hit_only = evaluated["predicted_category"].tolist()
    acc_hit_only = accuracy_score(y_true_hit_only, y_pred_hit_only)
    macro_f1_hit_only = f1_score(y_true_hit_only, y_pred_hit_only, average="macro", labels=known_categories, zero_division=0)
    print(f"BLAST baseline (hit-only, {len(evaluated):,}/{len(all_query_ids):,} queries): "
          f"accuracy {acc_hit_only:.4f} | macro-F1 {macro_f1_hit_only:.4f}")

    report = classification_report(y_true_full, y_pred_full, labels=known_categories, zero_division=0, output_dict=True)
    print(classification_report(y_true_full, y_pred_full, labels=known_categories, zero_division=0))
    cm_labels = known_categories + [NO_HIT]
    cm = confusion_matrix(y_true_full, y_pred_full, labels=cm_labels)

    best_hits.to_csv(out_dir / "blast_baseline_predictions.tsv", sep="\t", index=False)
    with open(out_dir / "blast_baseline_results.json", "w") as f:
        json.dump({
            "known_categories": known_categories,
            "n_test_sequences": len(all_query_ids),
            "n_with_hit": len(best_hits),
            "n_no_hit": n_no_hit,
            "accuracy_full_coverage": acc_full,
            "macro_f1_full_coverage": macro_f1_full,
            "accuracy_hit_only": acc_hit_only,
            "macro_f1_hit_only": macro_f1_hit_only,
            "classification_report_full_coverage": report,
            "confusion_matrix_labels": cm_labels,
            "confusion_matrix_full_coverage": cm.tolist(),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=cm_labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_dir / "blast_baseline_confusion_matrix.png", dpi=200)

    print(f"Saved results to {out_dir}")


# ----------------------------------------------------
# Approach 2: LoRA FINE-TUNING — ProtT5 itself is adapted (not just a
# classifier on top of frozen embeddings). LoRA adapters target the encoder's
# q/k/v/o/wi/wo Linear layers (~1.8% of the 1.2B params actually trainable);
# modules_to_save=["classifier"] keeps the classification head fully
# trainable too (verified empirically — without it, the head stays frozen
# at random init and never learns anything). More expensive than the simple
# classifier above (real GPU backprop, not just an embedding extraction +
# sklearn fit), but has a higher ceiling since the encoder's own
# representations can shift during training, not just the head on top of
# them. This is what __main__ currently runs.
# ----------------------------------------------------
class ProtT5ClassificationHead(nn.Module):
    """Same dense -> tanh -> out_proj shape as ESMC's classification head."""
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class ProtT5ForSequenceClassification(nn.Module):
    """Wraps T5EncoderModel (self.encoder) + a classification head (self.classifier).
    Naming matters: LoRA's modules_to_save=["classifier"] matches this field name."""
    def __init__(self, checkpoint, num_labels):
        super().__init__()
        # bf16, not the fp32 default — this is a ~1.2B param encoder, and full
        # precision roughly doubles memory use for no accuracy benefit here.
        # Ampere (RTX 3060) has native bf16 support, no loss-scaling needed
        # unlike fp16.
        self.encoder = T5EncoderModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
        self.classifier = ProtT5ClassificationHead(self.encoder.config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.float()  # upcast to fp32 before the classifier head — the
        # head is tiny (~1M params) so this costs nothing, and keeps its weights
        # (constructed in default fp32) numerically consistent with its input
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        # pooled is exposed here (not just logits/loss) so the same forward pass used
        # for classification can also be reused for fine-tuned embedding extraction.
        return type("Output", (), {"loss": loss, "logits": logits, "pooled": pooled})()


def build_lora_model(num_labels, device):
    from peft import LoraConfig, get_peft_model

    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, do_lower_case=False)
    model = ProtT5ForSequenceClassification(CHECKPOINT, num_labels)
    model.encoder.gradient_checkpointing_enable()  # trade compute for memory — 3B params is a lot on a 12GB GPU

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,  # bumped from 0.01 — ProtT5's frozen features were already
        # strong (66-73% depending on setup), unlike ESMC's weak zero-shot baseline
        # (~44.5%), so this run needs more regularization to avoid overfitting fast
        target_modules=["q", "k", "v", "o", "wi", "wo"],  # verified against the actual ProtT5-XL encoder's Linear layer names
        # Verified empirically (not by analogy to ESMC): target_modules above doesn't
        # match "dense"/"out_proj" at all, so without modules_to_save the ENTIRE
        # classifier head (both layers) stays frozen at random init, not just part of
        # it. PEFT wraps this module and trains a separate modules_to_save.default
        # copy — confirmed both dense and out_proj get requires_grad=True there.
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    return model, tokenizer


def run_epoch(model, tokenizer, pairs, category2idx, device, train, batch_size, optimizer=None, seed=None, desc=""):
    from tqdm import tqdm
    model.train() if train else model.eval()
    total_loss, total_count = 0.0, 0
    all_labels, all_preds = [], []
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for batch in tqdm(make_batches(pairs, batch_size, shuffle_batches=train, seed=seed), desc=desc):
            seqs = [s for s, pid, cat in batch]
            label_ids = torch.tensor([category2idx[cat] for s, pid, cat in batch], dtype=torch.long).to(device)
            prepped = prott5_prep(seqs)
            inputs = tokenizer.batch_encode_plus(prepped, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
            out = model(inputs.input_ids, inputs.attention_mask, labels=label_ids)
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


def lora_finetune_pipeline(n_epochs=20, train_batch_size=8, patience=4):
    records, known_categories, category_map = load_known_category_records()
    category2idx = {cat: i for i, cat in enumerate(known_categories)}

    # records are (seq, phrog_id, split) -> convert to (seq, phrog_id, category) for run_epoch's label lookup
    def to_pairs(split_name):
        return [(seq, pid, category_map.get(pid)) for seq, pid, split in records if split == split_name]

    train_pairs = to_pairs("train")
    val_pairs = to_pairs("val")
    test_pairs = to_pairs("test")
    print(f"Train: {len(train_pairs):,} | Val: {len(val_pairs):,} | Test: {len(test_pairs):,}")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = build_lora_model(len(known_categories), device)
    # lr dropped an order of magnitude from the ESMC config (1e-4 -> 2e-5).
    # ESMC's zero-shot baseline was weak (~44.5% macro-F1) so an aggressive LR made
    # sense to move it a lot; ProtT5's frozen features were already strong
    # (66-73%), so the same large step size overshot a near-good optimum instead
    # of refining it — val peaked at epoch 1 and got worse every epoch after.
    # Only pass trainable params (not the whole model) — frozen params never get
    # gradients so AdamW would just waste memory tracking momentum for them.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)

    best_adapter_path = RESULTS_DIR / "prott5_lora_category_classifier_best"
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
    for epoch in range(n_epochs):
        avg_train, train_n, train_acc, train_f1, _, _ = run_epoch(
            model, tokenizer, train_pairs, category2idx, device, train=True,
            batch_size=train_batch_size, optimizer=optimizer, seed=epoch, desc=f"Epoch {epoch+1} train"
        )
        avg_val, val_n, val_acc, val_f1, _, _ = run_epoch(
            model, tokenizer, val_pairs, category2idx, device, train=False,
            batch_size=train_batch_size, desc=f"Epoch {epoch+1} val"
        )
        history["train_loss"].append(avg_train); history["val_loss"].append(avg_val)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1); history["val_f1"].append(val_f1)
        print(f"Epoch {epoch+1} | train loss {avg_train:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} "
              f"| val loss {avg_val:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        # Track the best checkpoint by val macro-F1, not just whatever epoch training
        # happens to stop at — the ESMC run's final epoch was actually slightly worse
        # than an earlier one, and this avoids repeating that.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            model.save_pretrained(best_adapter_path)
            print(f"  New best val macro-F1: {best_val_f1:.4f} — saved checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{patience} epoch(s) "
                  f"(best so far: {best_val_f1:.4f})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no val macro-F1 improvement for {patience} consecutive epochs.")
                break

    # Reload the best checkpoint (not necessarily the last epoch trained) before
    # final test evaluation, so the reported test numbers reflect the best model.
    from peft import PeftModel
    base_model = ProtT5ForSequenceClassification(CHECKPOINT, len(known_categories))
    model = PeftModel.from_pretrained(base_model, best_adapter_path).to(device)
    print(f"Reloaded best checkpoint (val macro-F1 {best_val_f1:.4f}) for final test evaluation")

    avg_test, test_n, test_acc, test_f1, test_labels, test_preds = run_epoch(
        model, tokenizer, test_pairs, category2idx, device, train=False,
        batch_size=train_batch_size, desc="Test evaluation"
    )
    print(f"Final test loss {avg_test:.4f} | acc {test_acc:.4f} | macro-F1 {test_f1:.4f}")

    report = classification_report(test_labels, test_preds, labels=list(range(len(known_categories))),
                                    target_names=known_categories, zero_division=0, output_dict=True)
    print(classification_report(test_labels, test_preds, labels=list(range(len(known_categories))),
                                 target_names=known_categories, zero_division=0))
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(known_categories))))

    adapter_path = RESULTS_DIR / "prott5_lora_category_classifier"
    model.save_pretrained(adapter_path)
    with open(RESULTS_DIR / "prott5_lora_finetune_results.json", "w") as f:
        json.dump({
            "known_categories": known_categories,
            "history": history,
            "test_loss": avg_test,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_classification_report": report,
            "test_confusion_matrix": cm.tolist(),
        }, f, indent=2)
    print(f"Saved LoRA adapter to {adapter_path}")

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.axhline(avg_test, color="red", linestyle="--", label=f"test ({avg_test:.4f})")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prott5_lora_loss_curve.png", dpi=200)

    plt.figure()
    plt.plot(history["train_f1"], label="train")
    plt.plot(history["val_f1"], label="val")
    plt.axhline(test_f1, color="red", linestyle="--", label=f"test ({test_f1:.4f})")
    plt.xlabel("Epoch"); plt.ylabel("Macro F1"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prott5_lora_f1_curve.png", dpi=200)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=known_categories).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prott5_lora_test_confusion_matrix.png", dpi=200)

    # ----------------------------------------------------
    # Extract & save fine-tuned embeddings for train+val+test (the best
    # checkpoint, already reloaded above) — same role as
    # esmc300m_finetuned_embeddings.npy played for ESMC: this is what feeds
    # the test-set t-SNE plot and any other embedding-space analysis.
    # ----------------------------------------------------
    extract_finetuned_embeddings(model, tokenizer, train_pairs, val_pairs, test_pairs, device)

    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


def extract_finetuned_embeddings(model, tokenizer, train_pairs, val_pairs, test_pairs, device, embed_batch_size=8, checkpoint_every=50):
    from tqdm import tqdm

    labeled_records = (
        [(seq, pid, "train") for seq, pid, cat in train_pairs]
        + [(seq, pid, "val") for seq, pid, cat in val_pairs]
        + [(seq, pid, "test") for seq, pid, cat in test_pairs]
    )
    category_by_pid = {pid: cat for seq, pid, cat in (train_pairs + val_pairs + test_pairs)}
    labeled_records.sort(key=lambda r: len(r[0]))

    embed_output_path = RESULTS_DIR / "prott5_lora_finetuned_embeddings.npy"
    meta_output_path = RESULTS_DIR / "prott5_lora_finetuned_embeddings_meta.tsv"
    progress_path = RESULTS_DIR / "prott5_lora_finetuned_embeddings.progress.json"

    if progress_path.exists() and embed_output_path.exists():
        start_idx = json.loads(progress_path.read_text())["next_index"]
        mm = np.lib.format.open_memmap(embed_output_path, mode="r+")
        print(f"Resuming embedding extraction: {start_idx:,}/{len(labeled_records):,} already done.")
    else:
        start_idx = 0
        mm = None
        with open(meta_output_path, "w") as f:
            f.write("phrog_id\tsplit\tcategory\n")

    model.eval()
    with torch.inference_mode():
        for batch_num, i in enumerate(range(start_idx, len(labeled_records), embed_batch_size)):
            chunk = labeled_records[i:i + embed_batch_size]
            seqs = [c[0] for c in chunk]
            prepped = prott5_prep(seqs)
            inputs = tokenizer.batch_encode_plus(prepped, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
            out = model(inputs.input_ids, inputs.attention_mask)
            pooled_np = out.pooled.float().cpu().numpy()

            if mm is None:
                mm = np.lib.format.open_memmap(
                    embed_output_path, mode="w+", dtype="float32",
                    shape=(len(labeled_records), pooled_np.shape[1]),
                )
            mm[i:i + len(chunk)] = pooled_np
            with open(meta_output_path, "a") as f:
                for _, pid, split in chunk:
                    f.write(f"{pid}\t{split}\t{category_by_pid[pid]}\n")

            if batch_num % checkpoint_every == 0:
                mm.flush()
                progress_path.write_text(json.dumps({"next_index": i + len(chunk)}))
                print(f"  {i + len(chunk):,}/{len(labeled_records):,} embedded")

    mm.flush()
    print(f"Saved fine-tuned ProtT5 embeddings {mm.shape} to {embed_output_path}")


if __name__ == "__main__":
    # ACTIVE PATH: LoRA fine-tuning (Approach 2), not the simple classifier
    # (Approach 1) — the simple classifier already ran and its results are
    # saved; call linear_probe_pipeline() instead of the line below if you
    # ever need to regenerate them.
    lora_finetune_pipeline()
