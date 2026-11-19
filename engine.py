import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup
transformers.logging.set_verbosity_error()      # hides warnings and the load report
transformers.logging.disable_progress_bar()
import optuna
import plotly
optuna.logging.set_verbosity(optuna.logging.WARNING)
from torch.optim import AdamW
from tqdm.auto import tqdm
import time
import gc, sys
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

cuda = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
print(cuda)

def set_seed(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

def clear_gpu(verbose=True):
    g = globals()
    # 1. Delete global variables holding models, optimizers, schedulers or GPU tensors
    removed = []
    for name, obj in list(g.items()):
        if name.startswith("_"):
            continue
        if (isinstance(obj, (nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler))
                or (torch.is_tensor(obj) and obj.is_cuda)):
            del g[name]
            removed.append(name)

    # 2. Clear hidden references: crashed-cell tracebacks and notebook output history
    sys.last_traceback = sys.last_value = sys.last_type = None
    try:
        get_ipython().run_line_magic("reset", "-f out")
    except NameError:
        pass   # not running in a notebook

    # 3. Free the memory and hand it back to the driver
    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    if verbose:
        print("Removed:", removed or "nothing")
        print(f"Allocated: {allocated:.3f} GB | Reserved: {reserved:.3f} GB")
    return allocated

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(cuda), labels.to(cuda)
            out = model(pixel_values=images, labels=labels)
            loss_sum += out.loss.item() * len(labels)
            correct += (out.logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total      

import os
import time
import contextlib

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

def fully_train_model(model, train_loader, val_loader, **kwargs):
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    epochs = kwargs["epochs"]
    # bf16=True -> full fine-tuning computes in bf16 like QLoRA does (fair speed/memory comparison)
    bf16 = kwargs.get("bf16", False)
    # extra info stored in every history row, e.g. {"seed": 42, "lora_r": 8, "lora_alpha": 16}
    meta = kwargs.get("meta", {})

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_num = sum(p.numel() for p in model.parameters())
    weights_gb = model.get_memory_footprint() / 1e9  # constant, so computed once

    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext

    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss_sum, correct, n = 0.0, 0, 0
        torch.cuda.reset_peak_memory_stats()  # per-epoch peak (training only, read before validation)
        pbar = tqdm(train_loader, desc=f"Epoch: {epoch}", leave=False)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for images, labels, _, _ in pbar:
            images = images.to(cuda)
            labels = labels.to(cuda)
            with autocast():
                out = model(pixel_values=images, labels=labels)
            loss = out.loss

            total_loss_sum += loss.item() * len(labels)
            correct += (out.logits.argmax(-1) == labels).sum().item()
            n += len(labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=f"{total_loss_sum / n:.4f}")

        torch.cuda.synchronize()
        train_time = time.perf_counter() - start
        train_peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        train_peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        val_loss, val_acc = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": total_loss_sum / n,
            "train_accuracy": correct / n,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "train_time_s": train_time,
            "images_per_s": n / train_time,
            "peak_mem_allocated_gb": train_peak_alloc,
            "peak_mem_reserved_gb": train_peak_reserved,
            "weights_mem_gb": weights_gb,
            "trainable_params": trainable_params_num,
            "total_params": total_params_num,
            "trainable_pct": 100 * trainable_params_num / total_params_num,
            "bf16": bf16,
            **meta,
        })

        h = history[-1]
        print(f"Epoch {epoch} | train_loss {h['train_loss']:.4f} | train_acc {h['train_accuracy']:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | {train_time:.1f}s | "
              f"peak {train_peak_alloc:.2f} GB")

    return model, pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader, bf16=False):
    """Test-set predictions + inference speed/memory."""
    model.eval()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    y_true, probs = [], []
    for images, labels, _, _ in loader:
        with autocast():
            logits = model(pixel_values=images.to(cuda)).logits
        probs.append(logits.float().softmax(-1).cpu())
        y_true.append(labels)
    torch.cuda.synchronize()
    t = time.perf_counter() - start

    probs = torch.cat(probs).numpy()
    y_true = torch.cat(y_true).numpy()
    stats = {
        "test_acc": float((probs.argmax(1) == y_true).mean()),
        "test_top5_acc": float((np.argsort(-probs, 1)[:, :5] == y_true[:, None]).any(1).mean()),
        "infer_images_per_s": len(y_true) / t,
        "infer_peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    return y_true, probs, stats


def dir_size_mb(path):
    """Checkpoint size on disk (LoRA adapter vs full model)."""
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e6


def save_predictions(y_true, probs, save_dir, id2label=None):
    """One CSV per model -> F1, confusion matrix, McNemar, top-5 all computed offline."""
    df = pd.DataFrame(probs, columns=[f"p_{i}" for i in range(probs.shape[1])])
    df["y_true"] = y_true
    df["y_pred"] = probs.argmax(1)
    if id2label:
        df["true_name"] = df["y_true"].map(id2label)
        df["pred_name"] = df["y_pred"].map(id2label)
    df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
    return df

# def fully_train_model(model,train_loader, val_loader, **kwargs):
#     lr = kwargs["lr"]
#     #batch_size = kwargs["batch_size"]
#     weight_decay = kwargs["weight_decay"]
#     epochs = kwargs["epochs"]
    
#     optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     steps = epochs * len(train_loader)
#     schedular = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

#     trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     history = []
#     torch.cuda.reset_peak_memory_stats()
#     for epoch in range(epochs):
#         model.train()
#         total_loss_sum = 0
#         n = 0
#         pbar = tqdm(train_loader, desc=f"Epoch: {epoch}", leave=False)

#         torch.cuda.synchronize()
#         start = time.perf_counter()
#         for images, labels, _, _ in pbar:
#             images = images.to(cuda)
#             labels = labels.to(cuda)
#             loss = model(pixel_values = images, labels= labels).loss
            
#             total_loss_sum += loss.item() * len(labels)
#             n += len(labels)
#             loss.backward()
#             optimizer.step()
#             schedular.step()
#             optimizer.zero_grad()
#             pbar.set_postfix(loss=f"{total_loss_sum / n:.4f}")

#         torch.cuda.synchronize()
#         train_time = time.perf_counter() - start

#         train_loss = total_loss_sum / n
#         val_loss, val_acc = evaluate(model, val_loader)

#         history.append({
#             "epoch" : epoch,
#             "train_loss": train_loss,
#             "val_loss": val_loss,
#             "val_accuracy": val_acc,
#             "train_time_s" : train_time,
#             "images_per_s" : n / train_time,
#             "peak_mem_allocated_gb" : torch.cuda.max_memory_allocated() / 1e9,
#             "peak_mem_reserved_gb" : torch.cuda.max_memory_reserved() / 1e9,
#             "weights_mem_consumption" : model.get_memory_footprint(),
#             "trainable_params" : trainable_params_num
#         })

#         h = history[-1]
#         print(f"Epoch {epoch} | train_loss {h['train_loss']:.4f} | val_loss {val_loss:.4f} | "
#               f"val_acc {val_acc:.4f} | {train_time:.1f}s | peak {h['peak_mem_allocated_gb']:.2f} GB")
        
#     return model, pd.DataFrame(history)

def objective(model_name, id2label, label2id, train_loader,val_loader, search_epochs,random_state=RANDOM_STATE,trial=None):
    set_seed(random_state)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    
    #weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    weight_decay = 0.05

    #label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    label_smoothing = 0
    
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=len(id2label), 
        id2label=id2label, 
        label2id=label2id,
        ignore_mismatched_sizes=True
        ).to(cuda)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = search_epochs * len(train_loader)

    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    torch.cuda.reset_peak_memory_stats()
    history = []
    try:
        for epoch in range(search_epochs):
            model.train()
            epoch_loss_sum = 0
            n = 0
            pbar = tqdm(train_loader, desc=f"Trial {trial.number} | epoch {epoch}", leave=False)
            for images, labels, _, _ in pbar:
                logits = model(pixel_values=images.to(cuda)).logits
                loss = F.cross_entropy(logits, labels.to(cuda), label_smoothing=label_smoothing)
                epoch_loss_sum += loss.item() * len(labels)
                n += len(labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss=f"{epoch_loss_sum / n:.4f}")

            train_loss = epoch_loss_sum / n
            val_loss, val_acc = evaluate(model, val_loader)

            history.append({
                "epoch" : epoch+1,
                "train_loss" : train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            trial.set_user_attr("history", history)
            trial.report(val_loss, epoch)

            print(f"Trial {trial.number} | epoch {epoch} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val_loss
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()   

def build_qlora_model(model_name, id2label, label2id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"],
    )
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(id2label), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    # Always use a fresh 37-class head (the loader may keep the checkpoint's 1000-class weights)
    model.classifier = nn.Linear(model.config.hidden_size, len(id2label)).to(model.device)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0,
        target_modules="all-linear",
        exclude_modules=["classifier"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, lora_config)

def objective_qlora(model_name,id2label,label2id,train_loader, val_loader,search_epochs=3, random_state=RANDOM_STATE,trial=None):
    set_seed(seed=random_state)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    torch.cuda.reset_peak_memory_stats()
    model = build_qlora_model(model_name=model_name, id2label=id2label, label2id=label2id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=0.05)
    steps = search_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    history = []
    try:
        for epoch in range(search_epochs):
            model.train()
            loss_sum, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Trial {trial.number} | epoch {epoch}", leave=False)
            for images, labels, _, _ in pbar:
                loss = model(pixel_values=images.to(cuda), labels=labels.to(cuda)).loss
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                loss_sum += loss.item() * len(labels)
                n += len(labels)
                pbar.set_postfix(loss=f"{loss_sum / n:.4f}")

            train_loss = loss_sum / n
            val_loss, val_acc = evaluate(model, val_loader)
            history.append({"epoch": epoch, "train_loss": loss_sum / n,
                            "val_loss": val_loss, "val_accuracy": val_acc})
            trial.set_user_attr("history", history)
            trial.set_user_attr("peak_memory_gb", torch.cuda.max_memory_allocated() / 1e9)

            trial.report(val_loss, epoch)
            print(f"Trial {trial.number} | epoch {epoch+1} | train_loss {train_loss:.4f} | "
                              f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        return val_loss
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
