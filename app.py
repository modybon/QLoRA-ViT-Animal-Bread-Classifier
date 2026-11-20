"""
Gradio demo: upload a cat/dog photo -> top-5 breeds from all 4 models side by side.
Hover over a breed name to see a sample training image of that breed.
Run from the project folder (same place as data_pipeline.py and trained_models/):
    python app.py
"""
import os
import io
import glob
import time
import base64

import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from transformers import ViTForImageClassification, BitsAndBytesConfig
from peft import PeftModel

from data_pipeline import get_breads_names, eval_tf

# ---------------- Config ----------------
RANDOM_STATE = 42
DATA_ROOT = "data"          # <-- set to the same root folder data_pipeline.py uses (avoids re-downloading)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = {"small": "facebook/deit-small-patch16-224",
               "base":  "facebook/deit-base-patch16-224"}
RUNS = [("small", "full"), ("small", "qlora"), ("base", "full"), ("base", "qlora")]
PATH = lambda size, mode: f"trained_models/deit_{size}_{mode}_seed{RANDOM_STATE}"
LABEL = lambda size, mode: f"DeiT-{size.capitalize()} · {'Full' if mode == 'full' else 'QLoRA'}"

breed_names = get_breads_names()
ID2LABEL = {i: n for i, n in enumerate(breed_names)}
LABEL2ID = {n: i for i, n in enumerate(breed_names)}
pretty = lambda n: n.replace("_", " ").title()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["classifier"],
)


# ---------------- Sample image per breed (for hover) ----------------
THUMB_DIR = "breed_thumbs"   # cache: one small JPEG per breed, built once
THUMB_SIZE = 320


def file_to_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def build_thumbnails():
    """One training image per breed, keyed by class index. Built once, then loaded from THUMB_DIR."""
    cached = {i: os.path.join(THUMB_DIR, f"{i}.jpg") for i in range(len(breed_names))}

    if not all(os.path.exists(p) for p in cached.values()):
        print("Creating breed thumbnails (first run only)...")
        os.makedirs(THUMB_DIR, exist_ok=True)
        ds = OxfordIIITPet(root=DATA_ROOT, split="trainval", download=False)  # never re-download
        if [c.lower() for c in ds.classes] != [pretty(n).lower() for n in breed_names]:
            print("WARNING: torchvision class order differs from get_breads_names(); hover images may be wrong.")
        done = set()
        for path, label in zip(ds._images, ds._labels):
            if label not in done:
                img = Image.open(path).convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(cached[label], format="JPEG", quality=90)
                done.add(label)
            if len(done) == len(breed_names):
                break

    return {i: file_to_data_uri(p) for i, p in cached.items()}


print("Building breed thumbnails...")
THUMBS = build_thumbnails()


# ---------------- Loading ----------------
def load_full(size):
    return ViTForImageClassification.from_pretrained(PATH(size, "full")).to(DEVICE).eval()


def load_qlora(size):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAMES[size],
        num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.classifier = nn.Linear(model.config.hidden_size, len(ID2LABEL)).to(model.device)  # same as training
    model = PeftModel.from_pretrained(model, PATH(size, "qlora"))  # LoRA weights + trained head
    return model.eval()


print("Loading models...")
MODELS = {}
for size, mode in RUNS:
    MODELS[(size, mode)] = load_full(size) if mode == "full" else load_qlora(size)
    print(f"  loaded {LABEL(size, mode)}")

# Warm-up: one-time CUDA setup shouldn't be counted in the first timing
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
    for m in MODELS.values():
        for _ in range(3):
            m(pixel_values=dummy)
if DEVICE == "cuda":
    torch.cuda.synchronize()
print("Warm-up done.")


# ---------------- Rendering ----------------
def render_card(title, top_probs, top_ids):
    rows = []
    for p, i in zip(top_probs, top_ids):
        name = pretty(ID2LABEL[i])
        rows.append(f"""
        <div class="pred-row">
          <span class="breed">{name}
            <span class="tip"><img src="{THUMBS.get(i, '')}"><br>{name}</span>
          </span>
          <div class="bar"><div class="fill" style="width:{p * 100:.1f}%"></div></div>
          <span class="pct">{p:.1%}</span>
        </div>""")
    return f'<div class="pred-card"><div class="pred-title">{title}</div>{"".join(rows)}</div>'


CSS = """
.pred-html, .pred-html * { overflow: visible !important; }
.pred-card { border: 1px solid #e1e0d9; border-radius: 10px; padding: 12px 14px; }
.pred-title { font-weight: 700; margin-bottom: 8px; }
.pred-row { display: grid; grid-template-columns: 190px 1fr 56px; align-items: center;
            gap: 8px; margin: 6px 0; font-size: 14px; }
.breed { position: relative; cursor: help; border-bottom: 1px dotted #898781; width: fit-content; }
.tip { display: none; position: absolute; left: 0; top: 130%; z-index: 1000;
       width: 280px; background: white; border: 1px solid #c3c2b7; border-radius: 8px; padding: 6px;
       box-shadow: 0 6px 20px rgba(0,0,0,.22); text-align: center; font-size: 13px; color: #0b0b0b; }
.tip img { width: 100% !important; max-width: none !important; height: auto !important;
           border-radius: 4px; display: block; margin: 0 0 4px 0; }
.breed:hover .tip { display: block; }
.bar { background: #eeede8; border-radius: 4px; height: 10px; }
.fill { background: #eb6834; border-radius: 4px; height: 10px; }
.pct { text-align: right; font-variant-numeric: tabular-nums; }
"""


# ---------------- Inference ----------------
@torch.no_grad()
def classify(image, n_repeats=10):
    if image is None:
        return [""] * len(RUNS) + [""]

    x = eval_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    cards, rows, top1 = [], [], set()
    for key in RUNS:
        model = MODELS[key]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            for _ in range(n_repeats):
                logits = model(pixel_values=x).logits
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / n_repeats

        probs = logits.float().softmax(-1)[0].cpu()
        top = probs.topk(5)
        ids, ps = top.indices.tolist(), top.values.tolist()
        cards.append(render_card(LABEL(*key), ps, ids))
        top1.add(ids[0])
        rows.append(f"| {LABEL(*key)} | {pretty(ID2LABEL[ids[0]])} | {ps[0]:.1%} | {ms:.1f} ms |")

    agree = "✅ All 4 models agree." if len(top1) == 1 else "⚠️ Models disagree."
    table = ("| Model | Top-1 breed | Confidence | Inference time (avg of 10) |\n|---|---|---|---|\n"
             + "\n".join(rows) + f"\n\n{agree}")
    return cards + [table]


# ---------------- UI ----------------
with gr.Blocks(title="Pet Breed Classifier: Full vs QLoRA", css=CSS) as demo:
    gr.Markdown(
        "# 🐱🐶 Pet Breed Classifier — Full fine-tuning vs QLoRA\n"
        "Upload a cat or dog photo. Each of the 4 models predicts its top-5 breeds "
        "out of the 37 Oxford-IIIT Pet breeds. **Hover over a breed name** to see a sample image."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Pet photo", height=320)
            btn = gr.Button("Classify", variant="primary")
            example_files = sorted(glob.glob("examples/*.[jp][pn]g"))
            if example_files:
                gr.Examples(example_files, inputs=image_in)
        with gr.Column(scale=2):
            with gr.Row():
                outs = [gr.HTML(elem_classes="pred-html") for _ in RUNS[:2]]
            with gr.Row():
                outs += [gr.HTML(elem_classes="pred-html") for _ in RUNS[2:]]
    summary = gr.Markdown()

    btn.click(classify, inputs=image_in, outputs=outs + [summary])
    image_in.change(classify, inputs=image_in, outputs=outs + [summary])

if __name__ == "__main__":
    demo.launch()
