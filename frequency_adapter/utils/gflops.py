#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Minimal: batch_size=1, forward once, use thop to print MACs and FLOPs.

import argparse, json, inspect
import torch
import torch.nn as nn
from frequency_adapter.models import FreqApdapterModel, CLIPAdapterCLIPModel
# 可选：若安装了 transformers，就能直接用 HF 的 CLIPModel
try:
    from transformers import CLIPModel, CLIPProcessor
    _HAS_TF = True
except Exception:
    _HAS_TF = False

from thop import profile  # pip install thop

def dummy_inputs(image_size=336, context_length=77, device="cuda", dtype=torch.float16):
    imgs = torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)  # bs=1
    vocab = 49408
    tokens = torch.randint(0, vocab, (1, context_length), device=device)
    attn = torch.ones_like(tokens)
    return imgs, tokens, attn

def call_style_is_kw(model: nn.Module) -> bool:
    sig = inspect.signature(model.forward)
    params = set(sig.parameters.keys())
    return {"pixel_values", "input_ids", "attention_mask"}.issubset(params)

def run(model, image_size, context_length, device, dtype):
    model = model.to(device).eval()
    imgs, tokens, attn = dummy_inputs(image_size, context_length, device, dtype)

    # 用一个小包装器把 kwargs/positional 统一给 thop
    use_kw = call_style_is_kw(model)

    class Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, *a):
            if use_kw:
                return self.m(pixel_values=imgs, input_ids=tokens, attention_mask=attn)
            else:
                return self.m(imgs, tokens)

    wrapped = Wrap(model)

    with torch.no_grad():
        macs, _ = profile(wrapped, inputs=(torch.empty(0, device=device),), verbose=False)  # dummy占位
    flops = macs * 2  # 常规近似：FLOPs ≈ 2 × MACs
    out = {
        "MACs_G": float(macs) / 1e9,
        "FLOPs_G": float(flops) / 1e9,
        "image_size": image_size,
        "context_length": context_length,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "call_style": "keyword" if use_kw else "positional"
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser(description="Minimal THOP GFLOPs for CLIP forward (bs=1)")
    ap.add_argument("--hf_model", type=str, default=None,
                    help="e.g. openai/clip-vit-base-patch16；若提供则用HF加载")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--context-length", type=int, default=77)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    ap.add_argument('--peft_type', type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.hf_model:
        if not _HAS_TF:
            raise RuntimeError("需要 transformers：pip install transformers")
        if args.peft_type in ['freq_adapter', 'org_freq_adapter']:
            model = FreqApdapterModel.from_pretrained(args.hf_model, torch_dtype=dtype)
            model.set_adapter(enable_mcfa=True, enable_mgfa=True, use_freq_adapter=True)
        elif args.peft_type in [None]:
            model = CLIPModel.from_pretrained(args.hf_model, torch_dtype=dtype)
        elif args.peft_type in ['clip_adapter']:
            model = CLIPAdapterCLIPModel.from_pretrained(args.hf_model, torch_dtype=dtype)
        # 可选：从 processor 推断默认尺寸（不想要就注释掉）
        try:
            proc = CLIPProcessor.from_pretrained(args.hf_model)
            sz = getattr(getattr(proc, "image_processor", None), "size", None)
            if isinstance(sz, dict) and "shortest_edge" in sz:
                args.image_size = int(sz["shortest_edge"])
            elif isinstance(sz, int):
                args.image_size = int(sz)
        except Exception:
            pass
    else:
        # 没给 --hf_model 时，假定你的模型在这儿手动创建（示例Dummy；改成你的模型即可）
        class DummyCLIP(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 16, 16)
                self.img_proj = nn.Linear(64*14*14, 512)
                self.emb = nn.Embedding(49408, 512)
                self.txt_proj = nn.Linear(512*args.context_length, 512)
                self.sim = nn.CosineSimilarity(dim=-1)
            def forward(self, img, tokens):
                v = self.img_proj(self.conv(img).flatten(1))
                t = self.txt_proj(self.emb(tokens).reshape(tokens.size(0), -1))
                return self.sim(v, t)
        model = DummyCLIP()

    run(model, args.image_size, args.context_length, device, dtype)

if __name__ == "__main__":
    main()
# python -m frequency_adapter.utils.gflops --hf_model openai/clip-vit-large-patch14-336 --image-size 336 --context-length 77 --dtype fp16
# freq adapter
# {
#   "MACs_G": 181.310580736,
#   "FLOPs_G": 362.621161472,
#   "image_size": 336,
#   "context_length": 77,
#   "device": "cuda",
#   "dtype": "float16",
#   "call_style": "keyword"
# }
# clip
# {
#   "MACs_G": 181.25805056,
#   "FLOPs_G": 362.51610112,
#   "image_size": 336,
#   "context_length": 77,
#   "device": "cuda",
#   "dtype": "float16",
#   "call_style": "keyword"
# }
# clip adapter
# {
#   "MACs_G": 181.258574848,
#   "FLOPs_G": 362.517149696,
#   "image_size": 336,
#   "context_length": 77,
#   "device": "cuda",
#   "dtype": "float16",
#   "call_style": "keyword"
# }