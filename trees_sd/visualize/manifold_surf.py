#!/usr/bin/env python3
"""
Manifold surfing between two SD1.5 LoRAs (PEFT format) by interpolating adapter weights.

Produces:
  - strip.png: interpolation strip alpha=0..1
  - traits.csv: simple image metrics per alpha
  - traits_edge_density.png / traits_green_ratio.png / traits_texture.png (optional, if matplotlib installed)

Example (Windows Git Bash):
  python -m trees_sd.visualize.manifold_surf ^
    --base_model "C:/Users/talake2/Desktop/trees-stable-diffusion/model_cache/sd-legacy_stable-diffusion-v1-5" ^
    --lora_a "C:/Users/talake2/Desktop/trees-stable-diffusion/lora-outputs/inaturalist_sd1.5/acer/checkpoint-1000" ^
    --lora_b "C:/Users/talake2/Desktop/trees-stable-diffusion/lora-outputs/inaturalist_sd1.5/pinus/checkpoint-1000" ^
    --out_dir "C:/Users/talake2/Desktop/manifold_surf_test" ^
    --prompt "A real-world iNaturalist field photograph of a tree, documentary ecological style, realistic colors" ^
    --negative_prompt "illustration, drawing, painting, sketch, cartoon, anime, 3d render, cgi, logo, caption, text, watermark, abstract, surreal" ^
    --n 11 --seed 123 --steps 28 --guidance 7.0 --dtype fp16 --save_individual
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from diffusers import StableDiffusionPipeline
from peft import PeftModel
from safetensors.torch import load_file


# -----------------------------
# PEFT loader + interpolation
# -----------------------------
def load_peft_state(ckpt_dir: Path, weight_name: str = "adapter_model.safetensors") -> dict:
    fp = ckpt_dir / weight_name
    if not fp.exists():
        raise FileNotFoundError(f"Missing {weight_name} in {ckpt_dir}")
    return load_file(str(fp))  # dict[str, torch.Tensor] on CPU


def intersect_keys(sd_a: dict, sd_b: dict) -> list[str]:
    keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))
    if not keys:
        raise ValueError("No overlapping LoRA keys between A and B state dicts.")
    return keys


def interpolate_state_dict(sd_a: dict, sd_b: dict, keys: list[str], alpha: float) -> dict:
    out = {}
    for k in keys:
        ta = sd_a[k]
        tb = sd_b[k]
        if ta.shape != tb.shape:
            raise ValueError(f"Shape mismatch for key {k}: {ta.shape} vs {tb.shape}")
        # keep on CPU here; caller can move to device/dtype
        out[k] = (1.0 - alpha) * ta + alpha * tb
    return out

def normalize_peft_key(k: str) -> str:
    # PEFT sometimes uses "...lora_A.default.weight" vs checkpoint "...lora_A.weight"
    k = k.replace(".lora_A.weight", ".lora_A.default.weight")
    k = k.replace(".lora_B.weight", ".lora_B.default.weight")
    return k

def encode_prompt_sd15(
        pipe: StableDiffusionPipeline,
        prompt: str,
        negative_prompt: str | None,
        device: torch.device,
        torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Encode prompt(s) into text encoder hidden states for SD1.5
    Returns:
    - prompt_embeds: (1, seq_len, dim)
    - negative_prompt_embeds: (or None)
    """
    # Prompt
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok.input_ids.to(device)
    with torch.inference_mode():
        prompt_embeds = pipe.text_encoder(input_ids)[0].to(dtype=torch_dtype)

    neg_embeds = None
    if negative_prompt is not None and len(negative_prompt) > 0:
        ntok = pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        ninput_ids = ntok.input_ids.to(device)
        with torch.inference_mode():
            neg_embeds = pipe.text_encoder(ninput_ids)[0].to(dtype=torch_dtype)

    return prompt_embeds, neg_embeds

# Linear interpolation helper
def lerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return (1.0 - t) * a + t * b


def build_key_mapping(sd_keys: list[str], live_keys: list[str]) -> dict[str, str]:
    """
    Map checkpoint keys -> live parameter names.
    Returns dict: ckpt_key -> live_param_name
    """
    live_set = set(live_keys)
    mapping: dict[str, str] = {}

    # Pre-index suffixes for faster lookup (optional but nice)
    # Here we do a simple scan; 256 keys is tiny.
    for k in sd_keys:
        k1 = k
        k2 = normalize_peft_key(k)

        if k1 in live_set:
            mapping[k] = k1
            continue
        if k2 in live_set:
            mapping[k] = k2
            continue

        # suffix match (handles differences in prefixes)
        hits = [lk for lk in live_keys if lk.endswith(k1)]
        if hits:
            mapping[k] = hits[0]
            continue
        hits = [lk for lk in live_keys if lk.endswith(k2)]
        if hits:
            mapping[k] = hits[0]
            continue

    return mapping

def resolve_checkpoint_dir(p: Path, weight_name: str) -> Path:
    """
    Allow passing either:
      - .../genus/checkpoint-1000
      - .../genus  (we auto-pick latest checkpoint-*)
    """
    p = Path(p)
    if (p / weight_name).exists():
        return p

    ckpts = []
    for c in p.glob("checkpoint-*"):
        try:
            step = int(c.name.split("-")[-1])
        except Exception:
            step = -1
        if (c / weight_name).exists():
            ckpts.append((step, c))

    if ckpts:
        ckpts.sort(key=lambda x: x[0])
        return ckpts[-1][1]

    return p

# Save GIF outputs for better visualization of the manifold (optional)
def save_gif(frames: list[Image.Image], out_path: Path, fps: float = 6.0, loop: int = 0):
    """
    Save a looping GIF from PIL Images.
    fps: frames per second
    loop: 0 = loop forever
    """
    if not frames:
        raise ValueError("No frames to save")

    duration_ms = int(round(1000.0 / fps))
    first, *rest = [im.convert("P", palette=Image.ADAPTIVE) for im in frames]

    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=loop,
        optimize=False,   # set True if you want smaller but slower saving
        disposal=2,       # better for full-frame updates
    )

# -----------------------------
# Image metrics ("traits")
# -----------------------------
def _to_float01_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def green_stats(rgb01: np.ndarray) -> dict:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    eps = 1e-6
    return {
        "green_mean": float(g.mean()),
        "green_std": float(g.std()),
        "green_ratio": float((g / (r + g + b + eps)).mean()),
    }


def edge_density(rgb01: np.ndarray, thresh: float = 0.20) -> float:
    """
    Simple Sobel edge density on grayscale:
      - compute gradient magnitude (Sobel)
      - threshold
      - return fraction of pixels considered "edges"
    """
    gray = (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    def conv2(img, k):
        h, w = img.shape
        out = np.zeros_like(img)
        padded = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
        for y in range(h):
            for x in range(w):
                patch = padded[y:y+3, x:x+3]
                out[y, x] = float((patch * k).sum())
        return out

    gx = conv2(gray, kx)
    gy = conv2(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)

    m = mag / (mag.max() + 1e-6)
    return float((m > thresh).mean())


def laplacian_variance(rgb01: np.ndarray) -> float:
    """
    Texture proxy: variance of Laplacian of grayscale.
    Higher => more high-frequency texture (bark-like / detailed foliage).
    """
    gray = (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)

    h, w = gray.shape
    out = np.zeros_like(gray)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    for y in range(h):
        for x in range(w):
            patch = padded[y:y+3, x:x+3]
            out[y, x] = float((patch * k).sum())

    return float(out.var())


def compute_traits(img: Image.Image) -> dict:
    rgb01 = _to_float01_rgb(img)
    traits = green_stats(rgb01)
    traits["edge_density"] = edge_density(rgb01, thresh=0.20)
    traits["laplacian_var"] = laplacian_variance(rgb01)
    return traits


# -----------------------------
# Strip rendering
# -----------------------------
def make_strip(images: list[Image.Image], pad: int = 8, bg=(15, 15, 15)) -> Image.Image:
    if not images:
        raise ValueError("No images provided")
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError("All images must be same size")

    total_w = len(images) * w + (len(images) + 1) * pad
    total_h = h + 2 * pad
    canvas = Image.new("RGB", (total_w, total_h), bg)

    x = pad
    for im in images:
        canvas.paste(im, (x, pad))
        x += w + pad
    return canvas


def linspace_0_1(n: int) -> list[float]:
    if n < 2:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model",
        default="runwayml/stable-diffusion-v1-5",
        help="HF model ID or local directory (your cached sd-legacy path is fine).",
    )
    ap.add_argument("--lora_a", required=True, help="Path to LoRA A checkpoint dir (or genus dir containing checkpoint-*).")
    ap.add_argument("--lora_b", required=True, help="Path to LoRA B checkpoint dir (or genus dir containing checkpoint-*).")
    ap.add_argument("--lora_weight_name", default="adapter_model.safetensors", help="PEFT LoRA filename inside checkpoint dir.")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--prompt_a", default="a photo of a tree")
    ap.add_argument("--prompt_b", default=None, help="If set, interpolate prompt embeddings from A->B across alpha.")
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n", type=int, default=11, help="Number of alpha points between 0 and 1")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--save_individual", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_a = resolve_checkpoint_dir(Path(args.lora_a), args.lora_weight_name)
    lora_b = resolve_checkpoint_dir(Path(args.lora_b), args.lora_weight_name)

    if not (lora_a / args.lora_weight_name).exists():
        raise FileNotFoundError(f"LoRA A file not found: {lora_a / args.lora_weight_name}")
    if not (lora_b / args.lora_weight_name).exists():
        raise FileNotFoundError(f"LoRA B file not found: {lora_b / args.lora_weight_name}")

    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    device = torch.device(args.device)

    # 1) Load base pipeline ONCE
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 2) Wrap UNet with PEFT adapter structure from LoRA A
    #    (This loads the adapter config and makes the UNet "LoRA-aware".)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_a), is_trainable=False).to(device)
    pipe.unet.eval()

    # 3) Load LoRA state dicts (CPU) and compute key intersection once
    sd_a = load_peft_state(lora_a, weight_name=args.lora_weight_name)
    sd_b = load_peft_state(lora_b, weight_name=args.lora_weight_name)
    keys = intersect_keys(sd_a, sd_b)

    # Sanity Check - are the two LoRAs actually different? Measure mean absolute difference across all overlapping tensors.
    diff = np.mean([(sd_a[k].float() - sd_b[k].float()).abs().mean().item() for k in keys])
    print("Mean |A-B| per tensor:", diff)

    print(f"LoRA A: {lora_a}")
    print(f"LoRA B: {lora_b}")
    print(f"PEFT keys: A={len(sd_a)}  B={len(sd_b)}  overlap={len(keys)}")

    if len(keys) < 100:
        print("WARNING: very small overlap in LoRA keys. Are these adapters compatible / trained with same target modules?")

    # 4) Fixed initial latents for reproducibility (same noise for all alphas)
    latent_h = args.height // 8
    latent_w = args.width // 8
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents0 = torch.randn(
        (1, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch_dtype,
    )

    alphas = linspace_0_1(args.n)
    images: list[Image.Image] = []
    rows: list[dict] = []

    # Map live parameter names -> parameter references
    live = dict(pipe.unet.named_parameters())
    live_keys = list(live.keys())

    key_map = build_key_mapping(keys, live_keys)
    print(f"Key map coverage: {len(key_map)}/{len(keys)}")

    if len(key_map) < len(keys):
        missing = [k for k in keys if k not in key_map][:10]
        raise RuntimeError(
            f"Could not map {len(keys) - len(key_map)} LoRA keys to live params. "
            f"First missing examples: {missing}"
        )

    # Sanity: ensure at least one mapped key is a LoRA param
    example_ckpt_key = keys[0]
    example_live_key = key_map[example_ckpt_key]
    print(f"Example mapping: {example_ckpt_key} -> {example_live_key}")

    # -----------------------------
    # Prompt interpolation setup
    # -----------------------------
    prompt_a = args.prompt_a
    prompt_b = args.prompt_b if args.prompt_b is not None else args.prompt_a

    prompt_embeds_a, neg_embeds_a = encode_prompt_sd15(
        pipe=pipe,
        prompt=prompt_a,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        device=device,
        torch_dtype=torch_dtype,
    )
    prompt_embeds_b, neg_embeds_b = encode_prompt_sd15(
        pipe=pipe,
        prompt=prompt_b,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        device=device,
        torch_dtype=torch_dtype,
    )

    # Sanity (shapes should match)
    assert prompt_embeds_a.shape == prompt_embeds_b.shape
    if (neg_embeds_a is not None) and (neg_embeds_b is not None):
        assert neg_embeds_a.shape == neg_embeds_b.shape
    
    # SURF THE MANIFOLD! 
    # Iterate over alphas, apply interpolated LoRA weights, generate image, compute traits, save results.
    for i, a in enumerate(alphas):

        # Build interpolated LoRA weights on CPU
        sd_mix = interpolate_state_dict(sd_a, sd_b, keys, a)

        with torch.no_grad():
            for k in keys:
                live_name = key_map[k]
                target = live[live_name]

                mixed = (1.0 - a) * sd_a[k] + a * sd_b[k]
                target.copy_(mixed.to(device=target.device, dtype=target.dtype))

        # debug AFTER applying LORA mix
        name, p0 = next((n, p) for n, p in pipe.unet.named_parameters() if "lora_" in n)
        checksum = float(p0.detach().float().abs().mean().cpu())
        print("DEBUG", a, name, checksum)

        # Reset latents each time so only LoRA mix changes the outcome
        latents = latents0.clone()

        # Interpolate prompt embeddings (A->B) across alpha
        prompt_embeds = lerp(prompt_embeds_a, prompt_embeds_b, float(a))

        # Keep negative prompt fixed for now
        negative_prompt_embeds = neg_embeds_a

        out = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            latents=latents,
        )

        img = out.images[0]
        images.append(img)

        traits = compute_traits(img)
        row = {
            "index": i,
            "alpha": float(a),
            "w_a": float(1.0 - a),
            "w_b": float(a),
            **traits,
        }
        rows.append(row)

        if args.save_individual:
            img.save(out_dir / f"frame_{i:02d}_alpha_{a:.2f}.png")

        print(
            f"[{i+1}/{len(alphas)}] alpha={a:.2f}  edge_density={traits['edge_density']:.4f}  "
            f"green_ratio={traits['green_ratio']:.4f}  lap_var={traits['laplacian_var']:.6f}"
        )

    # Save strip
    strip = make_strip(images, pad=8)
    strip_path = out_dir / "strip.png"
    strip.save(strip_path)

    # # Save GIF as boomerang vis
    # gif_path = out_dir / "strip.gif"
    # frames = images + images[-2:0:-1]  # avoids duplicating endpoints twice
    # save_gif(frames, gif_path, fps=6.0, loop=0)
    # print(f"Saved: {gif_path}")

    gif_path = out_dir / "strip.gif"
    save_gif(images, gif_path, fps=6.0, loop=0)
    print(f"Saved: {gif_path}")

    # Save traits CSV
    csv_path = out_dir / "traits.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Optional plots
    if plt is not None:
        al = [r["alpha"] for r in rows]
        ed = [r["edge_density"] for r in rows]
        gr = [r["green_ratio"] for r in rows]
        lv = [r["laplacian_var"] for r in rows]

        plt.figure()
        plt.plot(al, ed, marker="o")
        plt.xlabel("alpha (LoRA mix: A->B)")
        plt.ylabel("edge_density (Sobel)")
        plt.title("Leafiness proxy: edge density vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "traits_edge_density.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(al, gr, marker="o")
        plt.xlabel("alpha (LoRA mix: A->B)")
        plt.ylabel("green_ratio")
        plt.title("Greenness proxy: green_ratio vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "traits_green_ratio.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(al, lv, marker="o")
        plt.xlabel("alpha (LoRA mix: A->B)")
        plt.ylabel("laplacian_var")
        plt.title("Texture proxy: Laplacian variance vs alpha")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "traits_texture.png", dpi=160)
        plt.close()

    print(f"\nSaved: {strip_path}")
    print(f"Saved: {csv_path}")
    if plt is not None:
        print(f"Saved: {out_dir / 'traits_edge_density.png'}")
        print(f"Saved: {out_dir / 'traits_green_ratio.png'}")
        print(f"Saved: {out_dir / 'traits_texture.png'}")


if __name__ == "__main__":
    main()