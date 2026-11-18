---
title: Maa Sitala Passport Photo Studio
emoji: 📸
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: true
suggested_hardware: zero-a10g
---

# 📸 Advance Passport Size Photo Studio V10 (`Gradio 6.32.0` + `BRIA RMBG-2.0` + `CatVTON AnatomicalBodyFit™` + `True 8K`)

[![Hugging Face Public Showcase Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20ZeroGPU-Public%20Showcase%20Live-blue)](https://huggingface.co/spaces/AnupamSahoo/Advance-Passport-Photo-Studio)
[![GitHub Showcase Repository](https://img.shields.io/badge/GitHub-anupamsahoo4508%2FAdvance--Passport--Photo--Studio-181717?logo=github)](https://github.com/anupamsahoo4508/Advance-Passport-Photo-Studio)
[![Gradio SDK](https://img.shields.io/badge/Gradio-6.32.0-orange)](https://www.gradio.app/)

Next-generation **Cloud-Native AI & Biophysical Studio Engine** engineered for **Hugging Face ZeroGPU Cloud** (`0 MB` local PC model downloads):
- **Public Developer Showcase (Preview on ZeroGPU — No Dataset Required):** [`AnupamSahoo/Advance-Passport-Photo-Studio`](https://huggingface.co/spaces/AnupamSahoo/Advance-Passport-Photo-Studio)
- **Professional Production Studio:** `DevAnupam2008/Passport-Studio-V2`
- **GitHub Portfolio Repository:** [`anupamsahoo4508/Advance-Passport-Photo-Studio`](https://github.com/anupamsahoo4508/Advance-Passport-Photo-Studio)

## ✨ V10 Studio Master Highlights

1. **👗 Two-Stage Canonical `CatVTON` + `AnatomicalBodyFit™` Virtual Try-On (`src/core/vton_ai_engine.py`)**
   - Native `BF16` UNet (`zhengchong/CatVTON` `mix-48k-1024`) + `FP32` `sd-vae-ft-mse` + `SegFormer-B2` 18-part human parsing (`mattmdjaga/segformer_b2_clothes`) with Neck-Locked $C^1$ Anatomical Body Tailoring (`_fit_rendered_vton_to_subject_body_768`) for men, boys, women, girls, veiled (`veli`/`dupatta`) portraits, and official couples.
2. **✂️ BRIA AI `RMBG-2.0` + `RMBG-1.4` Dual-Pass Studio Matting (`src/core/bria_bg_engine.py`)**
   - Powered by **BRIA AI `RMBG-2.0`** (`BiRefNet` multi-scale portrait matting at `1024×1024` on ZeroGPU CUDA, compatible with `transformers 5.x`), edge-guided full-resolution alpha upsampling, closed-form color decontamination, and green-foliage despill.
3. **💎 Native PyTorch CUDA `CodeFormer(adain=True)` + `4x-UltraSharp` + `IdentityMarkLock™` (`src/core/neural_hub.py` & `src/core/studio_lighting_ai.py`)**
   - Adaptive `CodeFormer` fidelity weighting (`adain=True`) fused with **Frangi/Hessian `IdentityMarkLock™`** and **`OrnamentLock™`** so **100% of natural skin colour, moles, beauty spots, sindoor, bindi, nose-pins, facial scars, and expression lines** are preserved without waxy over-smoothing, up to **True 8K (`7680px`)**.
4. **💡 12-Zone `StudioAutoDirectorAI` Spatial Relighting & Instant `60 FPS` Client Live Colour (`src/core/studio_lighting_ai.py` & `app.py`)**
   - Automatically measures 12 facial lighting zones, skin ITA° angle, flash glare, under-eye/eyeglass shadow pockets, and sclera white-balance cast, finished with **Björn Ottosson Perceptual Oklab** studio color grading and instant `60 FPS` HTML5 Canvas live manual colour sliders.
5. **📱 Zero-HF-Login Phone QR Camera Shoot (`MSV8` Chunked AES-256-GCM) & 4-Digit PIN Customer Share Vault (`src/shared/mobile_qr_and_share.py`)**
   - Self-contained ISO/IEC 18004 QR generator, `MSV8` multi-chunk encrypted mobile camera relay (immune to `HTTP 413` payload limits), and 4-digit PIN customer share vault with 1x–8x Deep Zoom.

- **Full Mathematical & Cloud Architecture Blueprint:** See [`MASTER_ARCHITECTURE_AND_SCIENCE_PLAN.md`](./MASTER_ARCHITECTURE_AND_SCIENCE_PLAN.md).
