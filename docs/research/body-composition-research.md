# Research (L3): privacy-preserving body-fat & muscle-definition estimation
Date: 2026-11-01 · Scope: can PhysiqueGap estimate body-fat % and visible muscle definition from body images, privately, with free tooling and a labelled dataset?

## 1. Framing
Three sub-questions:
1. What methods estimate body-fat % (BF%) from images or body shape, and how accurate are they vs. DXA?
2. Is there a public labelled dataset for BF% or muscle definition from images?
3. How do we keep it private (images never leave the device) and stay inside free LLM limits?

## 2. Findings

### 2.1 Image -> BF% works in industry, but the data is private
- Amazon's smartphone method (front + back photos in an A-pose, background removed, CNN pre-trained on shape then fine-tuned on DXA) reached MAE 2.34% vs DXA on 134 adults, beating consumer and professional BIA (3.1–5.9% MAE) [1]. Images are processed **in the cloud** and the data is not public [1].
- A 2026 study ("PhotoScan") pre-trained on 35,323 UK Biobank participants, fine-tuned on 677 DXA-scanned people, and reported ~2.1% MAE vs DXA on two cohorts [2]. No public data release stated [2].
- An open-source on-device model (DenseNet-201, ExecuTorch on Android) estimates **BMI** (not BF%) from 85k photos, MAPE 7.9%; the training data is proprietary and can't be released for consent reasons [3]. It proves on-device CNN inference on phones is practical [3].

### 2.2 No public labelled BF% / definition image dataset exists
- A 2025 paper states directly that no public datasets exist for vision-based BF% estimation; they scraped 282 Reddit images, labels self-reported (~43% with DEXA references), ResNet MAE 3.34% [4].
- A GitHub project scraped 1,022 Reddit images (men only), labels = **perceived** BF% from comments, ResNet RMSE ≈ 3.2 points, no licence and no consent handling [5].
- Muscle-definition: psychology uses figure-rating scales (e.g., Body Image Matrix of Thinness and Muscularity) that rate fat and muscularity on separate axes [6] — useful as a **label scheme**, but they are drawings, not a training set.
- Conclusion: any real-photo labelled set we can legally use must be **collected with consent** or be **synthetic**.

### 2.3 Shape-only BF% is surprisingly good and needs no image texture
- Relative Fat Mass: RFM = 64 − 20·(height/waist) + 12·sex (sex = 0 men, 1 women), validated on NHANES (n = 3,456) vs DXA [7].
  - R² vs DXA BF%: men 0.75 (BMI 0.61), women 0.69 (BMI 0.65); ~89–92% of estimates within 20% of DXA [7].
  - Weaknesses: less accurate in lean people, declines with age, DXA itself biased, only US ethnic groups tested [7].
- We already estimate waist girth from front + side silhouettes (BodyM gives ground-truth waist to calibrate it). So RFM is free once the silhouette pipeline works.
- Implication: RFM is the honest numeric anchor; visual cues refine it, they don't replace it.

### 2.4 Synthetic data is the realistic path to definition labels
- MakeHuman assets (base mesh, targets, proxies) are CC0, so exported bodies carry no licence restriction [8]. It has muscle and weight sliders — the slider values become labels.
- Large synthetic human sets (e.g., SynBody, 1.2M images) vary shape through SMPL-X betas but don't provide fat/muscle labels [9] -> not directly usable.
- Domain gap (renders vs. real skin, lighting) is the main risk; standard mitigation is heavy augmentation + a small real fine-tune/validation set.

### 2.5 Energy estimates
- Among RMR equations, Mifflin-St Jeor was most accurate: off by > 10% for 22% of subjects vs 33% for Harris-Benedict [10]. Katch-McArdle needs lean mass (which we'd have only as an estimate) -> use Mifflin by default.

### 2.6 Free LLMs — limits and privacy
- Groq free tier (Sept 2026): gpt-oss-120b/20b, Qwen 3.6/3.8 27B (preview); ~30 RPM, 1,000 req/day, 8k tokens/min; Llama models removed from free tier Aug 2026 [11].
- OpenRouter free (Oct 2026): ~17 free models incl. vision ones (Gemma 4, Nemotron VL); 20 req/min [12]. Daily cap 50/day without ≥ 10 credits, 1,000/day after [13].
- Of 19 free OpenRouter text models, 9 hosts may train on prompts and only 4 have zero retention [13]. -> **Never send images to free hosted LLMs**; send only numbers + text, and prefer no-train endpoints.

### 2.7 Legal (India)
- DPDP Rules 2025: notices must itemise data and purposes, withdrawal must be as easy as consent; children's data requires verifiable parental consent; most obligations apply from mid-2027 [14]. Our 18+ gate + no server storage keeps us simple.

## 3. Where sources disagree / open questions
- **Image vs shape:** industry CNNs report ~2.1–2.3% MAE [1][2]; RFM is validated by R² not MAE [7], so direct comparison is apples-to-oranges. We must measure both on the same validation set.
- **Perceived vs measured labels:** Reddit-style datasets [4][5] learn what people *think* BF% looks like, not DXA truth.
- **Synthetic -> real transfer** for definition is unproven for this task; treat as experiment (ADR-014 gate).
- **Lean athletes** are where RFM is weakest [7] and where definition cues are strongest — the fusion is most valuable there and least validated.

## 4. Bottom line (what we build)
1. **BF% v1 = RFM** from silhouette-estimated waist + user height + sex, shown as a range. Optional tape-measure input overrides waist.
2. **Definition score v2 = on-device CNN** (MobileNetV3-Small, ONNX Runtime Web) trained on MakeHuman synthetic renders with muscle/fat slider labels, validated on a small consented real set. Outputs 1–5 per region; only scores leave the device.
3. **Fusion:** definition score nudges BF% within the RFM range; never outputs a single overconfident number.
4. **Energy:** Mifflin-St Jeor × activity factor -> cut/bulk/recomp target ranges with safety floors.
5. **LLMs:** text-only, Groq primary, OpenRouter fallback on no-train endpoints.
6. **Do not** use scraped Reddit images (no consent, ToS risk, perceived labels, male-only).

## References
1. Majmudar et al., Smartphone camera based assessment of adiposity (medRxiv 2021 / npj Digit Med 2022) — https://medrxiv.org/content/10.1101/2021.06.10.21258595v1.full (peer-reviewed version: https://pmc.ncbi.nlm.nih.gov/articles/PMC9243018)
2. Beyond BMI: Smartphone Body Composition Phenotyping for Cardiometabolic Risk Assessment (arXiv 2603.27017) — https://arxiv.org/abs/2603.27017v2
3. Digital Scale: Open-Source On-Device BMI Estimation (arXiv 2508.20534) — https://arxiv.org/html/2508.20534v1
4. Multimodal AI for Body Fat Estimation: CV and Anthropometry with DEXA Benchmarks (arXiv 2511.17576) — https://arxiv.org/html/2511.17576v1
5. Body-Fat-Regression-from-Reddit-Image-Dataset (GitHub) — https://github.com/Kayse-Johnson/Body-Fat-Regression-from-Reddit-Image-Dataset
6. Arkenau et al., Body Image Matrix of Thinness and Muscularity — Male Bodies (J Clin Psychol 2020) — https://onlinelibrary.wiley.com/doi/full/10.1002/jclp.22933 (page not retrievable; cited from search listing only)
7. Woolcott & Bergman, Relative fat mass (RFM) (Sci Rep 2018) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6054651
8. MakeHuman FAQ — licence change 2020 — https://static.makehumancommunity.org/oldsite/faq/what_changed_regarding_the_license_in_2020.html
9. SynBody (ICCV 2023) — https://arxiv.org/html/2303.17368v2
10. Validation of several established equations for RMR in obese and nonobese people (PubMed 12963943) — https://pubmed.ncbi.nlm.nih.gov/12963943/
11. Groq Free Tier 2026 (third-party summary of Groq docs) — https://klymentiev.com/blog/groq-pricing
12. OpenRouter Free Models list (CostGoat, Oct 2026) — https://costgoat.com/pricing/openrouter-free-models
13. TrueFoundry, OpenRouter free models: what's actually free — https://www.truefoundry.com/blog/openrouter-free-models-whats-actually-free-and-what-it-costs-you
14. The Leaflet, India's DPDP Rules 2025: a detailed reading — https://theleaflet.in/amp/story/explainer/indias-new-digital-personal-data-protection-rules-2025-a-detailed-reading
15. BodyM dataset — https://registry.opendata.aws/bodym/
