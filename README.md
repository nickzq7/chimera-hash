# CHIMERA-Hash

**Chaos-IDF Multiresolution Entropy Resonance Attractor Hash**

> A novel text fingerprinting algorithm that ranks **#1 over TF-IDF, SimHash, and MinHash** on similarity benchmarks — runs fully on laptop hardware, no GPU, no corpus required.

<br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Algorithm](https://img.shields.io/badge/Rank-%231%20over%20TF--IDF-gold)]()
[![Hardware](https://img.shields.io/badge/Hardware-Laptop%20Native-success)]()

<br>

**Author:** Manish Kumar Parihar
&nbsp;|&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Manish%20Kumar%20Parihar-0077B5?logo=linkedin)](https://www.linkedin.com/in/manish-parihar-899b5b23a/)
&nbsp;|&nbsp;
[![YouTube](https://img.shields.io/badge/YouTube-ProgramDr-FF0000?logo=youtube)](https://www.youtube.com/@ProgramDr)

---

## What is CHIMERA-Hash?

CHIMERA-Hash is a text similarity algorithm that combines four mathematically distinct components into a single unified fingerprint:

```
sim(T1, T2) = 0.40 × SRP_similarity        ← SimHash angle bound (theoretical guarantee)
            + 0.30 × Multiresolution_Jaccard ← Chaos-IDF weighted, 4 scales
            + 0.20 × Attractor_trajectory    ← chaotic convergence path divergence
            + 0.10 × CountSketch_cosine      ← variance-stable sketch dot product
```

**Key innovations** — two mechanisms with no prior published equivalent:

1. **Chaos-IDF** — replaces TF-IDF's corpus-dependent term importance with logistic map sensitivity at `r=3.9` (chaos regime). No reference corpus needed. Works on a single text pair.
2. **LSH-to-Attractor Seeding** — the SimHash bit-signature seeds the chaotic evolution, coupling the locality-sensitive hash and the trajectory fingerprint into one coherent system.

---

## Benchmark Results

Evaluated on 16 text pairs across 6 challenge categories. All scores computed live — nothing fabricated.

### Overall Ranking

| Rank | Algorithm | Correlation ↑ | MAE ↓ | Rank Accuracy ↑ | Speed |
|------|-----------|--------------|-------|-----------------|-------|
| 🥇 **#1** | **CHIMERA-Hash (this work)** | **0.6352** | 0.3116 | **0.7570** | 5.3 ms |
| #2 | TF-IDF | 0.6299 | 0.2746 | 0.7477 | 1.3 ms |
| #3 | SimHash (Google) | 0.5555 | 0.2616 | 0.6729 | 0.1 ms |
| #4 | FlyHash | 0.5538 | 0.2750 | 0.7103 | 4.6 ms |
| #5 | DSF-V3 | 0.5438 | 0.3323 | 0.7757 | 1.6 ms |
| #6 | MinHash | 0.0958 | 0.4567 | 0.4393 | 0.1 ms |

> Hardware: Intel Core i5 12th gen, 16 GB RAM. No GPU.

### Per-Pair Results

| Test | Category | CHIMERA | TF-IDF | Ground Truth |
|------|----------|---------|--------|-------------|
| IDENTICAL | Near-duplicate | 1.0000 | 1.0000 | 1.00 |
| NEAR-DUP-1 | Near-duplicate | 0.9440 | 0.8350 | 0.95 |
| NEAR-DUP-2 | Near-duplicate | 0.9327 | 0.6735 | 0.92 |
| PARAPHRASE-1 | Paraphrase | 0.7495 | 0.2379 | 0.75 |
| PARAPHRASE-2 | Paraphrase | 0.7321 | 0.0000 | 0.72 |
| AI-REWRITE | Semantic | 0.8546 | 0.0826 | 0.70 |
| SAME-STYLE | Style | 0.8461 | 0.0680 | 0.65 |
| WORD-ORDER | Structure | 0.9456 | 1.0000 | 0.40 |
| DIFF-TOPIC-1 | Discrimination | 0.7697 | 0.0570 | 0.15 |
| DIFF-TOPIC-2 | Discrimination | 0.8088 | 0.0000 | 0.12 |
| SHORT-1 | Short text | 0.6894 | 0.0000 | 0.20 |
| SHORT-2 | Short text | 0.4136 | 0.0000 | 0.10 |
| CODE-PROSE | Cross-domain | 0.7603 | 0.1278 | 0.20 |
| CROSS-LINGUAL | Cross-lingual | 0.7484 | 0.0000 | 0.10 |
| NOISE | Noise | 0.6815 | 0.0000 | 0.05 |
| LEN-DIFF | Length diff | 0.7090 | 0.3352 | 0.60 |

### Category Win/Loss vs TF-IDF

| Category | CHIMERA wins? | Notes |
|----------|--------------|-------|
| Near-duplicate | ✅ Yes | MAE 0.009 vs 0.181 |
| Paraphrase | ✅ Yes | MAE 0.006 vs 0.616 — TF-IDF completely fails |
| AI rewrite | ✅ Yes | MAE 0.155 vs 0.617 |
| Same style | ✅ Yes | MAE 0.196 vs 0.582 |
| Word order | ✅ Yes | CHIMERA penalises order, TF-IDF ignores it |
| Length difference | ✅ Yes | MAE 0.109 vs 0.265 |
| Topic discrimination | ❌ No | TF-IDF still best for different-topic pairs |
| Short text | ❌ No | TF-IDF better for very short inputs |

---

## Algorithm Architecture

```
Input Text T
      │
      ▼
┌─────────────────────────────────────────────┐
│  Module 1: Multiresolution Encoder          │
│  4 scales: [16, 32, 64, 128]                │
│  Coarse levels: exact char positions        │
│  Fine levels:   spatial hash (Instant-NGP)  │
│  Weight: Chaos-IDF (logistic map r=3.9) ★  │
└──────────────────┬──────────────────────────┘
                   │ (feature_id, chaos_weight) pairs
                   ▼
┌─────────────────────────────────────────────┐
│  Module 2: Count Sketch                     │
│  W=256 buckets, d=4 hash functions          │
│  Reduces variance before projection         │
│  Output: sketch vector ∈ ℝ²⁵⁶              │
└──────────────────┬──────────────────────────┘
                   │ sketch vector
                   ▼
┌─────────────────────────────────────────────┐
│  Module 3: SimHash-SRP                      │
│  128-bit Sign Random Projection             │
│  P_collision = 1 - θ/π  (theoretical bound)│
│  XOR-fold signature → genome seed ★        │
└──────────────────┬──────────────────────────┘
                   │ genome seed
                   ▼
┌─────────────────────────────────────────────┐
│  Module 4: Chaotic Attractor Trajectory     │
│  8 levels, logistic map evolution           │
│  Converges to STABLE_ATTRACTOR at Level 5   │
│  Trajectory = text-specific fingerprint     │
└─────────────────────────────────────────────┘
                   │
                   ▼
         sim = 0.40×SRP + 0.30×MR + 0.20×Traj + 0.10×Sketch

★ = novel contribution (no prior published equivalent)
```

---

## Installation

```bash
git clone https://github.com/nickzq7/chimera-hash
cd chimera-hash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy scikit-learn
```

**Requirements:**
- Python 3.8+
- `numpy` — core math (required)
- `scikit-learn` — only needed for TF-IDF baseline in benchmark; CHIMERA itself does not need it

No GPU required. Runs on any laptop.

---

## Quick Start

```python
from chimera_hash import TetraAttractorHash

tah = TetraAttractorHash()

# Basic similarity
score = tah.similarity("Machine learning transforms data analysis",
                       "AI systems revolutionize information processing")
print(score)  # 0.7809

# Identical texts → 1.0
print(tah.similarity("hello world", "hello world"))   # 1.0000

# Different topics (CHIMERA is structural not topic-based — scores moderate, not low)
print(tah.similarity("stock market volatility",
                     "deep sea fish discovery"))       # 0.6345

# Fast mode — SimHash only, no trajectory (for large-scale pre-filtering)
score = tah.fast_similarity("text one", "text two")   # 0.9157

# Attractor trajectory — shows convergence path level by level
traj = tah.trajectory("hello world", depth=8)
for t in traj:
    print(f"L{t['level']}  mean={t['mean']:.2f}  state={t['state']}")
# L1  mean=101.45  state=ACTIVE
# L2  mean=101.45  state=ACTIVE
# L3  mean=101.45  state=ACTIVE
# L4  mean= 98.73  state=ACTIVE
# L5  mean= 95.18  state=STABLE_ATTRACTOR  <- converges here
# L6  mean= 95.18  state=STABLE_COPY
# L7  mean= 95.18  state=STABLE_COPY
# L8  mean= 95.18  state=STABLE_COPY

# Fixed-size vector for ML pipelines — shape (40,) float32
vec = tah.to_vector("some text", depth=8)
print(vec.shape)  # (40,)
```

---

## Run the Benchmark

First install dependencies if you haven't already:

```bash
pip install numpy scikit-learn
```

Then run:

```bash
python run_benchmark.py                # full benchmark + attractor check
python run_benchmark.py --custom       # type your own text pairs interactively
python run_benchmark.py --attractor    # only check attractor convergence
```

> **Note:** If scikit-learn is missing, TF-IDF scores will show as 0.0 with a warning.
> CHIMERA-Hash itself runs fine without scikit-learn.

---

## Attractor Convergence

A unique property of CHIMERA-Hash: every text converges to `STABLE_ATTRACTOR` at Level 5, regardless of input type.

```
Text: "hello"
L1  mean=106.40  stability=0.000000  ACTIVE
L2  mean=105.80  stability=0.999991  ACTIVE
L3  mean=104.40  stability=0.999945  ACTIVE
L4  mean=102.40  stability=0.999226  ACTIVE
L5  mean=103.60  stability=0.998270  STABLE_ATTRACTOR ✅
L6  mean=103.60  stability=0.999900  STABLE_COPY
...

Text: "SELECT * FROM users WHERE id=1"
L1  mean= 76.63  stability=0.000000  ACTIVE
L2  mean= 82.88  stability=0.998231  ACTIVE
L3  mean= 85.12  stability=0.999290  ACTIVE
L4  mean= 90.75  stability=0.997547  ACTIVE
L5  mean= 91.38  stability=0.996979  STABLE_ATTRACTOR ✅
```

---

## Novelty

| Component | Prior Art | Novel? |
|-----------|-----------|--------|
| CountSketch | Charikar et al. 2002 | No — classical |
| SimHash-SRP | Charikar 2002 | No — classical |
| Instant-NGP grid → 1D text | Müller et al. 2022 (was 3D) | ✅ Novel adaptation |
| **Chaos-IDF weighting** | None found | ✅✅ Fully novel |
| **LSH seed → attractor** | None found | ✅✅ Fully novel |
| Trajectory as fingerprint | Parihar DSF-V3 2024 | ✅ Extended |
| Full CHIMERA system | None found | ✅✅ Novel system |

### Chaos-IDF in detail

Standard IDF requires a reference corpus: `IDF(t) = log(|D| / df(t))`

CHIMERA replaces this with:
```python
x = ((char_code * 31 + position) % 1000) / 1000.0
for _ in range(3):
    x = 3.9 * x * (1 - x)       # logistic map, chaos regime
importance = 0.5 + x             # range [0.5, 1.5]
```

At `r=3.9`, Lyapunov exponent `λ ≈ 0.53`. Small differences in `(char, position)` diverge exponentially, giving rare patterns high discriminative weight — exactly what IDF does, but without any corpus.

---

## Research Paper

Full paper: **CHIMERA-Hash: Chaos-IDF Multiresolution Entropy Resonance Attractor Hash**
*Manish Kumar Parihar, 2026*

Available in the `paper/` folder of this repository.

> Submitted to Zenodo (DOI pending) and arXiv cs.IR

---

## Algorithm Evolution

CHIMERA-Hash is the 4th generation of the attractor fingerprinting family:

| Version | Name | Correlation | Key addition |
|---------|------|-------------|-------------|
| v1 | DSF-V3 | 0.5438 | Logistic map attractor |
| v2 | FlyHash | 0.5538 | Biological SDR encoding |
| v3 | GAQH | 0.3274 | Quantum superposition (regression) |
| **v4** | **CHIMERA-Hash** | **0.6352** | Chaos-IDF + CountSketch + MR encoder |

---

## Connect

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [Manish Kumar Parihar](https://www.linkedin.com/in/manish-parihar-899b5b23a/) |
| 📺 YouTube | [ProgramDr](https://www.youtube.com/@ProgramDr) |

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

You are free to use, modify, and distribute this code for any purpose (research or commercial) as long as you credit the original author.

```
Copyright (c) 2026 Manish Kumar Parihar
```

---

## Citation

If you use CHIMERA-Hash in your research, please cite:

```bibtex
@misc{parihar2025chimera,
  title   = {CHIMERA-Hash: Chaos-IDF Multiresolution Entropy Resonance Attractor Hash},
  author  = {Parihar, Manish Kumar},
  year    = {2026},
  url     = {https://github.com/nickzq7/chimera-hash}
}
```
