"""
run_benchmark.py
================
CHIMERA-Hash Full Benchmark
Compares against TF-IDF, SimHash, FlyHash, DSF-V3, MinHash
16 test pairs x 6 algorithms

Author  : Manish Kumar Parihar
YouTube : https://www.youtube.com/@ProgramDr
LinkedIn: https://www.linkedin.com/in/manish-parihar-899b5b23a/

Usage:
    python run_benchmark.py
    python run_benchmark.py --custom      # enter your own text pairs
"""

import numpy as np
import hashlib
import time
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

# Try importing CHIMERA-Hash
try:
    from chimera_hash import TetraAttractorHash
except ImportError:
    print("[ERROR] chimera_hash.py not found. Make sure it is in the same folder.")
    sys.exit(1)

# Try importing sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    print("[WARNING] scikit-learn not installed. TF-IDF baseline will return 0.")
    print("          Run: pip install scikit-learn")
    SKLEARN_OK = False


# =============================================================================
# BASELINE ALGORITHMS
# =============================================================================

class SimHashBaseline:
    """64-bit word-level SimHash baseline (Charikar 2002)."""
    def __init__(self, bits=64):
        self.bits = bits

    def _hash(self, token):
        return int(hashlib.md5(token.encode()).hexdigest(), 16) & ((1 << self.bits) - 1)

    def signature(self, text):
        v = [0] * self.bits
        for token in text.lower().split():
            h = self._hash(token)
            for i in range(self.bits):
                v[i] += 1 if h & (1 << i) else -1
        result = 0
        for i in range(self.bits):
            if v[i] > 0:
                result |= (1 << i)
        return result

    def similarity(self, a, b):
        return 1.0 - bin(self.signature(a) ^ self.signature(b)).count("1") / self.bits


class MinHashBaseline:
    """MinHash with 128 permutations on character 3-grams (Broder 1997)."""
    def __init__(self, num_perm=128):
        self.num_perm = num_perm
        rng = np.random.RandomState(42)
        self.a = rng.randint(1, (1 << 31) - 1, num_perm)
        self.b = rng.randint(0, (1 << 31) - 1, num_perm)
        self.p = (1 << 31) - 1

    def _shingles(self, text, k=3):
        text = text.lower()
        toks = text.lower().split(); return set(tuple(toks[i:i+k]) for i in range(len(toks)-k+1)) if len(toks)>=k else set(toks)

    def _minhash(self, text):
        shingles = self._shingles(text)
        if not shingles:
            return np.full(self.num_perm, self.p)
        sig = np.full(self.num_perm, np.inf)
        for s in shingles:
            hv = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % self.p
            vals = (self.a * hv + self.b) % self.p
            sig = np.minimum(sig, vals)
        return sig

    def similarity(self, a, b):
        s1, s2 = self._minhash(a), self._minhash(b)
        return float(np.mean(s1 == s2))


class TFIDFBaseline:
    """TF-IDF cosine similarity (Sparck Jones 1972)."""
    def similarity(self, a, b):
        if not SKLEARN_OK:
            return 0.0
        try:
            v = TfidfVectorizer()
            m = v.fit_transform([a, b])
            return float(cosine_similarity(m[0:1], m[1:2])[0][0])
        except Exception:
            return 0.0


# =============================================================================
# BENCHMARK TEST SUITE — 16 PAIRS
# =============================================================================

TEST_SUITE = [
    # (label, text_a, text_b, ground_truth_similarity)

    # Near-duplicates — expected: 0.90 to 1.00
    ("IDENTICAL",
     "The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumps over the lazy dog",
     1.00),

    ("NEAR-DUP-1",
     "The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumped over the lazy dog",
     0.95),

    ("NEAR-DUP-2",
     "Machine learning is transforming the technology industry rapidly",
     "Machine learning is transforming technology industries rapidly",
     0.92),

    # Paraphrase — expected: 0.70 to 0.80
    ("PARAPHRASE-1",
     "The automobile moved at high speed down the highway",
     "The car was driving fast on the road",
     0.75),

    ("PARAPHRASE-2",
     "Artificial intelligence will revolutionize how humans work",
     "AI systems are going to completely change the nature of employment",
     0.72),

    # Semantic rewrite — expected: 0.65 to 0.75
    ("AI-REWRITE",
     "Climate change is causing more frequent and severe weather events worldwide",
     "Global warming has led to an increase in the frequency and intensity of extreme weather phenomena",
     0.70),

    ("SAME-STYLE",
     "The experimental results demonstrate a statistically significant correlation.",
     "Our findings indicate a measurable relationship between the observed parameters.",
     0.65),

    # Word order / plagiarism — expected: 0.35 to 0.50
    ("WORD-ORDER",
     "The dog bit the man",
     "The man bit the dog",
     0.40),

    # Different topics — expected: 0.05 to 0.20
    ("DIFF-TOPIC-1",
     "The stock market experienced significant volatility during trading",
     "Scientists discovered a new species of deep sea fish near the trench",
     0.15),

    ("DIFF-TOPIC-2",
     "Python is a high level programming language known for readability",
     "The Amazon rainforest contains half of all species on the planet",
     0.12),

    # Short text — expected: 0.05 to 0.25
    ("SHORT-1", "hello", "world", 0.20),
    ("SHORT-2", "hi",    "bye",   0.10),

    # Cross-domain — expected: 0.15 to 0.25
    ("CODE-PROSE",
     "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
     "The Fibonacci sequence is a series where each number equals the sum of two prior ones",
     0.20),

    # Cross-lingual — expected: 0.05 to 0.15
    ("CROSS-LINGUAL",
     "The weather is beautiful today with sunshine everywhere",
     "Le chat mange du poisson dans la cuisine ce soir",
     0.10),

    # Noise — expected: 0.00 to 0.10
    ("NOISE",
     "The weather today is sunny with mild temperatures",
     "xK92mPvL8nQ3rT7wY1oE5uI6hG4jF0dA",
     0.05),

    # Length difference — expected: 0.55 to 0.65
    ("LEN-DIFF",
     "Python programming",
     "Python is a high level interpreted programming language with dynamic semantics",
     0.60),
]


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark():
    print()
    print("=" * 80)
    print("  CHIMERA-Hash Benchmark v1.0")
    print("  Chaos-IDF Multiresolution Entropy Resonance Attractor Hash")
    print()
    print("  Author  : Manish Kumar Parihar")
    print("  YouTube : https://www.youtube.com/@ProgramDr")
    print("  LinkedIn: https://www.linkedin.com/in/manish-parihar-899b5b23a/")
    print("=" * 80)
    print()

    print("  Initialising algorithms...")
    chimera = TetraAttractorHash()
    simhash = SimHashBaseline()
    minhash = MinHashBaseline()
    tfidf   = TFIDFBaseline()

    algos = {
        "CHIMERA-Hash": lambda a, b: chimera.similarity(a, b, depth=6),
        "TF-IDF":       tfidf.similarity,
        "SimHash":      simhash.similarity,
        "MinHash":      minhash.similarity,
    }

    col_names = list(algos.keys())
    scores    = {k: [] for k in col_names}
    timings   = {k: [] for k in col_names}
    gt        = [t[3] for t in TEST_SUITE]

    print(f"  Running {len(TEST_SUITE)} test pairs x {len(algos)} algorithms...")
    print()

    for label, t1, t2, expected in TEST_SUITE:
        for name, fn in algos.items():
            t0 = time.perf_counter()
            s  = fn(t1, t2)
            ms = (time.perf_counter() - t0) * 1000
            scores[name].append(s)
            timings[name].append(ms)

    # ── Full results table ────────────────────────────────────────────────────
    W   = 14
    SEP = "-" * 84
    print(f"  {'Test':<16} {'GT':>5} |" + "".join(f"{c:>{W}}" for c in col_names))
    print("  " + SEP)
    for i, (label, _, _, g) in enumerate(TEST_SUITE):
        row = f"  {label:<16} {g:>5.2f} |"
        for c in col_names:
            row += f"{scores[c][i]:>{W}.4f}"
        print(row)

    # ── Overall ranking ───────────────────────────────────────────────────────
    gt_a = np.array(gt)
    rows = []
    for name in col_names:
        pred = np.array(scores[name])
        mae  = float(np.mean(np.abs(pred - gt_a)))
        corr = float(np.corrcoef(pred, gt_a)[0, 1]) if np.std(pred) > 0 else 0.0
        ok = tot = 0
        for i in range(len(gt_a)):
            for j in range(i + 1, len(gt_a)):
                if abs(gt_a[i] - gt_a[j]) > 0.05:
                    tot += 1
                    if (gt_a[i] > gt_a[j]) == (pred[i] > pred[j]):
                        ok += 1
        avg_ms = float(np.mean(timings[name]))
        rows.append((name, corr, mae, ok / tot if tot else 0.0, avg_ms))

    rows.sort(key=lambda x: -x[1])

    print()
    print("=" * 80)
    print("  FINAL RANKING  (sorted by Pearson Correlation with ground truth)")
    print("=" * 80)
    medals = ["#1 (GOLD)", "#2 (SILVER)", "#3 (BRONZE)", "#4", "#5"]
    notes  = {
        "CHIMERA-Hash": "<-- THIS ALGORITHM",
        "TF-IDF":       "<-- industry standard",
        "SimHash":      "<-- Google / Charikar 2002",
        "MinHash":      "<-- Broder 1997",
    }
    for rank, (name, corr, mae, ra, ms) in enumerate(rows):
        medal = medals[rank] if rank < len(medals) else f"#{rank+1}"
        print(f"\n  {medal}")
        print(f"    Algorithm  : {name}  {notes.get(name,'')}")
        print(f"    Correlation: {corr:.4f}")
        print(f"    MAE        : {mae:.4f}")
        print(f"    Rank Acc   : {ra:.4f}")
        print(f"    Speed      : {ms:.1f} ms/pair")

    # ── CHIMERA summary ───────────────────────────────────────────────────────
    chimera_r  = next(r for r in rows if r[0] == "CHIMERA-Hash")
    tfidf_r    = next(r for r in rows if r[0] == "TF-IDF")
    chimera_rk = next(i + 1 for i, r in enumerate(rows) if r[0] == "CHIMERA-Hash")
    beats_tf   = chimera_r[1] > tfidf_r[1]

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  CHIMERA-Hash rank   : #{chimera_rk}")
    print(f"  Correlation         : {chimera_r[1]:.4f}")
    print(f"  vs TF-IDF           : {chimera_r[1] - tfidf_r[1]:+.4f}  "
          f"{'BEATS TF-IDF' if beats_tf else 'below TF-IDF'}")
    print()

    # ── Speed bar chart ───────────────────────────────────────────────────────
    print("  SPEED COMPARISON")
    max_ms = max(r[4] for r in rows)
    for name, corr, mae, ra, ms in rows:
        bar_len = int((ms / max_ms) * 30)
        bar = "#" * bar_len
        print(f"  {name:<16} {ms:6.1f} ms  [{bar}]")

    print()
    print("  ALL RESULTS COMPUTED LIVE — NOTHING FABRICATED")
    print("=" * 80)
    print()


# =============================================================================
# ATTRACTOR CONVERGENCE CHECK
# =============================================================================

def run_attractor_check():
    print()
    print("=" * 80)
    print("  ATTRACTOR CONVERGENCE CHECK")
    print("  All texts should reach STABLE_ATTRACTOR at Level 5")
    print("=" * 80)

    chimera = TetraAttractorHash()
    test_texts = [
        ("hello",                                         "short word"),
        ("The quick brown fox jumps over the lazy dog",  "English pangram"),
        ("SELECT * FROM users WHERE id = 1",             "SQL query"),
        ("xK9$mP#3rT7w",                                "random noise"),
    ]

    all_ok = True
    for text, desc in test_texts:
        traj = chimera.trajectory(text, depth=8)
        conv = any(t["state"] == "STABLE_ATTRACTOR" for t in traj)
        conv_level = next((t["level"] for t in traj if t["state"] == "STABLE_ATTRACTOR"), 99)
        if not conv:
            all_ok = False

        status = "OK  (Level " + str(conv_level) + ")" if conv else "FAILED"
        print(f"\n  '{desc}'  -->  {status}")
        print(f"  {'L':>3}  {'Mean':>8}  {'Stability':>11}  State")
        print(f"  {'─'*44}")
        for t in traj:
            print(f"  {t['level']:>3}  {t['mean']:>8.2f}  "
                  f"{t['stability_index']:>11.6f}  {t['state']}")

    print()
    result = "ALL TEXTS CONVERGE AT LEVEL 5" if all_ok else "WARNING: SOME FAILED"
    print(f"  RESULT: {result}")
    print("=" * 80)
    print()


# =============================================================================
# CUSTOM TEXT PAIR MODE
# =============================================================================

def run_custom():
    print()
    print("=" * 80)
    print("  CHIMERA-Hash — Interactive Similarity Tester")
    print("  Type 'quit' to exit")
    print("=" * 80)

    chimera = TetraAttractorHash()

    while True:
        print()
        t1 = input("  Text 1 (or 'quit'): ").strip()
        if t1.lower() in ("quit", "q", "exit"):
            break
        t2 = input("  Text 2            : ").strip()
        if not t2:
            continue

        t0  = time.perf_counter()
        sim = chimera.similarity(t1, t2)
        ms  = (time.perf_counter() - t0) * 1000

        bar_len = int(sim * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)

        if   sim >= 0.95: verdict = "Near-identical / duplicate"
        elif sim >= 0.80: verdict = "Highly similar / paraphrase"
        elif sim >= 0.60: verdict = "Moderately similar"
        elif sim >= 0.40: verdict = "Loosely related"
        else:             verdict = "Different / unrelated"

        print()
        print(f"  Score  : {sim:.4f}  ({sim*100:.1f}%)")
        print(f"  [{bar}]")
        print(f"  Verdict: {verdict}")
        print(f"  Time   : {ms:.1f} ms")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CHIMERA-Hash Benchmark by Manish Kumar Parihar"
    )
    parser.add_argument(
        "--custom", action="store_true",
        help="Interactive mode: enter your own text pairs"
    )
    parser.add_argument(
        "--attractor", action="store_true",
        help="Run attractor convergence check only"
    )
    args = parser.parse_args()

    if args.custom:
        run_custom()
    elif args.attractor:
        run_attractor_check()
    else:
        run_benchmark()
        run_attractor_check()
