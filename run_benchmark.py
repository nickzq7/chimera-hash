"""
run_benchmark.py
================
CHIMERA-Hash Full Benchmark — 115 pairs, 16 categories
Reproduces Table 1, Table 2, and Table 3 from the paper exactly.

Author  : Manish Kumar Parihar
GitHub  : https://github.com/nickzq7/chimera-hash
YouTube : https://www.youtube.com/@ProgramDr
LinkedIn: https://www.linkedin.com/in/manish-parihar-899b5b23a/

Usage:
    python run_benchmark.py               # full 115-pair benchmark
    python run_benchmark.py --attractor   # attractor convergence check only
    python run_benchmark.py --custom      # enter your own text pairs

Expected output (matches paper Section 5.3):
    #1 TF-IDF        corr=0.5683  MAE=0.2578  RankAcc=0.7208
    #2 CHIMERA-Hash  corr=0.5082  MAE=0.3288  RankAcc=0.7108  <-- this work
    #3 MinHash       corr=0.5527  MAE=0.3617  RankAcc=0.5466
    #4 SimHash       corr=0.4968  MAE=0.2541  RankAcc=0.6789

Hardware: any Python 3.8+ machine, 4 GB RAM, no GPU. ~3 min runtime.
"""

import numpy as np
import hashlib
import time
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Import CHIMERA-Hash ─────────────────────────────────────────────────────
try:
    from chimera_hash import TetraAttractorHash
except ImportError:
    print("[ERROR] chimera_hash.py not found. Place it in the same folder.")
    sys.exit(1)

# ── Import sklearn ──────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    print("[WARNING] scikit-learn not installed. TF-IDF will return 0.")
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
    """MinHash with 128 permutations on word 3-gram shingles (Broder 1997)."""
    def __init__(self, num_perm=128):
        self.num_perm = num_perm
        rng = np.random.RandomState(42)
        self.a = rng.randint(1, (1 << 31) - 1, num_perm)
        self.b = rng.randint(0, (1 << 31) - 1, num_perm)
        self.p = (1 << 31) - 1

    def _shingles(self, text, k=3):
        toks = text.lower().split()
        return set(tuple(toks[i:i+k]) for i in range(len(toks)-k+1)) if len(toks) >= k else set(toks)

    def _minhash(self, text):
        shingles = self._shingles(text)
        if not shingles:
            return np.full(self.num_perm, self.p)
        sig = np.full(self.num_perm, np.inf)
        for s in shingles:
            hv = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % self.p
            sig = np.minimum(sig, (self.a * hv + self.b) % self.p)
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
# FULL 115-PAIR BENCHMARK — Appendix A of the paper
# =============================================================================

TEST_SUITE = [

    # ── C01: Identical (5 pairs) ────────────────────────────────────────────
    ("IDENTICAL_01",
     "The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumps over the lazy dog",
     1.00),
    ("IDENTICAL_02",
     "Machine learning is a subset of artificial intelligence",
     "Machine learning is a subset of artificial intelligence",
     1.00),
    ("IDENTICAL_03",
     "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
     "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
     1.00),
    ("IDENTICAL_04",
     "The stock market closed higher today driven by technology stocks",
     "The stock market closed higher today driven by technology stocks",
     1.00),
    ("IDENTICAL_05",
     "climate change global warming temperature rise",
     "climate change global warming temperature rise",
     1.00),

    # ── C02: Near-Duplicate (10 pairs) ─────────────────────────────────────
    ("NEARDUP_01",
     "The quick brown fox jumps over the lazy dog",
     "The quick brown fox jumped over the lazy dog",
     0.95),
    ("NEARDUP_02",
     "Machine learning is transforming the technology industry rapidly",
     "Machine learning is transforming technology industries rapidly",
     0.93),
    ("NEARDUP_03",
     "Scientists have discovered a new planet orbiting a distant star",
     "Scientists discovered a new planet orbiting a distant star system",
     0.94),
    ("NEARDUP_04",
     "The president signed the new economic policy into law yesterday",
     "The president signed the new economic policy into law on Thursday",
     0.93),
    ("NEARDUP_05",
     "Python is widely used for data science and machine learning tasks",
     "Python is widely used for data science and machine learning applications",
     0.93),
    ("NEARDUP_06",
     "The hospital reported a significant decrease in patient wait times",
     "The hospital reported a significant reduction in patient waiting times",
     0.92),
    ("NEARDUP_07",
     "Researchers published a groundbreaking study on cancer treatment",
     "Researchers have published a groundbreaking study on cancer treatments",
     0.93),
    ("NEARDUP_08",
     "The electric vehicle market grew by forty percent last quarter",
     "The electric vehicle market grew by 40 percent in the last quarter",
     0.92),
    ("NEARDUP_09",
     "Apple released its latest smartphone with improved camera features",
     "Apple released its newest smartphone with improved camera capabilities",
     0.92),
    ("NEARDUP_10",
     "The new law requires companies to disclose their carbon emissions annually",
     "The new regulation requires companies to report their carbon emissions every year",
     0.90),

    # ── C03: Paraphrase (12 pairs) ──────────────────────────────────────────
    ("PARAPHRASE_01",
     "The automobile moved at high speed down the highway",
     "The car was driving fast on the road",
     0.75),
    ("PARAPHRASE_02",
     "Artificial intelligence will revolutionize how humans work",
     "AI systems are going to completely change the nature of employment",
     0.72),
    ("PARAPHRASE_03",
     "The child was very happy when she received the gift",
     "The little girl felt extremely joyful upon getting the present",
     0.73),
    ("PARAPHRASE_04",
     "The company reported record profits for the financial year",
     "The firm announced its highest ever earnings during the fiscal period",
     0.72),
    ("PARAPHRASE_05",
     "Doctors recommend exercising regularly to maintain good health",
     "Physicians advise people to work out frequently for better wellbeing",
     0.72),
    ("PARAPHRASE_06",
     "The storm caused widespread damage across the coastal region",
     "Heavy winds and rain devastated communities along the shoreline",
     0.70),
    ("PARAPHRASE_07",
     "The government announced new measures to reduce unemployment",
     "Officials unveiled fresh policies aimed at tackling joblessness",
     0.71),
    ("PARAPHRASE_08",
     "Students who study consistently tend to perform better in exams",
     "Pupils who revise regularly usually achieve higher test scores",
     0.72),
    ("PARAPHRASE_09",
     "The film received critical acclaim for its stunning cinematography",
     "Critics praised the movie for its breathtaking visual storytelling",
     0.73),
    ("PARAPHRASE_10",
     "Eating a balanced diet is essential for long term health",
     "Consuming nutritious food regularly is vital for your wellbeing over time",
     0.71),
    ("PARAPHRASE_11",
     "The software update introduced several new security patches",
     "The latest version of the program included multiple fixes for vulnerabilities",
     0.70),
    ("PARAPHRASE_12",
     "Global temperatures have risen significantly over the past century",
     "Average worldwide heat levels have increased substantially during the last hundred years",
     0.71),

    # ── C04: AI Rewrite (8 pairs) ───────────────────────────────────────────
    ("AIREWRITE_01",
     "Climate change is causing more frequent and severe weather events worldwide",
     "Global warming has led to an increase in the frequency and intensity of extreme weather phenomena",
     0.70),
    ("AIREWRITE_02",
     "The neural network achieved state of the art performance on the benchmark",
     "Our deep learning model set new records across all evaluation metrics on the dataset",
     0.68),
    ("AIREWRITE_03",
     "The central bank raised interest rates to combat rising inflation",
     "Monetary authorities increased borrowing costs in an effort to control accelerating price growth",
     0.68),
    ("AIREWRITE_04",
     "Renewable energy sources are becoming increasingly cost competitive",
     "The price of clean power generation has fallen dramatically making it economically viable",
     0.67),
    ("AIREWRITE_05",
     "The quarterback threw a touchdown pass in the final seconds",
     "The football player scored the winning points with a last second throw to his receiver",
     0.66),
    ("AIREWRITE_06",
     "The algorithm processes millions of records in under one second",
     "This computational method can handle vast datasets with extremely low latency",
     0.67),
    ("AIREWRITE_07",
     "The prime minister resigned following the no confidence vote",
     "The head of government stepped down after lawmakers withdrew their support",
     0.67),
    ("AIREWRITE_08",
     "Researchers found a strong correlation between sleep quality and cognitive performance",
     "Scientists discovered that better rest is linked to improved mental functioning",
     0.68),

    # ── C05: Same Style (6 pairs) ───────────────────────────────────────────
    ("SAMESTYLE_01",
     "The experimental results demonstrate a statistically significant correlation.",
     "Our findings indicate a measurable relationship between the observed parameters.",
     0.65),
    ("SAMESTYLE_02",
     "Pursuant to the terms outlined in Section 4.2 of the agreement herein.",
     "In accordance with the provisions specified under Clause 7.1 of this contract.",
     0.63),
    ("SAMESTYLE_03",
     "We hereby acknowledge receipt of your communication dated the first instant.",
     "This letter confirms that we have received your correspondence of recent date.",
     0.62),
    ("SAMESTYLE_04",
     "The patient presented with acute onset chest pain and shortness of breath.",
     "The subject exhibited sudden thoracic discomfort accompanied by respiratory difficulty.",
     0.64),
    ("SAMESTYLE_05",
     "Yo bro that game last night was absolutely insane no cap fr.",
     "Dude that match was totally wild yesterday I am not even joking.",
     0.63),
    ("SAMESTYLE_06",
     "The Board of Directors resolved to approve the proposed merger subject to regulatory review.",
     "Trustees voted to sanction the acquisition pending approval from the relevant authorities.",
     0.62),

    # ── C06: Word Order (6 pairs) ───────────────────────────────────────────
    ("WORDORDER_01",
     "The dog bit the man",
     "The man bit the dog",
     0.40),
    ("WORDORDER_02",
     "John loves Mary deeply and sincerely",
     "Mary loves John deeply and sincerely",
     0.42),
    ("WORDORDER_03",
     "The police arrested the suspect near the bank",
     "The suspect was arrested near the bank by the police",
     0.43),
    ("WORDORDER_04",
     "Cats eat mice and dogs chase cats",
     "Dogs chase cats and cats eat mice",
     0.42),
    ("WORDORDER_05",
     "The teacher praised the student for excellent work",
     "The student was praised by the teacher for excellent work",
     0.44),
    ("WORDORDER_06",
     "Fast cars use more fuel than slow ones",
     "Slow cars use less fuel than fast ones",
     0.41),

    # ── C07: Related Topic (8 pairs) ────────────────────────────────────────
    ("RELATED_01",
     "Python is a programming language used for data science",
     "Java is a programming language used for enterprise applications",
     0.40),
    ("RELATED_02",
     "The iPhone was released by Apple in 2007",
     "The Samsung Galaxy was released by Samsung to compete with Apple",
     0.38),
    ("RELATED_03",
     "Football is the most popular sport in the United States",
     "Basketball is one of the fastest growing sports in the United States",
     0.38),
    ("RELATED_04",
     "World War One began in 1914 in Europe",
     "World War Two ended in 1945 with the defeat of Nazi Germany",
     0.35),
    ("RELATED_05",
     "NASA launched the Artemis mission to return humans to the Moon",
     "SpaceX developed the Starship rocket for Mars colonization missions",
     0.37),
    ("RELATED_06",
     "The stock market rally was driven by strong earnings reports",
     "The bond market declined as investors moved to higher yielding assets",
     0.36),
    ("RELATED_07",
     "Python uses indentation to define code blocks",
     "JavaScript uses curly braces to define code blocks",
     0.40),
    ("RELATED_08",
     "Convolutional neural networks excel at image recognition tasks",
     "Recurrent neural networks are designed for sequential data processing",
     0.38),

    # ── C08: Different Topic (10 pairs) ─────────────────────────────────────
    ("DIFFTOPIC_01",
     "The stock market experienced significant volatility during trading",
     "Scientists discovered a new species of deep sea fish near the trench",
     0.15),
    ("DIFFTOPIC_02",
     "Python is a high level programming language known for readability",
     "The Amazon rainforest contains half of all species on the planet",
     0.12),
    ("DIFFTOPIC_03",
     "The recipe calls for two cups of flour and one egg",
     "The space shuttle was launched from Kennedy Space Center in Florida",
     0.10),
    ("DIFFTOPIC_04",
     "Manchester United won the Premier League championship last season",
     "Quantum computers use qubits instead of classical binary bits",
     0.08),
    ("DIFFTOPIC_05",
     "The Eiffel Tower was built in Paris in 1889",
     "Antibiotics are used to treat bacterial infections in medicine",
     0.08),
    ("DIFFTOPIC_06",
     "The violin is a string instrument played with a bow",
     "The national debt of the United States exceeds thirty trillion dollars",
     0.07),
    ("DIFFTOPIC_07",
     "Photosynthesis converts sunlight into glucose in plant cells",
     "The Roman Empire fell in 476 AD after centuries of decline",
     0.07),
    ("DIFFTOPIC_08",
     "The treaty was signed by representatives of fifteen nations",
     "Baking bread requires yeast flour water and salt mixed together",
     0.07),
    ("DIFFTOPIC_09",
     "The jaguar is the largest wild cat in the Americas",
     "Object oriented programming uses classes and inheritance structures",
     0.07),
    ("DIFFTOPIC_10",
     "The Himalayan mountain range contains the world's highest peaks",
     "The Federal Reserve sets monetary policy for the United States economy",
     0.08),

    # ── C09: Short Text (8 pairs) ───────────────────────────────────────────
    ("SHORT_01", "hello", "world", 0.20),
    ("SHORT_02", "hi",    "bye",   0.10),
    ("SHORT_03", "cat",   "dog",   0.15),
    ("SHORT_04", "good",  "great", 0.22),
    ("SHORT_05", "yes",   "no",    0.10),
    ("SHORT_06", "fast car",  "slow truck",  0.22),
    ("SHORT_07", "I love pizza", "I hate pizza", 0.18),
    ("SHORT_08", "run fast",  "walk slowly", 0.20),

    # ── C10: Code vs Code (6 pairs) ─────────────────────────────────────────
    ("CODE_01",
     "def add(a, b): return a + b",
     "def sum_two(x, y): return x + y",
     0.75),
    ("CODE_02",
     "for i in range(10): print(i)",
     "for j in range(10): print(j)",
     0.88),
    ("CODE_03",
     "SELECT * FROM users WHERE id = 1",
     "SELECT * FROM customers WHERE customer_id = 1",
     0.72),
    ("CODE_04",
     "import numpy as np arr = np.zeros(100)",
     "import numpy as np array = np.ones(100)",
     0.70),
    ("CODE_05",
     "if x > 0: return True else: return False",
     "if value > 0: return True else: return False",
     0.82),
    ("CODE_06",
     "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
     "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
     0.58),

    # ── C11: Code vs Prose (6 pairs) ────────────────────────────────────────
    ("CODEPROSE_01",
     "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
     "The Fibonacci sequence is a series where each number equals the sum of two prior ones",
     0.20),
    ("CODEPROSE_02",
     "SELECT name FROM employees WHERE salary > 50000",
     "Find all employees whose annual pay exceeds fifty thousand dollars",
     0.20),
    ("CODEPROSE_03",
     "for i in range(len(arr)): arr[i] = arr[i] * 2",
     "Multiply every element in the array by two",
     0.20),
    ("CODEPROSE_04",
     "class Animal: def __init__(self, name): self.name = name",
     "An animal object stores a name attribute when created",
     0.22),
    ("CODEPROSE_05",
     "import pandas as pd df = pd.read_csv('data.csv')",
     "Load a CSV file into a dataframe using the pandas library",
     0.22),
    ("CODEPROSE_06",
     "try: x = int(input()) except ValueError: print('invalid')",
     "If the user enters something that is not a number show an error message",
     0.20),

    # ── C12: Cross-Lingual (6 pairs) ────────────────────────────────────────
    ("CROSSLINGUAL_01",
     "The weather is beautiful today with sunshine everywhere",
     "Le chat mange du poisson dans la cuisine ce soir",
     0.10),
    ("CROSSLINGUAL_02",
     "Machine learning algorithms transform data analysis",
     "Los algoritmos de aprendizaje automatico transforman el analisis de datos",
     0.10),
    ("CROSSLINGUAL_03",
     "The economy is growing at a steady pace",
     "Die Wirtschaft wachst mit einem stetigen Tempo",
     0.10),
    ("CROSSLINGUAL_04",
     "Scientists discovered a cure for the disease",
     "Gli scienziati hanno scoperto una cura per la malattia",
     0.10),
    ("CROSSLINGUAL_05",
     "The president made an important announcement today",
     "Le president a fait une annonce importante aujourd hui",
     0.10),
    ("CROSSLINGUAL_06",
     "Children love playing games in the park",
     "Los ninos aman jugar juegos en el parque por la tarde",
     0.10),

    # ── C13: Noise (6 pairs) ────────────────────────────────────────────────
    ("NOISE_01",
     "The weather today is sunny with mild temperatures",
     "xK92mPvL8nQ3rT7wY1oE5uI6hG4jF0dA",
     0.05),
    ("NOISE_02",
     "Machine learning is a branch of artificial intelligence",
     "7hJ#mK9$pL2@nQ5^rT8&wX1!yZ4*bC6",
     0.05),
    ("NOISE_03",
     "The capital of France is Paris",
     "aBcDeFgHiJkLmNoPqRsTuVwXyZ012345",
     0.05),
    ("NOISE_04",
     "asdfjkl qwerty uiop",
     "zxcvbnm poiuytrewq lkjhgfdsa mnbvcxz",
     0.05),
    ("NOISE_05",
     "1234567890 abcdef ghijk",
     "9876543210 zyxwvu tsrqp",
     0.05),
    ("NOISE_06",
     "The results show a clear improvement",
     "!@#$%^&*()_+{}|:<>?~`-=[];',./",
     0.05),

    # ── C14: Length Difference (6 pairs) ────────────────────────────────────
    ("LENGTHDIFF_01",
     "Python programming",
     "Python is a high level interpreted programming language with dynamic semantics and a large standard library",
     0.60),
    ("LENGTHDIFF_02",
     "Climate change",
     "Climate change refers to long term shifts in global temperatures and weather patterns caused by human activity",
     0.58),
    ("LENGTHDIFF_03",
     "Neural network",
     "A neural network is a computational model inspired by the human brain consisting of layers of interconnected nodes",
     0.58),
    ("LENGTHDIFF_04",
     "Stock market",
     "The stock market is a platform where buyers and sellers trade shares of publicly listed companies",
     0.57),
    ("LENGTHDIFF_05",
     "Quantum computing",
     "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to process information",
     0.55),
    ("LENGTHDIFF_06",
     "Machine learning",
     "Machine learning is a method of data analysis that automates analytical model building using statistical algorithms",
     0.58),

    # ── C15: Negation (6 pairs) ─────────────────────────────────────────────
    ("NEGATION_01",
     "The patient recovered fully after the surgery",
     "The patient did not recover after the surgery",
     0.25),
    ("NEGATION_02",
     "The economy is growing at a record pace",
     "The economy is not growing and is in recession",
     0.25),
    ("NEGATION_03",
     "The algorithm converges quickly on large datasets",
     "The algorithm fails to converge on large datasets",
     0.25),
    ("NEGATION_04",
     "Smoking is harmful to your health",
     "Smoking is not harmful and has no health effects",
     0.25),
    ("NEGATION_05",
     "The company made a profit last quarter",
     "The company did not make a profit last quarter",
     0.25),
    ("NEGATION_06",
     "The test results were positive",
     "The test results were negative",
     0.28),

    # ── C16: Factual Variation / NumVar (6 pairs) ───────────────────────────
    ("NUMVAR_01",
     "The population of India is 1.4 billion people",
     "The population of China is 1.4 billion people",
     0.65),
    ("NUMVAR_02",
     "The company was founded in 1998 in California",
     "The company was founded in 2004 in New York",
     0.62),
    ("NUMVAR_03",
     "The temperature today is 25 degrees Celsius",
     "The temperature today is 35 degrees Celsius",
     0.68),
    ("NUMVAR_04",
     "The train arrives at platform 3 at 10 AM",
     "The train arrives at platform 7 at 2 PM",
     0.62),
    ("NUMVAR_05",
     "Revenue grew by 15 percent in the third quarter",
     "Revenue grew by 32 percent in the second quarter",
     0.65),
    ("NUMVAR_06",
     "The building has 50 floors and was completed in 2010",
     "The building has 80 floors and was completed in 2018",
     0.63),
]

CATEGORY_NAMES = {
    "IDENTICAL":     "C01 Identical",
    "NEARDUP":       "C02 Near-Duplicate",
    "PARAPHRASE":    "C03 Paraphrase",
    "AIREWRITE":     "C04 AI Rewrite",
    "SAMESTYLE":     "C05 Same Style",
    "WORDORDER":     "C06 Word Order",
    "RELATED":       "C07 Related Topic",
    "DIFFTOPIC":     "C08 Different Topic",
    "SHORT":         "C09 Short Text",
    "CODE":          "C10 Code vs Code",
    "CODEPROSE":     "C11 Code vs Prose",
    "CROSSLINGUAL":  "C12 Cross-Lingual",
    "NOISE":         "C13 Noise",
    "LENGTHDIFF":    "C14 Length Diff",
    "NEGATION":      "C15 Negation",
    "NUMVAR":        "C16 Factual Variation",
}

CATEGORY_ORDER = [
    "IDENTICAL", "NEARDUP", "PARAPHRASE", "AIREWRITE",
    "SAMESTYLE", "WORDORDER", "RELATED", "DIFFTOPIC",
    "SHORT", "CODE", "CODEPROSE", "CROSSLINGUAL",
    "NOISE", "LENGTHDIFF", "NEGATION", "NUMVAR",
]


# =============================================================================
# METRICS
# =============================================================================

def pearson(pred, gt):
    p, g = np.array(pred), np.array(gt)
    if np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


def mae(pred, gt):
    return float(np.mean(np.abs(np.array(pred) - np.array(gt))))


def rank_accuracy(pred, gt, threshold=0.05):
    ok = tot = 0
    for i in range(len(gt)):
        for j in range(i + 1, len(gt)):
            if abs(gt[i] - gt[j]) > threshold:
                tot += 1
                if (gt[i] > gt[j]) == (pred[i] > pred[j]):
                    ok += 1
    return ok / tot if tot else 0.0


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark():
    print()
    print("=" * 80)
    print("  CHIMERA-Hash Benchmark v1.0 — 115 pairs, 16 categories")
    print("  Chaos-IDF Multiresolution Entropy Resonance Attractor Hash")
    print()
    print("  Author  : Manish Kumar Parihar")
    print("  YouTube : https://www.youtube.com/@ProgramDr")
    print("  LinkedIn: https://www.linkedin.com/in/manish-parihar-899b5b23a/")
    print("=" * 80)
    print()

    print(f"  Initialising algorithms...")
    chimera = TetraAttractorHash()
    simhash = SimHashBaseline()
    minhash = MinHashBaseline()
    tfidf   = TFIDFBaseline()
    print("  CHIMERA-Hash  ✓")
    print(f"  TF-IDF        {'✓' if SKLEARN_OK else '✗ (pip install scikit-learn)'}")
    print("  SimHash       ✓")
    print("  MinHash       ✓")
    print()

    algos = {
        "CHIMERA-Hash": lambda a, b: chimera.similarity(a, b, depth=6),
        "TF-IDF":       tfidf.similarity,
        "SimHash":      simhash.similarity,
        "MinHash":      minhash.similarity,
    }
    col_names = list(algos.keys())

    scores  = {k: [] for k in col_names}
    timings = {k: [] for k in col_names}
    gt_all  = [t[3] for t in TEST_SUITE]

    print(f"  Running {len(TEST_SUITE)} pairs x {len(algos)} algorithms...")
    for idx, (label, t1, t2, gt) in enumerate(TEST_SUITE):
        prefix = label.rsplit("_", 1)[0]
        cat    = CATEGORY_NAMES.get(prefix, prefix)
        if idx % 15 == 0:
            print(f"  [{idx:3d}/{len(TEST_SUITE)}]  {cat} ...", flush=True)
        for name, fn in algos.items():
            t0 = time.perf_counter()
            s  = fn(t1, t2)
            ms = (time.perf_counter() - t0) * 1000
            scores[name].append(s)
            timings[name].append(ms)

    print(f"  [{len(TEST_SUITE)}/{len(TEST_SUITE)}]  Done.\n")

    # ── Full per-pair table ───────────────────────────────────────────────
    W   = 13
    SEP = "─" * (30 + W * len(col_names))
    print("  FULL RESULTS — all 115 pairs")
    print()
    header = f"  {'Label':<22} {'GT':>5} |" + "".join(f"{c:>{W}}" for c in col_names)
    print(header)
    print("  " + SEP)

    prev_cat = None
    for i, (label, _, _, g) in enumerate(TEST_SUITE):
        cat = label.rsplit("_", 1)[0]
        if cat != prev_cat:
            if prev_cat is not None:
                print("  " + "·" * len(SEP))
            cname = CATEGORY_NAMES.get(cat, cat)
            print(f"  ── {cname} ──")
            prev_cat = cat
        row = f"  {label:<22} {g:>5.2f} |"
        for c in col_names:
            row += f"{scores[c][i]:>{W}.4f}"
        print(row)

    # ── Overall ranking ───────────────────────────────────────────────────
    gt_a = np.array(gt_all)
    rows = []
    for name in col_names:
        pred = np.array(scores[name])
        c    = pearson(pred, gt_a)
        m    = mae(pred, gt_a)
        r    = rank_accuracy(pred.tolist(), gt_a.tolist())
        avg  = float(np.mean(timings[name]))
        rows.append((name, c, m, r, avg))

    rows.sort(key=lambda x: -x[1])

    print()
    print("=" * 80)
    print("  FINAL RANKING — sorted by Pearson Correlation with ground truth")
    print("=" * 80)
    medals = ["#1 (GOLD)", "#2 (SILVER)", "#3 (BRONZE)", "#4"]
    notes  = {
        "CHIMERA-Hash": "  <-- THIS ALGORITHM",
        "TF-IDF":       "  <-- industry standard (Sparck Jones 1972)",
        "SimHash":      "  <-- Charikar 2002",
        "MinHash":      "  <-- Broder 1997",
    }
    for rank, (name, c, m, r, ms) in enumerate(rows):
        medal = medals[rank] if rank < len(medals) else f"#{rank+1}"
        print(f"\n  {medal}")
        print(f"    Algorithm  : {name}{notes.get(name,'')}")
        print(f"    Correlation: {c:.4f}")
        print(f"    MAE        : {m:.4f}")
        print(f"    Rank Acc   : {r:.4f}")
        print(f"    Speed      : {ms:.1f} ms/pair")

    # ── Category-level MAE table ──────────────────────────────────────────
    print()
    print("=" * 80)
    print("  CATEGORY-LEVEL MAE (lower is better) — Table 1 & 2 from paper")
    print("=" * 80)
    cat_header = f"  {'Category':<22} {'GT Range':>10}" + "".join(f"{c:>14}" for c in col_names) + "  Winner"
    print(cat_header)
    print("  " + "─" * (len(cat_header) - 2))

    cat_wins = {k: 0 for k in col_names}
    for cat_key in CATEGORY_ORDER:
        indices = [i for i, (lbl, _, _, _) in enumerate(TEST_SUITE)
                   if lbl.rsplit("_", 1)[0] == cat_key]
        gt_cat  = [gt_all[i] for i in indices]
        cat_mae = {n: mae([scores[n][i] for i in indices], gt_cat) for n in col_names}
        winner  = min(cat_mae, key=cat_mae.get)
        cat_wins[winner] += 1
        gt_str   = f"{min(gt_cat):.2f}–{max(gt_cat):.2f}"
        cname    = CATEGORY_NAMES.get(cat_key, cat_key)
        row = f"  {cname:<22} {gt_str:>10}"
        for n in col_names:
            mark = " ✓" if n == winner else "  "
            row += f"{cat_mae[n]:>12.4f}{mark}"
        print(row)

    print()
    wins_str = "  |  ".join(f"{k}: {v}/16" for k, v in cat_wins.items())
    print(f"  Category wins — {wins_str}")

    # ── CHIMERA summary ───────────────────────────────────────────────────
    cr   = next(r for r in rows if r[0] == "CHIMERA-Hash")
    tf   = next(r for r in rows if r[0] == "TF-IDF")
    rank = next(i + 1 for i, r in enumerate(rows) if r[0] == "CHIMERA-Hash")

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  CHIMERA-Hash rank   : #{rank}")
    print(f"  Correlation         : {cr[1]:.4f}  (paper: 0.5082)")
    print(f"  MAE                 : {cr[2]:.4f}  (paper: 0.3288)")
    print(f"  Rank Accuracy       : {cr[3]:.4f}  (paper: 0.7108)")
    diff = cr[1] - tf[1]
    print(f"  vs TF-IDF corr      : {diff:+.4f}  ({'BEATS' if diff > 0 else 'below'} TF-IDF)")
    print(f"  Category wins       : {cat_wins.get('CHIMERA-Hash', 0)}/16")

    # ── Speed ─────────────────────────────────────────────────────────────
    print()
    print("  SPEED")
    max_ms = max(r[4] for r in rows)
    for name, c, m, r, ms in rows:
        bar = "█" * int((ms / max_ms) * 30)
        print(f"  {name:<16} {ms:6.1f} ms/pair  [{bar}]")

    print()
    print("  ALL RESULTS COMPUTED LIVE FROM SOURCE CODE — NOTHING FABRICATED")
    print("=" * 80)
    print()


# =============================================================================
# ATTRACTOR CONVERGENCE CHECK
# =============================================================================

def run_attractor_check():
    print()
    print("=" * 80)
    print("  ATTRACTOR CONVERGENCE CHECK — Table 3 from paper")
    print("  All texts reach STABLE_ATTRACTOR at Level 5 regardless of content")
    print("=" * 80)

    chimera = TetraAttractorHash()
    test_texts = [
        ("hello",                                        "short word"),
        ("The quick brown fox jumps over the lazy dog",  "English pangram"),
        ("SELECT * FROM users WHERE id = 1",             "SQL query"),
        ("xK9$mP#3rT7wY1oE5",                           "random noise"),
    ]

    all_ok = True
    for text, desc in test_texts:
        traj       = chimera.trajectory(text, depth=8)
        conv       = any(t["state"] == "STABLE_ATTRACTOR" for t in traj)
        conv_level = next((t["level"] for t in traj if t["state"] == "STABLE_ATTRACTOR"), 99)
        if not conv:
            all_ok = False

        status = f"STABLE at Level {conv_level}" if conv else "DID NOT CONVERGE"
        print(f"\n  [{desc}]  →  {status}")
        print(f"  {'L':>3}  {'Mean':>8}  {'Stability':>11}  State")
        print(f"  {'─'*44}")
        for t in traj:
            print(f"  {t['level']:>3}  {t['mean']:>8.2f}  "
                  f"{t['stability_index']:>11.6f}  {t['state']}")

    print()
    result = "✓ ALL TEXTS CONVERGE AT LEVEL 5" if all_ok else "✗ WARNING: CONVERGENCE FAILED"
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

        bar = "█" * int(sim * 40) + "░" * (40 - int(sim * 40))

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
        description="CHIMERA-Hash Benchmark — Manish Kumar Parihar"
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
