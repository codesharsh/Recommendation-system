# 🎬 Movie Recommendation System

An end-to-end machine learning project implementing and comparing multiple recommendation strategies — from a simple popularity baseline to collaborative filtering, matrix factorization, content-based filtering, and a switching hybrid system.

Built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (a public stand-in for Netflix-style rating data, since Netflix's original dataset is no longer publicly available).

---

## 📌 Project Goals

- Build and benchmark multiple recommender system approaches from scratch
- Understand *why* certain methods work better than others, not just implement them
- Handle real-world challenges: sparsity, cold-start users/items, temporal data leakage
- Produce an honest, evidence-based comparison — including cases where "fancier" methods underperformed simpler ones

---

## 🗂️ Dataset

**MovieLens ml-latest-small**
- 100,836 ratings
- 610 users
- 9,742 movies
- 3,683 tag applications
- Rating scale: 0.5–5.0
- Source: [GroupLens](https://grouplens.org/datasets/movielens/)

> A larger-scale extension of this project using **ml-25m** (25M ratings, 162K users, 62K movies) is planned/in progress — see [Future Work](#-future-work).

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, NumPy |
| Recommender algorithms | scikit-surprise (KNN, SVD), scikit-learn (TF-IDF, cosine similarity) |
| Evaluation | RMSE, MAE, Precision@K, Recall@K |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 🧭 Approach Overview

The project builds up in layers, with each phase benchmarked against the ones before it:

1. **Popularity Baseline** — IMDb-style Bayesian weighted rating
2. **Collaborative Filtering** — User-based & item-based KNN (cosine, Pearson)
3. **Matrix Factorization** — SVD, tuned via GridSearchCV
4. **Content-Based Filtering** — Genre (multi-hot) + tag (TF-IDF) similarity
5. **Hybrid Model** — 3-tier switching system combining all of the above

### Why time-based train/test split

Ratings were split chronologically (80% earliest / 20% most recent) rather than randomly, to avoid **temporal data leakage** — a model should only ever be evaluated on its ability to predict the *future*, not fill in gaps from the past. This is more realistic but does introduce a measurable cold-start effect in evaluation (documented below).

---

## 📊 Results

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| Popularity Baseline | 1.038 | 0.813 | Bayesian weighted rating; 15.09% test movies unseen in training |
| Item-Based CF (Cosine) | 1.077 | 0.850 | Underperformed baseline — sparsity hurts neighborhood methods |
| User-Based CF (Pearson) | 1.076 | 0.851 | Same issue as above |
| Item-Based KNNWithMeans | 1.069 | 0.844 | Best of the KNN variants, still below baseline |
| **SVD (GridSearchCV tuned)** | **1.005** | **0.777** | **Best-performing model overall** |
| Content-Based (Genre + TF-IDF + popularity tiebreak) | Precision@10: 0.0179 · Recall@10: 0.0078 | — | Weak standalone; genre signal is coarse (90%+ vector collisions) |
| Hybrid (SVD + Popularity fallback) | 1.016 | 0.792 | Slightly below pure SVD — see findings below |

*(Full experiment details, including hyperparameter search results, are in the `notebooks/` folder.)*

---

## 🔍 Key Findings

**1. Basic KNN-based collaborative filtering underperformed the popularity baseline.**
With ~99% sparsity in the user-item matrix and a small user base (610 users), neighborhood-based similarity is unreliable — most user/item pairs simply don't have enough overlapping ratings to compute meaningful similarity.

**2. Matrix factorization (SVD) was the clear winner.**
By learning shared latent factors across the entire ratings matrix rather than relying on direct pairwise overlap, SVD handled sparsity far better than KNN — improving RMSE by ~3% over the baseline and ~6% over the best KNN variant.

**3. Content-based filtering alone was weak — and the reason was diagnosable, not mysterious.**
With only ~20 genre categories, over 90% of movies shared an identical genre vector. Adding TF-IDF features from user tags helped only the ~16% of movies that had tags at all (most had none), and actually slightly reduced Recall@10 — a useful lesson that adding features isn't automatically better without addressing coverage and dimensionality balance.

**4. The dominant cold-start problem was *new users*, not new movies.**
75.9% of test-set users had **zero** rating history in the training period — a consequence of the small dataset combined with a strict time-based split. This is a bigger and different problem than the baseline's 15% *movie* coverage gap.

**5. The hybrid's popularity fallback slightly underperformed SVD's native handling of unknown users.**
Surprise's SVD implementation already applies a learned per-movie bias term even for unseen users, which turned out to be a more refined "cold-start strategy" than a hand-built popularity heuristic. The hybrid's real value is therefore in **serving users the model literally cannot predict for at all** (via a content-based "pick your favorites" onboarding flow) — not in improving aggregate RMSE.

---

## 🧩 Hybrid System Design

The final hybrid uses a 3-tier switching strategy based on how much is known about a user:

```
Tier 1 — User has rating history        → SVD prediction (best accuracy)
Tier 2 — New user, provides preferences  → Content-based (genre similarity to stated favorites)
Tier 3 — No information available        → Popularity baseline (safe default)
```

Tier 2 mirrors real onboarding flows (e.g., "pick a few movies you like") and could not be evaluated against this dataset's historical test set, since MovieLens has no such interaction recorded — it is demonstrated functionally rather than scored quantitatively.

---

## 📁 Project Structure

```
movie-recommender/
│
├── data/
│   └── raw/                      # MovieLens CSVs (not committed — see Setup)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_collaborative_filtering.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_content_based.ipynb
│   └── 06_hybrid_model.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── baseline.py
│   ├── collaborative_filtering.py
│   ├── matrix_factorization.py
│   ├── content_based.py
│   ├── hybrid.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender

# Set up environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Download the dataset:**
```python
import urllib.request, zipfile, os

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
os.makedirs("data/raw", exist_ok=True)
urllib.request.urlretrieve(url, "data/raw/ml-latest-small.zip")

with zipfile.ZipFile("data/raw/ml-latest-small.zip", 'r') as z:
    z.extractall("data/raw")
```

Then run the notebooks in order (`01` → `06`), or import functions directly from `src/`.

---

## 🚧 Future Work

- [ ] Re-run the full pipeline on **ml-25m** (25M ratings) to test how findings hold up at scale — particularly whether KNN's sparsity problem and the cold-start proportions change
- [ ] Address KNN's scalability limits on larger datasets (subsampling or approximate nearest-neighbor methods)
- [ ] Experiment with a weighted hybrid (blended SVD + content-based scores) instead of pure switching
- [ ] Explore Neural Collaborative Filtering as a deep learning comparison point

---

## 📚 References

- [IMDb Ratings FAQ](https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV) — weighted rating (Bayesian estimate) formula
- [GroupLens / MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- [Surprise Library Documentation](https://surpriselib.com/)
