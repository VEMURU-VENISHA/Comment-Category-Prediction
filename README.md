# 💬 Comment Category Prediction Challenge

A machine learning project to predict the internal category assigned to user comments on an online platform, using text content, metadata, and behavioral signals.

---
## 📌 Problem Statement
Online platforms internally classify user comments into categories for moderation and handling. Given a comment along with its metadata (reactions, emoticons, topic flags, internal signals), the goal is to **predict the label (0, 1, 2, or 3)** assigned by the platform.
This is a **multi-class text classification** problem.

---

## 📂 Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 198,000 | Training data with labels |
| `test.csv` | 102,000 | Test data (no labels) |
| `Sample.csv` | 102,000 | Sample submission format |
> ⚠️ Raw data files are **not included** in this repository due to size (112 MB).  
> Download them from the competition page and place them in the appropriate input directory.
### 🗂️ Feature Columns
| Column | Type | Description |
|--------|------|-------------|
| `comment` | text | Raw comment content |
| `created_date` | datetime | When the comment was posted |
| `post_id` | int | ID of the parent thread/post |
| `emoticon_1` | int | Count of emoticon group 1 |
| `emoticon_2` | int | Count of emoticon group 2 |
| `emoticon_3` | int | Count of emoticon group 3 |
| `upvote` | int | Positive reactions received |
| `downvote` | int | Negative reactions received |
| `if_1` | int | Hidden internal platform feature 1 |
| `if_2` | int | Hidden internal platform feature 2 |
| `race` | object | Reference to group identity (NaN if absent) |
| `religion` | object | Reference to belief-related topic (NaN if absent) |
| `gender` | object | Reference to gender topic (NaN if absent) |
| `disability` | bool | Reference to ability-related topic |
| `label` | int | **Target variable** — category 0, 1, 2, or 3 |

---

## 📊 Exploratory Data Analysis

### Label Distribution
| Label | Count | Percentage |
|-------|-------|------------|
| 0 | ~114,173 | 57.7% |
| 2 | ~62,440 | 31.5% |
| 1 | ~15,918 | 8.0% |
| 3 | ~5,469 | 2.8% |

> The dataset is **highly imbalanced** — label 0 dominates, while label 3 is rare.

### Key Observations
- `race`, `religion`, and `gender` columns have ~73% missing values → filled with `"none"`
- One row in `comment` was null → filled with `"none"`
- `if_2` ranges from 3 to 1833, suggesting a platform-specific engagement score
- `upvote` median is 1; most comments receive very few votes
- Engineered features: `vote_diff` (upvote − downvote) and `vote_ratio` (upvote / (downvote + 1))

---

## 🛠️ Approach

### 1. Preprocessing
- Filled missing values in `race`, `religion`, `gender`, `comment` with `"none"`
- Engineered `vote_diff`, `vote_ratio`, and `text_length` features
- Applied `TF-IDF` vectorization on the `comment` column

### 2. Feature Pipeline (`ColumnTransformer`)
| Transformer | Applied To | Details |
|-------------|------------|---------|
| `TfidfVectorizer` (word) | `comment` | max 80k features, unigrams+bigrams, sublinear TF |
| `TfidfVectorizer` (char) | `comment` | max 30k features, char 3–6 grams |
| `OneHotEncoder` | `race`, `religion`, `gender`, `disability` | handles unknown categories |
| `StandardScaler` | `upvote`, `downvote`, `emoticon_*`, `if_1`, `if_2` | sparse-compatible scaling |

### 3. Models Evaluated

| Model | Training Accuracy |
|-------|------------------|
| LinearSVC (TF-IDF only) | 91.2% |
| Logistic Regression | 82.2% |
| SGDClassifier (basic) | 73.6% |
| **SGDClassifier (full pipeline, elasticnet)** | **Final Model ✅** |

### 4. Final Model
```
SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    alpha=0.000015,
    l1_ratio=0.15,
    max_iter=5000,
    tol=1e-5,
    class_weight='balanced',
    random_state=42
)
```
- `class_weight='balanced'` used to handle class imbalance
- Combined word-level and character-level TF-IDF for richer text features
- Categorical features one-hot encoded; numerical features scaled

---

## 📁 Repository Structure

```
comment-category-prediction/
│
├── notebook.ipynb          # Full EDA + modeling notebook
├── submission.csv          # Final predictions for test set
├── README.md               # This file
```

---



## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Or install all at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📬 Submission Format

The `submission.csv` file contains two columns:

| ID | label |
|----|-------|
| 1 | 2 |
| 2 | 2 |
| 3 | 0 |
| ... | ... |

102,000 rows total, one prediction per test comment.

---

## 🧠 Key Takeaways

- **Text is the strongest signal** — TF-IDF on comment content dominates performance
- **Character n-grams** help capture spelling patterns and emoticon-like sequences
- **Class imbalance** is significant (57% label 0) — `class_weight='balanced'` is essential
- **Metadata features** (upvotes, emoticons, internal flags) provide useful supplementary signal

---

## 👤 Author
> Submitted as part of the **Comment Category Prediction Challenge** on Kaggle.
