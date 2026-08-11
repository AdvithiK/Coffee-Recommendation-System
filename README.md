# Coffee Recommendation System

A content-based recommendation engine that matches users to coffee drinks based on their taste profile, using TF-IDF text vectorization and cosine similarity.

## How it works

Each user has a **taste profile** (favorite flavors + preferred drink types) and each **drink** has its own flavor and type attributes. The algorithm:

1. **Builds text features** — combines each user's flavor and drink-type preferences into a single string, and does the same for each drink's flavor/type attributes.
2. **Vectorizes with TF-IDF** — fits a shared `TfidfVectorizer` across both users and drinks so they land in the same vector space.
3. **One-hot encodes categorical attributes** — `temperature` (hot/iced) and `milk` (type/none) are encoded separately and concatenated onto the TF-IDF vectors with `scipy.sparse.hstack`.
4. **Scores via cosine similarity** — for a given `userId`, the algorithm compares that user's combined vector against every drink's vector and returns the top-k most similar drinks.

```python
recommend_drinks("anikalovescoffee", top_k=3)
```

## Tech stack

- **Python** — pandas, NumPy
- **scikit-learn** — `TfidfVectorizer`, `OneHotEncoder`, `cosine_similarity`, `MinMaxScaler`
- **SciPy** — sparse matrix concatenation (`hstack`)
- **matplotlib / seaborn** — exploratory data visualization

## Data files

| File | Description |
|---|---|
| `users_large.csv` | User accounts (`userId`, `email`, `passwordHash`, `firstName`, `lastName`) |
| `taste_profiles_large.csv` | Per-user taste preferences (`temperature`, `flavor_1-3`, `drinkType_1-3`, `milk`) |
| `drinks_large.csv` | Drink catalog (`drinkName`, `drinkType`, `temperature`, `milk`, `flavor 1-2`, `cost`, `cals`, `coffeeShop`) |
| `order_history_large.csv` | Past orders per user (`userId`, `drink`, `cafeName`) |

## Getting started

```bash
git clone https://github.com/AdvithiK/Coffee-Recommendation-System.git
cd Coffee-Recommendation-System
pip install pandas numpy scikit-learn scipy matplotlib seaborn
python copy_of_coffee_reccomendation_algorithm.py
```

Make sure the four CSV files are in the same directory as the script — they're loaded via relative paths.

To get recommendations for a specific user, call:

```python
recommend_drinks(user_id, top_k=5)
```

where `user_id` matches a `userId` in `taste_profiles_large.csv`.

## Limitations

- Currently a single-script prototype (originally exported from a Colab notebook) rather than a packaged module
- Cold-start problem: users need a filled-out taste profile to get recommendations; there's no fallback for brand-new users yet
- No evaluation metrics (precision@k, etc.) implemented yet
