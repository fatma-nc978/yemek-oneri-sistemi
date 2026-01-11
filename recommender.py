import numpy as np
import os
import pickle
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity


# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "reindexed")

RECIPES_PATH = os.path.join(DATA_DIR, "recipes_small_reindexed_translated.csv")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions_small_reindexed.csv")


# LOAD DATA
def load_assets():
    recipes_df = pd.read_csv(RECIPES_PATH)
    interactions_df = pd.read_csv(INTERACTIONS_PATH)

    with open(os.path.join(MODEL_DIR, "svd_model_v1.pkl"), "rb") as f:
        svd_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "knn_user_model.pkl"), "rb") as f:
        user_knn_model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "user_recipe_sparse.pkl"), "rb") as f:
        user_recipe_sparse = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "tfidf_matrix_tr.pkl"), "rb") as f:
        tfidf_matrix_tr = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer_tr.pkl"), "rb") as f:
        tfidf_vectorizer_tr = pickle.load(f)

    print(" recipes_df:", recipes_df.shape)
    print(" interactions_df:", interactions_df.shape)
    print(" modeller yüklendi")

    return (
        recipes_df,
        interactions_df,
        svd_model,
        user_knn_model,
        user_recipe_sparse,
        tfidf_matrix_tr,
        tfidf_vectorizer_tr
    )

(
    recipes_df,
    interactions_df,
    svd_model,
    user_knn_model,
    user_recipe_sparse,
    tfidf_matrix_tr,
    tfidf_vectorizer_tr
) = load_assets()



# LANGUAGE HELPERS
def lang_cols(lang):
    return {
        "name": "name_tr" if lang == "tr" else "name",
        "desc": "description_tr" if lang == "tr" else "description",
        "tags": "tags_tr" if lang == "tr" else "tags",
        "ingredients": "ingredients_tr" if lang == "tr" else "ingredients",
        "steps": "steps_tr" if lang == "tr" else "steps",
    }


def parse_list_field(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")):
            try:
                out = ast.literal_eval(s)
                return [str(x).strip() for x in out] if isinstance(out, list) else []
            except Exception:
                pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


# POPULAR (dataset tabanlı)
def get_top_rated_recipes(top_n=20, min_votes=0, lang="tr"):
    cols = lang_cols(lang)
    stats = (
        interactions_df
        .groupby("recipe_id")
        .agg(
            avg_rating=("rating", "mean"),
            vote_count=("rating", "count")
        )
        .reset_index()
    )
    # daha yumuşak filtre
    stats = stats[stats["vote_count"] >= min_votes]

    # Ağırlıklı skor
    stats["score"] = stats["avg_rating"] * np.log1p(stats["vote_count"])

    stats = stats.sort_values("score", ascending=False).head(top_n)

    merged = stats.merge(
        recipes_df,
        left_on="recipe_id",
        right_on="id",
        how="inner"
    )

    results = []
    for _, r in merged.iterrows():
        results.append({
            "id": int(r["id"]),
            "name": r[cols["name"]],
            "minutes": int(r["minutes"]) if pd.notna(r["minutes"]) else None,
            "n_ingredients": int(r["n_ingredients"]) if pd.notna(r["n_ingredients"]) else None,
            "avg_rating": round(float(r["avg_rating"]), 1)
        })

    return results


# eski isimle çağrılan yerler için ALIAS
def get_popular_recipes(top_n=10, min_votes=0, lang="tr"):
    return get_top_rated_recipes(top_n=top_n, min_votes=min_votes, lang=lang)


# DETAIL
def get_recipe_detail(recipe_id, lang="tr"):
    recipe = recipes_df[recipes_df["id"] == recipe_id]
    if recipe.empty:
        return None

    row = recipe.iloc[0]
    cols = lang_cols(lang)

    ingredients = parse_list_field(row.get(cols["ingredients"]))
    steps = parse_list_field(row.get(cols["steps"]))
    tags = parse_list_field(row.get(cols["tags"]))

    return {
        "id": int(recipe_id),
        "name": row.get(cols["name"], ""),
        "description": row.get(cols["desc"], ""),
        "minutes": int(row["minutes"]) if not pd.isna(row["minutes"]) else None,
        "ingredients": ingredients,
        "steps": steps,
        "tags": tags,
        "n_ingredients": len(ingredients),
        "n_steps": len(steps),
    }


def content_based_for_user(user_id, conn, top_n=10):
    user_vector = build_user_profile_vector(user_id, conn)
    if user_vector is None:
        return []

    sims = cosine_similarity(user_vector, tfidf_matrix_tr)[0]
    ranked = sims.argsort()[::-1]

    interacted = get_user_interacted_recipe_ids(conn, user_id)

    results = []
    for idx in ranked:
        if int(idx) not in interacted:
            results.append(int(idx))
        if len(results) >= top_n:
            break

    return results


# USER INTERACTIONS
def get_user_interacted_recipe_ids(conn, user_id):
    recipe_ids = set()

    rows = conn.execute(
        "SELECT recipe_id FROM ratings WHERE user_id=?",
        (user_id,)
    ).fetchall()
    recipe_ids.update([int(r["recipe_id"]) for r in rows])

    rows = conn.execute(
        "SELECT recipe_id FROM favorites WHERE user_id=?",
        (user_id,)
    ).fetchall()
    recipe_ids.update([int(r["recipe_id"]) for r in rows])

    rows = conn.execute(
        "SELECT recipe_id FROM saved WHERE user_id=?",
        (user_id,)
    ).fetchall()
    recipe_ids.update([int(r["recipe_id"]) for r in rows])

    rows = conn.execute(
        "SELECT recipe_id FROM comments WHERE user_id=?",
        (user_id,)
    ).fetchall()
    recipe_ids.update([int(r["recipe_id"]) for r in rows])

    return recipe_ids


def recommend_recipes_for_user(user_code, n_recommendations=20, lang="tr"):
   
    try:
        user_idx = int(user_code)
    except:
        return []

    if user_idx >= user_recipe_sparse.shape[0]:
        return []

    #  En benzer kullanıcıları bul
    distances, indices = user_knn_model.kneighbors(
        user_recipe_sparse[user_idx],
        n_neighbors=20
    )

    similar_users = indices.flatten()

    #  Benzer kullanıcıların etkileştiği tarifler
    candidate_scores = {}

    for sim_user in similar_users:
        user_rows = interactions_df[
            interactions_df["user_id"] == sim_user
        ]

        for _, row in user_rows.iterrows():
            rid = int(row["recipe_id"])
            rating = float(row["rating"])

            # ağırlıklandır
            candidate_scores[rid] = candidate_scores.get(rid, 0) + rating

    if not candidate_scores:
        return []

    #  Kullanıcının zaten etkileştiği tarifleri çıkar
    from app import get_db_connection
    conn = get_db_connection()

    interacted = get_user_interacted_recipe_ids(conn, user_code)
    interacted_ids = get_user_interacted_recipe_ids(conn, user_code)

    conn.close()

    #  Filtrele + sırala
    filtered = {
        rid: score
        for rid, score in candidate_scores.items()
        if rid not in interacted_ids
    }

    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    #  Tarif detaylarını getir
    results = []
    for rid, _ in ranked[:n_recommendations]:
        recipe = get_recipe_detail(rid, lang)
        if recipe:
            results.append(recipe)

    return results


# USER INTERACTIONS 
def get_user_interactions_with_weights(conn, user_id):
    """
    (recipe_id, weight) list döndürür. (profil vektörü için)
    """
    pairs = []

    # rating - weight = rating
    rows = conn.execute("SELECT recipe_id, rating FROM ratings WHERE user_id=?", (user_id,)).fetchall()
    for r in rows:
        pairs.append((int(r["recipe_id"]), float(r["rating"])))

    # favorites - sabit ağırlık
    rows = conn.execute("SELECT recipe_id FROM favorites WHERE user_id=?", (user_id,)).fetchall()
    for r in rows:
        pairs.append((int(r["recipe_id"]), 4.0))

    # saved - sabit ağırlık
    rows = conn.execute("SELECT recipe_id FROM saved WHERE user_id=?", (user_id,)).fetchall()
    for r in rows:
        pairs.append((int(r["recipe_id"]), 3.0))

    # comments - sabit ağırlık
    rows = conn.execute("SELECT recipe_id FROM comments WHERE user_id=?", (user_id,)).fetchall()
    for r in rows:
        pairs.append((int(r["recipe_id"]), 3.0))

    return pairs



def build_user_profile_vector(user_id, conn):
    interactions = get_user_interactions_with_weights(conn, user_id)

    if not interactions:
        return None

    vectors = []
    weights = []

    for recipe_id, weight in interactions:
        idx = int(recipe_id)

        if idx >= tfidf_matrix_tr.shape[0]:
            continue

        vectors.append(tfidf_matrix_tr[idx].toarray()[0])  # dense
        weights.append(float(weight))

    if not vectors:
        return None

    vectors = np.array(vectors)
    weights = np.array(weights)

    user_profile = np.average(vectors, axis=0, weights=weights)
    return user_profile.reshape(1, -1)


def get_popularity_scores(conn, recipe_ids):
    """
    0-1 normalize popularity skoru döndürür.
    DB'den: favorites + saved + ratings_count (+ avg_rating küçük katkı)
    """
    if not recipe_ids:
        return {}

    placeholders = ",".join(["?"] * len(recipe_ids))

    # favorites count
    fav_rows = conn.execute(
        f"SELECT recipe_id, COUNT(*) c FROM favorites WHERE recipe_id IN ({placeholders}) GROUP BY recipe_id",
        tuple(recipe_ids)
    ).fetchall()
    fav = {int(r["recipe_id"]): int(r["c"]) for r in fav_rows}

    # saved count
    sav_rows = conn.execute(
        f"SELECT recipe_id, COUNT(*) c FROM saved WHERE recipe_id IN ({placeholders}) GROUP BY recipe_id",
        tuple(recipe_ids)
    ).fetchall()
    saved = {int(r["recipe_id"]): int(r["c"]) for r in sav_rows}

    # ratings count + avg rating
    rat_rows = conn.execute(
        f"SELECT recipe_id, COUNT(*) cnt, AVG(rating) avg_r FROM ratings WHERE recipe_id IN ({placeholders}) GROUP BY recipe_id",
        tuple(recipe_ids)
    ).fetchall()
    rcount = {int(r["recipe_id"]): int(r["cnt"]) for r in rat_rows}
    ravg = {int(r["recipe_id"]): float(r["avg_r"]) for r in rat_rows}

    raw = {}
    for rid in recipe_ids:
        rid = int(rid)
        # log ile şişmeyi azaltıyoruz
        score = np.log1p(fav.get(rid, 0)) + np.log1p(saved.get(rid, 0)) + np.log1p(rcount.get(rid, 0))
        # avg rating katkısı küçük (0-5 -> 0-1)
        score += 0.2 * (ravg.get(rid, 0.0) / 5.0)
        raw[rid] = float(score)

    vals = np.array(list(raw.values()), dtype=float)
    if len(vals) == 0:
        return {}

    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {rid: 0.0 for rid in raw}

    return {rid: (v - mn) / (mx - mn) for rid, v in raw.items()}



def get_content_scores_for_candidates(user_id, conn, candidate_ids, lang="tr"):
    """
    Aday tariflere 0-1 normalize content similarity döndürür.
    """
    if not candidate_ids:
        return {}

    # senin fonksiyonun (TR/EN ayrımı varsa ona göre düzenle)
    user_vec = build_user_profile_vector(user_id, conn)  # (1, D) veya None
    if user_vec is None:
        return {int(rid): 0.0 for rid in candidate_ids}

    # TR/EN matris seçimi (senin dosyalarında var)
    matrix = tfidf_matrix_tr


    # cosine similarity tüm tariflere hesaplanabilir ama adaylarla sınırlayalım
    sims = {}
    # vektörler sparse ise vstack/indekslemeyi senin yapına göre gerekirse uyarlarsın
    cand_idx = [int(rid) for rid in candidate_ids if int(rid) < matrix.shape[0]]
    if not cand_idx:
        return {int(rid): 0.0 for rid in candidate_ids}

    sim_vals = cosine_similarity(user_vec, matrix[cand_idx])[0]

    for rid, s in zip(cand_idx, sim_vals):
        sims[int(rid)] = float(s)

    # normalize 0-1
    vals = np.array(list(sims.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {rid: 0.0 for rid in sims}
    return {rid: (v - mn) / (mx - mn) for rid, v in sims.items()}


def get_svd_scores(user_id, candidate_ids):
    """
    Aday tariflere 0-1 normalize SVD tahmin skoru döndürür.
    """
    if not candidate_ids:
        return {}

    preds = {}
    for rid in candidate_ids:
        rid = int(rid)
        try:
            est = float(svd_model.predict(str(user_id), str(rid)).est)
        except Exception:
            # user/recipe bilinmiyorsa düşük ver
            est = 0.0
        preds[rid] = est

    vals = np.array(list(preds.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {rid: 0.0 for rid in preds}
    return {rid: (v - mn) / (mx - mn) for rid, v in preds.items()}


def get_interaction_count(user_id, conn):
    # ratings + favorites + saved + comments toplam sayısı
    r = conn.execute("SELECT COUNT(*) c FROM ratings WHERE user_id=?", (user_id,)).fetchone()["c"]
    f = conn.execute("SELECT COUNT(*) c FROM favorites WHERE user_id=?", (user_id,)).fetchone()["c"]
    s = conn.execute("SELECT COUNT(*) c FROM saved WHERE user_id=?", (user_id,)).fetchone()["c"]
    c = conn.execute("SELECT COUNT(*) c FROM comments WHERE user_id=?", (user_id,)).fetchone()["c"]
    return int(r) + int(f) + int(s) + int(c)


def hybrid_recommendations_adaptive(user_id=None, conn=None, top_n=20, lang="tr"):
    """
    Gerçek hibrit:
    - cold start: popularity
    - az etkileşim: content + popularity
    - çok etkileşim: svd + content + popularity
    Dönen: recipe_id listesi
    """

    # Güvenli top_n
    try:
        top_n = int(top_n)
    except:
        top_n = 20
    if top_n <= 0:
        top_n = 20

    
    # 0) Kullanıcının zaten etkileştiği tarifleri al (filtre için)
    interacted = set()
    if user_id and conn:
        try:
            interacted = set(get_user_interacted_recipe_ids(conn, user_id))
        except:
            interacted = set()

    # 1) Aday havuzu oluştur
    # Popülerlerden az ama güvenilir
    # Content tabanlıdan geniş
    popular = get_top_rated_recipes(top_n=50, min_votes=3, lang=lang)  # dict list
    popular_ids = []
    for r in popular:
        try:
            if r and "id" in r:
                popular_ids.append(int(r["id"]))
        except:
            pass

    candidate_ids = set(popular_ids)

    interaction_count = 0
    if user_id and conn:
        try:
            interaction_count = int(get_interaction_count(user_id, conn))
        except:
            interaction_count = 0

        # content tabanlı adaylar (daha geniş)
        try:
            content_ids = content_based_for_user(user_id, conn, top_n=200)
        except:
            content_ids = []

        # content_ids bazen dict vb gelirse sağlamlaştır
        cleaned = []
        for x in content_ids:
            try:
                cleaned.append(int(x))
            except:
                pass

        candidate_ids.update(cleaned)

    # Eğer aday havuzu boşsa direkt boş dön
    if not candidate_ids:
        return []

    # 1.5) Adaylardan "interacted" olanları çıkar
    # (Slider’da kullanıcının kendi beğendikleri görünmesin)
    candidate_ids = [rid for rid in candidate_ids if rid not in interacted]

    # Aşırı filtre yüzünden boş kalırsa: filtreyi gevşet (en azından popüler dönsün)
    if not candidate_ids:
        candidate_ids = list(set(popular_ids))

    # Stabil sıralama (debug daha kolay)
    candidate_ids = sorted(set(candidate_ids))

    # 2) Skorları al
    # popularity scores: conn varsa DB’den, yoksa sıfır
    if conn:
        try:
            pop_scores = get_popularity_scores(conn, candidate_ids)
        except:
            pop_scores = {rid: 0.0 for rid in candidate_ids}
    else:
        pop_scores = {rid: 0.0 for rid in candidate_ids}

    # 3) Hibrit ağırlıklandırma
    if (not user_id) or (not conn) or interaction_count == 0:
        # cold start: sadece popularity
        final = {rid: float(pop_scores.get(rid, 0.0)) for rid in candidate_ids}

    elif interaction_count < 5:
        # az etkileşim: content + pop
        try:
            cont_scores = get_content_scores_for_candidates(user_id, conn, candidate_ids, lang=lang)
        except:
            cont_scores = {rid: 0.0 for rid in candidate_ids}

        final = {
            rid: 0.65 * float(cont_scores.get(rid, 0.0)) + 0.35 * float(pop_scores.get(rid, 0.0))
            for rid in candidate_ids
        }

    else:
        # çok etkileşim: svd + content + pop
        try:
            cont_scores = get_content_scores_for_candidates(user_id, conn, candidate_ids, lang=lang)
        except:
            cont_scores = {rid: 0.0 for rid in candidate_ids}

        try:
            svd_scores = get_svd_scores(user_id, candidate_ids)
        except:
            svd_scores = {rid: 0.0 for rid in candidate_ids}

        # SVD tamamen boş/0 ise: content+pop’a düş (stabil olsun)
        if not svd_scores or all(float(svd_scores.get(rid, 0.0)) == 0.0 for rid in candidate_ids):
            final = {
                rid: 0.70 * float(cont_scores.get(rid, 0.0)) + 0.30 * float(pop_scores.get(rid, 0.0))
                for rid in candidate_ids
            }
        else:
            final = {
                rid: 0.50 * float(svd_scores.get(rid, 0.0))
                   + 0.30 * float(cont_scores.get(rid, 0.0))
                   + 0.20 * float(pop_scores.get(rid, 0.0))
                for rid in candidate_ids
            }

    # 4) Sıralayıp top_n döndür
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)

    # skorlar 0 ise bile id dönsün diye sadece id seçiyoruz
    return [rid for rid, _ in ranked[:top_n]]
