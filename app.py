from flask import Flask, render_template, request, session, redirect, url_for
import sqlite3
from recommender import load_assets, MODEL_DIR
import os
from recommender import recipes_df
import re
from flask import flash 
import pandas as pd
from surprise import Reader, Dataset, SVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pickle
from recommender import get_popularity_scores
from recommender import interactions_df
from recommender import (
    recommend_recipes_for_user,
    content_based_for_user,
    get_recipe_detail,
    get_top_rated_recipes
)

RECOMMENDATION_CACHE = {}

app = Flask(__name__)
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# DB
def get_db_connection():
    conn = sqlite3.connect(os.path.join(BASE_DIR, "recipes_users.db"))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                recipe_id INTEGER NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                recipe_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                UNIQUE(user_id, recipe_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                recipe_id INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                UNIQUE(user_id, recipe_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                recipe_id INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                UNIQUE(user_id, recipe_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        conn.commit()


# DİL
@app.route("/set_lang/<code>")
def set_lang(code):
    session["lang"] = code
    return redirect(request.referrer or "/")



# HELPERS
def parse_ingredients(text: str):
    if not text:
        return []

    text = text.lower().strip()

    for sep in ["+", "/", "|", ";"]:
        text = text.replace(sep, ",")

    text = re.sub(r"\s+", " ", text)

    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if len(p) >= 2]

    uniq = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)

    return uniq


def get_popular_recipes_db(top_n=20, lang="tr"):
    conn = get_db_connection()

    rows = conn.execute("""
        SELECT
            recipe_id,
            COUNT(*) AS rating_count,
            ROUND(AVG(rating), 1) AS avg_rating
        FROM ratings
        GROUP BY recipe_id
        HAVING rating_count >= 5
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT ?
    """, (top_n,)).fetchall()

    conn.close()

    recipes = []
    for r in rows:
        recipe = get_recipe_detail(r["recipe_id"], lang)
        if recipe:
            recipe["avg_rating"] = r["avg_rating"]
            recipes.append(recipe)

    return recipes

def get_model_comments(recipe_id, limit=10):
    rows = interactions_df[
        (interactions_df["recipe_id"] == recipe_id) &
        (interactions_df["review_tr"].notnull())
    ].head(limit)

    comments = []
    for _, r in rows.iterrows():
        comments.append({
            "username": "Food.com kullanıcısı",
            "comment": r["review_tr"],
            "date": r["date"],
            "source": "model"
        })

    return comments


def get_model_avg_rating(recipe_id):
    rows = interactions_df[
        (interactions_df["recipe_id"] == recipe_id) &
        (interactions_df["rating"].notnull())
    ]

    if rows.empty:
        return None

    return round(rows["rating"].mean(), 1)



def get_final_avg_rating(recipe_id):
    model_rating = get_model_avg_rating(recipe_id)

    conn = get_db_connection()
    row = conn.execute("""
        SELECT ROUND(AVG(rating), 1) as avg
        FROM ratings
        WHERE recipe_id=?
    """, (recipe_id,)).fetchone()
    conn.close()

    db_rating = float(row["avg"]) if row and row["avg"] is not None else None

    return db_rating if db_rating is not None else model_rating



def get_trending_recipes(days=7, top_n=20, lang="tr"):
    conn = get_db_connection()

    rows = conn.execute("""
        SELECT
            r.recipe_id,
            COUNT(DISTINCT r.id) AS rating_count,
            COUNT(DISTINCT c.id) AS comment_count
        FROM ratings r
        LEFT JOIN comments c ON c.recipe_id = r.recipe_id
        WHERE r.created_at >= datetime('now', ?)
        GROUP BY r.recipe_id
    """, (f"-{days} days",)).fetchall()

    conn.close()

    scored = []
    for row in rows:
        score = row["rating_count"] * 2 + row["comment_count"]
        recipe = get_recipe_detail(row["recipe_id"], lang)
        if recipe:
            recipe["trend_score"] = score
            scored.append(recipe)

    scored = sorted(scored, key=lambda x: x["trend_score"], reverse=True)
    return scored[:top_n]

def normalize_recipe(r, lang="tr"):
    rid = r.get("id")

    #  Ortalama puan (DB + model)
    avg_rating = r.get("avg_rating")
    if avg_rating is None:
        avg_rating = get_final_avg_rating(rid)


    #  Malzeme sayısı
    n_ingredients = r.get("n_ingredients")

    if n_ingredients is None:
        row = recipes_df[recipes_df["id"] == rid]
        if not row.empty:
            try:
                ingredients = row.iloc[0]["ingredients"]
                if isinstance(ingredients, list):
                    n_ingredients = len(ingredients)
                else:
                    n_ingredients = len(str(ingredients).split(","))
            except:
                n_ingredients = None

    return {
        "id": rid,
        "name": r.get("name"),
        "minutes": r.get("minutes", 0),
        "image": r.get("image"),
        "n_ingredients": n_ingredients,
        "avg_rating": avg_rating
    }


def get_user_interacted_recipe_ids(conn, uid):
    ids = set()

    for table in ["ratings", "favorites", "saved", "comments"]:
        rows = conn.execute(
            f"SELECT DISTINCT recipe_id FROM {table} WHERE user_id=?",
            (uid,)
        ).fetchall()
        ids.update([r["recipe_id"] for r in rows])

    return ids



# HOME (4 slider dolacak)
from recommender import get_top_rated_recipes
from recommender import content_based_for_user
from recommender import hybrid_recommendations_adaptive

import random 
# ANA SAYFA (HOME) - AKILLI ID EŞLEŞTİRME 
@app.route("/")
def home():
    user_id = session.get("user_id")
    lang = session.get("lang", "tr")
    user_prefs = session.get("user_preferences", []) 

    TAG_MAP = {
        "kahvalti": "kahvaltı", "aksam-yemegi": "ana yemek", "tatli": "tatlılar",
        "saglikli": "sağlıklı", "italyan": "avrupa", "turk": "orta doğu",
        "atistirmalik": "rahat yemek", "vegan": "vejetaryen", "diyet": "diyet", "kolay": "kolay"
    }

    # 1. POPÜLER TARİFLERİ OLUŞTUR
    popular_raw = get_popular_recipes_db(top_n=60, lang=lang)
    if not popular_raw:
        popular_raw = get_top_rated_recipes(top_n=60, min_votes=1, lang=lang)
    
    if not popular_raw:
        from recommender import recipes_df
        indices = random.sample(range(len(recipes_df)), min(20, len(recipes_df)))
        for idx in indices:
            row = recipes_df.iloc[idx]
            popular_raw.append({"id": int(row["id"]), "name": row.get(f"name_{lang}", row["name"]), "avg_rating": 4.5})

    combined_map = {normalize_recipe(r, lang)['id']: normalize_recipe(r, lang) for r in popular_raw}
    all_popular = list(combined_map.values())
    all_popular.sort(key=lambda x: (-1 * (x.get('avg_rating', 0) or 0), x['id']))
    popular_recipes = all_popular[:20]

    #  ÖNBELLEK KONTROLÜ 
    if user_id and user_id in RECOMMENDATION_CACHE:
        cached = RECOMMENDATION_CACHE[user_id]
        return render_template("index.html", popular_recipes=popular_recipes, 
                             hybrid_recipes=cached['hybrid'], content_recipes=cached['content'], 
                             user_cf_recipes=cached['user_cf'])

    # 2. HESAPLAMA YAP
    hybrid_recipes, content_recipes, user_cf_recipes = [], [], []

    if user_id:
        try:
            conn = get_db_connection()
            # ID'leri Integer setine çevir
            raw_interacted = get_user_interacted_recipe_ids(conn, user_id)
            interacted = {int(rid) for rid in raw_interacted}
            
            from recommender import recipes_df # ID dönüşümü için lazım

            # --- YARDIMCI FONKSİYON: ID mi Index mi? ---
            def resolve_recipe(val):
                # 1. Önce ID olarak dene
                r = get_recipe_detail(val, lang)
                if r: return r
                
                # 2. Bulamazsan Index (Sıra No) olarak dene
                try:
                    real_id = recipes_df.iloc[int(val)]["id"]
                    r = get_recipe_detail(real_id, lang)
                    if r: return r
                except:
                    pass
                return None

            # --- A) Content Based ---
            c_vals = content_based_for_user(user_id, conn, top_n=30)
            # Modelden gelen değerleri (Index veya ID) çözümle
            for val in c_vals:
                recipe = resolve_recipe(val)
                if recipe and recipe['id'] not in interacted:
                    content_recipes.append(normalize_recipe(recipe, lang))
                if len(content_recipes) >= 10: break

            # B) User CF
            u_raw = recommend_recipes_for_user(user_id, 30, lang)
            for r in u_raw:
                if int(r["id"]) not in interacted:
                    user_cf_recipes.append(normalize_recipe(r, lang))
                if len(user_cf_recipes) >= 10: break
            
            # C) Hybrid
            h_vals = hybrid_recommendations_adaptive(user_id, conn, 30, lang)
            for val in h_vals:
                recipe = resolve_recipe(val)
                if recipe and recipe['id'] not in interacted:
                    hybrid_recipes.append(normalize_recipe(recipe, lang))
                if len(hybrid_recipes) >= 10: break
            
            conn.close()
        except Exception as e:
            print("Öneri Hatası:", e)

    # 3. FALLBACK (DOLDURMA) - BOŞ KALIRSA DEVREYE GİRER
    
    # A) Hibrit Doldur (Anketten)
    if not hybrid_recipes and user_prefs:
        tag_col = "tags_tr" if lang == "tr" else "tags"
        target_tags = [TAG_MAP.get(p, p) for p in user_prefs]
        matched_ids = set()
        from recommender import recipes_df
        for _, row in recipes_df.iterrows():
            if any(t in str(row.get(tag_col, "")).lower() for t in target_tags): matched_ids.add(row["id"])
            if len(matched_ids) >= 40: break
        
        # Etkileşimde olmayanları seç
        pool = [mid for mid in sorted(list(matched_ids)) if user_id and int(mid) not in interacted]
        if not pool: pool = sorted(list(matched_ids))

        if pool:
            rng = random.Random(user_id if user_id else "guest")
            selected = rng.sample(pool, min(len(pool), 10))
            for rid in selected:
                r = get_recipe_detail(rid, lang)
                if r: hybrid_recipes.append(normalize_recipe(r, lang))

    # B) User CF Doldur (Popülerden)
    if not user_cf_recipes and popular_recipes:
        pool = sorted(popular_recipes, key=lambda x: x['id'])
        shift = int(user_id) % len(pool) if user_id else 0
        shifted_pool = pool[shift:] + pool[:shift]
        user_cf_recipes = [p for p in shifted_pool if user_id and int(p['id']) not in interacted][:10]
        if not user_cf_recipes: user_cf_recipes = shifted_pool[:10]

    # C) Content Doldur
    if not content_recipes:
        source = hybrid_recipes if hybrid_recipes else popular_recipes
        pool = sorted(source, key=lambda x: x['id'])
        rng = random.Random(str(user_id) + "_content" if user_id else "g_cont")
        rng.shuffle(pool)
        content_recipes = pool[:10]

    if not hybrid_recipes: hybrid_recipes = popular_recipes[:10]

    # 4. SONUCU KAYDET
    if user_id:
        RECOMMENDATION_CACHE[user_id] = {
            'hybrid': hybrid_recipes[:20], 'content': content_recipes[:20], 'user_cf': user_cf_recipes[:20]
        }

    return render_template("index.html", popular_recipes=popular_recipes,
        hybrid_recipes=hybrid_recipes[:20], content_recipes=content_recipes[:20],
        user_cf_recipes=user_cf_recipes[:20])


# ÖNERİ SAYFASI 
@app.route("/recommend", methods=["POST"])
def recommend_page():
    ingredients = request.form.get("ingredients", "")
    recipe_type = request.form.get("recipe_type", "")
    notes = request.form.get("notes", "")

    result_text = "Buraya sonra modelden gelen gerçek sonuçları yazacağız."
    return render_template("recommend.html", result_text=result_text)


@app.route("/yorumlarim")
def comments_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    lang = session.get("lang", "tr")

    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT recipe_id, comment, created_at
        FROM comments
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (uid,)
    ).fetchall()
    conn.close()

    comments = []
    for r in rows:
        recipe = get_recipe_detail(r["recipe_id"], lang)
        if recipe:
            comments.append({
                "recipe": recipe,
                "comment": r["comment"],
                "created_at": r["created_at"]
            })

    return render_template(
        "comments.html",
        comments=comments,
        comment_count=len(comments)
    )


@app.route("/puanlamalarim")
def puanlamalar_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    lang = session.get("lang", "tr")

    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT recipe_id, rating, created_at
        FROM ratings
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    ).fetchall()
    conn.close()

    rated_recipes = []

    for row in rows:
        recipe = get_recipe_detail(row["recipe_id"], lang)
        if recipe:
            recipe["rating"] = row["rating"]
            recipe["rated_at"] = row["created_at"]
            rated_recipes.append(recipe)

    return render_template(
        "puanlamalar.html",
        rated_recipes=rated_recipes,
        rating_count=len(rated_recipes)
    )



@app.route("/settings")
def settings():
    if "user_id" not in session:
        return redirect(url_for("login"))

    lang = session.get("lang", "tr")

    return render_template("settings.html", lang=lang)


@app.route("/security", methods=["GET", "POST"])
def security():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        uid = session["user_id"]
        new_password = request.form.get("password")

        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET password=? WHERE id=?",
            (new_password, uid)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("account"))

    return render_template("security.html")



# MALZEME SORGU
@app.route("/malzeme_sorgu", methods=["GET", "POST"])
def malzeme_sorgu():
    recipes = None
    lang = session.get("lang", "tr")

    if request.method == "POST":
        raw_input = request.form.get("ingredients", "")
        ingredients_list = parse_ingredients(raw_input)

        ingredient_col = "ingredients_tr" if lang == "tr" else "ingredients"

        matched_recipes = []

        if ingredients_list:
            for _, row in recipes_df.iterrows():
                recipe_ingredients = str(row.get(ingredient_col, "")).lower()

                match_count = sum(
                    1 for ing in ingredients_list if ing in recipe_ingredients
                )

                if match_count > 0:
                    matched_recipes.append({
                        "id": int(row["id"]),
                        "name": row.get("name_tr") or row.get("name"),
                        "description": row.get("description_tr") or row.get("description"),
                        "tags": row.get("tags_tr") if isinstance(row.get("tags_tr"), list) else [],
                        "date": "—",
                        "score": match_count
                    })

        recipes = sorted(
            matched_recipes,
            key=lambda x: x["score"],
            reverse=True
        )[:30]

    return render_template("malzeme_sorgu.html", recipes=recipes)



# REGISTER / LOGIN
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_db_connection()
        try:
            cur = conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )
            conn.commit()
            session["user_id"] = cur.lastrowid

        except sqlite3.IntegrityError:
            conn.close()
            return "<h3 style='color:white'>Bu kullanıcı adı veya mail zaten kayıtlı!</h3>"

        conn.close()
        return redirect(url_for("onboarding"))

    return render_template("register.html")


@app.route("/onboarding")
def onboarding():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("onboarding.html")


@app.route("/onboarding_complete", methods=["POST"])
def onboarding_complete():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Kullanıcının seçtiği etiketleri al (Liste olarak)
    preferences = request.form.getlist("preferences")
    
    # Bu tercihleri session'a (oturuma) kaydet
    session["user_preferences"] = preferences

    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, password)
        ).fetchone()
        conn.close()

        if user is None:
            return "<h3 style='color:white'>Giriş başarısız! Kullanıcı veya şifre hatalı.</h3>"

        session["user_id"] = user["id"]

        """
        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET last_login=datetime('now','localtime') WHERE id=?",
            (user["id"],)
        )
        conn.commit()"""

        conn.close()
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# PROFİL
@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]

    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

    saved_count = conn.execute("SELECT COUNT(*) FROM saved WHERE user_id=?", (uid,)).fetchone()[0]
    liked_count = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id=?", (uid,)).fetchone()[0]
    comment_count = conn.execute("SELECT COUNT(*) FROM comments WHERE user_id=?", (uid,)).fetchone()[0]

    conn.close()

    if user is None:
        return "<h3 style='color:white'>Kullanıcı bulunamadı!</h3>"

    return render_template(
        "profile.html",
        user=user,
        saved_count=saved_count,
        liked_count=liked_count,
        comment_count=comment_count
    )


@app.route("/account")
def account():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    lang = session.get("lang", "tr")

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, username, email FROM users WHERE id=?",
        (uid,)
    ).fetchone()
    conn.close()

    return render_template(
        "account.html",
        user=user,
        lang=lang
    )

@app.route("/account/update_username", methods=["POST"])
def update_username():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    new_username = request.form.get("username", "").strip()
    if not new_username:
        return redirect(url_for("account"))

    conn = get_db_connection()
    try:
        conn.execute("UPDATE users SET username=? WHERE id=?", (new_username, uid))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return "<h3 style='color:white'>Bu kullanıcı adı zaten kullanılıyor!</h3>"
    conn.close()
    return redirect(url_for("account"))


@app.route("/account/update_email", methods=["POST"])
def update_email():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    new_email = request.form.get("email", "").strip()
    if not new_email:
        return redirect(url_for("account"))

    conn = get_db_connection()
    try:
        conn.execute("UPDATE users SET email=? WHERE id=?", (new_email, uid))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return "<h3 style='color:white'>Bu e-posta zaten kullanılıyor!</h3>"
    conn.close()
    return redirect(url_for("account"))



# TARİF DETAY
@app.route("/tarif/<int:id>")
def tarif_detay(id):
    lang = session.get("lang", "tr")
    uid = session.get("user_id")

    #  Tarif verisi
    recipe_data = get_recipe_detail(id, lang)
    if recipe_data is None:
        return "<h3>Tarif bulunamadı</h3>"

    #  DB bağlantısı
    conn = get_db_connection()

    #  Ortalama puan (model + db)
    avg_rating = get_final_avg_rating(id)

    #  YORUMLAR
    comments_list = []

    #  A) MODEL  yorumları
    model_rows = interactions_df[
        (interactions_df["recipe_id"] == id) &
        (interactions_df["review_tr"].notnull())
    ].head(10)

    for _, r in model_rows.iterrows():
        comments_list.append({
            "username": "Food.com kullanıcısı",
            "comment": r["review_tr"],
            "date": r["date"],
            "rating": round(float(r["rating"]), 1) if r["rating"] else None,
            "source": "model"
        })

    #  B) UYGULAMA kullanıcı yorumları
    rows = conn.execute("""
        SELECT c.comment, c.created_at, u.username
        FROM comments c
        JOIN users u ON u.id = c.user_id
        WHERE c.recipe_id=?
        ORDER BY c.id DESC
    """, (id,)).fetchall()

    for r in rows:
        comments_list.append({
            "username": r["username"],
            "comment": r["comment"],
            "date": r["created_at"],
            "rating": None,
            "source": "user"
        })

    #  Kullanıcı durumları
    is_favorite = False
    is_saved = False
    my_rating = None

    if uid:
        is_favorite = conn.execute(
            "SELECT 1 FROM favorites WHERE user_id=? AND recipe_id=?",
            (uid, id)
        ).fetchone() is not None

        is_saved = conn.execute(
            "SELECT 1 FROM saved WHERE user_id=? AND recipe_id=?",
            (uid, id)
        ).fetchone() is not None

        r = conn.execute(
            "SELECT rating FROM ratings WHERE user_id=? AND recipe_id=?",
            (uid, id)
        ).fetchone()
        my_rating = float(r["rating"]) if r else None

    conn.close()

    #  Template
    return render_template(
        "tarif_detay.html",
        recipe=recipe_data,
        comments_list=comments_list,
        is_favorite=is_favorite,
        is_saved=is_saved,
        my_rating=my_rating,
        avg_rating=avg_rating
    )




# KAYDEDİLENLER
@app.route("/kaydedilenler")
def kaydedilenler_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    lang = session.get("lang", "tr")

    conn = get_db_connection()

    rows = conn.execute("""
        SELECT recipe_id, created_at
        FROM saved
        WHERE user_id=?
        ORDER BY id DESC
    """, (uid,)).fetchall()

    conn.close()

    saved_recipes = []

    for row in rows:
        recipe = get_recipe_detail(row["recipe_id"], lang)
        if recipe:
            recipe["saved_at"] = row["created_at"]
            saved_recipes.append(recipe)

    return render_template(
        "kaydedilenler.html",
        saved_recipes=saved_recipes,
        saved_count=len(saved_recipes)
    )



# FAVORİLER
@app.route("/favoriler")
def favoriler_page():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    lang = session.get("lang", "tr")

    conn = get_db_connection()
    rows = conn.execute(
        "SELECT recipe_id, created_at FROM favorites WHERE user_id=? ORDER BY id DESC",
        (uid,)
    ).fetchall()
    conn.close()

    liked_recipes = []
    for row in rows:
        recipe = get_recipe_detail(row["recipe_id"], lang)
        if recipe:
            recipe["liked_at"] = row["created_at"]
            liked_recipes.append(recipe)

    return render_template(
        "favoriler.html",
        liked_recipes=liked_recipes,
        liked_count=len(liked_recipes)
    )

def get_most_favorited_recipes(top_n=30, lang="tr"):
    conn = get_db_connection()

    rows = conn.execute("""
        SELECT recipe_id, COUNT(*) AS fav_count
        FROM favorites
        GROUP BY recipe_id
        ORDER BY fav_count DESC
        LIMIT ?
    """, (top_n,)).fetchall()

    conn.close()

    results = []
    for r in rows:
        recipe = get_recipe_detail(r["recipe_id"], lang)
        if recipe:
            recipe["fav_count"] = r["fav_count"]
            results.append(recipe)

    return results



@app.route("/en-begenilenler")
def most_favorited_page():
    lang = session.get("lang", "tr")
    recipes = get_most_favorited_recipes(lang=lang)
    return render_template("en_begenilenler.html", recipes=recipes)



def get_most_saved_recipes(top_n=30, lang="tr"):
    conn = get_db_connection()

    rows = conn.execute("""
        SELECT recipe_id, COUNT(*) AS save_count
        FROM saved
        GROUP BY recipe_id
        ORDER BY save_count DESC
        LIMIT ?
    """, (top_n,)).fetchall()

    conn.close()

    results = []
    for r in rows:
        recipe = get_recipe_detail(r["recipe_id"], lang)
        if recipe:
            recipe["save_count"] = r["save_count"]
            results.append(recipe)

    return results



@app.route("/en-cok-kaydedilenler")
def most_saved_page():
    lang = session.get("lang", "tr")
    recipes = get_most_saved_recipes(lang=lang)
    return render_template("en_cok_kaydedilenler.html", recipes=recipes)



@app.route("/etiket/<category>/<tag>")
def recipes_by_tag(category, tag):
    lang = session.get("lang", "tr")
    
    tag_col = "tags_tr" if lang == "tr" else "tags"

    TAG_MAP_TR = {
        "kahvalti": "kahvaltı",
        "ogle-yemegi": "öğle yemeği",
        "aksam-yemegi": "akşam yemeği",
        "atistirmalik": "atıştırmalık",
        "tatli": "tatlı",
        "yan-yemek": "yan yemek",

        # 🌍 Mutfaklar
        "turk": "türk",
        "italyan": "italyan",
        "asya": "asya",
        "meksika": "meksika",
        "amerikan": "amerikan",

        # 🥗 Diyet & Sağlık
        "vejetaryen": "vejetaryen",
        "vegan": "vegan",
        "fit-saglikli": "diyet",
        "glutensiz": "glutensiz",
        "dusuk-karbonhidrat": "düşük karbonhidrat",

        # 🍳 Pişirme Yöntemi
        "firin": "fırın",
        "tava": "tava",
        "air-fryer": "fritöz",
        "izgara-bbq": "ızgara",
        "mikrodalga": "mikrodalga",
        "yavas-pisirme": "yapma zamanı",

        # 🌤️ Mevsimsel
        "yaz": "yaz",
        "kis": "kış",
        "ilkbahar": "ilkbahar",
        "sonbahar": "sonbahar",


    }

    real_tag = TAG_MAP_TR.get(tag, tag).lower()
    matched = []

    for _, row in recipes_df.iterrows():
        tags = row.get(tag_col)
        if not tags:
            continue

        # tags string ise listeye çevir
        if isinstance(tags, str):
            tags_list = [t.strip().lower() for t in tags.split(",")]
        else:
            tags_list = [str(t).lower() for t in tags]

        #  ASIL DOĞRU KONTROL
        if any(real_tag in t for t in tags_list):
            matched.append({
                "id": int(row["id"]),
                "name": row["name_tr"] if lang == "tr" else row["name"],
                "minutes": row["minutes"],
                "n_ingredients": row["n_ingredients"]
            })

    return render_template(
        "etiket_listesi.html",
        recipes=matched,
        title=real_tag.title()
    )


# TREND TARİFLER
@app.route("/trend")
def trend_page():
    lang = session.get("lang", "tr")

    recipes = get_trending_recipes(
        days=7,
        top_n=30,
        lang=lang
    )

    return render_template(
        "trend.html",
        recipes=recipes
    )



# SEARCH
@app.route("/search")
def search():
    q = request.args.get("q", "").strip().lower()
    lang = session.get("lang", "tr")

    results = []

    if q:
        for _, row in recipes_df.iterrows():
            name = (row.get("name_tr") if lang == "tr" else row.get("name")) or ""
            desc = (row.get("description_tr") if lang == "tr" else row.get("description")) or ""

            text = f"{name} {desc}".lower()

            if q in text:
                results.append({
                    "id": int(row["id"]),
                    "name": name,
                    "description": desc
                })

    return render_template(
        "search_results.html",
        query=q,
        results=results
    )



# FAVORITES (EKLE / SİL)
@app.route("/favorite/<int:recipe_id>", methods=["POST"])
def toggle_favorite(recipe_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    conn = get_db_connection()

    exists = conn.execute(
        "SELECT 1 FROM favorites WHERE user_id=? AND recipe_id=?",
        (uid, recipe_id)
    ).fetchone()

    if exists:
        conn.execute(
            "DELETE FROM favorites WHERE user_id=? AND recipe_id=?",
            (uid, recipe_id)
        )
    else:
        conn.execute(
            "INSERT INTO favorites (user_id, recipe_id) VALUES (?, ?)",
            (uid, recipe_id)
        )

    conn.commit()
    conn.close()
    return redirect(url_for("tarif_detay", id=recipe_id))



# SAVED (EKLE / SİL)
@app.route("/save/<int:recipe_id>", methods=["POST"])
def toggle_saved(recipe_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    conn = get_db_connection()

    exists = conn.execute(
        "SELECT 1 FROM saved WHERE user_id=? AND recipe_id=?",
        (uid, recipe_id)
    ).fetchone()

    if exists:
        conn.execute(
            "DELETE FROM saved WHERE user_id=? AND recipe_id=?",
            (uid, recipe_id)
        )
    else:
        conn.execute(
            "INSERT INTO saved (user_id, recipe_id) VALUES (?, ?)",
            (uid, recipe_id)
        )

    conn.commit()
    conn.close()
    return redirect(url_for("tarif_detay", id=recipe_id))



# RATINGS (EKLE / GÜNCELLE)
@app.route("/rate/<int:recipe_id>", methods=["POST"])
def rate_recipe(recipe_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    uid = session["user_id"]
    rating = request.form.get("rating")

    try:
        rating = float(rating)
    except:
        return redirect(url_for("tarif_detay", id=recipe_id))

    if rating < 1:
        rating = 1
    if rating > 5:
        rating = 5

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO ratings (user_id, recipe_id, rating)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, recipe_id)
        DO UPDATE SET rating=excluded.rating,
                      created_at=datetime('now','localtime')
    """, (uid, recipe_id, rating))

    conn.commit()
    conn.close()
    return redirect(url_for("tarif_detay", id=recipe_id))



# COMMENTS (EKLE)
@app.route("/comment/<int:recipe_id>", methods=["POST"])
def add_comment(recipe_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    text = request.form.get("comment", "").strip()
    if not text:
        return redirect(url_for("tarif_detay", id=recipe_id))

    uid = session["user_id"]
    conn = get_db_connection()

    conn.execute(
        "INSERT INTO comments (user_id, recipe_id, comment) VALUES (?, ?, ?)",
        (uid, recipe_id, text)
    )

    conn.commit()
    conn.close()
    return redirect(url_for("tarif_detay", id=recipe_id))



import recommender 
from flask import flash # Mesaj göstermek için
@app.route("/update_recommendations", methods=["POST"])
def update_recommendations():
    if "user_id" not in session: return redirect(url_for("login"))

    # 1. ORİJİNAL VERİ YAPISINI KORUMAK İÇİN CSV'Yİ YÜKLEME
    from recommender import interactions_df as original_csv_df
    
    # 2. YENİ VERİLERİ DB'DEN ÇEK
    conn = get_db_connection()
    df_r = pd.read_sql("SELECT user_id, recipe_id, rating, created_at as date FROM ratings", conn)
    
    df_f = pd.read_sql("SELECT user_id, recipe_id, created_at as date FROM favorites", conn)
    df_f["rating"] = 5.0
    
    df_s = pd.read_sql("SELECT user_id, recipe_id, created_at as date FROM saved", conn)
    df_s["rating"] = 4.0
    conn.close()

    # 3. VERİLERİ BİRLEŞTİR
    new_interactions = pd.concat([df_r, df_f, df_s], ignore_index=True)
    
    # Eğer veri yoksa dön
    if len(new_interactions) < 3:
        flash("Yeterli veri yok. Biraz daha etkileşim yapın!", "warning")
        return redirect(url_for("profile"))

    # 4. EKSİK SÜTUNLARI DOLDUR 
    # Orijinal df'de olup bizde olmayan sütunları boş/varsayılan değerle doldur
    for col in original_csv_df.columns:
        if col not in new_interactions.columns:
            new_interactions[col] = "" 
            
    # Sütun sırasını orijinaliyle aynı yap 
    new_interactions = new_interactions[original_csv_df.columns]

    # 5. GEÇMİŞ + BUGÜN BİRLEŞTİR 
    full_training_data = pd.concat([original_csv_df, new_interactions], ignore_index=True)
    
    # En son oyu baz alarak çakışmaları temizleme
    full_training_data = full_training_data.groupby(["user_id", "recipe_id"], as_index=False).last()

    try:
        # 6. MODELLERİ EĞİT
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(full_training_data[['user_id', 'recipe_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        new_svd = SVD()
        new_svd.fit(trainset)

        pivot = full_training_data.pivot(index='user_id', columns='recipe_id', values='rating').fillna(0)
        sparse_matrix = csr_matrix(pivot.values)
        new_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        new_knn.fit(sparse_matrix)

        # 7. DİSKE KAYDET
        with open(os.path.join(MODEL_DIR, "svd_model_v1.pkl"), "wb") as f: pickle.dump(new_svd, f)
        with open(os.path.join(MODEL_DIR, "knn_user_model.pkl"), "wb") as f: pickle.dump(new_knn, f)
        with open(os.path.join(MODEL_DIR, "user_recipe_sparse.pkl"), "wb") as f: pickle.dump(sparse_matrix, f)

        # 8. CANLI HAFIZAYI GÜNCELLE
        recommender.svd_model = new_svd
        recommender.user_knn_model = new_knn
        recommender.user_recipe_sparse = sparse_matrix
        
        recommender.interactions_df = full_training_data 

        # 9. CACHE SİL
        uid = session["user_id"]
        if uid in RECOMMENDATION_CACHE: del RECOMMENDATION_CACHE[uid]

        flash(" Öneri motoru başarıyla güncellendi! Yeni beğenilerin işlendi.", "success")

    except Exception as e:
        print("Hata:", e)
        flash(f"Hata: {str(e)}", "error")

    return redirect(url_for("profile"))

# =========================
# UYGULAMAYI BAŞLAT 
# =========================
if __name__ == "__main__":
    init_db()
    app.run(debug=True)