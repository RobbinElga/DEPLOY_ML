import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ================================================
# 📌 LOAD PAKET MODEL REKOMENDASI TEMAN (.pkl)
# ================================================
with open("model/cluster_friend_recommender.pkl", "rb") as f:
    friendrec_package = pickle.load(f)

scaler = friendrec_package["scaler"]
model = friendrec_package["model"]
feature_cols = friendrec_package["feature_cols"]
cluster_names = friendrec_package["cluster_names"]
friend_distribution = friendrec_package["friend_distribution"]

# ================================================
# 📌 Dummy/Database user_features (HARUS kamu ganti dengan database asli!)
# ================================================
# NOTE:
user_features = pd.read_csv("model/user_features_cluster_final_named.csv")  

# ================================================
# 📌 Fungsi rekomendasi teman
# ================================================
def recommend_friends_by_cluster(
    user_id,
    df_user,
    max_total=10,
    distribution_map=friend_distribution,
    random_state=42
):

    # Pastikan user ada
    row = df_user[df_user["user_id"] == user_id]
    if row.empty:
        return None, None, None  # nanti di-route kita handle error
    
    row = row.iloc[0]

    current_cluster = int(row["cluster_final"])
    current_name = row.get("cluster_final_name", str(current_cluster))

    if current_cluster not in distribution_map:
        dist = {current_cluster: max_total}
    else:
        dist = distribution_map[current_cluster]

    rec_rows = []
    total_added = 0

    rng = np.random.RandomState(random_state)

    for target_cluster, target_n in dist.items():
        if total_added >= max_total:
            break

        remaining_slots = max_total - total_added
        n_to_take = min(target_n, remaining_slots)

        candidates = df_user[
            (df_user["cluster_final"] == target_cluster) &
            (df_user["user_id"] != user_id)
        ]

        if candidates.empty:
            continue

        if len(candidates) <= n_to_take:
            sampled = candidates
        else:
            sampled = candidates.sample(n=n_to_take, random_state=rng)

        rec_rows.append(sampled)
        total_added += len(sampled)

    if not rec_rows:
        return current_cluster, current_name, pd.DataFrame()

    rec_df = (
        pd.concat(rec_rows, axis=0)
        .drop_duplicates(subset=["user_id"])
        .head(max_total)
    )

    return current_cluster, current_name, rec_df


# ================================================
# 📌 ROUTE API
# ================================================

@app.route("/")
def index():
    return {"status": "SUCCESS", "message": "service is up"}, 200


# -----------------------------------------------
# 🚀 ROUTE: Rekomendasi Teman
# -----------------------------------------------
@app.route("/recommend-friends", methods=["GET"])
def recommend_friends():
    try:
        user_id = request.args.get("user_id", type=int)

        if user_id is None:
            return jsonify({
                "status": "ERROR",
                "message": "Parameter 'user_id' wajib diisi"
            }), 400
        
        current_cluster, current_name, rec_df = recommend_friends_by_cluster(
            user_id=user_id,
            df_user=user_features,
            max_total=10
        )

        if current_cluster is None:
            return jsonify({
                "status": "ERROR",
                "message": f"User dengan id {user_id} tidak ditemukan"
            }), 404

        # Convert recommendation df ke JSON list
        recommendations = []
        if rec_df is not None and len(rec_df) > 0:
            for _, row in rec_df.iterrows():
                recommendations.append({
                    "user_id": int(row["user_id"]),
                    "username": row.get("username", None),
                    "cluster": int(row["cluster_final"]),
                    "cluster_name": row.get("cluster_final_name", "")
                })

        return jsonify({
            "status": "SUCCESS",
            "user_id": user_id,
            "user_cluster": current_cluster,
            "user_cluster_name": current_name,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }), 200

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500


# ================================================
# RUN SERVER
# ================================================
if __name__ == "__main__":
    app.run(debug=True)
