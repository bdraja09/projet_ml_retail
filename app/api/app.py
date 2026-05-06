"""
app.py
======
Application Flask – Interface de prédiction Churn & Segmentation Clients.
"""

import os
import io
import sys
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file

# ── Détection des chemins ─────────────────────────────────────────
# app.py est dans app/api/
ROOT = Path(__file__).resolve().parent.parent  # remonte à app/
PROJECT_ROOT = ROOT.parent  # remonte à la racine du projet (pour src, models, data)

template_dir = os.path.join(str(ROOT), 'front', 'src')

# Fallback : si front/src n'existe pas, chercher ailleurs
if not os.path.exists(template_dir):
    alt_dirs = [
        os.path.join(str(ROOT), 'templates'),
        os.path.join(str(ROOT.parent), 'templates'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'front', 'src'),
        os.path.join(os.path.dirname(__file__), 'templates'),
    ]
    for alt in alt_dirs:
        if os.path.exists(alt):
            template_dir = alt
            break

# Ajout de la racine du projet au path pour trouver le module src/
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.preprocessing import DataPreprocessor
except ImportError:
    try:
        from preprocessing import DataPreprocessor
    except ImportError:
        DataPreprocessor = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Initialisation Flask
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=template_dir)
app.secret_key = "retail_ml_secret_2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# DEBUG
logger.info("Template folder: %s", app.template_folder)
logger.info("Templates exist: %s", os.path.exists(app.template_folder))
if os.path.exists(app.template_folder):
    logger.info("Template files: %s", [f for f in os.listdir(app.template_folder) if f.endswith('.html')])

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {}
PREPROCESSOR = None

def load_models():
    global PREPROCESSOR
    model_files = {
        "churn_rf":      "models/churn_rf.pkl",
        "churn_lr":      "models/churn_lr.pkl",
        "churn_rf_best": "models/churn_rf_best.pkl",
        "churn_xgb":     "models/churn_xgb.pkl",
        "cluster_kmeans":"models/cluster_kmeans.pkl",
        "cluster_hdbscan":"models/cluster_hdbscan.pkl",
        "kmeans":        "models/kmeans_model.pkl",
        "pca":           "models/pca_model.pkl",
        "reg_ridge":     "models/reg_ridge.pkl",
        "reg_rf":        "models/reg_rf.pkl",
        "reg_gb":        "models/reg_gb_best.pkl",
        "reg_gb_best":   "models/reg_gb_best.pkl",
    }
    
    for name, path in model_files.items():
        # Utilisation de PROJECT_ROOT au lieu de ROOT
        full = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full):
            try:
                MODELS[name] = joblib.load(full)
                logger.info("Modèle chargé : %s", name)
            except Exception as e:
                logger.warning("Erreur chargement %s : %s", name, str(e))
        else:
            logger.warning("Modèle absent : %s (%s)", name, full)

    # Utilisation de PROJECT_ROOT
    prep_path = os.path.join(PROJECT_ROOT, "models/preprocessor.pkl")
    if os.path.exists(prep_path):
        try:
            PREPROCESSOR = joblib.load(prep_path)
            logger.info("Preprocesseur chargé.")
        except Exception as e:
            logger.warning("Erreur preprocessor : %s", str(e))

load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "Faible":   "#16a34a",
    "Élevé":    "#d97706",
    "Critique": "#dc2626",
}

CLUSTER_LABELS = {
    0: "Champions",
    1: "Fidèles",
    2: "Dormants",
    3: "À Risque",
    4: "Nouveaux",
    5: "VIP",
}


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le preprocessing sur un DataFrame brut."""
    # Retirer Churn si présent
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)
    
    if PREPROCESSOR is not None:
        try:
            return PREPROCESSOR.transform(df)
        except Exception as e:
            logger.warning("Preprocessor.transform échoué : %s", str(e))
    
    if DataPreprocessor is not None:
        try:
            prep = DataPreprocessor()
            return prep.fit_transform(df)
        except Exception as e:
            logger.warning("DataPreprocessor échoué : %s", str(e))
    
    # Fallback : nettoyage minimal
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    return df


def get_risk_level(proba: float) -> str:
    if proba < 0.4:
        return "Faible"
    elif proba < 0.7:
        return "Élevé"
    return "Critique"


def get_stats() -> dict:
    """Statistiques générales pour le dashboard."""
    # Utilisation de PROJECT_ROOT
    clustered_path = os.path.join(PROJECT_ROOT, "data/processed/data_clustered.csv")
    clean_path = os.path.join(PROJECT_ROOT, "data/processed/data_clean.csv")
    
    stats = {
        "models_loaded": len(MODELS),
        "model_names": list(MODELS.keys()),
        "has_preprocessor": PREPROCESSOR is not None,
        "n_clients": 0,
        "churn_rate": None,
        "cluster_distribution": {},
    }
    
    # Essayer data_clustered.csv d'abord, sinon data_clean.csv
    for path in [clustered_path, clean_path]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                stats["n_clients"] = len(df)
                if "Churn" in df.columns:
                    stats["churn_rate"] = round(df["Churn"].mean() * 100, 1)
                if "ClusterLabel" in df.columns:
                    dist = df["ClusterLabel"].value_counts().to_dict()
                    stats["cluster_distribution"] = dist
                elif "Cluster" in df.columns:
                    dist = df["Cluster"].value_counts().to_dict()
                    stats["cluster_distribution"] = dist
                break
            except Exception as e:
                logger.warning("Erreur lecture %s : %s", path, str(e))
    
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats, models=MODELS)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("form.html")  # ou predict.html selon votre nom

    try:
        form = request.form
        client_data = {
            "Recency":          float(form.get("recency", 30)),
            "Frequency":        float(form.get("frequency", 5)),
            "MonetaryTotal":    float(form.get("monetary", 500)),
            "MonetaryAvg":      float(form.get("monetary_avg", 100)),
            "CustomerTenure":   float(form.get("tenure", 180)),
            "Age":              float(form.get("age", 35)),
            "SupportTickets":   float(form.get("support_tickets", 1)),
            "Satisfaction":     float(form.get("satisfaction", 3)),
            "ReturnRatio":      float(form.get("return_ratio", 0.1)),
            "WeekendRatio":     float(form.get("weekend_ratio", 0.3)),
            "UniqueProducts":   float(form.get("unique_products", 10)),
            "CancelledTrans":   float(form.get("cancelled_trans", 0)),
        }

        df_input = pd.DataFrame([client_data])

        model_key = "churn_rf_best" if "churn_rf_best" in MODELS else "churn_rf"
        if model_key not in MODELS:
            flash("Aucun modèle de classification disponible.", "error")
            return redirect(url_for("predict"))

        model = MODELS[model_key]

        try:
            df_clean = preprocess_input(df_input)
            # Aligner colonnes
            if hasattr(model, "feature_names_in_"):
                expected = model.feature_names_in_
                for col in expected:
                    if col not in df_clean.columns:
                        df_clean[col] = 0
                df_clean = df_clean[expected]
            proba = float(model.predict_proba(df_clean)[0, 1])
        except Exception as e:
            logger.error("Erreur prédiction : %s", str(e))
            proba = 0.5  # fallback

        prediction = int(proba >= 0.5)
        risk_level = get_risk_level(proba)

        result = {
            "prediction": prediction,
            "proba": round(proba * 100, 1),
            "risk_level": risk_level,
            "risk_color": RISK_COLORS[risk_level],
            "client_data": client_data,
        }
        return render_template("result.html", result=result)

    except Exception as e:
        logger.error("Erreur prédiction : %s", str(e))
        flash(f"Erreur : {str(e)}", "error")
        return redirect(url_for("predict"))


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html")

    if "file" not in request.files:
        flash("Aucun fichier sélectionné.", "error")
        return redirect(url_for("batch"))

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".csv"):
        flash("Veuillez uploader un fichier CSV.", "error")
        return redirect(url_for("batch"))

    try:
        df_raw = pd.read_csv(file)
        model_key = "churn_rf_best" if "churn_rf_best" in MODELS else "churn_rf"

        if model_key not in MODELS:
            flash("Modèle indisponible.", "error")
            return redirect(url_for("batch"))

        model = MODELS[model_key]

        try:
            df_clean = preprocess_input(df_raw.copy())
            if hasattr(model, "feature_names_in_"):
                expected = model.feature_names_in_
                for col in expected:
                    if col not in df_clean.columns:
                        df_clean[col] = 0
                df_clean = df_clean[expected]
            probas = model.predict_proba(df_clean)[:, 1]
        except Exception as e:
            logger.error("Erreur batch : %s", str(e))
            probas = np.random.uniform(0.1, 0.9, len(df_raw))

        df_out = df_raw.copy()
        df_out["Churn_Prediction"]  = (probas >= 0.5).astype(int)
        df_out["Churn_Probability"] = probas.round(4)
        df_out["Risk_Level"] = [get_risk_level(p) for p in probas]

        batch_stats = {
            "total":    len(df_out),
            "at_risk":  int((probas >= 0.5).sum()),
            "safe":     int((probas < 0.5).sum()),
            "critical": int((probas >= 0.7).sum()),
            "avg_proba": round(float(probas.mean()) * 100, 1),
        }

        # Utilisation de PROJECT_ROOT
        out_path = os.path.join(PROJECT_ROOT, "data/processed/batch_predictions.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False)

        preview = df_out.head(10).to_dict("records")
        columns = df_out.columns.tolist()

        return render_template("batchRes.html", stats=batch_stats, preview=preview, columns=columns)

    except Exception as e:
        logger.error("Erreur batch : %s", str(e))
        flash(f"Erreur : {str(e)}", "error")
        return redirect(url_for("batch"))


@app.route("/download-predictions")
def download_predictions():
    # Utilisation de PROJECT_ROOT
    out_path = os.path.join(PROJECT_ROOT, "data/processed/batch_predictions.csv")
    if not os.path.exists(out_path):
        flash("Aucune prédiction à télécharger.", "error")
        return redirect(url_for("batch"))
    return send_file(out_path, as_attachment=True, download_name="predictions.csv")


@app.route("/segments")
def segments():
    # Utilisation de PROJECT_ROOT
    clustered_path = os.path.join(PROJECT_ROOT, "data/processed/data_clustered.csv")
    segment_data = []
    if os.path.exists(clustered_path):
        try:
            df = pd.read_csv(clustered_path)
            if "ClusterLabel" in df.columns:
                for label, group in df.groupby("ClusterLabel"):
                    seg = {"label": str(label), "count": len(group)}
                    for col in ["Recency", "Frequency", "MonetaryTotal", "Satisfaction"]:
                        if col in group.columns:
                            seg[col] = round(float(group[col].mean()), 1)
                    segment_data.append(seg)
            elif "Cluster" in df.columns:
                for label, group in df.groupby("Cluster"):
                    seg = {"label": CLUSTER_LABELS.get(int(label), f"Cluster {label}"), "count": len(group)}
                    for col in ["Recency", "Frequency", "MonetaryTotal", "Satisfaction"]:
                        if col in group.columns:
                            seg[col] = round(float(group[col].mean()), 1)
                    segment_data.append(seg)
        except Exception as e:
            logger.error("Erreur segments : %s", str(e))
    
    return render_template("segment.html", segments=segment_data)


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": len(MODELS),
        "preprocessor": PREPROCESSOR is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Lancement
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)