"""
regression.py
=============
Régression pour prédire la valeur monétaire future d'un client (MonetaryTotal).

Modèles entraînés :
    1. Ridge Regression (régularisation L2)
    2. Random Forest Regressor
    3. Gradient Boosting Regressor
    4. Optimisation GridSearchCV sur le meilleur modèle

Métriques : MAE, RMSE, R²

Sorties :
    - models/reg_ridge.pkl
    - models/reg_rf.pkl
    - models/reg_gb_best.pkl
    - reports/regression_comparison.png
    - reports/actual_vs_predicted.png
    - reports/residuals.png
"""

import os
import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# Chargement & Préparation
# ─────────────────────────────────────────────────────────────────────────────

def load_regression_data(
    train_path: str = "data/train_test/X_train.csv",
    test_path:  str = "data/train_test/X_test.csv",
    target_col: str = "MonetaryTotal",
) -> tuple:
    """
    Charge les splits train/test.

    Si MonetaryTotal n'est pas dans X_train (déjà droppée en preprocessing),
    tente de la récupérer depuis data/processed/data_clean.csv.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier manquant : {p}. Lancez preprocessing.py d'abord.")

    X_train = pd.read_csv(train_path)
    X_test  = pd.read_csv(test_path)

    # Cas 1 : target disponible dans X_train (données complètes)
    if target_col in X_train.columns and target_col in X_test.columns:
        # Identifier les colonnes catégorielles
        cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

        if cat_cols:
            logger.info("Encodage One-Hot des colonnes : %s", cat_cols)
            
            # Concaténer pour fit sur tout
            X_combined = pd.concat([X_train, X_test], axis=0)
            
            # One-Hot Encoding
            X_encoded = pd.get_dummies(X_combined, columns=cat_cols, drop_first=True)
            
            # Resplit
            split_idx = len(X_train)
            X_train = X_encoded.iloc[:split_idx].reset_index(drop=True)
            X_test = X_encoded.iloc[split_idx:].reset_index(drop=True)
            y_train = X_train.pop(target_col)
            y_test  = X_test.pop(target_col)
            logger.info("Target '%s' extraite de X_train/X_test.", target_col)

    # Cas 2 : reconstruire depuis data_clean.csv
    elif os.path.exists("../data/processed/data_clean.csv"):
        logger.warning(
            "'%s' absente de X_train. Reconstruction depuis data_clean.csv.", target_col
        )
        df_clean = pd.read_csv("../data/processed/data_clean.csv")
        if target_col not in df_clean.columns:
            raise ValueError(
                f"'{target_col}' introuvable dans data_clean.csv. "
                "Vérifiez que MonetaryTotal n'est pas dans COLS_TO_DROP."
            )
        from sklearn.model_selection import train_test_split
        X_all = df_clean.drop(columns=["Churn", target_col], errors="ignore")
        y_all = df_clean[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )
    else:
        raise ValueError(
            f"'{target_col}' introuvable. Assurez-vous que preprocessing.py "
            "conserve MonetaryTotal ou adaptez target_col."
        )

    # Nettoyage des NaN
    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

    logger.info("Train : %s | Test : %s", X_train.shape, X_test.shape)
    logger.info("Target — min=%.1f | max=%.1f | mean=%.1f",
                y_train.min(), y_train.max(), y_train.mean())
    return X_train, X_test, y_train, y_test



# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regressor(
    model,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    name: str,
) -> dict:
    """Calcule et affiche les métriques de régression."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Cross-val sur train
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    separator = "─" * 52
    print(f"\n{separator}")
    print(f"  MODÈLE : {name}")
    print(separator)
    print(f"  MAE   : {mae:>10.2f} £")
    print(f"  RMSE  : {rmse:>10.2f} £")
    print(f"  R²    : {r2:>10.4f}")
    print(f"  CV R² (5-fold) : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    return {"name": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv_r2.mean(),
            "y_pred": y_pred}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results: list[dict],
    save_path: str = "reports/regression_comparison.png",
) -> None:
    """Bar chart comparatif des métriques des modèles."""
    names = [r["name"] for r in results]
    r2s   = [r["R2"]   for r in results]
    rmses = [r["RMSE"]  for r in results]
    maes  = [r["MAE"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#2563eb", "#dc2626", "#16a34a"]

    for ax, vals, title, color in zip(
        axes,
        [r2s, rmses, maes],
        ["R² (↑)", "RMSE £ (↓)", "MAE £ (↓)"],
        colors,
    ):
        bars = ax.bar(names, vals, color=color, alpha=0.8, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Comparaison des modèles de régression", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()
    logger.info("Comparaison modèles sauvegardée → %s", save_path)


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = "../reports/actual_vs_predicted.png",
) -> None:
    """Scatter actual vs predicted + ligne identité."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#2563eb")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Prédiction parfaite")
    ax.set_xlabel("Valeur réelle (£)")
    ax.set_ylabel("Valeur prédite (£)")
    ax.set_title(f"Réel vs Prédit — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()
    logger.info("Scatter réel vs prédit sauvegardé → %s", save_path)


def plot_residuals(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = "reports/residuals.png",
) -> None:
    """Distribution des résidus + résidus vs valeurs prédites."""
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribution des résidus
    axes[0].hist(residuals, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Résidu (£)")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution des résidus", fontweight="bold")

    # Résidus vs prédits
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=12, color="#dc2626")
    axes[1].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Valeur prédite (£)")
    axes[1].set_ylabel("Résidu (£)")
    axes[1].set_title("Résidus vs Prédits", fontweight="bold")
    axes[1].grid(alpha=0.2)

    plt.suptitle(f"Analyse des résidus — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()
    logger.info("Analyse résidus sauvegardée → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── Données ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_regression_data()

    results = []

    # ── 1. Ridge Regression ───────────────────────────────────────────────────
    logger.info("\n[1/3] Entraînement Ridge Regression…")
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=1.0)),
    ])
    ridge_pipeline.fit(X_train, y_train)
    r = evaluate_regressor(ridge_pipeline, X_train, X_test, y_train, y_test, "Ridge")
    results.append(r)

    # ── 2. Random Forest Regressor ────────────────────────────────────────────
    logger.info("\n[2/3] Entraînement Random Forest Regressor…")
    rf_reg = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    r = evaluate_regressor(rf_reg, X_train, X_test, y_train, y_test, "Random Forest")
    results.append(r)

    # ── 3. Gradient Boosting + GridSearch ─────────────────────────────────────
    logger.info("\n[3/3] Optimisation Gradient Boosting…")
    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_
    logger.info("Meilleurs paramètres GB : %s", grid_search.best_params_)
    r = evaluate_regressor(best_gb, X_train, X_test, y_train, y_test, "GradientBoosting")
    results.append(r)

    # ── Visualisations ────────────────────────────────────────────────────────
    # Trouver le meilleur modèle
    best_result = max(results, key=lambda x: x["R2"])
    logger.info("\nMeilleur modèle : %s  (R²=%.4f)", best_result["name"], best_result["R2"])

    plot_model_comparison(results)
    plot_actual_vs_predicted(y_test, best_result["y_pred"], best_result["name"])
    plot_residuals(y_test, best_result["y_pred"], best_result["name"])

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    joblib.dump(ridge_pipeline, "../models/reg_ridge.pkl")
    joblib.dump(rf_reg,         "../models/reg_rf.pkl")
    joblib.dump(best_gb,        "../models/reg_gb_best.pkl")

    print("\n" + "─" * 52)
    print("  Ridge sauvegardé         → models/reg_ridge.pkl")
    print("  RF Regressor sauvegardé  → models/reg_rf.pkl")
    print("  GB best sauvegardé       → models/reg_gb_best.pkl")
    print("  Rapports                 → reports/")
    print("─" * 52)


if __name__ == "__main__":
    main()