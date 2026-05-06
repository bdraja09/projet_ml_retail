"""
clustering.py
=============
Segmentation clients par clustering (K-Means) avec réduction dimensionnelle (ACP/PCA).

Pipeline :
    1. Chargement des données preprocessées
    2. ACP (PCA) — réduction 2D / 3D pour visualisation
    3. Méthode du coude + Silhouette Score → choix optimal de k
    4. K-Means clustering
    5. Profiling des clusters (caractéristiques métier)
    6. Visualisations : scatter 2D/3D, radar chart, distribution

Sorties :
    - models/kmeans_model.pkl
    - models/pca_model.pkl
    - data/processed/data_clustered.csv
    - reports/pca_variance.png
    - reports/elbow_silhouette.png
    - reports/clusters_2d.png
    - reports/clusters_3d.png
    - reports/cluster_profiles.png
"""

import os
import sys
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

CLUSTER_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
CLUSTER_LABELS  = {
    0: "Champions",
    1: "Fidèles",
    2: "Dormants",
    3: "À Risque",
    4: "Nouveaux",
    5: "VIP",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_data(
    path: str = "data/processed/data_clean.csv",
    train_path: str = "data/train_test/X_train.csv",
) -> pd.DataFrame:
    """
    Charge les données preprocessées.
    Priorité : data_clean.csv → X_train.csv (sans la target).
    """
    for p in [path, train_path]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # Supprimer la target si présente
            df = df.drop(columns=["Churn"], errors="ignore")
            logger.info("Données chargées depuis %s  shape=%s", p, df.shape)
            return df
    raise FileNotFoundError(
        "Aucun fichier de données trouvé. "
        "Lancez d'abord src/preprocessing.py."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Préparation des features pour le clustering
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Sélectionne et normalise les features numériques pour le clustering.

    Returns
    -------
    (X_scaled, feature_names)
    """
    # Conserver uniquement les colonnes numériques
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].copy()

    # Nettoyage des NaN / Inf résiduels
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Features pour clustering : %d  |  Observations : %d", len(num_cols), len(X))
    return X_scaled, num_cols


# ─────────────────────────────────────────────────────────────────────────────
# ACP / PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(
    X_scaled: np.ndarray,
    n_components_max: int = 10,
    save_path: str = "reports/pca_variance.png",
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calcule l'ACP complète et trace la variance expliquée cumulée.

    Returns
    -------
    (pca_full, X_2d, X_3d)
        pca_full : PCA fittée sur toutes les composantes
        X_2d     : Projection sur les 2 premières composantes
        X_3d     : Projection sur les 3 premières composantes
    """
    n_comp = min(n_components_max, X_scaled.shape[1])
    pca_full = PCA(n_components=n_comp, random_state=42)
    X_pca    = pca_full.fit_transform(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    var_cum   = np.cumsum(var_ratio)

    # ── Graphique variance expliquée ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Variance individuelle
    axes[0].bar(
        range(1, n_comp + 1), var_ratio * 100,
        color="#2563eb", alpha=0.8, edgecolor="white"
    )
    axes[0].set_xlabel("Composante principale")
    axes[0].set_ylabel("Variance expliquée (%)")
    axes[0].set_title("Variance expliquée par composante", fontweight="bold")
    axes[0].set_xticks(range(1, n_comp + 1))

    # Variance cumulée
    axes[1].plot(range(1, n_comp + 1), var_cum * 100, "o-", color="#dc2626", lw=2)
    axes[1].axhline(80, color="gray", linestyle="--", alpha=0.6, label="Seuil 80 %")
    axes[1].axhline(95, color="gray", linestyle=":",  alpha=0.6, label="Seuil 95 %")
    axes[1].fill_between(range(1, n_comp + 1), var_cum * 100, alpha=0.15, color="#dc2626")
    axes[1].set_xlabel("Nombre de composantes")
    axes[1].set_ylabel("Variance cumulée (%)")
    axes[1].set_title("Variance cumulée (ACP)", fontweight="bold")
    axes[1].set_xticks(range(1, n_comp + 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()
    logger.info("Graphique ACP sauvegardé → %s", save_path)

    # Résumé
    for k in [2, 3, 5]:
        if k <= n_comp:
            logger.info("  %d composantes → %.1f %% de variance", k, var_cum[k - 1] * 100)

    # Projections 2D et 3D
    X_2d = X_pca[:, :2]
    X_3d = X_pca[:, :3] if X_pca.shape[1] >= 3 else np.hstack([X_pca, np.zeros((len(X_pca), 1))])

    return pca_full, X_2d, X_3d


# ─────────────────────────────────────────────────────────────────────────────
# Méthode du coude + Silhouette
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    save_path: str = "reports/elbow_silhouette.png",
) -> int:
    """
    Détermine le k optimal via la méthode du coude (inertie) et le Silhouette Score.

    Returns
    -------
    int : k optimal recommandé (meilleur silhouette score)
    """
    inertias, silhouettes, db_scores = [], [], []

    logger.info("Recherche du k optimal (k = %d à %d)…", k_range.start, k_range.stop - 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        logger.info("  k=%d | Inertie=%.0f | Silhouette=%.3f | DB=%.3f",
                    k, km.inertia_, silhouettes[-1], db_scores[-1])

    best_k = k_range[np.argmax(silhouettes)]

    # ── Graphique ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Méthode du coude
    axes[0].plot(k_range, inertias, "o-", color="#2563eb", lw=2)
    axes[0].axvline(best_k, color="#dc2626", linestyle="--", alpha=0.7, label=f"k optimal = {best_k}")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Inertie (WCSS)")
    axes[0].set_title("Méthode du coude", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Silhouette Score
    axes[1].plot(k_range, silhouettes, "s-", color="#16a34a", lw=2)
    axes[1].axvline(best_k, color="#dc2626", linestyle="--", alpha=0.7, label=f"k optimal = {best_k}")
    axes[1].set_xlabel("Nombre de clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Score de silhouette", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Détermination du k optimal → k = {best_k}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.show()
    logger.info("Graphique coude/silhouette sauvegardé → %s  (k optimal = %d)", save_path, best_k)

    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# K-Means
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(X_scaled: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """Entraîne le modèle K-Means avec le k optimal."""
    logger.info("Entraînement K-Means avec k=%d…", k)
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    logger.info("K-Means k=%d | Silhouette=%.4f | Davies-Bouldin=%.4f", k, sil, db)

    return km, labels


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    save_path: str = "reports/clusters_2d.png",
) -> None:
    """Scatter 2D des clusters dans l'espace PCA."""
    k = len(np.unique(labels))
    palette = CLUSTER_PALETTE[:k]

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, color in enumerate(palette):
        mask = labels == i
        label_name = CLUSTER_LABELS.get(i, f"Cluster {i}")
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, alpha=0.5, s=18, label=f"{label_name} (n={mask.sum()})"
        )

    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    ax.set_title("Segmentation clients — ACP 2D", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    logger.info("Scatter 2D sauvegardé → %s", save_path)


def plot_clusters_3d(
    X_3d: np.ndarray,
    labels: np.ndarray,
    save_path: str = "reports/clusters_3d.png",
) -> None:
    """Scatter 3D des clusters dans l'espace PCA."""
    k = len(np.unique(labels))
    palette = CLUSTER_PALETTE[:k]

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for i, color in enumerate(palette):
        mask = labels == i
        label_name = CLUSTER_LABELS.get(i, f"Cluster {i}")
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
            c=color, alpha=0.4, s=12, label=label_name
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("Segmentation clients — ACP 3D", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    logger.info("Scatter 3D sauvegardé → %s", save_path)


def plot_cluster_profiles(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    save_path: str = "reports/cluster_profiles.png",
) -> pd.DataFrame:
    """
    Heatmap des moyennes normalisées par cluster (profil comportemental).

    Returns
    -------
    pd.DataFrame : Tableau de profiling des clusters
    """
    df_temp = df_original[feature_cols].copy()
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan).fillna(df_temp.median())
    df_temp["Cluster"] = labels

    # Profil moyen par cluster
    profiles = df_temp.groupby("Cluster")[feature_cols].mean()

    # Top 10 features les plus discriminantes (variance inter-cluster)
    top_features = profiles.std().nlargest(10).index.tolist()
    profiles_top = profiles[top_features]

    # Normalisation min-max pour la heatmap
    normalized = (profiles_top - profiles_top.min()) / (
        profiles_top.max() - profiles_top.min() + 1e-8
    )

    fig, ax = plt.subplots(figsize=(13, max(4, len(normalized) * 0.8)))
    sns.heatmap(
        normalized,
        annot=profiles_top.round(1),
        fmt="g",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Score normalisé (0–1)"},
    )
    ax.set_title("Profils des clusters — Top 10 features discriminantes",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    logger.info("Profils clusters sauvegardés → %s", save_path)

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── Données ───────────────────────────────────────────────────────────────
    df = load_processed_data()
    X_scaled, feature_names = prepare_features(df)

    # ── ACP ───────────────────────────────────────────────────────────────────
    pca_model, X_2d, X_3d = run_pca(X_scaled)

    # ── Choix de k ────────────────────────────────────────────────────────────
    best_k = find_optimal_k(X_scaled)

    # ── K-Means ───────────────────────────────────────────────────────────────
    km_model, labels = run_kmeans(X_scaled, best_k)

    # ── Visualisations ────────────────────────────────────────────────────────
    plot_clusters_2d(X_2d, labels)
    plot_clusters_3d(X_3d, labels)
    profiles = plot_cluster_profiles(df, labels, feature_names)

    # ── Affichage profils ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PROFIL MOYEN PAR CLUSTER")
    print("─" * 60)
    print(profiles.to_string())

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    print("\n  Distribution :")
    for u, c in zip(unique, counts):
        print(f"    Cluster {u} ({CLUSTER_LABELS.get(u, '?')}) : {c} clients ({100*c/len(labels):.1f} %)")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    df["Cluster"]      = labels
    df["ClusterLabel"] = [CLUSTER_LABELS.get(l, f"Cluster {l}") for l in labels]
    df.to_csv("../data/processed/data_clustered.csv", index=False)

    joblib.dump(km_model,  "../models/kmeans_model.pkl")
    joblib.dump(pca_model, "../models/pca_model.pkl")

    print("\n" + "─" * 60)
    print(" K-Means sauvegardé      → models/kmeans_model.pkl")
    print(" PCA sauvegardée         → models/pca_model.pkl")
    print(" Données clusterisées    → data/processed/data_clustered.csv")
    print("─" * 60)


if __name__ == "__main__":
    main()