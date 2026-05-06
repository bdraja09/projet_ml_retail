"""
utils.py
========
Fonctions utilitaires pour le projet ML Retail – Prédiction du Churn.

Expose :
    - quick_clean()          : nettoie un DataFrame brut en une ligne
    - split_and_preprocess() : split stratifié + preprocessing sans fuite de données
    - save_preprocessor()    : sérialise le preprocesseur entraîné
    - load_preprocessor()    : recharge un preprocesseur sérialisé
"""

import os
import importlib
import logging
from typing import Tuple

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement dynamique de DataPreprocessor (compatible notebook & package)
# ─────────────────────────────────────────────────────────────────────────────

def _load_data_preprocessor():
    """Importe DataPreprocessor depuis src.preprocessing ou preprocessing."""
    for module_name in ("src.preprocessing", "preprocessing"):
        try:
            module = importlib.import_module(module_name)
            return module.DataPreprocessor
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Impossible d'importer DataPreprocessor. "
        "Vérifiez que src/preprocessing.py est accessible."
    )


DataPreprocessor = _load_data_preprocessor()


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions publiques
# ─────────────────────────────────────────────────────────────────────────────

def quick_clean(
    df: pd.DataFrame,
    save_path: str = "../data/processed/cleaned_dataset.csv",
) -> pd.DataFrame:
    """
    Nettoie un DataFrame brut via le preprocesseur complet.

    Parameters
    ----------
    df        : DataFrame brut (issu de pd.read_csv)
    save_path : Chemin de sauvegarde du résultat nettoyé

    Returns
    -------
    pd.DataFrame : Dataset nettoyé
    """
    if df is None or df.empty:
        raise ValueError("Le DataFrame fourni est vide ou None.")

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_clean.to_csv(save_path, index=False)

    logger.info("Dataset nettoyé sauvegardé → %s  shape=%s", save_path, df_clean.shape)
    return df_clean


def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Pipeline complète : split stratifié → preprocessing séparé train/test.

    Le preprocesseur est fitté **uniquement** sur le train set, puis appliqué
    en transform-only sur le test set, ce qui garantit l'absence de fuite de
    données (data leakage).

    Parameters
    ----------
    df           : DataFrame complet (features + target)
    target_col   : Nom de la colonne cible (défaut : 'Churn')
    test_size    : Part réservée au test (défaut : 0.20)
    random_state : Graine aléatoire pour la reproductibilité

    Returns
    -------
    (X_train_clean, X_test_clean, y_train, y_test, preprocessor)
    """
    # ── Vérifications ────────────────────────────────────────────────────────
    if df is None or df.empty:
        raise ValueError("Le DataFrame est vide ou None.")
    if target_col not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_col}' introuvable. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    churn_rate = df[target_col].mean()
    if churn_rate < 0.05 or churn_rate > 0.95:
        logger.warning(
            "Déséquilibre de classes important détecté : %.1f %% de classe 1.",
            churn_rate * 100,
        )

    # ── Séparation features / target ─────────────────────────────────────────
    X, y = df.drop(target_col, axis=1), df[target_col]
    logger.info("Dimensions initiales : %s", X.shape)
    logger.info("Distribution cible :\n%s", y.value_counts(normalize=True).round(3))

    # ── Split stratifié ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split — Train : %d | Test : %d", len(X_train), len(X_test))

    # ── Preprocessing sans fuite de données ──────────────────────────────────
    preprocessor = DataPreprocessor()

    logger.info("=== fit_transform sur TRAIN ===")
    X_train_clean = preprocessor.fit_transform(X_train)

    logger.info("=== transform seul sur TEST ===")
    X_test_clean = preprocessor.transform(X_test)

    # ── Contrôle de cohérence ─────────────────────────────────────────────────
    if X_train_clean.shape[1] != X_test_clean.shape[1]:
        # Aligner les colonnes (cas rare avec get_dummies sur train/test déséquilibrés)
        X_train_clean, X_test_clean = X_train_clean.align(
            X_test_clean, join="left", axis=1, fill_value=0
        )
        logger.warning("Colonnes train/test désynchronisées – alignement appliqué.")

    logger.info(
        "Preprocessing terminé — %d features | NaN train=%d | NaN test=%d",
        X_train_clean.shape[1],
        X_train_clean.isna().sum().sum(),
        X_test_clean.isna().sum().sum(),
    )
    return X_train_clean, X_test_clean, y_train, y_test, preprocessor


def save_preprocessor(
    preprocessor,
    filepath: str = "../models/preprocessor.pkl",
) -> str:
    """
    Sérialise le preprocesseur entraîné (réutilisation en production / API).

    Parameters
    ----------
    preprocessor : Instance DataPreprocessor déjà fittée
    filepath     : Chemin de sauvegarde (.pkl)

    Returns
    -------
    str : Chemin absolu du fichier sauvegardé
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(preprocessor, filepath)
    logger.info("Preprocesseur sauvegardé → %s", filepath)
    return os.path.abspath(filepath)


def load_preprocessor(filepath: str = "../models/preprocessor.pkl"):
    """
    Charge un preprocesseur précédemment sérialisé.

    Parameters
    ----------
    filepath : Chemin vers le fichier .pkl

    Returns
    -------
    DataPreprocessor : Instance prête à l'emploi
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocesseur introuvable : {filepath}")
    preprocessor = joblib.load(filepath)
    logger.info("Preprocesseur chargé depuis %s", filepath)
    return preprocessor