"""
preprocessing.py
================
Module de preprocessing pour le projet ML Retail – Prédiction du Churn.

Pipeline :
    1. Nettoyage des outliers
    2. Parsing et extraction de features temporelles
    3. Feature engineering métier
    4. Suppression des colonnes inutiles
    5. Imputation des valeurs manquantes (médiane)
    6. Encodage des variables catégorielles

Pattern sklearn : fit() sur train → transform() sur test (zéro fuite de données).
"""

import os
import sys
import warnings
import logging

import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes redondantes ou à forte cardinalité inutile
COLS_TO_DROP = [
    "CustomerID",
    "NewsletterSubscribed",
    "LastLoginIP",
    "UniqueInvoices",
    "NegativeQuantityCount",
    "UniqueDescriptions",
    "MonetaryMin",
    "MonetaryMax",
    "MinQuantity",
    "MaxQuantity",
    "TotalQuantity",
    "TotalTransactions",
    "AvgLinesPerInvoice",
    "FirstPurchaseDaysAgo",
    "CustomerTenureDays",
]

# Candidats ordinal-encoding (ordre implicite dans les données)
ORDINAL_CANDIDATES = [
    "RFMSegment",
    "AgeCategory",
    "SpendingCat",
    "LoyaltyLevel",
    "ChurnRisk",
    "BasketSize",
]


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """
    Preprocesseur complet pour le dataset clients retail.

    Utilisation
    -----------
    >>> prep = DataPreprocessor()
    >>> X_train_clean = prep.fit_transform(X_train)
    >>> X_test_clean  = prep.transform(X_test)
    """

    def __init__(self):
        self.cols_to_drop: list[str] = COLS_TO_DROP
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Colonnes identifiées lors du fit
        self.num_cols: list[str] | None = None
        self.cat_cols: list[str] | None = None
        self.ordinal_cols: list[str] | None = None
        self.onehot_cols: list[str] | None = None
        self._is_fitted: bool = False

    # ── Étapes de transformation ────────────────────────────────────────────

    @staticmethod
    def _clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """Corrige les valeurs aberrantes métier."""
        df = df.copy()
        if "SupportTickets" in df.columns:
            df.loc[df["SupportTickets"] < 0, "SupportTickets"] = 0
            df.loc[df["SupportTickets"] >= 999, "SupportTickets"] = np.nan
        if "Satisfaction" in df.columns:
            df.loc[df["Satisfaction"] > 5, "Satisfaction"] = np.nan
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Extrait des features temporelles depuis RegistrationDate."""
        df = df.copy()
        if "RegistrationDate" in df.columns:
            reg = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")
            df["RegYear"]    = reg.dt.year
            df["RegMonth"]   = reg.dt.month
            df["RegDay"]     = reg.dt.day
            df["RegWeekday"] = reg.dt.weekday
            df.drop(columns=["RegistrationDate"], inplace=True)
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """Crée de nouvelles features métier à partir des colonnes existantes."""
        df = df.copy()

        cols = df.columns.tolist()

        if {"MonetaryTotal", "Recency"}.issubset(cols):
            df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

        if {"MonetaryTotal", "Frequency"}.issubset(cols):
            df["AvgBasketValue"] = df["MonetaryTotal"] / df["Frequency"].replace(0, np.nan)

        if {"Recency", "CustomerTenure"}.issubset(cols):
            df["TenureRatio"] = df["Recency"] / (df["CustomerTenure"] + 1)

        if "Recency" in cols:
            df["IsDormant"] = (df["Recency"] > 180).astype(int)

        if "ReturnRatio" in cols:
            df["HighReturner"] = (df["ReturnRatio"] > 0.5).astype(int)

        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les colonnes inutiles présentes dans le dataset."""
        existing = [c for c in self.cols_to_drop if c in df.columns]
        return df.drop(columns=existing, errors="ignore")

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identifie les colonnes numériques / catégorielles / encodage."""
        self.num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "Churn"
        ]
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.ordinal_cols = [c for c in ORDINAL_CANDIDATES if c in self.cat_cols]
        self.onehot_cols  = [c for c in self.cat_cols if c not in self.ordinal_cols]

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encode les colonnes catégorielles."""
        if not self.onehot_cols:
            return df
        cols_present = [c for c in self.onehot_cols if c in df.columns]
        if not cols_present:
            return df
        df = pd.get_dummies(df, columns=cols_present, drop_first=False)
        return df

    # ── API publique ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, y=None) -> "DataPreprocessor":
        """Calcule les paramètres de transformation sur le jeu d'entraînement."""
        logger.info("DataPreprocessor.fit() — apprentissage sur le train set")
        df_work = (
            df.copy()
            .pipe(self._clean_outliers)
            .pipe(self._parse_dates)
            .pipe(self._feature_engineering)
            .pipe(self._drop_columns)
        )
        self._identify_column_types(df_work)
        if self.num_cols:
            self.num_imputer.fit(df_work[self.num_cols])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Applique la transformation (sans re-fitter) sur n'importe quel split."""
        if not self._is_fitted:
            raise RuntimeError("Le preprocesseur doit d'abord être fitté (appeler .fit()).")

        df_work = (
            df.copy()
            .pipe(self._clean_outliers)
            .pipe(self._parse_dates)
            .pipe(self._feature_engineering)
            .pipe(self._drop_columns)
        )

        # Imputation numérique
        num_present = [c for c in (self.num_cols or []) if c in df_work.columns]
        if num_present:
            df_work[num_present] = self.num_imputer.transform(df_work[num_present])

        # Nettoyage des infinis résiduels
        num_all = df_work.select_dtypes(include=[np.number]).columns
        df_work[num_all] = (
            df_work[num_all]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(df_work[num_all].median())
        )

        # Encodage catégoriel
        df_work = self._encode_categoricals(df_work)

        return df_work

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit puis transform en une seule passe."""
        return self.fit(df, y).transform(df, y)

    def split_and_save(
        self,
        data_clean: pd.DataFrame,
        save_dir: str = "data/train_test",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """
        Effectue le split stratifié et sauvegarde les 4 fichiers CSV.

        Returns
        -------
        tuple : (X_train, X_test, y_train, y_test)
        """
        if "Churn" not in data_clean.columns:
            raise ValueError("La colonne 'Churn' est absente du dataset.")

        X = data_clean.drop("Churn", axis=1)
        y = data_clean["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        os.makedirs(save_dir, exist_ok=True)
        X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{save_dir}/X_test.csv",  index=False)
        y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{save_dir}/y_test.csv",   index=False)

        logger.info(
            "Split sauvegardé dans '%s' — Train: %d | Test: %d",
            save_dir, len(X_train), len(X_test),
        )
        return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH      = "../data/raw/customers.csv"
    PROCESSED_PATH = "../data/processed/data_clean.csv"

    if not os.path.exists(DATA_PATH):
        logger.error("Fichier introuvable : %s", DATA_PATH)
        sys.exit(1)

    logger.info("Chargement de %s …", DATA_PATH)
    raw = pd.read_csv(DATA_PATH)

    preprocessor = DataPreprocessor()
    clean = preprocessor.fit_transform(raw)

    os.makedirs("../models", exist_ok=True)
    joblib.dump(preprocessor, "../models/preprocessor.pkl")
    logger.info("Preprocesseur sauvegardé → ../models/preprocessor.pkl")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    logger.info("Dataset nettoyé → %s  %s", PROCESSED_PATH, clean.shape)

    preprocessor.split_and_save(clean)