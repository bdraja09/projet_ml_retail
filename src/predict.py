"""
predict.py
==========
Script de prédiction unifié supportant :
    - Classification Churn (RF, LR, XGBoost, GridSearch)
    - Clustering (segments clients)
    - Régression (MonetaryTotal)

Usage :
    python src/predict.py --input data/raw/customers.csv --model models/churn_rf.pkl --output predictions.csv
    python src/predict.py --input data/raw/customers.csv --model models/cluster_kmeans.pkl --mode cluster
    python src/predict.py --input data/raw/customers.csv --model models/reg_rf.pkl --mode regression
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib


try:
    from preprocessing import DataPreprocessor # pyright: ignore[reportMissingImports]
except ImportError:
    # Si exécuté depuis la racine du projet
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import DataPreprocessor # pyright: ignore[reportMissingImports]
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Configuration des modèles disponibles
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Classification Churn
    'churn_rf': {'path': '../models/churn_rf.pkl', 'mode': 'classification', 'proba': True},
    'churn_lr': {'path': '../models/churn_lr.pkl', 'mode': 'classification', 'proba': True},
    'churn_rf_best': {'path': '../models/churn_rf_best.pkl', 'mode': 'classification', 'proba': True},
    'churn_xgb': {'path': '../models/churn_xgb.pkl', 'mode': 'classification', 'proba': True},
    
    # Clustering
    'cluster_kmeans': {'path': '../models/cluster_kmeans.pkl', 'mode': 'cluster', 'proba': False},
    'cluster_hdbscan': {'path': '../models/cluster_hdbscan.pkl', 'mode': 'cluster', 'proba': False},
    
    # Régression MonetaryTotal
    'reg_ridge': {'path': '../models/reg_ridge.pkl', 'mode': 'regression', 'proba': False},
    'reg_rf': {'path': '../models/reg_rf.pkl', 'mode': 'regression', 'proba': False},
    'reg_gb_best': {'path': '../models/reg_gb_best.pkl', 'mode': 'regression', 'proba': False},
}


# ─────────────────────────────────────────────────────────────────────────────
# Prétraitement robuste
# ─────────────────────────────────────────────────────────────────────────────

def load_preprocessor():
    """Charge le preprocessor sauvegardé ou crée un fallback."""
    prep_path = '../models/preprocessor.pkl'
    if os.path.exists(prep_path):
        print(f"Preprocessor chargé : {prep_path}")
        return joblib.load(prep_path)
    else:
        print("Aucun preprocessor.pkl trouvé. Utilisation de DataPreprocessor() brut.")
        return DataPreprocessor()


def preprocess_input(df, preprocessor=None):
    """
    Applique le preprocessing et nettoie les données.
    Gère les colonnes textuelles résiduelles.
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()
    
    # Tentative de transformation
    try:
        df_clean = preprocessor.transform(df)
    except (ValueError, TypeError):
        # Fallback : fit_transform si transform échoue (nouvelles données)
        print(" Transform échoué, fallback sur fit_transform...")
        df_clean = preprocessor.fit_transform(df)
    
    # Nettoyage des colonnes textuelles résiduelles
    if isinstance(df_clean, pd.DataFrame):
        text_cols = df_clean.select_dtypes(include=['object', 'string', 'category']).columns
        if len(text_cols) > 0:
            print(f" Colonnes textuelles supprimées : {list(text_cols)}")
            df_clean = df_clean.drop(columns=text_cols)
        
        # Forcer numérique
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
        df_clean = df_clean.fillna(df_clean.median())
    
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction par mode
# ─────────────────────────────────────────────────────────────────────────────

def predict_classification(model, df_clean, model_name):
    """Prédiction classification avec probabilités."""
    predictions = model.predict(df_clean)
    
    result = {
        'Churn_Prediction': predictions,
        'Churn_Label': np.where(predictions == 1, 'Parti', 'Fidèle')
    }
    
    # Probabilités si disponibles
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(df_clean)
            result['Churn_Probability'] = proba[:, 1]
            result['Churn_Confidence'] = np.max(proba, axis=1)
        except Exception as e:
            print(f"Probabilités indisponibles : {e}")
    
    print(f"   Clients à risque (Churn=1) : {predictions.sum()} / {len(predictions)}")
    return result


def predict_clustering(model, df_clean, model_name):
    """Prédiction clustering avec noms de segments."""
    predictions = model.predict(df_clean)
    
    # Mapping des clusters vers noms lisibles (ajustez selon votre analyse)
    cluster_names = {
        0: 'Fidèles',
        1: 'À risque',
        2: 'Nouveaux',
        3: 'VIP',
        4: 'Dormants',
        -1: 'Bruit'  # HDBSCAN
    }
    
    result = {
        'Cluster_ID': predictions,
        'Cluster_Name': [cluster_names.get(c, f'Cluster_{c}') for c in predictions]
    }
    
    # Distances au centroïde si disponible (KMeans)
    if hasattr(model, 'transform'):
        distances = model.transform(df_clean)
        result['Distance_To_Centroid'] = np.min(distances, axis=1)
    
    # Statistiques
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"   Répartition des clusters :")
    for u, c in zip(unique, counts):
        name = cluster_names.get(u, f'Cluster_{u}')
        print(f"      {name} (ID={u}) : {c} clients ({100*c/len(predictions):.1f}%)")
    
    return result


def predict_regression(model, df_clean, model_name):
    """Prédiction régression valeur monétaire."""
    predictions = model.predict(df_clean)
    
    result = {
        'MonetaryTotal_Predicted': predictions,
        'MonetaryTotal_Rounded': np.round(predictions, 2)
    }
    
    # Score de confiance via cross-val si disponible
    if hasattr(model, 'score'):
        print(f"   R² du modèle (entraînement) : consultez les rapports")
    
    print(f"   Valeur moyenne prédite : {predictions.mean():.2f} £")
    print(f"   Min : {predictions.min():.2f} £ | Max : {predictions.max():.2f} £")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def predict_csv(input_path, model_path=None, model_key=None, output_path=None, mode=None):
    """
    Charge un CSV brut, applique le preprocessing et prédit.
    
    Parameters
    ----------
    input_path : str
        Chemin du CSV d'entrée
    model_path : str, optional
        Chemin direct du modèle .pkl
    model_key : str, optional
        Clé du modèle dans MODEL_REGISTRY (ex: 'churn_rf', 'cluster_kmeans')
    output_path : str, optional
        Chemin de sortie (défaut: data/processed/predictions.csv)
    mode : str, optional
        'classification' | 'cluster' | 'regression' (auto-détecté si model_key)
    """
    # ── Validation entrée ────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} introuvable")
    
    # ── Résolution du modèle ─────────────────────────────────────────────────
    if model_key and model_key in MODEL_REGISTRY:
        info = MODEL_REGISTRY[model_key]
        model_path = info['path']
        mode = info['mode']
        print(f"Modèle sélectionné : {model_key} ({mode})")
    
    if not model_path or not os.path.exists(model_path):
        available = [f"{k} ({v['path']})" for k, v in MODEL_REGISTRY.items()]
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Modèles disponibles :\n" + "\n".join(f"   • {m}" for m in available)
        )
    
    if not mode:
        # Auto-détection par nom de fichier
        mode = 'classification'  # défaut
        if 'cluster' in model_path.lower():
            mode = 'cluster'
        elif 'reg' in model_path.lower():
            mode = 'regression'
    
    # ── Chargement données ───────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"\n{'='*60}")
    print(f"Données chargées : {df.shape}")
    print(f"Mode : {mode.upper()}")
    print(f"Modèle : {model_path}")
    print(f"{'='*60}")
    target_col = 'Churn'
    if target_col in df.columns:
        print(f"Colonne '{target_col}' retirée des features (cible)")
        df_features = df.drop(columns=[target_col])
    else:
        df_features = df.copy()
    
    # Utiliser df_features (sans Churn) pour le preprocessing
    preprocessor = load_preprocessor()
    df_clean = preprocess_input(df_features, preprocessor)
    
    # ── Preprocessing ────────────────────────────────────────────────────────
    preprocessor = load_preprocessor()
    df_clean = preprocess_input(df_features, preprocessor)
    print(f"Features après preprocessing : {df_clean.shape}")
    
    # ── Chargement modèle ────────────────────────────────────────────────────
    model = joblib.load(model_path)
    print(f"Modèle chargé : {type(model).__name__}")
    
    # ── Prédiction selon le mode ─────────────────────────────────────────────
    if mode == 'classification':
        predictions = predict_classification(model, df_clean, model_path)
    elif mode == 'cluster':
        predictions = predict_clustering(model, df_clean, model_path)
    elif mode == 'regression':
        predictions = predict_regression(model, df_clean, model_path)
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    
    # ── Assemblage résultat ──────────────────────────────────────────────────
    for col, values in predictions.items():
        df[col] = values
    
    # ── Sauvegarde ───────────────────────────────────────────────────────────
    if output_path is None:
        suffix = mode.replace('classification', 'churn').replace('regression', 'monetary')
        output_path = f'../data/processed/predictions_{suffix}.csv'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Prédictions sauvegardées : {output_path}")
    print(f"   Colonnes ajoutées : {list(predictions.keys())}")
    print(f"{'='*60}")
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Prédiction Churn / Clustering / Régression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Classification Churn (défaut)
  python src/predict.py --input data/raw/customers.csv --model models/churn_rf.pkl
  
  # Via clé de registre
  python src/predict.py --input data/raw/customers.csv --model-key churn_xgb
  
  # Clustering
  python src/predict.py --input data/raw/customers.csv --model-key cluster_kmeans --mode cluster
  
  # Régression valeur monétaire
  python src/predict.py --input data/raw/customers.csv --model-key reg_gb_best --mode regression
  
  # Sortie personnalisée
  python src/predict.py --input clients.csv --model-key churn_rf_best --output results/mes_predictions.csv
        """
    )
    
    parser.add_argument('--input', default='../data/raw/customers.csv', help='CSV entrée')
    parser.add_argument('--model', default=None, help='Chemin modèle .pkl')
    parser.add_argument('--model-key', default=None, choices=list(MODEL_REGISTRY.keys()),
                        help='Clé modèle prédéfinie')
    parser.add_argument('--mode', choices=['classification', 'cluster', 'regression'],
                        help='Mode de prédiction (auto-détecté par défaut)')
    parser.add_argument('--output', default=None, help='CSV sortie')
    
    args = parser.parse_args()
    
    if not args.model and not args.model_key:
        args.model_key = 'churn_rf'  # défaut
    
    predict_csv(
        input_path=args.input,
        model_path=args.model,
        model_key=args.model_key,
        output_path=args.output,
        mode=args.mode
    )


if __name__ == "__main__":
    main()