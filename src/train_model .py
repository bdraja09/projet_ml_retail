"""
Entraînement et évaluation des modèles de classification Churn.
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                             confusion_matrix, roc_curve)
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_splits(path='data/train_test'):
    """Charge les données préparées."""
    X_train = pd.read_csv(f'{path}/X_train.csv')
    X_test = pd.read_csv(f'{path}/X_test.csv')
    y_train = pd.read_csv(f'{path}/y_train.csv').squeeze()
    y_test = pd.read_csv(f'{path}/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def clean_numeric_data(X_train, X_test):
    """
    Supprime les colonnes textuelles et force la conversion numérique.
    Garantit l'alignement des colonnes entre train et test.
    """
    # 1. Identifier et supprimer les colonnes non numériques
    for df, name in [(X_train, 'train'), (X_test, 'test')]:
        text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        if text_cols:
            print(f"Colonnes textuelles supprimées de {name}: {text_cols}")
            df.drop(columns=text_cols, inplace=True)
    
    # 2. Forcer la conversion numérique (remplace les erreurs par NaN)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    # 3. Aligner les colonnes (garder l'intersection)
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # 4. Remplacer les NaN par la médiane de chaque colonne
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    print(f"Données nettoyées : {X_train.shape} | {X_test.shape}")
    return X_train, X_test


def evaluate_model(model, X_test, y_test, name):
    """Affiche les métriques d'un modèle."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"MODÈLE : {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Parti']))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix :\n{cm}")
    
    return y_proba


def plot_roc_curves(models_dict, X_test, y_test):
    """Trace les courbes ROC comparatives."""
    plt.figure(figsize=(8, 6))
    
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlabel('Taux Faux Positifs')
    plt.ylabel('Taux Vrais Positifs')
    plt.title('Courbes ROC comparatives')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/roc_curves.png')
    plt.show()


def plot_feature_importance(model, X_train, top_n=15):
    """Affiche l'importance des features (Random Forest)."""
    importance = pd.Series(
        model.feature_importances_, 
        index=X_train.columns
    ).sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title(f'Top {top_n} features importantes')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.show()
    return importance


def main():
    # Création des dossiers
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Chargement
    print("Chargement des données...")
    X_train, X_test, y_train, y_test = load_splits()
    X_train, X_test = clean_numeric_data(X_train, X_test)
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    # ==================== MODÈLE 1 : RANDOM FOREST ====================
    print("\nEntraînement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        max_depth=10
    )
    rf.fit(X_train, y_train)
    proba_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    plot_feature_importance(rf, X_train)
    
    # ==================== MODÈLE 2 : RÉGRESSION LOGISTIQUE ====================
    print("\nEntraînement Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    )
    lr.fit(X_train, y_train)
    proba_lr = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    
    # ==================== MODÈLE 3 : GRIDSEARCHCV ====================
    print("\nGridSearchCV Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres : {grid.best_params_}")
    print(f"Meilleur CV ROC-AUC : {grid.best_score_:.4f}")
    proba_grid = evaluate_model(grid.best_estimator_, X_test, y_test, "RF GridSearch")
    
    # ==================== COMPARAISON ROC ====================
    models_dict = {
        'Random Forest': rf,
        'Logistic Regression': lr,
        'RF GridSearch': grid.best_estimator_
    }
    plot_roc_curves(models_dict, X_test, y_test)
    
    # ==================== SAUVEGARDE ====================
    joblib.dump(rf, 'models/churn_rf.pkl')
    joblib.dump(lr, 'models/churn_lr.pkl')
    joblib.dump(grid.best_estimator_, 'models/churn_rf_best.pkl')
    
    print(f"\n{'='*50}")
    print("MODÈLES SAUVEGARDÉS dans models/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()