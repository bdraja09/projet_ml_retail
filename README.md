#  Analyse Comportementale Clientèle — Atelier Machine Learning

##  Description
Projet de data science appliqué à un e-commerce de cadeaux.  
L'objectif est d'analyser le comportement des clients à partir de **52 features** issues de transactions réelles pour :
- Segmenter la clientèle (Clustering)
- Prédire le départ des clients (Classification — Churn)
- Estimer les dépenses futures (Régression — MonetaryTotal)

---

##  Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Raja_BD_Oss/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
# Création
python -m venv venv

venv\Scripts\activate

source venv/bin/activate
```

### 3. Installer les dépendances
```
pip install -r requirements.txt
```

## Structure du projet

```
projet_ml_retail/
│
├── data/
│   ├── raw/              # Données brutes originales (ne pas modifier)
│   ├── processed/        # Données nettoyées et encodées
│   └── train_test/       # Données splittées (X_train, X_test, y_train, y_test)
│
├── notebooks/            # Notebooks Jupyter pour le prototypage et l'exploration
│
├── src/                  # Scripts Python prêts pour la production
│   ├── preprocessing.py  # Nettoyage, encodage, normalisation
│   ├── train_model.py    # Entraînement des modèles ML
│   ├── predict.py        # Prédictions sur nouvelles données
│   └── utils.py          # Fonctions utilitaires partagées
│
├── models/               # Modèles sauvegardés (.pkl, .joblib)
│
├── app/                  # Application web Flask
│   ├── app.py            # Serveur Flask principal
│   └── templates/        # Pages HTML
│
├── reports/              # Rapports, graphiques et visualisations
│
├── requirements.txt      # Dépendances Python du projet
├── README.md             # Documentation (ce fichier)
└── .gitignore            # Fichiers exclus du suivi Git
```

## Modèles utilisés
Clustering : KMeans, DBSCAN 
Classification (Churn) : Random Forest, XGBoost, Régression Logistique 
Régression (MonetaryTotal) : Linear Regression, Ridge, XGBoost Regressor 

---

## Auteur
- **Nom** : BOUABIDI RAJA  
- **Formation** : Atelier Machine Learning  
- **Année** : 2025/2026