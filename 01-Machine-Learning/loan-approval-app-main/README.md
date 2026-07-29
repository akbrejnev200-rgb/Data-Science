# LoanPredict AI — Prédiction d'approbation de prêt bancaire

Application Streamlit de prédiction d'approbation de prêt, comparant deux modèles de classification (Régression Logistique et Random Forest) sur un dataset public de demandes de crédit.

**[Voir la démo en ligne](https://loan-approval-app-txp6suqmdfpxuuqkmspb9s.streamlit.app/)**

## Ce que fait l'application

- **Exploration des données** : distribution des demandes (approuvées/refusées), analyse des valeurs manquantes, distributions des variables numériques (revenu, montant du prêt, durée), matrice de corrélation entre variables
- **Prédiction interactive** : simulation d'une demande de prêt avec choix du modèle (Régression Logistique ou Random Forest), affichage de la probabilité d'approbation et des facteurs analysés
- **Performance des modèles** : comparaison des deux modèles sur le jeu de test (accuracy, precision, recall, F1, AUC), matrice de confusion, importance des variables, tableau récapitulatif avec recommandation

## Pipeline de données

- Imputation des valeurs manquantes (mode pour les variables catégorielles, médiane pour le montant du prêt)
- Feature engineering : ratio mensualité/revenu (EMI-to-Income), transformation logarithmique des revenus et montants, agrégation des revenus du demandeur et du co-demandeur
- Encodage des variables catégorielles et one-hot encoding de la zone géographique du bien
- Standardisation des variables pour la régression logistique

## Résultats (jeu de test, 123 observations)

| Métrique | Régression Logistique | Random Forest |
|---|---|---|
| Accuracy | 85,4% | 82,9% |
| Precision | 83,2% | 87,2% |
| Recall | 98,8% | 88,2% |
| F1-Score | 90,3% | 87,7% |
| AUC | 0,867 | 0,837 |

Dataset : 615 demandes de prêt (491 en entraînement, 123 en test), 14 variables après feature engineering. La régression logistique obtient le meilleur AUC et reste préférable en contexte bancaire pour son interprétabilité.

## Stack technique

Python · Streamlit · Scikit-learn · Pandas · Plotly · Joblib

## Lancer le projet en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
