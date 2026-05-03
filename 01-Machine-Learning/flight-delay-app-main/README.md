### Flight Delay Intelligence Platform

Application Streamlit de prédiction et analyse des retards aériens aux États-Unis.

## Application en ligne
> https://flight-delay-app-xrr9auhhzwalgp66xvwabt.streamlit.app/

## Dataset
- **Source :** US Bureau of Transportation Statistics
- **Couverture :** 50 000 vols · 18 compagnies · 2019–2023

## Modèle ML
- **Algorithme :** Random Forest (100 arbres)
- **Accuracy :** 93%
- **AUC-ROC :** 0.913
- **Variable cible :** Vol retardé de plus de 15 min (Oui/Non)

## Pages
- **Accueil** — KPIs, carte des aéroports, évolution 2019–2023
- **Exploration** — Carte des routes, retards par compagnie/heure/jour
- **Analyse Approfondie** — Corrélations, heatmaps, insights
- **Prédiction ML** — Simulateur : mon vol sera-t-il retardé ?

## Lancement local
```bash
pip install -r requirements.txt
streamlit run app.py
```
