# NeoGarden Assistant — kit de démarrage

Assistant RAG pour une boutique de jardinage fictive. Streamlit + LangChain + FAISS + Mistral.

## Fichiers de ce kit

| Fichier | Rôle |
|---|---|
| `NeoGarden_FAQ.docx` | ~950 mots — livraison, paiement, commandes, SAV |
| `NeoGarden_CGU.docx` | ~1160 mots — 16 articles, dont RGPD et garanties |
| `NeoGarden_Politique_Retour.docx` | ~900 mots — rétractation, exclusions, remboursement |
| `Catalogue_Jardin_Amazon-2 - Catalogue.csv` | 38 produits, 13 catégories |
| `requirements.txt` | dépendances, dont deux absentes des imports du code |
| `golden_set.csv` | 15 questions de test à passer avant la démo |

Les marques du catalogue (Verdania, Torvald, Aquaflow, BotaniK, Terraluxe, Ferroline, Solaris Garden) sont **inventées**. C'est délibéré : inventer des prix et des specs pour de vraies marques serait faux, et dans un contexte de démo devant un recruteur c'est une mauvaise idée.

Les trois documents sont **cohérents entre eux** : franchise de port à 59 €, rétractation à 14 jours, garantie légale à 2 ans, garantie de reprise des plantes à 30 jours. Un RAG qui répond n'importe quoi révélera donc un vrai problème, pas une incohérence du corpus.

## Installation

Place tous ces fichiers dans le **même dossier que `app.py`**, puis :

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Colle ta clé Mistral dans la barre latérale.

> **Fais-le ce soir, pas demain.** Le premier lancement télécharge le modèle CamemBERT (plusieurs centaines de Mo) puis calcule les embeddings des ~180 chunks. Compte 3 à 6 minutes. Ensuite le modèle est en cache local et le démarrage est rapide.

## Passer le golden set

Pose les 15 questions à la main dans l'interface, remplis la colonne `statut` avec OK ou KO, et note ton score.

La question 15 est la plus importante : le système doit répondre qu'il ne sait pas. Si à la place il invente un chiffre d'affaires, tu as une hallucination à montrer — et savoir la nommer vaut mieux que ne pas l'avoir vue.

La question 14 croise FAQ et catalogue. Elle a de bonnes chances d'échouer avec `k=4` en recherche purement vectorielle. C'est normal et c'est un bon exemple à citer.

## Limites connues — à énoncer avant qu'on te les trouve

1. **Le retriever ignore l'historique.** `create_retrieval_chain` recherche sur `input` seul. Question de suivi pronominale (« il coûte combien ? ») → retrieval hors sujet. Correctif : `create_history_aware_retriever`.
2. **Recherche purement vectorielle, k=4 fixe.** Pas de BM25, pas de reranker. Les références produit exactes et les questions croisant deux sources sont mal servies.
3. **Index FAISS non persisté.** Recalculé à chaque démarrage à froid au lieu d'être sauvegardé sur disque.
4. **Aucun guardrail.** Ni détection de prompt injection, ni masquage PII, ni contrôle d'ancrage en sortie.
5. **Aucun tracing.** Ni tokens, ni latence, ni coût par requête.
6. **Chunking à 1000/150 par défaut**, choisi sans mesure.
7. **Le cache est indexé sur la clé API.** Corriger une faute de frappe dans la clé relance tout le calcul d'embeddings.

## Deux améliorations à fort rendement, si tu as le temps

**Afficher les sources** (~10 min). `response['context']` contient déjà les documents récupérés. Les afficher dans un `st.expander` sous la réponse transforme la démo de « un chatbot » en « un RAG dont je peux prouver l'ancrage ».

**Corriger le retriever historique** (~20 min). C'est le bug qu'un Tech Lead trouve en trente secondes, parce que poser une question de suivi est le réflexe naturel de n'importe qui.

Si tu ne fais ni l'un ni l'autre : ne pose pas de question de suivi pendant la démo, et cite les deux points comme limites identifiées.
