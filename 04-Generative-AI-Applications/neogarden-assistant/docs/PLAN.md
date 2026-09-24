# Plan de restructuration — NeoGarden Assistant

Objectif : transformer le projet actuel (un fichier `app.py` unique) en dépôt
GitHub propre, sans changer le comportement de l'application. Une phase à la
fois, validation à chaque étape.

**Décisions déjà prises (à ne pas re-discuter) :**
- Emplacement : dans le monorepo `Data-Science`, sous
  `04-Generative-AI-Applications/neogarden-assistant/` (pas de repo séparé,
  choix de l'utilisateur, comme le projet MLOps)
- Le notebook d'exploration (Mistral local) n'est **pas** inclus dans le repo
- Les `except Exception` larges sont remplacés par des erreurs plus précises

**Conséquences du monorepo (à garder en tête pour les phases suivantes) :**
- Tout ce qui doit être « à la racine du repo » l'est pour `Data-Science`, pas
  pour ce dossier : `.pre-commit-config.yaml` (Phase 3) et
  `.github/workflows/` (Phase 6). On les **restreint à ce dossier** (`files:`
  pour pre-commit, `paths:` pour GitHub Actions) afin de ne pas toucher aux
  autres projets du repo.
- `pyproject.toml`, `.env.example`, `.gitignore`, `src/`, `tests/` restent
  dans le dossier du projet : on travaille depuis
  `04-Generative-AI-Applications/neogarden-assistant/`.
- Les PR se font sur `Data-Science` (branche de phase → `main`).
- Ordre : un `pyproject.toml` **minimal** est créé au début de la Phase 2 (pour
  que `rag_garden` soit importable), puis complété en Phase 3.
- `Lancer_NeoGarden.bat` contient un chemin personnel en dur (venv local) :
  à rendre portable ou remplacer par une commande documentée en Phase 2,
  quand `app.py` est déplacé.

**Git, pour toutes les phases :** une branche par phase
(`feat/xxx`, `refactor/xxx`, `test/xxx`, `ci/xxx`…), commits au format
Conventional Commits, PR à la fin de chaque phase (voir « Workflow Git » en
bas de ce document).

---

## Étape 0 — Mise en place (avant la Phase 1)

**Pourquoi une étape 0 :** avant de pouvoir appliquer la Phase 1 (hygiène du
dépôt), il faut un dépôt. Ce n'est pas une des 7 phases demandées, c'est la
préparation matérielle.

- Copier le projet de `OneDrive\Bureau\RAG Jardin\` vers
  `C:\Users\akbre\projets\neogarden-assistant\` (hors OneDrive : Git et la
  synchronisation cloud ne font pas bon ménage — fichiers "en ligne
  seulement", verrous, lenteurs).
- Le notebook `.ipynb` ne fait **pas** partie de la copie (décision prise).
- `git init`, premier commit `chore: import du projet existant (avant
  restructuration)` — un point de départ figé, pour pouvoir comparer
  "avant/après" à tout moment.
- Vérification manuelle rapide : aucun `.docx` ne contient de clé API collée
  par erreur (peu probable, mais rapide à vérifier).
- Puis fusion dans `Data-Science` sous `04-Generative-AI-Applications/
  neogarden-assistant/` avec `git subtree` : l'historique est conservé, et le
  dossier local autonome `projets\neogarden-assistant\` ne sert plus que de
  sauvegarde (la suite se fait dans le clone de `Data-Science`).

**Validation :** je te montre `git log` et `git status`, tu confirmes avant
la suite.

---

## Phase 1 — Sécurité et hygiène du dépôt

**Branche :** `chore/repo-hygiene`

- `.env.example` : liste les variables attendues (`OPENROUTER_API_KEY=`),
  sans valeur réelle.
- `.gitignore` : déjà correct pour `.env`/`.venv`/`__pycache__`/`streamlit.log`
  — j'ajoute la règle pour l'index FAISS qu'on va générer en Phase 2
  (dossier `vectorstore/` ou équivalent, jamais versionné : c'est un
  artefact recalculable, pas du code).
- `src/rag_garden/config.py` créé dès cette phase (juste le chargement des
  variables d'env + les chemins) — le reste de la config (chunk_size, k...)
  arrive en Phase 2 avec le code qui les utilise.
- Confirmation qu'aucun secret n'est dans l'historique Git (trivial ici :
  l'étape 0 vient de créer l'historique, donc rien à nettoyer).

**Validation :** l'appli tourne toujours (`streamlit run app.py` inchangé à
ce stade), `git log` propre, je résume les changements.

---

## Phase 2 — Restructuration en modules

**Branche :** `refactor/modular-structure`

La partie la plus grosse. Découpage de `app.py` (236 lignes, tout mélangé)
en modules à responsabilité unique :

| Fichier | Contenu repris de `app.py` |
|---|---|
| `src/rag_garden/ingestion.py` | chargement des 3 `.docx` + du catalogue CSV, construction du texte par produit, chunking |
| `src/rag_garden/embeddings.py` | embeddings CamemBERT + construction **et persistance sur disque** de l'index FAISS (aujourd'hui recalculé à chaque démarrage) |
| `src/rag_garden/retriever.py` | le retriever (`as_retriever(search_kwargs={"k": ...})`) |
| `src/rag_garden/prompts.py` | `system_prompt` et `contextualize_q_system_prompt` |
| `src/rag_garden/generation.py` | configuration du LLM (`ChatOpenAI`/OpenRouter) + `invoke_with_retry` avec gestion d'erreurs **précisée** (erreur réseau / erreur API / réponse invalide, au lieu d'un `except Exception` générique) |
| `src/rag_garden/memory.py` | conversion historique Streamlit ↔ messages LangChain (`HumanMessage`/`AIMessage`) |
| `src/rag_garden/pipeline.py` | assemble tout — remplace `init_rag_chain()` |
| `src/rag_garden/config.py` | complété : `chunk_size=1000`, `chunk_overlap=150`, `k=4`, noms des modèles, chemins des fichiers de données — tout ce qui est en dur aujourd'hui |
| `app/streamlit_app.py` | ce qui reste : `st.chat_input`, `st.session_state`, affichage — **aucune logique métier** |
| `scripts/build_index.py` | nouveau : construit l'index FAISS et le sauvegarde sur disque, pour ne plus le recalculer à chaque démarrage à froid |

- `data/` : les `.docx` + le `.csv` catalogue déplacés ici (données, pas du code).
- Chaque fonction : type hints + docstring courte (1-2 lignes, ce qu'elle
  fait et pourquoi, pas un roman).
- `print()` → `logging` (niveau INFO pour le déroulement normal, WARNING/ERROR
  pour les erreurs gérées).
- `guardrails.py` ne contient ici que le raccourci « salutations » (comportement
  existant, déplacé). Les vrais garde-fous changeraient le comportement : décision
  prise (point d'attention n°1) de les ajouter **après l'évaluation**, en
  Phase 5bis, pour pouvoir les mesurer.

**Validation :** je relance l'appli (`streamlit run app/streamlit_app.py`),
je repose les 15 questions du golden set à la main pour vérifier qu'aucun
comportement n'a changé, je te montre le résultat avant de continuer.

---

## Phase 3 — Qualité de code

**Branche :** `chore/tooling`

- `pyproject.toml` : dépendances **avec versions figées** (le
  `requirements.txt` actuel n'en a aucune).
- Ruff configuré (lint + format), tout ce qu'il remonte est corrigé.
- `pre-commit` avec le hook Ruff (le fichier de config est à la racine de
  `Data-Science`, restreint à ce dossier via `files:`).
- Je t'explique les règles Ruff activées au fur et à mesure qu'elles
  remontent quelque chose (pas une liste abstraite à l'avance).

**Validation :** `ruff check .` et `ruff format --check .` propres, appli
toujours fonctionnelle.

---

## Phase 4 — Tests

**Branche :** `test/unit-tests`

Tests `pytest`, **aucun appel réseau réel** (LLM et embeddings mockés) :

- `test_ingestion.py` : le chunking découpe correctement un document de test
- `test_retriever.py` : le retriever renvoie les k documents attendus (FAISS
  factice ou mocké)
- `test_prompts.py` : le prompt final contient bien le contexte et la
  question
- `test_memory.py` : conversion historique Streamlit → messages LangChain
- `test_guardrails.py` : le raccourci « salutations » (seul garde-fou existant
  à ce stade ; les autres arrivent en Phase 5bis, avec leurs propres tests).
  **Bug connu, corrigé test d'abord** (point d'attention n°6) : `is_greeting("Bonjour !")`
  renvoie faux à cause de l'espace avant « ! ». On écrit le test **avant** le
  correctif, on constate qu'il échoue (rouge), puis un commit `fix:`
  (`rstrip(" !.?")`) le fait passer (vert).
- `test_generation.py` (ajouté, hors liste initiale) : la politique de retry
  sur erreurs LLM — erreurs simulées (openai.*, et la vraie `ValueError(dict)`
  que LangChain lève pour une erreur renvoyée dans un corps HTTP 200), sans
  aucun appel réseau. Reprend et formalise le script de vérification jetable
  utilisé en Phase 2 : c'est la logique la plus fragile du projet (une
  régression de résilience y avait été introduite puis corrigée pendant la
  Phase 2), elle mérite une couverture permanente.

Je t'expliquerai concrètement, avec un exemple du projet, la différence
test unitaire / test d'intégration à ce moment-là (demandé dans tes
consignes).

**Validation :** `pytest` passe, rapide (aucun test ne dépend d'une clé API).

---

## Phase 5 — Évaluation du RAG

**Branche :** `feat/evaluation`

- `evaluation/golden_set.csv` : reprend tes 15 questions existantes, colonne
  `statut` à remplir automatiquement par le script (plus à la main).
- Ajout de quelques questions **hors périmètre** (pour vérifier que le
  système refuse de répondre plutôt que d'halluciner — ta question 15
  actuelle en est déjà une, j'en ajoute 2-3 autres dans le même esprit), et
  quelques **questions d'attaque** (tentatives d'injection de prompt) : elles
  servent de banc d'essai à la Phase 5bis.
- `evaluation/evaluate.py` :
  - **Retrieval** : précision/recall — les bons chunks sont-ils dans le
    top-k ?
  - **Génération** : fidélité au contexte, pertinence, refus correct quand
    l'info manque — via LLM-as-a-judge (un modèle note la réponse).
  - Produit un rapport lisible (markdown ou texte), pas juste un chiffre brut.
- **Expérience de prompt** (point d'attention n°2) : score de référence avec le
  prompt actuel (règles 3 et 4 collées sur une ligne), puis correction du saut
  de ligne dans un commit `fix:` séparé et comparaison avant/après.
- Ce script reste **hors CI** (il appelle une vraie API, comme demandé).

**Validation :** rapport généré, je te montre les résultats et ce qu'ils
disent des limites déjà identifiées (`LIMITES_RAG_EXEMPLES.md`).

**Note (2026-09-23) — contrainte réelle découverte en cours de route :**
OpenRouter limite les modèles gratuits à 50 requêtes/jour pour tout le
compte (tous modèles confondus), pas 50 par modèle. Une évaluation complète
(20 questions × avant/après correctif) demande ~80 appels, largement
au-dessus. Décision prise avec l'utilisateur : l'outillage (`evaluate.py`,
golden set enrichi, juge LLM, correctif du prompt) est livré et validé sur
un **sous-ensemble réduit de 6 questions représentatives** (id 1, 3, 10, 14,
15, 18 — couvrent FAQ, Politique de retour, Catalogue, la fusion entre deux
sources, une question hors périmètre et une tentative d'injection),
`evaluation/rapport_reduit_avant.md` et `rapport_reduit_apres.md`.
L'évaluation complète sur les 20 questions sera relancée plus tard (étalée
sur deux jours, gratuitement) et remplacera ces rapports réduits.

---

## Phase 5bis — Garde-fous (après l'évaluation)

**Branche :** `feat/guardrails`

Pourquoi ici et pas en Phase 2 : un garde-fou **refuse** des entrées, donc il
change le comportement, et un garde-fou non mesuré peut refuser de vraies
questions (faux positifs). On les ajoute une fois le banc d'essai de la
Phase 5 en place, pour mesurer leur effet avant/après.

- **Injection de prompt** (entrée) : détection par motifs des attaques
  évidentes, réponse de refus polie. Première ligne de défense, contournable :
  à documenter comme telle.
- **Hors sujet** (entrée) : ~~seuil sur le score de similarité du retriever~~
  **changé après mesure.** Un seuil de distance FAISS a été testé sur les 20
  questions du golden set + des questions de suivi courtes : il pénalisait
  autant une question de suivi légitime ("Et combien ça coûte ?", distance
  12.3) qu'une vraie question hors sujet ("capitale de l'Australie ?",
  distance 12.5) — chevauchement trop important pour être fiable. Remplacé
  par une absence de recouvrement avec le **vocabulaire du corpus** (mots
  significatifs des documents sources) : mesuré sur le même golden set,
  0 faux positif sur les 14 questions normales, 3/3 questions hors périmètre
  détectées, et les questions de suivi courtes passent correctement.
- **Réponse ancrée** (sortie) : ~~juge LLM, réutilise l'outillage de la
  Phase 5~~ **changé après discussion.** Un juge LLM à chaque réponse
  doublerait le coût de chaque question réelle posée dans l'application (pas
  seulement en évaluation) — risque accru d'épuiser le quota gratuit
  d'OpenRouter, y compris pendant une démo. Remplacé par une heuristique
  locale (recouvrement lexical entre la réponse et le contexte retrouvé),
  gratuite, validée par tests unitaires (exemples fabriqués : réponse fidèle,
  réponse inventée, refus légitime).
- Chaque garde-fou est mesuré : injection et hors sujet sur le texte des 20
  questions du golden set (aucun appel API, calcul local) ; ancrage par tests
  unitaires. Confirmé aussi de bout en bout avec `evaluation/evaluate.py` sur
  le sous-ensemble réduit (`evaluation/rapport_reduit_garde_fous.md`) : les
  questions hors périmètre et d'attaque sont bloquées avant tout appel LLM
  (0 appel juge nécessaire pour elles), sans regression sur les questions
  normales.
- Tests unitaires (mocks) pour chacun (`tests/test_guardrails.py`).

**Validation :** rapport avant/après ; on ne garde un garde-fou que si son
taux de faux positifs sur le golden set est acceptable. Ici : 0 faux positif
mesuré sur l'ensemble du golden set (20 questions) pour l'injection et le
hors sujet.

---

## Phase 6 — CI/CD

**Branche :** `ci/github-actions`

- `.github/workflows/neogarden-ci.yml` (à la racine de `Data-Science`, filtré
  sur ce dossier avec `paths:`) : à chaque push/PR → install dépendances,
  `ruff check`, `ruff format --check`, `pytest`.
- Badge de statut dans le README.
- Explication de CI/CD et de chaque étape du workflow à ce moment-là.

**Validation :** premier run vert sur GitHub, je te montre le lien.

---

## Phase 7 — Documentation

**Branche :** `docs/readme-and-learnings`

- `README.md` : une **version propre a déjà été écrite en Phase 2** (demande de
  l'utilisateur : l'ancien « kit de démarrage » contenait des notes personnelles
  et des chiffres faux). Elle couvre déjà objectif, schémas Mermaid, RAG vs
  fine-tuning, installation, lancement, données, limites et feuille de route.
  La Phase 7 l'enrichit de ce qui n'existe pas encore : commandes de test et de
  lint, badge de CI, **résultats chiffrés de l'évaluation**, garde-fous, et
  passe la feuille de route à jour.
- `docs/APPRENTISSAGES.md` : une notion par entrée (linter, formateur, tests,
  mocks, CI/CD, branches, PR, LLM-as-a-judge...), avec question d'entretien
  probable + réponse courte pour chacune.

**Validation :** relecture ensemble, dernière PR vers `main`.

---

## Workflow Git (rappel, appliqué à chaque phase)

```bash
git checkout -b feat/nom-de-la-phase
# ... travail, commits Conventional Commits (feat:, fix:, refactor:, test:, ci:, docs:) ...
git push -u origin feat/nom-de-la-phase
```
Puis sur GitHub : ouvrir une Pull Request vers `main`, relire le diff,
merger (je t'explique le geste précis quand on y arrive, pas maintenant en
abstrait).

## Ce qui ne change à aucun moment
- Le comportement de l'appli (mêmes réponses aux mêmes questions)
- Les données (FAQ, CGU, politique de retour, catalogue) : contenu inchangé,
  seulement déplacées dans `data/`
