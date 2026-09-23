# Apprentissages — NeoGarden Assistant

Une notion par entrée, telle qu'elle a été réellement appliquée dans ce
projet (pas une définition abstraite), avec une question d'entretien
probable et une réponse courte.

---

## Linter (Ruff)

Un linter lit le code **sans l'exécuter** et repère les erreurs probables
(imports inutilisés, variables non définies) et les entorses de style.
Ruff fait aussi office de formateur (comme Black), en un seul outil rapide.

**Question probable :** « Quelle différence entre un linter et un
formateur ? »
**Réponse courte :** le formateur change la mise en forme (indentation,
retours à la ligne) sans changer le sens du code ; le linter repère des
problèmes potentiels (parfois de vrais bugs, comme un import oublié) sans
les corriger automatiquement — sauf les correctifs sûrs (`--fix`).

---

## Tests unitaires vs tests d'intégration

Un test unitaire vérifie **une fonction isolée**, sans rien de réel autour
(`test_guardrails.py::test_is_greeting_ignores_space_before_punctuation` —
une fonction pure, texte en entrée, booléen en sortie). Un test
d'intégration vérifie que **plusieurs pièces marchent ensemble**
(`test_retriever.py` — un vrai index FAISS, un vrai retriever LangChain,
seuls les embeddings sont simulés pour éviter de télécharger un modèle).

**Question probable :** « Comment savoir si un test est unitaire ou
d'intégration ? »
**Réponse courte :** ce n'est pas une question de présence de réseau, mais
de nombre de composants réels impliqués. Un test unitaire isole une pièce ;
un test d'intégration vérifie leur assemblage.

---

## Mocks / doublures de test

Remplacer une dépendance réelle (API, base de données) par une version
factice et contrôlée, pour tester la logique sans dépendre d'un service
externe. Exemple : `tests/test_generation.py` construit de vraies
exceptions `openai.*` à la main (sans réseau) pour vérifier que la
politique de retry réagit correctement à chaque cas (429, 500, erreur
définitive...).

**Question probable :** « Pourquoi ne pas juste appeler la vraie API dans
les tests ? »
**Réponse courte :** lenteur, coût, non-déterminisme (un LLM ne répond
jamais deux fois pareil), et impossibilité de forcer un cas précis (comme
une panne réseau) de façon fiable et répétable.

---

## TDD (rouge → vert)

Écrire le test **avant** le correctif, constater qu'il échoue (rouge),
puis corriger le code pour qu'il passe (vert). Fait concrètement dans ce
projet pour le bug `is_greeting("Bonjour !")` : un commit `test:` ajoute le
test et le montre en échec, puis un commit `fix:` séparé le fait passer.

**Question probable :** « Quel est l'intérêt d'écrire le test avant le
correctif, plutôt qu'après ? »
**Réponse courte :** ça prouve que le test détecte vraiment le bug (s'il
passait déjà avant le correctif, il ne testerait rien) et documente le bug
de façon reproductible.

---

## CI/CD (intégration continue)

Lancer automatiquement les vérifications (lint, tests) sur un serveur
distant à chaque push ou pull request, plutôt que de compter sur chacun de
les lancer à la main. Implémenté avec GitHub Actions
(`.github/workflows/neogarden-ci.yml`) : install, `ruff check`,
`ruff format --check`, `pytest`.

**Question probable :** « Pourquoi l'évaluation du RAG n'est pas dans la
CI ? »
**Réponse courte :** elle appelle une vraie API (lente, payante même si
gratuite ici via quota, non déterministe) — la CI doit rester rapide et
reproductible à chaque commit. L'évaluation reste un outil qu'on lance à
la demande.

---

## Git : branches, Pull Requests, Conventional Commits

Une branche par unité de travail (ici, une par phase), jamais de commit
direct sur `main`. Une Pull Request propose la fusion et affiche le diff
complet avant qu'il ne devienne définitif. Les messages de commit suivent
un préfixe qui dit *quel type* de changement (`feat:` nouveauté, `fix:`
correction de bug, `test:`, `docs:`, `ci:`, `chore:` tâche d'outillage).

**Question probable :** « Pourquoi ne pas pousser directement sur main ? »
**Réponse courte :** la PR permet une relecture avant que le changement
soit définitif, garde un historique clair par unité de travail, et
correspond à la pratique standard en entreprise.

---

## LLM-as-a-judge

Utiliser un second LLM pour noter automatiquement les réponses du premier,
plutôt qu'une comparaison de texte stricte (impossible, un LLM ne répond
jamais mot pour mot pareil) ou une relecture humaine de chaque réponse
(pas soutenable à l'échelle). Implémenté dans `evaluation/evaluate.py` :
un second modèle reçoit la question, le contexte, la réponse attendue et
la réponse réelle, et renvoie un verdict structuré (JSON).

**Question probable :** « Pourquoi un LLM différent comme juge, pas le
même modèle ? »
**Réponse courte :** un modèle qui évalue ses propres réponses a tendance
à se trouver bon (biais d'auto-évaluation). Idéalement un fournisseur
différent aussi, pour plus d'indépendance — compromis assumé ici (même
fournisseur, modèle beaucoup plus gros) faute de modèles gratuits fiables
d'autres fournisseurs au moment du test.

---

## RAG : deux sources d'erreur distinctes

Un RAG (Retrieval-Augmented Generation) peut échouer à deux endroits
différents, et il faut savoir lequel pour corriger le bon problème : le
**retrieval** (les bons documents ne sont pas retrouvés) ou la
**génération** (les bons documents sont là, mais le LLM ne les exploite
pas bien). `evaluate.py` les mesure séparément.

**Question probable :** « Comment savoir si un RAG rate à cause de la
recherche ou de la génération ? »
**Réponse courte :** vérifier si le bon document apparaît dans les
passages retrouvés (top-k). S'il y est mais que la réponse est quand même
mauvaise, le problème est dans la génération (ou le prompt) ; s'il n'y est
pas, le problème est dans le retrieval (chunking, embeddings, k trop
petit...).

---

## Gestion précise des erreurs (vs `except Exception`)

Un `except Exception` générique masque aussi bien un problème réseau
temporaire qu'un vrai bug. `src/rag_garden/generation.py` distingue les
erreurs **temporaires** (réseau coupé, 429, 5xx — on réessaie) des erreurs
**définitives** (clé invalide, requête refusée — on ne réessaie pas, ça ne
servirait à rien), y compris quand l'erreur arrive dans le corps d'une
réponse HTTP 200 (cas réel rencontré avec OpenRouter/LangChain).

**Question probable :** « Pourquoi ne pas juste attraper toutes les
exceptions et réessayer ? »
**Réponse courte :** réessayer une erreur définitive (clé invalide) gaspille
du temps et masque un vrai problème de configuration ; ne jamais réessayer
une erreur temporaire dégrade l'expérience pour rien.

---

## Limites de débit (rate limiting) et quotas API

Une API, même gratuite, a des limites : dans ce projet, OpenRouter limite
les modèles gratuits à 50 requêtes par jour **pour tout le compte**, pas
par modèle. Une politique de retry doit distinguer une surcharge passagère
(retenter avec un délai) d'un quota journalier épuisé (retenter ne sert à
rien avant la réinitialisation).

**Question probable :** « Ton système gère bien les pannes, mais que se
passe-t-il si le quota gratuit est épuisé ? »
**Réponse courte :** le retry échoue proprement après un nombre de
tentatives limité et lève une erreur explicite plutôt que de boucler
indéfiniment ; concrètement rencontré pendant ce projet, résolu en
réduisant le périmètre d'un test plutôt qu'en payant.

---

## Hooks pre-commit

Un script qui s'exécute automatiquement **avant** chaque commit local, pour
bloquer un commit qui casserait le lint ou le format — avant même que ça
n'atteigne la CI. Testé en conditions réelles pendant ce projet : un commit
volontairement cassé (import mal placé) a été refusé avec l'erreur exacte
affichée.

**Question probable :** « Quelle différence entre pre-commit et la CI ? »
**Réponse courte :** pre-commit s'exécute en local, avant même que le code
parte sur GitHub — retour immédiat, rien à pousser pour le savoir. La CI
s'exécute côté serveur, après le push, et sert de filet de sécurité pour
tout le monde (y compris si quelqu'un contourne le hook local).

---

## Impact mesurable d'un changement de prompt

Un changement de prompt système change le comportement du modèle — il ne
faut donc pas le modifier « à l'instinct », mais mesurer l'effet avant et
après. Fait concrètement dans ce projet : un bug connu du prompt (deux
règles collées sur la même ligne) a été corrigé dans un commit `fix:`
séparé, avec un score avant/après sur le même jeu de questions.

**Question probable :** « Comment sais-tu qu'un changement de prompt a
vraiment amélioré les choses ? »
**Réponse courte :** en le mesurant avec le même jeu de test avant et
après, pas en le supposant. Ici, le score est resté identique (5/6) : le
bug corrigé n'était pas la cause de l'échec restant, ce qui est une
information utile en soi (évite de croire à tort qu'on a réglé le
problème).
