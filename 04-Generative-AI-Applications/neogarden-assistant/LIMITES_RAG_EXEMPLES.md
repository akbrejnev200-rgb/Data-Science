# Exemples concrets de limites du RAG naïf — NeoGarden

Deux cas observés en testant l'app ce soir, utiles pour illustrer concrètement
les limites listées dans le README plutôt que de les citer abstraitement.

---

## Exemple 1 — Paiement en 3 fois (chunk dilué par du texte voisin)

**Question testée :** "peux-t-on payer en plusieurs fois ?"
**Réponse obtenue :** le bot dit ne pas savoir.
**Réalité :** l'info existe bel et bien, dans `NeoGarden_CGU.docx`, Article 7 — Paiement :

> « Le paiement en trois fois sans frais est proposé pour les paniers compris
> entre 150 et 2 000 euros. »

**Pourquoi ça rate :** ce paragraphe est court (~320 caractères). Avec
`chunk_size=1000`, le splitter le fusionne dans le même chunk que la fin de
l'Article 6 et le début de l'Article 8 — Livraison (« France métropolitaine,
Corse, DOM, Belgique... »), un texte quasi identique à celui présent dans
plusieurs autres chunks (FAQ, catalogue). Ce contenu voisin « dilue » le
vecteur du chunk vers le thème livraison plutôt que paiement. Avec `k=4` fixe,
ce chunk se fait doubler par d'autres chunks plus nettement orientés sur un
seul thème.

**Limites README concernées :** n°2 (recherche vectorielle pure, k=4 fixe) et
n°6 (chunking à 1000/150, choisi sans mesure).

---

## Exemple 2 — Sensibilité à la formulation de la question

**Question A :** « connais-tu les frais de port en France ? »
→ Réponse correcte et complète : 4,90 € (standard, gratuit dès 59 €), 9,90 €
(express).

**Question B (même intention) :** « c'est combien vos prix de livraison ? »
→ Réponse incorrecte : « je ne dispose pas d'information », alors que le
chunk contenant la réponse existe dans l'index.

**Pourquoi ça rate :** chaque reformulation produit un vecteur d'embedding
légèrement différent. Le jeu des `k=4` chunks les plus proches peut donc
changer d'une formulation à l'autre, même pour une intention identique. La
recherche vectorielle pure n'est pas robuste aux paraphrases.

**Limite README concernée :** n°2 (recherche vectorielle pure, k=4 fixe).
