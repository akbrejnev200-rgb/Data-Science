# Analyse exploratoire

Quatre notebooks, un par bloc de la Phase 3 (`notebooks_avec_ecriture_dans_database/modelisation_*.py`) :
analyse exploratoire des donnees utilisees par chaque modele, puis diagnostic
d'apprentissage adapte au type de modele du bloc.

| Notebook | Bloc | Cible | Table source | Diagnostic d'apprentissage |
|---|---|---|---|---|
| `01_bloc_3a_volumes.ipynb` | 3A | `volume_entrant_jour` | `features_journalier` | Fenetre expansive chronologique (candidat XGBoost) |
| `02_bloc_3b_charge_etp.ipynb` | 3B | `charge_par_etp` | `features_journalier` | Fenetre expansive chronologique (XGBoost retenu) |
| `03_bloc_3c_retard.ipynb` | 3C | `est_en_retard` | `features_detail` | Fenetre expansive chronologique (Decision Tree retenu) |
| `04_bloc_3d_anomalies.ipynb` | 3D | (non supervise) | `features_detail` | Injection d'anomalies synthetiques : n_estimators et taille du train |

## Pourquoi pas `sklearn.model_selection.learning_curve` partout

Cette fonction tire des sous-echantillons aleatoires et fait une validation
croisee classique -- correct pour des observations independantes, mais faux
pour des donnees ordonnees dans le temps (elle melangerait passe et futur).
Les notebooks 3A/3B/3C utilisent donc une fenetre **expansive et
chronologique** faite maison : on entraine sur les N premiers jours/lignes du
train (jamais un tirage aleatoire) et on evalue systematiquement sur le meme
jeu de validation. Le notebook 3D (Isolation Forest) n'a pas de label
"anomalie" dans les donnees d'origine : la notion meme de score
train/validation ne s'applique pas a un modele non supervise, donc le
diagnostic repris de `modelisation_anomalies.py` mesure a la place un taux de
detection sur des anomalies injectees artificiellement.

## Reproduire

Chaque notebook se suffit a lui-meme (connexion DB, requetes, entrainements,
figures) et peut etre relance de bout en bout depuis ce dossier avec le
kernel `mon_projet_venv`. Les figures generees sont sauvegardees dans
`figures/<bloc>/` a chaque execution.

Duree approximative d'une execution complete : quelques secondes pour 3A/3B
(640 lignes), 3 a 5 minutes pour 3C/3D (donnees a la tache, ~940 000 lignes,
chargement + plusieurs reentrainements a taille croissante).

## Sources croisees

Les interpretations renvoient regulierement vers `docs/AMELIORATION_MODELES.md`
(comparaison avant/apres portage, chiffres de production) : ce notebook
explore ce que les *donnees* montrent en amont, `AMELIORATION_MODELES.md`
documente ce que les *modeles* obtiennent en sortie -- les deux se recoupent
mais ne redisent pas la meme chose.
