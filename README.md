# musicdetect — Ce que voit un détecteur de musique IA

Code du projet de M2 IA « *Ce que voit un détecteur de musique IA* » : des détecteurs de musique générée par IA
sont entraînés **sans aucun générateur**, en opposant de la musique réelle (FMA) à sa version ré-encodée par des
codecs neuronaux (EnCodec, DAC, Mimi), puis testés sur des générateurs jamais vus (FakeMusicCaps) et sous attaques
(vitesse, hauteur, MP3, rééchantillonnage, bruit).

> **État du dépôt, en deux lignes.** Le chemin numpy / scipy / scikit-learn (chaîne de traitement, M0, M1,
> évaluation, attaques) est exécuté et testé. Tout ce qui dépend de PyTorch (codecs neuronaux, M2, M3, M4) est
> écrit et relu contre le code des dépôts de référence, **mais n'a jamais été exécuté**. Détails : § 9.

## 1. Question et hypothèses

> Un détecteur appris uniquement sur de la musique réelle contre sa version ré-encodée par des codecs neuronaux
> généralise-t-il à des générateurs jamais vus, ou apprend-il un raccourci de jeu de données ? Et que devient-il
> sous changement de vitesse ou compression MP3 ?

| | Hypothèse (figée avant tout entraînement) | Comment le code la teste |
|---|---|---|
| **H1** | Artefacts du décodeur : on transfère mieux vers les générateurs dont le décodeur ressemble à celui qui a fabriqué les « faux ». | Entraîner avec un codec « oracle » vs un codec générique, évaluer sur la même colonne de FakeMusicCaps. `FakeprintDetector.weights()` et `CrossCorrDetector.filter_weights()` exposent ce que le modèle a appris. |
| **H2** | Transfert partiel : entraîné sur FMA vs FMA ré-encodé, le détecteur repère surtout MusicGen (basé sur EnCodec). | `evaluate --test fmc:<générateur>` pour les cinq générateurs, un codec d'entraînement par ligne. |
| **H3** | Raccourcis : une part du score vient du format, de la fréquence d'échantillonnage, du volume, de la durée ou du silence. | Détecteur M0 (descripteurs triviaux) après la chaîne commune ; ablation `audio.silence_policy`. |
| **H4** | Le spectre linéaire s'effondre après un changement de vitesse ; log-fréquence + max-pooling y résiste mieux, pas forcément au MP3. | M1 (linéaire) vs M3 (log-fréquence) sous `evaluate --attacks`. |

Les propositions complètes (longue, 2 pages), le dossier initial et le cours sont dans [`docs/`](docs/README.md).

## 2. Les détecteurs

| | Entrée | Modèle | Rôle | Fichier |
|---|---|---|---|---|
| **M0** | descripteurs triviaux : après la chaîne (`residual` : coupure spectrale, RMS, crête, silences, plancher de bruit, écrêtage, composante continue) ou métadonnées de fichier (`raw`) | régression logistique ou arbres (`hgb`) | mesurer les raccourcis (H3) | `detectors/m0_metadata.py` |
| **M1** | « fakeprint » : spectre moyen en dB, 1–8 kHz, enveloppe inférieure retranchée (recette de Deezer) | régression logistique (C réglable sur la validation) | référence linéaire, interprétable (H1) | `detectors/m1_fakeprint.py` |
| **M2** | spectrogramme en dB, fréquence en canaux | CNN 1D, convolutions strided, max-pooling global (PyTorch) | référence « boîte noire » | `detectors/m2_cnn.py`, `nn_models.py` |
| **M3** | fakeprint sur grille **log-fréquence** (taille impaire) | corrélation croisée avec un filtre appris + max ; perte BCE + λ·entropie croisée sur la position du pic ; augmentation par changement de vitesse | invariance vitesse / hauteur (H4) | `detectors/m3_crosscorr.py` |
| **M4** | segment de 10 s → recadrages de 5 s | SONICS SpecTTTra, **inférence seule**, probabilités moyennées | point de comparaison hors projet | `detectors/m4_sonics.py` |

Tous suivent l'interface `Detector` (`fit`, `score`, `save`, `load`) de `detectors/base.py`. Étiquette 1 = faux.

## 3. La chaîne commune (là où se jouent les raccourcis)

Un faux et son réel apparié passent par **les mêmes étapes**, pour qu'ils ne diffèrent que par le codec :

1. Décodage MP3 par ffmpeg, mono par moyenne explicite des canaux, fréquence native conservée.
2. **Branche faux** : x → rééchantillonnage à la fréquence native du codec f_c → codec → ancrage à 16 kHz.
   **Branche réel** : le même aller-retour x → f_c → 16 kHz **sans** codec (source nommée `real_sr<f_c>`).
3. Alignement du faux sur le réel par corrélation croisée FFT (retard borné à ±0,05 s).
4. Politique de silence `none` ou `trim`, décidée sur la branche réelle et appliquée aux deux.
5. Segmentation en 10 s (`segments_per_track`, gigue aléatoire), **mêmes débuts pour réel et faux**.
6. Normalisation du niveau par segment (RMS −23 dBFS, limite de crête 0,99, drapeau `limited`).
7. Stockage en WAV int16 à 16 kHz.

Les clips FakeMusicCaps (déjà en 16 kHz, 10 s) ne passent que par les étapes 4 à 6.
Implémentation : `audio/chain.py`.

## 4. Installation

Outils système : **uv**, **Python 3.13**, **ffmpeg ≥ 6** avec `libmp3lame` et le filtre `rubberband`. Liste
complète, versions et licences : [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

```bash
uv lock                        # génère uv.lock (non fourni : voir DEPENDENCIES.md)
uv sync                        # cœur + dev : M0, M1, évaluation, attaques, tests
uv sync --extra torch          # + codecs neuronaux, M2, M3
uv sync --extra all            # + M4 (SONICS)
uv run musicdetect selftest    # contrôle l'installation sur données synthétiques (~40 s, sans PyTorch)
uv run pytest                  # tests
```

Un GPU n'est pas requis pour entraîner (il aide pour les codecs) : voir `docs/DEPENDENCIES.md`, § 6. Sans GPU NVIDIA,
la recette d'index PyTorch CPU est en commentaire dans `pyproject.toml`.

**Ce qu'il reste à faire :** [`RESTE_A_FAIRE.md`](RESTE_A_FAIRE.md).

## 5. Données

- **FMA** (Free Music Archive) : `fma_metadata.zip` + `fma_small` (ou `fma_medium`), décompressés dans
  `paths.fma_root` (défaut `data/raw/fma`, attendu : `fma_metadata/tracks.csv` et `fma_small/…`).
  Découpe train / val / test **par artiste** (`GroupShuffleSplit` : aucun artiste dans deux jeux par construction ;
  `assert_group_disjoint` contrôle le cas du découpage `official` de FMA).
- **FakeMusicCaps** : archive décompressée dans `paths.fmc_root`. La disposition (un dossier par générateur)
  est une **hypothèse** : `musicdetect fmc-index` affiche ce qui a été trouvé ; adapter `fmc.generators` dans la
  configuration si les noms de dossier diffèrent.

## 6. Flux de travail

Toutes les commandes acceptent `-c fichier.yaml`, `--set section.cle=valeur` (répétable), `--tag nom` (ablations),
`--device`, `--jobs`. **Ces options se placent après le sous-commande.**

```bash
# 0. Essai rapide : quelques centaines de pistes, quelques époques (chiffres sans valeur scientifique)
C="-c configs/smoke.yaml --tag smoke"

# 1. Pistes retenues + découpe par artiste
uv run musicdetect manifest $C
# 2. Segments réels et faux, un ou plusieurs codecs (reprise automatique ; ajouter un codec ne perd pas les autres)
uv run musicdetect build-fma $C --codec encodec_24k_6kbps --codec dac_44k
# 3. FakeMusicCaps : inventaire à contrôler, puis passage dans la chaîne
uv run musicdetect fmc-index $C
uv run musicdetect build-fmc $C
# 4. Entraînement (M4 : rien à entraîner)
uv run musicdetect train $C --detector m1 --codec encodec_24k_6kbps
uv run musicdetect train $C --detector m3 --codec encodec_24k_6kbps --device cuda
# 5. Évaluation : un test par --test ; --attacks lance la grille sur le premier
uv run musicdetect evaluate $C --detector m1 --codec encodec_24k_6kbps \
    --test fma:encodec_24k_6kbps:test --test fmc:musicgen --test fmc:musicldm \
    --test fmc:audioldm2 --test fmc:stable_audio_open --test fmc:mustango --attacks
# 6. ΔAUC apparié entre deux détecteurs, puis tableau récapitulatif
uv run musicdetect compare results/smoke/scores/<a>.csv results/smoke/scores/<b>.csv
uv run musicdetect report $C
```

Jeux de test : `fma:<codec>[:val|test|train]` (réels et faux appariés) ou `fmc:<générateur>`. Si des clips
`fmc:real` existent, ils servent de réels ; sinon ce sont les réels FMA de test, et le contraste porte alors aussi
sur le format (confusion à signaler dans les résultats).

**Ablations.** `--tag silence_trim --set audio.silence_policy=trim` écrit dans des dossiers séparés
(`segments/fma_silence_trim`, `models/silence_trim`, `results/silence_trim`) sans toucher aux autres.

## 7. Structure du code

```
pyproject.toml              dépendances, extras (torch, sonics, all), outils (pytest, ruff)
configs/
  smoke.yaml                essai rapide (surcharges de default.yaml)
  phase1.yaml               phase 1 de la proposition : 2 000 pistes, ~4 codecs
  full.yaml                 phase 2 : FMA small en entier (8 000 pistes)
docs/                       cours, propositions de projet, dépendances (voir docs/README.md)
src/musicdetect/
  default.yaml              TOUS les paramètres, commentés
  config.py                 chargement YAML, fusion récursive, surcharges --set typées
  cli.py                    ligne de commande (argparse)
  pipeline.py               les étapes (manifest, build, train, evaluate, compare) et la disposition des dossiers
  selftest.py               auto-test de bout en bout sur données synthétiques
  utils.py                  tqdm avec repli
  audio/
    io.py                   ffmpeg / ffprobe, lecture et écriture WAV
    resample.py             soxr (défaut) ou scipy (repli de test)
    align.py                retard par corrélation croisée FFT
    chain.py                la chaîne commune (§ 3), segments, niveau, silences
    attacks.py              vitesse, hauteur, MP3, rééchantillonnage, bruit ; grille et transformation d'évaluation
  codecs/
    base.py                 interface Codec (native_sr, reconstruct)
    neural.py               EnCodec, DAC, Mimi, EnCodec de MusicGen (via transformers) — torch
    toy.py                  codec jouet à peigne de raies + identité (tests, sans torch)
  data/
    fma.py                  tracks.csv, sous-ensembles emboîtés, découpe par artiste
    fmc.py                  découverte des clips FakeMusicCaps
    segments.py             index CSV, SegmentSet (étiquettes, groupes, transformations)
    build.py                construction des segments (reprise, échecs consignés)
  detectors/
    features.py             fakeprint, enveloppe inférieure, grille log, descripteurs M0
    base.py                 interface Detector, extraction parallèle
    m0_metadata.py  m1_fakeprint.py  m2_cnn.py  m3_crosscorr.py  m4_sonics.py
    nn_models.py            SpecCNN, XCorrProbe, jeu de données PyTorch — torch
  eval/
    metrics.py              AUC, EER, TPR / FPR à seuil fixé
    bootstrap.py            bootstrap groupé, ΔAUC apparié, correction de Holm
    calibration.py          ECE, mise à l'échelle par température
    protocol.py             résumés, seuil de validation, grille d'attaques, résolution des jeux de test
tests/                      53 tests (+ test_torch.py, ignoré sans PyTorch)
```

Disposition des données produites (`paths.work_dir`, `paths.results_dir`) :

```
work_dir/manifest_fma.csv                       pistes retenues + découpe
work_dir/fmc_files.csv                          clips FakeMusicCaps trouvés
work_dir/segments/fma[_tag]/<source>/*.wav      réels appariés (real_sr<f_c>) et un dossier par codec
work_dir/index_fma[_tag].csv                    un segment par ligne : source, piste, groupe, découpe, drapeaux
work_dir/models/<tag|main>/<détecteur>__<codec>.{joblib|pt,json}   modèle + seuil et AUC de validation
results/<tag|main>/<modèle>.csv                 AUC + IC, EER, TPR / FPR, ECE par jeu de test
results/<tag|main>/attacks_<modèle>.csv         une ligne par attaque
results/<tag|main>/scores/*.csv                 scores par segment (entrée de `compare`)
```

## 8. Protocole d'évaluation

- **AUC** avec intervalle à 95 % par bootstrap **groupé** (groupes : artiste pour FMA, légende / source pour
  FakeMusicCaps), car des segments d'un même morceau ne sont pas indépendants. `above_chance` : l'IC exclut 0,5.
- **Seuil fixé une fois** sur les réels de validation propre (FPR cible 1 %), **jamais réajusté sous attaque** :
  une attaque qui fait dériver le FPR se voit.
- **EER**, **ECE** et **ΔAUC apparié** entre détecteurs (`compare`) sont calculés par la CLI. La **correction de Holm**
  (`eval.bootstrap.holm`) et la **mise à l'échelle par température** (`eval.calibration`) existent comme fonctions,
  testées, mais ne sont pas branchées sur la CLI : à appeler depuis un notebook.
- Grille d'attaques (`eval.attacks`) : vitesse 0,9 / 1,1 (étendue : 0,7 à 1,4), hauteur ±1 et ±2 demi-tons,
  MP3 64 à 320 kbit/s, rééchantillonnage 12 000 / 11 025 Hz, bruit blanc à 30 / 20 / 10 dB.

## 9. Ce qui a été testé, et ce qui ne l'a pas été

**Exécuté** (Python 3.13, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, ffmpeg 6.1.1) : 53 tests, dont un test
de bout en bout (`selftest`) : faux MP3 → chaîne → codec jouet → M0 / M1 → évaluation → attaques. Le test de la
prémisse de M3 (un changement de vitesse décale le fakeprint log-fréquence du retard prédit) et celui de la
convention de la corrélation croisée passent en numpy.

**Jamais exécuté** (PyTorch, `transformers`, `soxr`, `tqdm` absents de l'environnement d'écriture) :
- les codecs neuronaux (`codecs/neural.py`), M2, M3 (partie PyTorch), M4 ;
- le chemin `soxr` (les tests ont utilisé le repli scipy) ;
- `tests/test_torch.py` : écrit (formes, équivalence avec la référence numpy de M3, aller-retour d'entraînement),
  ignoré faute de PyTorch. **Premier réflexe sur votre machine : `uv run pytest`.**

La relecture contre les sources a vérifié les signatures d'appel de `transformers` (EnCodec, DAC, Mimi), le chargement
et la normalisation de SONICS (16 kHz, crops de 5 s, division par l'écart-type, 1 = faux). Elle n'a pas pu vérifier
l'exécution.

**Sans valeur scientifique** : tous les AUC de `selftest` (peigne de raies synthétique, quelques dizaines de segments).

## 10. Limites et hypothèses sur lesquelles le code repose

- **Ligne « VAE de Stable Audio Open (oracle) » de la proposition : non implémentée.** Seuls EnCodec 24 kHz,
  EnCodec 32 kHz de MusicGen (non testé), DAC et Mimi sont câblés ; ajouter un codec = une sous-classe de `Codec`
  et une entrée dans `codecs` de la configuration.
- Disposition de l'archive FakeMusicCaps supposée (§ 5). Chemins et noms de colonnes de FMA : ceux de `tracks.csv`
  à deux niveaux d'en-tête.
- Bornes de versions non éprouvées sur PyPI ; la résolution avec l'extra `sonics` (qui tire `librosa` et `numba`)
  peut plafonner numpy : voir `docs/DEPENDENCIES.md`.
- Le contraste réel / faux de `fmc:<générateur>` sans clips `fmc:real` mélange génération et format (§ 6).
- M3 reprend la perte de position de Barand avec λ = 0,05 ; le nom du point de contrôle de leur dépôt indique 0,1.

## 11. Licences

Ce dépôt réécrit des idées de dépôts **CC BY-NC 4.0** (`deezer/ismir25-ai-music-detector`,
`deezer/deepfake-detector`, `thomas0barand/robust-deepfake-detector`) ; Deezer précise deux demandes de brevet et
**réserve ses dérivés à la recherche, sans usage commercial**. Le jeu de données SONICS est CC BY-NC (code MIT).
FMA : code MIT, métadonnées CC BY 4.0, audio sous licence propre à chaque artiste. Mimi : CC BY 4.0 d'après la proposition (fiche du modèle, à reconfirmer) ; licence des poids
EnCodec non vérifiée. Aucune licence n'est choisie pour ce dépôt : à fixer avec votre encadrant.
