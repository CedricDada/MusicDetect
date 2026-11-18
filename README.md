# musicdetect

Code du projet « Que voit un détecteur de musique IA ? ». Des détecteurs sont entraînés **sans aucun
générateur** à distinguer de la musique réelle (FMA) de sa reconstruction par un codec neuronal (EnCodec), puis
testés sur cinq générateurs inconnus (FakeMusicCaps) et sous attaques (vitesse, MP3, etc.).

- Résultats, protocole et discussion : [`report/main.pdf`](report/main.pdf).
- Hypothèses et critères de réfutation, figés avant tout entraînement (commit `b0d72f8`) :
  [`preregistration.md`](preregistration.md).

## Installation

Outils système : **uv**, **Python 3.13**, **ffmpeg ≥ 6** avec `libmp3lame` (attaque MP3) et `rubberband`
(attaque de hauteur).

```bash
uv sync --extra torch          # cœur + codecs neuronaux, M2, M3
uv sync --extra all            # + M4 (SONICS)
uv run musicdetect selftest    # contrôle l'installation sur données synthétiques (~40 s)
uv run pytest                  # tests
```

`uv.lock` fige les versions. **M0 et M1 sont enregistrés par pickle** : les relire avec une autre version de
scikit-learn que celle du verrou (1.9.1) fait échouer M1 ou fausse M0.

## Données

- **FMA** : `fma_metadata.zip` et `fma_small.zip` décompressés dans `data/raw/fma/`
  (`fma_metadata/tracks.csv`, `fma_small/…`). Découpe apprentissage / validation / test **par artiste**.
- **FakeMusicCaps** : archive décompressée dans `data/raw/FakeMusicCaps/`. `musicdetect fmc-index` affiche ce qui
  a été trouvé ; adapter `fmc.generators` dans la configuration si les noms de dossier diffèrent. L'archive ne
  contient pas les extraits réels de MusicCaps : les négatifs d'un test `fmc:<générateur>` sont les réels FMA du
  test.

## Utilisation

Toutes les commandes acceptent `-c fichier.yaml`, `--set section.cle=valeur` (répétable), `--tag nom`,
`--device` et `--jobs` (4 recommandé : la valeur par défaut, 1, rend l'augmentation de M3 très lente). Ces options
se placent **après** la sous-commande. Tous les paramètres sont dans
[`src/musicdetect/default.yaml`](src/musicdetect/default.yaml) ; `configs/` contient les surcharges
(`smoke` : essai rapide ; `phase1` : l'étude du rapport).

```bash
C="-c configs/phase1.yaml --tag phase1"

uv run musicdetect manifest $C                                  # pistes retenues et découpe
uv run musicdetect build-fma $C --codec encodec_24k_6kbps --codec encodec_24k_12kbps --codec identity_24k
uv run musicdetect fmc-index $C && uv run musicdetect build-fmc $C

uv run musicdetect train $C --detector m1 --codec encodec_24k_6kbps
uv run musicdetect train -c configs/phase1.yaml --tag phase1_s1 --detector m3 --codec encodec_24k_6kbps \
    --set m3.seed=1 --jobs 4                        # une graine = un tag

uv run musicdetect evaluate $C --detector m1 --codec encodec_24k_6kbps \
    --test fma:encodec_24k_6kbps --test fma:identity_24k \
    --test fmc:musicgen --test fmc:stable_audio_open --test fmc:audioldm2 --test fmc:musicldm --test fmc:mustango \
    --attacks --attack-on fmc:musicgen              # grille d'attaques sur MusicGen

uv run musicdetect compare results/phase1/scores/<a>.csv results/phase1/scores/<b>.csv
uv run musicdetect report $C                        # results/<tag>/summary.csv
```

Jeux de test : `fma:<codec>` (réels et faux appariés, split test) ou `fmc:<générateur>`. Le test `fma:identity_24k`
est le contrôle négatif : tout détecteur doit y obtenir une AUC de 0,5. Chaque graine de M2 et M3 utilise son
propre `--tag` (`phase1_s1`, `phase1_s2`, …), et un tag lit ses propres index et segments
(`data/work/index_*_<tag>.csv`, `data/work/segments/*_<tag>`) : pour réutiliser les mêmes segments, copier les
index et lier les dossiers (lien symbolique ou jonction Windows) sous le nouveau tag.

## Reproduire les chiffres et les figures du rapport

Les analyses relisent les scores par segment (`results/<tag>/scores/`), sans recharger de modèle :

```bash
uv run python -m musicdetect.analysis --results results \
    --fma-index data/work/index_fma_phase1.csv --fmc-index data/work/index_fmc_phase1.csv \
    --manifest data/work/manifest_fma.csv --out results/analysis       # tests, Holm, figures, erreurs

uv run python -m musicdetect.peigne --model data/work/models/phase1/m1__encodec_24k_6kbps.joblib \
    --fma-index data/work/index_fma_phase1.csv --fma-dir data/work/segments/fma_phase1 \
    --fmc-index data/work/index_fmc_phase1.csv --fmc-dir data/work/segments/fmc_phase1 \
    --out results/analysis/peigne                                      # peigne de pics de M1

cd report && latexmk -pdf main.tex                                     # le rapport
```

[`notebooks/musicdetect_kaggle.ipynb`](notebooks/musicdetect_kaggle.ipynb) enchaîne ces étapes sur Kaggle en
reprenant ce qui est déjà calculé (utilisé pour la graine 3, les attaques et les analyses du rapport).

## Structure

```
configs/                    surcharges de default.yaml (smoke, phase1, full)
report/                     rapport LaTeX et figures
notebooks/                  session Kaggle reproductible
src/musicdetect/
  default.yaml              tous les paramètres, commentés
  cli.py  pipeline.py       ligne de commande ; étapes et disposition des dossiers
  analysis.py  peigne.py    analyses confirmatoires, figures ; lecture en fréquence de M1
  audio/                    ffmpeg, rééchantillonnage, alignement, chaîne commune, attaques
  codecs/                   EnCodec, DAC, Mimi (torch) ; codec identité et jouet (tests)
  data/                     FMA, FakeMusicCaps, index de segments, construction
  detectors/                M0 (raccourcis), M1 (fakeprint), M2 (CNN), M3 (log-fréquence), M4 (SONICS)
  eval/                     métriques, bootstrap groupé et apparié, Holm, calibration, protocole
tests/
```

Dossiers produits (ignorés par git) : `data/work/` (index, segments WAV, modèles), `results/`.

## Licences

Ce code réécrit des idées de dépôts **CC BY-NC 4.0** (`deezer/ismir25-ai-music-detector`,
`deezer/deepfake-detector`, `thomas0barand/robust-deepfake-detector`) ; Deezer précise deux demandes de brevet et
réserve ses dérivés à la recherche, sans usage commercial. SONICS : données CC BY-NC, code MIT. FMA : code MIT,
métadonnées CC BY 4.0, audio sous licence propre à chaque artiste. Aucune licence n'est choisie pour ce dépôt.
Le gabarit LaTeX `sleek` (`report/packages/`) est sous licence MIT.
