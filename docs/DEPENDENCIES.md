# Outils, bibliothèques et versions

Ce document répond à : *de quoi a-t-on besoin, et quelles versions mettre dans `uv.lock` ?*

> **`uv.lock` n'est pas dans l'archive.** L'environnement où le code a été écrit n'a pas accès à PyPI
> (`uv lock` échoue sur la résolution DNS) ; fabriquer un verrou à la main serait inventer des versions et des
> empreintes. Il se génère en une commande sur votre machine :
>
> ```bash
> uv lock                      # résout et écrit uv.lock (à committer)
> uv sync --extra all          # installe cœur + torch + sonics (ou --extra torch, ou rien pour le cœur seul)
> ```
>
> Les bornes de `pyproject.toml` ci-dessous sont ce qui est déclaré ; `uv.lock` en fixera les versions exactes.

## 1. Outils système (hors Python)

| Outil | Version | Pourquoi | Vérifié ici |
|---|---|---|---|
| **uv** | récent | environnement + verrou | non (pas de réseau) |
| **Python** | 3.13 (`.python-version`) ; `requires-python = ">=3.11,<3.14"` | le verrou de Barand est en 3.13 | 3.13.18 |
| **ffmpeg + ffprobe** | ≥ 6 | décodage MP3 → mono, MP3 aller-retour (attaque), `rubberband` (attaque de hauteur) | 6.1.1 |
| ↳ encodeur `libmp3lame` | — | attaque MP3 | présent |
| ↳ filtre `rubberband` (`librubberband`) | — | attaque de hauteur (durée conservée) | présent |
| **git** | — | le paquet `sonics` est installé depuis GitHub (commit épinglé) | oui |
| GPU NVIDIA + pilote | facultatif | M2, M3, M4 et les codecs neuronaux marchent sur CPU mais lentement ; les roues `torch` embarquent leur runtime CUDA | non |
| TeX Live (`pdflatex`, `babel-french`) | facultatif | recompiler `docs/` | `babel-french` absent ici : voir `docs/README.md` |

Place disque (d'après le README de FMA) : `fma_metadata.zip` 342 Mio ; `fma_small` 7,2 Gio (8 000 pistes) ;
`fma_medium` 22 Gio (25 000 pistes) ; `fma_large` 93 Gio. Les segments WAV produits par la chaîne (16 kHz, int16,
10 s ≈ 320 Ko) s'y ajoutent : comptez (pistes × segments × (1 + nombre de codecs) × 320 Ko).

Installer ffmpeg : `apt install ffmpeg` (Debian/Ubuntu), `brew install ffmpeg` (macOS). Contrôle :
`ffmpeg -hide_banner -encoders | grep libmp3lame` et `ffmpeg -hide_banner -filters | grep rubberband`.

## 2. Bibliothèques Python

### Cœur (toujours installées) — M0, M1, évaluation, attaques, tests

| Paquet | Borne dans `pyproject.toml` | Référence | Testé ici | Rôle |
|---|---|---|---|---|
| numpy | `>=2.3,<3` | Barand 2.3.5 | 2.5.3 | tout le signal |
| scipy | `>=1.17,<2` | Barand 1.17.0 | 1.18.1 | FFT, corrélation, fenêtres, repli de rééchantillonnage |
| scikit-learn | `>=1.8,<2` | Barand 1.8.0 | 1.9.1 | M0/M1, AUC, découpe par groupes |
| joblib | `>=1.5,<2` | Barand 1.5.3 | 1.6.0 | extraction parallèle |
| pandas | `>=2.2,<4` | — | 3.0.5 | index des segments, manifestes |
| pyyaml | `>=6.0,<7` | — | 6.0.3 | configuration |
| soxr | `>=1.0,<2` | Barand 1.0.0 | **non installé** | rééchantillonnage haute qualité (défaut) |
| tqdm | `>=4.67,<5` | Barand 4.67.1 | **non installé** | barres de progression (repli sans barre) |
| matplotlib | `>=3.10,<4` | Barand 3.10.8 | 3.11.2 | figures |

« Barand » = `requirements.lock` de `thomas0barand/robust-deepfake-detector` (Python 3.13). Pour pandas et
pyyaml, absents de ce verrou, la borne vient des API utilisées et n'a pas été éprouvée en version ancienne.

Groupe de développement (`uv sync` l'installe par défaut) : `pytest>=8.0` (testé : 9.1.1), `ruff>=0.6`.

### Extra `torch` — codecs neuronaux, M2, M3

| Paquet | Borne | Référence | Rôle |
|---|---|---|---|
| torch | `>=2.9,<3` | Barand 2.9.1 | M2, M3 |
| transformers | `>=4.57,<6` | — | `EncodecModel`, `DacModel`, `MimiModel` |
| huggingface-hub | `>=0.34` | — | téléchargement des poids |

### Extra `sonics` — M4 (inférence seule)

| Paquet | Borne | Rôle |
|---|---|---|
| torchaudio | `>=2.9,<3` | spectrogramme mel de SONICS |
| timm | `>=1.0.7` | exigé par `sonics` |
| sonics | `git+https://github.com/awsaf49/sonics.git@9156ffad…` (HEAD du 22 mai 2025, vérifié dans le dépôt) | `HFAudioClassifier` |

**Attention à la résolution.** `sonics` importe `librosa` dès l'import du paquet (via `sonics.utils.dataset`),
donc tire `numba` (+ `soundfile`, `audioread`, etc.). Le verrou de Barand associe `librosa 0.11.0` et
`numba 0.63.1` à `numpy 2.3.5` : avec l'extra `sonics`, `uv lock` risque donc de plafonner numpy autour de 2.3.x,
et peut-être scipy/pandas avec lui, sous les versions testées ici (2.5.3 pour numpy). C'est résoluble, mais **non
vérifié** : si la résolution échoue, la première chose à essayer est de retirer `sonics` du verrou principal et
de l'installer dans un second environnement (`uv sync --extra torch` d'un côté, `uv pip install` de `sonics` de l'autre).

### Choix d'exclusion (et pourquoi)

| Écarté | Remplacé par | Raison |
|---|---|---|
| `descript-audio-codec` / `descript-audiotools` | `transformers.DacModel` | `audiotools` impose `protobuf<3.20` : résolution fragile en Python 3.13 |
| `librosa`, `soundfile` (dans le cœur) | ffmpeg (décodage) + `scipy` + WAV maison | moins de dépendances natives ; `librosa` n'arrive que par `sonics` |
| `pyrubberband` | filtre `rubberband` de ffmpeg | pas de binaire `rubberband` séparé à installer |
| `lightning`, `torchmetrics`, `nnAudio` | PyTorch simple + `scipy` | M3 réécrit sans Lightning ; la STFT interpolée est calculée en numpy |
| `encodec` (paquet Meta) | `transformers.EncodecModel` | un seul chemin pour EnCodec, DAC, Mimi |

## 3. Poids téléchargés (Hugging Face Hub)

| Usage | Identifiant | Remarque |
|---|---|---|
| EnCodec 24 kHz | `facebook/encodec_24khz` | bandes 1,5 / 3 / 6 / 12 / 24 kbit/s |
| DAC 44,1 kHz | `descript/dac_44khz` | |
| Mimi | `kyutai/mimi` | 8 codebooks par défaut dans la configuration |
| EnCodec 32 kHz de MusicGen | `facebook/musicgen-small` (`audio_encoder`) | non testé |
| SONICS (M4) | `awsaf49/sonics-spectttra-gamma-5s` | 16 kHz, crops de 5 s |

## 4. Dépôts consultés pour s'en inspirer (état du clone, licence)

| Dépôt | Commit | Licence | Ce qui a servi |
|---|---|---|---|
| `deezer/ismir25-ai-music-detector` | dc8713a (2025-11-12) | CC BY-NC 4.0 + 2 demandes de brevet (déc. 2024) | recette du fakeprint (M1) |
| `deezer/deepfake-detector` | edc94ad (2026-01-28) | CC BY-NC 4.0 | `SimpleSpectrogramCNN` → M2 |
| `thomas0barand/robust-deepfake-detector` | c4aa1df (2026-07-24) | CC BY-NC 4.0 | `RobustDetector`, `LinearProj`, grille log → M3 |
| `awsaf49/sonics` | 9156ffa (2025-05-22) | code MIT (jeu de données CC BY-NC) | SpecTTTra, normalisation par l'écart-type → M4 |
| `facebookresearch/encodec` | 0e2d0ae (2023-06-20) | code MIT ; licence des poids non vérifiée | codec |
| `descriptinc/descript-audio-codec` | c7cfc5d (2023-07-20) | MIT | codec |
| `dolbylaboratories/neural-upsampling-artifacts-audio` | ab496c7 (2021-02-09) | MIT | motifs d'upsampling (cours) |
| `mdeff/fma` | 607364e (2021-09-07) | code MIT ; métadonnées CC BY 4.0 ; audio : licence par artiste | jeu de données réel |

**Conséquence de licence.** Le code de ce dépôt réécrit des idées de dépôts CC BY-NC : à traiter comme du code de
**recherche, sans usage commercial**. Les brevets Deezer interdisent explicitement un usage commercial des
dérivés. Le code a été écrit après lecture de ces dépôts (réécriture, sans copie de fichiers) ; il reste à traiter comme dérivé.

## 5. Ce que ce document ne garantit pas

- Aucune version n'a été interrogée sur PyPI (réseau fermé) ; les bornes viennent d'un verrou existant (Barand),
  des versions installées localement et de la lecture du code des dépôts ci-dessus.
- PyTorch, `transformers`, `soxr` et `tqdm` n'étaient pas installés : voir le README, « Ce qui a été testé ».
- Les noms d'API de `transformers` (`forward(..., bandwidth=)`, `n_quantizers=`, `num_quantizers=`) ont été
  relus dans le code source de la branche principale, pas exécutés.

## 6. Calcul : faut-il un GPU ?

**Non pour entraîner. Oui, c'est utile (sans être obligatoire) pour fabriquer les données avec les codecs.**
Les chiffres ci-dessous sont des **estimations analytiques** (taille des modèles de la configuration par défaut,
nombre d'opérations) ; **rien n'a été chronométré**, faute de PyTorch dans l'environnement d'écriture.

| Étape | GPU ? | Ce qu'on sait |
|---|---|---|
| M0, M1 | non | scikit-learn ; le fakeprint est une FFT numpy |
| M2 | non requis | 0,50 M de paramètres ; ~34 M d'opérations multiplication-addition par recadrage de 5 s. Phase 1, un codec : ~5 600 segments d'entraînement, soit de l'ordre de la minute par époque sur CPU (estimation) |
| M3 | non requis | **5 761 paramètres** (un seul filtre) ; ~33 M d'opérations par échantillon, 3 échantillons par segment (version propre + 2 accélérations) : quelques minutes par époque sur CPU (estimation). À mesurer : une convolution 1D à noyau de 5 761 points peut être plus lente sur CPU que ce calcul ne le suggère. Le coût principal est l'extraction (rééchantillonnage + fakeprint des variantes), sur CPU, parallélisable par `--jobs` |
| Codecs (`build-fma`) | **utile** | goulot d'étranglement : la phase 1 représente ~67 h d'audio à ré-encoder (tableau 4 de la proposition : 2 000 pistes × 4 configurations). DAC est nettement plus gros qu'EnCodec 24 kHz. La commande traite les pistes une par une (PyTorch parallélise à l'intérieur d'un calcul, pas entre pistes) |
| M4 (SONICS) | confortable | inférence seule ; possible sur CPU, plus lente |

**Mesurer avant de décider** (c'est le « point de décision de la semaine 1 » de la proposition) :

```bash
uv run musicdetect manifest  -c configs/smoke.yaml --tag t --set fma.max_tracks=20
time uv run musicdetect build-fma -c configs/smoke.yaml --tag t --set fma.max_tracks=20 --codec encodec_24k_6kbps
# durée totale ≈ (secondes par piste mesurées) × nombre de pistes × nombre de codecs
```

Montage réaliste sans machine GPU : **ré-encoder sur Colab (T4)**, comme le prévoit la proposition (15 à 25 h de T4
estimées), récupérer les segments (~6,4 Go en phase 1 : 2 000 pistes × 2 segments × 320 Ko × (1 réel + 4 codecs)),
puis entraîner M0 à M3 en local sur CPU. `build-fma` reprend là où il s'était arrêté : l'index est écrit toutes les 100 pistes,
une session Colab interrompue perd donc au plus ~100 pistes de calcul (à copier régulièrement sur Drive). L'entraînement de M2 et M3, lui, n'a ni reprise ni fp16 (la proposition
mentionne les deux pour Colab) : inutile tant que les modèles restent aussi petits.
