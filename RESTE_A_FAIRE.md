# Ce qu'il reste à faire

État au 8 novembre 2026. Cases à cocher ; l'ordre est celui du chemin critique. Les références `§` renvoient à la
proposition longue (`docs/projet/proposition_projet_longue.pdf`) et au [README](README.md).

**Chemin critique en une ligne :** installer → faire tourner les tests PyTorch → récupérer les données → essai
rapide chronométré → décisions avec l'encadrant → **figer le pré-enregistrement** → lancer la phase 1.
Rien ne doit être entraîné avant le pré-enregistrement daté (§3.2).

---

## A. Installation et vérification (avant tout calcul)

- [ ] `uv lock` puis `uv sync --extra all` ; committer `uv.lock`. Si la résolution échoue à cause de `sonics`
  (`librosa` → `numba` → plafond sur numpy), appliquer le plan B de `docs/DEPENDENCIES.md` § 2.
- [ ] `uv run pytest`. **`tests/test_torch.py` tourne ici pour la première fois** : c'est le premier vrai test de
  `nn_models.py`, M2 et M3. Toute erreur à ce stade est attendue ; elle est à corriger, pas à contourner.
- [ ] `uv run musicdetect selftest` (~40 s) ; `ffmpeg -hide_banner -encoders | grep libmp3lame` et
  `ffmpeg -hide_banner -filters | grep rubberband`.
- [ ] Télécharger FMA : `fma_metadata.zip` (342 Mio) et `fma_small.zip` (7,2 Gio) depuis
  `https://os.unil.cloud.switch.ch/fma/`, vérifier les empreintes SHA-1 données dans le README de FMA
  (`f0df49ff…` pour les métadonnées, `ade154f7…` pour `fma_small`), décompresser dans `data/raw/fma/`.
- [ ] Télécharger FakeMusicCaps (archive Zenodo, lien dans l'article arXiv 2409.10684), **relever sa licence**,
  décompresser dans `data/raw/FakeMusicCaps/`. Lancer `musicdetect fmc-index` et **lire la sortie** : adapter
  `fmc.generators` si les noms de dossier diffèrent ; noter si les clips MusicCaps réels sont dans l'archive
  (sinon le contraste `fmc:<générateur>` mélange génération et format, voir README § 6).
- [ ] **Essai rapide chronométré** (`configs/smoke.yaml`, 20 à 400 pistes, un codec EnCodec). C'est la mesure de la
  semaine 1 de la proposition : secondes par piste et par codec, donc durée de la phase 1 et besoin en GPU
  (`docs/DEPENDENCIES.md` § 6). À vérifier au passage :
  - `<index>.failures.csv` (à côté de l'index) vide ou presque ; part de segments `limited` faible ;
  - `delay_samples` petit et stable pour un même codec (sinon l'alignement est suspect) ;
  - écouter 3 couples réel / faux et regarder leurs spectrogrammes ;
  - **contrôle négatif** : évalué sur `fma:identity_24k` (le « faux » n'a traversé aucun codec), tout détecteur doit
    donner AUC = 0,5, comme dans `selftest` (construire d'abord `build-fma --codec identity_24k`, entraîner sur un vrai
    codec) ; un écart révèle une fuite dans la chaîne ;
  - M0 : sur `fma:<codec>`, un codec peut légitimement changer la coupure spectrale, donc un AUC élevé n'y prouve pas
    une fuite ; le raccourci de format se lit surtout sur `fmc:<générateur>` (H3).

## B. Décisions à prendre avec l'encadrant (§8.3)

- [ ] Seuils : IC à 95 %, « transfert fort » = AUC ≥ 0,90, correction de Holm, au moins 3 graines.
- [ ] Paliers A / B / C : minimum exigé, place de M3 et de H4 si le temps manque.
- [ ] Corpus : 2 000 pistes en phase 1 (`configs/phase1.yaml`), puis 8 000 (`configs/full.yaml`) ; combien de codecs.
- [ ] Calcul : T4 Colab suffisante, ou autre GPU / plus de disque ? (estimation de la proposition : 15 à 25 h de T4,
  à confirmer par la mesure ci-dessus).
- [ ] Format du pré-enregistrement : commit horodaté, document signé ou dépôt privé.
- [ ] Test d'écoute (palier C) : autorisation éthique, recrutement, consentement.
- [ ] Langue et format des rendus ; **licence du dépôt** (aucune n'est choisie ; les dépôts de référence sont en
  CC BY-NC 4.0, voir README § 11).

## C. À rédiger avant de lancer

- [ ] **Fichier de pré-enregistrement daté** (aucun modèle n'est fourni ; son contenu est décrit au §3.3) : les
  quatre hypothèses et leurs critères de réfutation, la répartition des générateurs en familles et leurs scores de
  similarité, la liste des comparaisons confirmatoires (une par hypothèse, corrigées par Holm), les seuils, **vos
  prédictions chiffrées d'AUC par générateur** et le plan d'analyse. À figer et committer **avant** `train`.
- [ ] Compléter les champs `[À compléter : …]` des propositions (nom, encadrant, établissement) et, dans le
  tableau des licences (§7.3), mettre à jour : licence de `thomas0barand/robust-deepfake-detector` =
  **CC BY-NC 4.0** (relevée dans le dépôt, la proposition la disait illisible), licence de l'archive FakeMusicCaps
  (à relever), licence des poids EnCodec (à relever). Recompiler les PDF avec `babel-french`.

## D. Compléter le code : écarts connus entre la proposition et le dépôt

Par ordre de priorité. Le dépôt n'implémente aujourd'hui ni les figures ni plusieurs lignes de la matrice.

- [ ] **Figures de résultats : aucune n'est produite** (`matplotlib` est installé mais inutilisé). Il faut au minimum :
  la matrice de généralisation (AUC ± IC par source d'entraînement × générateur, livrable explicite §8.1), les
  courbes d'AUC / FPR sous attaque, les filtres appris de M1 et M3 (`weights()`, `filter_weights()`, pour H1) et
  une courbe de calibration.
- [ ] **Sources d'entraînement supervisées sur FakeMusicCaps** (un détecteur par générateur, validation croisée sur
  les légendes ; leave-one-generator-out) : §6.2. `step_train` n'entraîne que sur des paires FMA / codec.
- [ ] **Oracle « VAE de Stable Audio Open »** (palier B, ligne H1 de la matrice) : à écrire comme sous-classe de
  `Codec` dans `codecs/neural.py` + une entrée dans `codecs`. Les poids sont peut-être soumis à acceptation de
  licence sur le Hub (à vérifier). L'oracle EnCodec 32 kHz de MusicGen est câblé mais non testé.
- [ ] **Plusieurs graines** (≥ 3 pour M2 et M3, les seuls entraînés par gradient ; M1 est déterministe) :
  `--set m2.seed=1 --tag seed1`, etc. Il manque l'agrégation (moyenne et écart-type entre graines, à rapporter
  *séparément* des IC bootstrap, §3.1).
- [ ] Brancher **Holm** (`eval.bootstrap.holm`) et la **mise à l'échelle par température**
  (`eval.calibration`) : les fonctions existent et sont testées, la CLI ne les appelle pas.
- [ ] **FPR par genre** (risque d'éthique et de décalage de domaine, §7.2-7.3) : `genre_top` est dans
  `manifest_fma.csv` mais pas dans l'index des segments ; jointure sur `track_id`.
- [ ] **Analyse d'erreurs** : lister les 20 segments les plus mal classés à partir de `results/*/scores/*.csv`
  (palier B, écoute des 20 extraits).
- [ ] **Notebook de démonstration** (livrable §8.1).
- [ ] Ablations non câblées : « bande passante » (44,1 kHz avant la chaîne) et MP3 à 44,1 kHz avant la chaîne
  (palier C). Déjà configurables : `audio.silence_policy`, `audio.roundtrip`, `audio.segments_per_track`, débit
  des codecs, `features.*`.
- [ ] Si l'entraînement devait dépasser quelques minutes : reprise de l'entraînement et fp16 (mentionnés pour
  Colab dans la proposition, absents du code).

## E. Expériences (calendrier §6.1)

| Semaine | Palier | Travail | Commandes (voir README § 6) |
|---|---|---|---|
| 1 | A | banc d'essai, M4 sur FMA et FakeMusicCaps, M0 brut, M1, mesure des codecs, pré-enregistrement | `build-fma` (essai), `train --detector m0 --variant raw`, `evaluate --detector m4 …` |
| 2 | A | ré-encoder FMA small (EnCodec 24 kHz, 32 kHz, DAC ; 2 à 3 débits), chaîne commune, contrôle de la chaîne | `manifest`, `build-fma --codec …`, `build-fmc` |
| 3 | A | M1 et M2, matrice de généralisation, H3, condition oracle MusicGen | `train` + `evaluate --test fmc:…` pour chaque (détecteur, codec) |
| 4 | B | attaques sur M1 et M2, M3 et H4, oracle Stable Audio Open, analyse d'erreurs, rapport | `evaluate … --attacks`, `train --detector m3` |
| bonus | C | Mimi, autres oracles, test d'écoute, test hors distribution (SONICS ou AI-OpenBMAT) | `--codec mimi_8cb` |

Ablations à planifier avec `--tag` : `silence_trim`, sans aller-retour de rééchantillonnage, débits de codec.

## F. Rendus

- [ ] Rapport : hypothèses datées, ablations, contrôles de raccourcis, analyse d'erreurs, compte rendu du test
  d'écoute (palier C), **rapport de calcul** (heures de GPU et stockage par expérience, à tenir dès la semaine 1).
- [ ] Un résultat négatif ou une hypothèse réfutée est un résultat valide s'il est mesuré proprement (§8.2).
- [ ] Ne pas redistribuer les reconstructions tant que les licences ne sont pas relevées (§7.3).
- [ ] Hygiène : `git init`, premier commit avec `uv.lock` (`.gitignore` fourni : `data/` et `results/` exclus).

## G. Risques connus (rien de nouveau à décider, juste à garder en tête)

- Le code PyTorch n'a jamais tourné (README § 9) ; le chemin `soxr` non plus (les tests ont utilisé le repli scipy).
- Disposition de l'archive FakeMusicCaps supposée ; recouvrement possible Stable Audio Open / FMA (§7.2).
- Bornes de versions non éprouvées sur PyPI.
- Les AUC de `selftest` n'ont aucune valeur scientifique (données synthétiques).
