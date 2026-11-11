# Pré-enregistrement — « Ce que voit un détecteur de musique IA »

**Date de gel :** à compléter au moment du commit (le timestamp Git fait foi).  
**Auteur :** étudiant M2 IA — projet personnel de préparation à la recherche.  
**Règle :** ce fichier est committé et daté **avant tout entraînement sur les données phase 1 (≥ 2 000 pistes)**.
Le smoke test (400 pistes, données synthétiques) a servi à vérifier l'installation ; ses AUC n'informent
pas les prédictions ci-dessous.

---

## 1. Cadre expérimental

**Question :** un détecteur appris uniquement sur de la musique réelle (FMA) contre sa version
ré-encodée par des codecs neuronaux généralise-t-il à des générateurs jamais vus (FakeMusicCaps),
ou apprend-il un raccourci de jeu de données ?

**Données d'entraînement :** FMA-small, split par artiste (train 70 % / val 10 % / test 20 %).  
**Données de test :** FakeMusicCaps — 5 générateurs : MusicGen, Stable Audio Open, AudioLDM2,
MusicLDM, Mustango (27 605 clips de 10 s, 5 521 par générateur).  
**Codecs d'entraînement phase 1 :** `encodec_24k_6kbps`, `encodec_24k_12kbps`.  
**Codecs phase full :** + `dac_44k`.

---

## 2. Familles de décodeurs (H1)

Chaque générateur est classé d'après son décodeur, à confirmer dans les papiers (§6 de la description).

| Générateur | Décodeur supposé | Famille | Lien avec EnCodec/DAC |
|---|---|---|---|
| MusicGen | Décodeur EnCodec (RVQ, suréchantillonnage par conv transposée) | **Codec-RVQ** | **Fort** |
| Stable Audio Open | Décodeur de son propre VAE (convolutions strided, activation Snake) | **VAE-waveform** | Faible |
| AudioLDM2 | VAE sur mel + vocodeur HiFi-GAN | **VAE-mel + vocodeur** | Faible |
| MusicLDM | HiFi-GAN ré-entraîné sur musique | **VAE-mel + vocodeur** | Faible |
| Mustango | Vocodeur (architecture à confirmer dans le papier) | **VAE-mel + vocodeur** | À déterminer |

**Score de similarité famille (à remplir avant d'entraîner) :**
- EnCodec 24 kHz vs MusicGen : **1** (même décodeur)
- EnCodec 24 kHz vs Stable Audio Open : **0** (VAE différent)
- EnCodec 24 kHz vs AudioLDM2 : **0**
- EnCodec 24 kHz vs MusicLDM : **0**
- EnCodec 24 kHz vs Mustango : **0** (à affiner après lecture du papier)

---

## 3. Hypothèses et critères de réfutation

### H1 — Les artefacts sont propres à l'architecture du décodeur

**Énoncé :** un détecteur entraîné sur FMA vs FMA-EnCodec transfère mieux vers MusicGen (décodeur
EnCodec) qu'vers les autres générateurs (décodeurs différents). La corrélation de rang (Spearman)
entre score de similarité de famille et AUC croisée est positive et significative.

**Critère de réfutation :** corrélation de Spearman ρ < 0 ou IC bootstrap à 95 % de ρ contenant 0,
calculé sur les 5 générateurs × 2 codecs d'entraînement.

**Comparaison confirmatoire :** AUC(EnCodec → MusicGen) − AUC(EnCodec → moyenne autres) > 0,
IC bootstrap à 95 % excluant 0.

---

### H2 — Transfert partiel : le détecteur EnCodec repère surtout MusicGen

**Énoncé :** un détecteur entraîné sur FMA vs FMA-EnCodec atteint AUC ≥ 0,75 sur MusicGen et
AUC < 0,65 sur au moins trois des quatre autres générateurs.

**Critère de réfutation :** AUC(EnCodec → MusicGen) < 0,75 OU AUC(EnCodec → autres) ≥ 0,75
sur au moins deux générateurs (IC bootstrap à 95 % excluant 0,5 dans les deux cas).

**Comparaison confirmatoire :** test unilatéral AUC(EnCodec → MusicGen) > 0,75 après correction
de Holm sur les 5 comparaisons (une par générateur).

---

### H3 — Une part de la précision vient de raccourcis de format

**Énoncé :** M0 (descripteurs triviaux après chaîne commune) atteint AUC significativement > 0,5
sur `fmc:<générateur>`, révélant un raccourci. Après application de la chaîne commune, l'AUC de M0
reste > 0,5 (raccourci résiduel, par exemple coupure spectrale du codec) mais inférieure à M1.

**Critère de réfutation :** IC bootstrap à 95 % de M0 contenant 0,5 **et** absence de chute d'AUC
de M1/M2 après correction du raccourci identifié.

**Comparaison confirmatoire :** AUC M0 > 0,5 sur `fmc:musicgen`, IC bootstrap à 95 % excluant 0,5.

---

### H4 — Le spectre linéaire s'effondre après changement de vitesse ; log-fréquence résiste mieux

**Énoncé :** sous attaque vitesse ×0,9 et ×1,1, l'AUC de M1 (spectre linéaire) chute de plus de
0,10 point par rapport à la baseline sans attaque ; l'AUC de M3 (log-fréquence + max-pooling)
chute de moins de 0,05 point. M3 ne résiste **pas** mieux au MP3 (nuance : invariance d'échelle
ne protège pas contre la compression).

**Critère de réfutation :** chute de M1 sous vitesse < 0,10 OU chute de M3 sous vitesse ≥ 0,10,
mesurée sur `fmc:musicgen` (IC bootstrap à 95 % des deux écarts excluant 0).

**Comparaison confirmatoire :** ΔAUC(M1, baseline→vitesse×1,1) apparié vs ΔAUC(M3,
baseline→vitesse×1,1), IC bootstrap à 95 % excluant 0.

---

## 4. Seuils décidés

| Paramètre | Valeur |
|---|---|
| Niveau de confiance des IC bootstrap | 95 % |
| Nombre de rééchantillonnages bootstrap | 1 000 (200 pour le smoke test) |
| Unité de rééchantillonnage | artiste (FMA) / légende (FakeMusicCaps) |
| Seuil "transfert fort" | AUC ≥ 0,75 |
| Seuil "transfert partiel" | AUC ∈ [0,60 ; 0,75[ avec IC excluant 0,5 |
| Seuil "pas de transfert" | IC bootstrap à 95 % de AUC contenant 0,5 |
| Correction multiplicités | Holm sur les 5 comparaisons confirmatoires (une par H) |
| Nombre de graines (M2, M3) | ≥ 3 ; rapporter moyenne ± écart-type séparément des IC bootstrap |

---

## 5. Prédictions chiffrées d'AUC (à remplir avant le premier `train` sur phase 1)

> Ces prédictions sont informées par le smoke test (400 pistes, sans valeur scientifique), la
> littérature (papier Deezer ISMIR 2025, papier de robustesse) et la connaissance des architectures.
> Elles sont volontairement imprécises ; l'intérêt est la direction, pas la valeur exacte.

### Détecteur M1 (fakeprint EnCodec 24 kHz) × 5 générateurs

| Générateur | AUC prédite | Raison |
|---|---|---|
| MusicGen | **0,80 – 0,90** | Décodeur EnCodec : artefacts identiques à l'entraînement |
| Stable Audio Open | **0,55 – 0,65** | VAE différent, peu d'artefacts communs |
| AudioLDM2 | **0,55 – 0,65** | Vocodeur HiFi-GAN, pas d'EnCodec |
| MusicLDM | **0,55 – 0,65** | HiFi-GAN, même famille qu'AudioLDM2 |
| Mustango | **0,55 – 0,70** | Architecture incertaine ; à affiner après lecture papier |

### Détecteur M0 (raccourcis) × MusicGen (représentatif)

| Détecteur | AUC prédite | Raison |
|---|---|---|
| M0 résiduel | **0,60 – 0,70** | Coupure spectrale codec visible malgré la chaîne commune |

### Détecteur M4 (SONICS pré-entraîné) × 5 générateurs

| Générateur | AUC prédite |
|---|---|
| MusicGen | **0,75 – 0,85** |
| Stable Audio Open | **0,65 – 0,75** |
| AudioLDM2 | **0,65 – 0,75** |
| MusicLDM | **0,65 – 0,75** |
| Mustango | **0,60 – 0,70** |

### Attaques sur M1 vs M3 (H4, sur fmc:musicgen)

| Attaque | ΔAUC M1 prédit | ΔAUC M3 prédit |
|---|---|---|
| Vitesse ×0,9 | **−0,15 à −0,25** | **−0,02 à −0,08** |
| Vitesse ×1,1 | **−0,15 à −0,25** | **−0,02 à −0,08** |
| MP3 128 kbps | **−0,05 à −0,15** | **−0,05 à −0,15** (comparable) |

---

## 6. Plan d'analyse

1. **Phase 1 (2 000 pistes, EnCodec 6 et 12 kbps) :** entraîner M0, M1, M2 ; évaluer sur les 5
   générateurs ; remplir la matrice 5×5 ; tester H2 et H3.
2. **Phase full (8 000 pistes, + DAC 44k) :** répéter sur 3 codecs ; tester H1 (corrélation de rang
   sur la matrice étendue).
3. **Robustesse (palier B) :** attaques vitesse + MP3 sur M1 et M2 ; entraîner M3 (3 graines) ;
   tester H4.
4. **Oracle MusicGen :** entraîner sur `encodec_32k_musicgen` ; vérifier AUC supérieure à EnCodec
   24 kHz sur MusicGen (confirmation H1).
5. **Analyse d'erreurs :** 20 segments les plus mal classés par générateur ; faux positifs par genre
   FMA ; empreintes moyennes M1 et filtres M3.
6. **Test d'écoute (palier C) :** 15 auditeurs × 20 extraits (réel ou IA ?) ; protocole décrit
   séparément ; à réaliser si le temps le permet.

---

## 7. Ce qui n'est pas prédit ici

- AUC de M2 et M3 sur chaque générateur (trop dépendantes de l'hyperréglage et du nombre de graines).
- Comportement sous attaque bruit et rééchantillonnage (H4 se concentre sur vitesse et MP3).
- Résultats sur DAC (phase full) : les artefacts DAC ne sont pas documentés aussi précisément
  qu'EnCodec dans la littérature disponible.
