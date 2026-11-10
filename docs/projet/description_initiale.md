# Projet audio — « Ce que voit un détecteur de musique IA »

**Dossier de projet + programme d'apprentissage**
Public : étudiant de M2 IA, à l'aise avec CNN, Transformers et PyTorch, **sans cours d'audio** (ni traitement du signal musical, ni VAE/GAN/diffusion audio). Les slides d'AI for Sound (G. Peeters) ne sont pas disponibles ; on les remplace par des cours ouverts (voir §4).

> Ce fichier est écrit pour **amorcer une nouvelle session de travail** dédiée à ce projet. Il est autonome : la session n'a pas besoin de l'historique de la conversation d'origine. Le mode d'emploi pour cette session est au §12.

---

## 0. Mode d'emploi et légende

### 0.1 Légende des liens
Tous les liens sont donnés en URL complète. Chaque lien porte une étiquette :

| Étiquette | Sens |
|---|---|
| **[V]** | Vérifié : page ouverte pendant la préparation, ou lien présent dans les listes que l'étudiant a explicitement autorisées, avec un titre cohérent. |
| **[R]** | Vu seulement dans un résultat de recherche (titre et auteurs cohérents), page non ouverte. |
| **[M]** | Non vérifié (mémoire ou déduction). À tester avant d'en dépendre. |

Les arXiv IDs sont à jour à la date de préparation (novembre 2026). Un lien qui ne marche pas doit être signalé, jamais remplacé par une adresse inventée.

### 0.2 Ce que ce fichier ne contient pas
- Les cours eux-mêmes : la nouvelle session les écrit, notion par notion (§12).
- Les slides de Peeters : **introuvables en ligne** (seules des pages de séminaires existent). Ne pas en chercher de substitut « équivalent » sous ce nom.
- Les chiffres des papiers récents (2025–2026) lus seulement par résumé : ils sont marqués comme tels. Les relire dans le papier avant de les citer dans un rapport.

### 0.3 Niveaux de maîtrise utilisés
- **Comprendre** : expliquer à l'oral, sans notes, à un camarade.
- **Savoir calculer/dériver** : refaire la formule ou le calcul sur papier.
- **Implémenter** : écrire le code minimal qui marche (NumPy/PyTorch).
- **Survoler** : savoir où c'est, à quoi ça sert, ne pas le refaire.

---

## 1. Le projet en bref

### 1.1 Question
Un détecteur appris uniquement sur de la **musique réelle** contre sa **version ré-encodée par des codecs neuronaux** généralise-t-il à des générateurs qu'il n'a jamais vus, ou apprend-il un **raccourci de jeu de données** ? Et que devient-il quand on change la vitesse ou qu'on compresse en MP3 ?

Le sujet est proche du travail de Deezer sur la détection de musique IA et se traite avec peu de GPU.

### 1.2 Principe expérimental (en une phrase)
Prendre de la vraie musique (FMA), fabriquer des « faux » **sans générateur de musique** en la faisant passer par un codec neuronal (EnCodec, DAC, Mimi), entraîner des détecteurs réel/faux, puis les tester sur de vrais faux générateurs (FakeMusicCaps : 5 modèles) et sous attaques (vitesse, hauteur, MP3, rééchantillonnage), en mesurant en parallèle ce que valent les raccourcis de métadonnées.

### 1.3 Faits vérifiés sur les sources (état à la préparation)

| Sujet | Fait | Source |
|---|---|---|
| Explication de Fourier (ISMIR 2025) | Les modules de déconvolution des générateurs laissent des **pics spectraux systématiques**, liés à l'architecture plutôt qu'aux données ou aux poids. Critère de détection simple et interprétable, plus de 99 % de précision dans plusieurs scénarios ; tests sur modèles ouverts et sur Suno et Udio. | https://arxiv.org/abs/2506.19108 **[V]** |
| Dépôt associé | `compute_fakeprints.py`, `train_test_regressor.py` ; utilise FMA et SONICS ; **CC BY-NC 4.0** ; mentionne **deux demandes de brevet** (déc. 2024). | https://github.com/deezer/ismir25-ai-music-detector **[V]** |
| Dépôt deepfake-detector | Entraîne un détecteur réel vs reconstruction par auto-encodeurs ; cite les codecs Musika et LAC ; CC-BY-NC-4.0 ; aucune instruction d'entraînement ; pas l'outil de production de Deezer. | https://github.com/deezer/deepfake-detector **[V]** |
| Robustesse (juillet 2026) | Les détecteurs actuels s'effondrent après changement de vitesse ou de hauteur. La méthode remappe le spectre sur un axe **logarithmique**, applique **un seul filtre de corrélation croisée appris + max-pooling** : l'invariance au facteur d'échelle devient structurelle et le facteur de vitesse est estimable. Checkpoints fournis ; licence illisible depuis la page. | https://arxiv.org/abs/2607.27454 **[V]** ; https://github.com/thomas0barand/robust-deepfake-detector **[V]** |
| FakeMusicCaps | 27 605 extraits de 10 s (~77 h), 12,9 Go sur Zenodo, générés depuis les légendes de MusicCaps par **MusicGen, MusicLDM, AudioLDM2, Stable Audio Open, Mustango**. | https://arxiv.org/abs/2409.10684 **[V]** ; https://zenodo.org/record/13732524/latest **[V]** |
| FMA | `fma_small` : 8 000 pistes de 30 s en **MP3** (7,2 Go), 8 genres, artiste et genre dans `tracks.csv`. Audio sous la licence choisie par chaque artiste ; usage recherche. | https://github.com/mdeff/fma **[V]** |
| SONICS | ~97 000 titres dont >49 000 générés par Suno/Udio. Le dépôt ne fournit que les titres synthétiques ; les vrais sont à télécharger par identifiant YouTube. Entraînement de référence : 150 Go de disque, 48 Go de GPU → **hors budget** ; on n'utilise que les **checkpoints pré-entraînés** (SpecTTTra, MIT) et, en extension, ses faux titres comme test hors distribution. | https://arxiv.org/abs/2408.14080 **[V]** ; https://github.com/awsaf49/sonics **[V]** |
| AI-OpenBMAT | 3 294 extraits d'une minute (54,9 h) imitant la télévision, avec continuations Suno v3.5 ; les détecteurs existants passent sous 60 % de F1 quand la musique est en arrière-plan ou courte. **Accès et licence non précisés** : à vérifier. | https://arxiv.org/abs/2602.06823 **[V]** |
| MusicGen | Utilise **EnCodec** comme tokenizer audio (RVQ), un LM Transformer monoétage et des schémas d'entrelacement de codebooks ; 32 kHz. (Confirme l'hypothèse « MusicGen s'appuie sur EnCodec ».) | https://arxiv.org/abs/2306.05284 **[R]** |

### 1.4 Hypothèses à **écrire avant** d'entraîner (pré-enregistrement)

| # | Hypothèse | Ce qui la réfute |
|---|---|---|
| **H1** | Les artefacts sont **propres à l'architecture** : un détecteur appris sur un générateur transfère mieux vers des générateurs dont le **décodeur lui ressemble** que vers les autres. On documente le décodeur de chacun des 5 modèles à partir de leurs papiers (§6) et on relie la similarité à l'AUC croisée. | Pas de lien monotone entre similarité de décodeur et AUC croisée (corrélation de rang nulle ou négative sur la matrice). |
| **H2** | Un détecteur appris sur **FMA vs FMA ré-encodé par EnCodec ou DAC** détecte *une partie* de FakeMusicCaps, surtout les modèles passant par un codec proche, et peu les autres. | AUC proche de 0,5 sur tous les générateurs, ou AUC aussi élevée sur les modèles sans codec proche que sur MusicGen. |
| **H3** | Une part de la précision « naïve » vient de **raccourcis** : format (MP3 vs WAV), fréquence d'échantillonnage native, bande passante, volume, durée. Un classifieur qui ne voit que ces métadonnées (M0) atteint déjà une AUC significative ; corriger des raccourcis la fait chuter. | L'IC bootstrap de l'AUC de M0 contient 0,5 *et* la correction des raccourcis ne change pas l'AUC des autres détecteurs. |
| **H4** | Les détecteurs à **spectre linéaire** s'effondrent après changement de vitesse ; la représentation **log-fréquence + max-pooling** y résiste mieux, mais pas nécessairement à la compression MP3 ni au bruit. | M3 ne résiste pas mieux que M1/M2 à la vitesse, ou M3 résiste aussi bien au MP3 (ce qui contredirait la nuance). |

*Note : la colonne « Ce qui la réfute » est une **proposition de ce dossier** ; la version d'origine du projet donne les quatre hypothèses sans condition de réfutation chiffrée. À valider et chiffrer en semaine 1.*

À faire en semaine 1 : **chiffrer** « significatif » et « mieux » (par exemple : « AUC de M0 dont l'intervalle de confiance bootstrap à 95 % exclut 0,5 » ; « écart d'AUC supérieur à l'IC »), puis dater et figer le texte avant tout entraînement.

Les hypothèses prévisibles sont acceptables : la valeur est dans la **mesure propre**, l'**ablation** et l'**analyse d'erreurs**.

### 1.5 Détecteurs à comparer

| Modèle | Entrée | Rôle |
|---|---|---|
| **M0** | Métadonnées seules (format, fréquence d'échantillonnage, durée, volume, bande passante) | Mesure du raccourci (contrôle négatif) |
| **M1** | Empreinte spectrale moyenne (« fakeprint ») + classifieur linéaire | Reproduction de l'article de Fourier |
| **M2** | Petit CNN (< 1 M de paramètres) sur spectrogramme | Baseline apprise |
| **M3** | Axe log-fréquence, un filtre de corrélation croisée, max-pooling | Mini-réimplémentation de l'article de robustesse |
| **M4** | SpecTTTra pré-entraîné de SONICS | Référence externe, **inférence seule** |

### 1.6 Données et chaîne de traitement
1. **Réel** : FMA-small (MP3, 30 s). Découpage **train/val/test par artiste** (jamais le même artiste des deux côtés).
2. **Faux de substitution** : ré-encoder FMA-small avec EnCodec, DAC et Mimi, à 2–3 débits. Plan B si trop lourd : 2 000 pistes au lieu de 8 000.
3. **Faux de test** : FakeMusicCaps (5 générateurs).
4. **Chaîne de normalisation commune** appliquée à *tout* (même rééchantillonnage, même passage MP3 si besoin, même normalisation de volume, même durée de segment), pour que M0 ne puisse pas « tricher » et pour prouver l'effet quand on la retire.

### 1.7 Protocole en 4 semaines

| Semaine | Travail |
|---|---|
| **1 — banc d'essai** | Relever format, fréquence d'échantillonnage, durée de chaque jeu ; lancer M4 sur FMA et FakeMusicCaps ; reproduire M1 ; écrire la proposition et les prédictions. |
| **2 — données** | Ré-encoder FMA-small (EnCodec, DAC, Mimi, 2–3 débits) ; split par artiste ; chaîne de normalisation commune. |
| **3 — expériences centrales** | Entraîner M1 et M2 ; matrice de généralisation 5×5 entre générateurs ; contrôles de raccourci (H3). |
| **4 — robustesse et restitution** | Attaques (vitesse, hauteur, MP3, rééchantillonnage) ; M3 ; analyse d'erreurs ; test d'écoute ; rapport et soutenance. |

### 1.8 Ablations (au moins deux axes, un seul facteur varié à la fois)

| Axe | Valeurs | Hypothèse testée |
|---|---|---|
| Faux d'entraînement | EnCodec seul, DAC seul, les deux, un générateur de FakeMusicCaps | H1, H2 |
| Représentation | spectre linéaire / axe log-fréquence | H4 |
| Durée du segment | 5, 10, 30 s | H3, H4 |
| Pics spectraux | bandes retirées / conservées | H1 (rôle causal) |
| Augmentations d'entraînement | aucune, vitesse, MP3 | H4 |

### 1.9 Analyse d'erreurs et évaluation
- Matrice de confusion **par générateur**.
- Faux positifs sur le réel **par genre FMA** (hypothèse à tester : la musique électronique tonale déclenche des pics spectraux naturels).
- Dérive du seuil de décision sous attaque.
- Empreintes moyennes par classe.
- Écoute de **20 extraits mal classés**.
- Métriques : **AUC, EER, taux de vrais positifs à 1 % de faux positifs**.
- Test d'écoute : **15 auditeurs × 20 extraits** (réel ou IA ?), comparaison avec le détecteur, et vérification que la difficulté humaine suit la confiance du détecteur.
- Rigueur : intervalles de confiance bootstrap **et** variance entre seeds (deux choses différentes, voir N9).

### 1.10 Extensions, budget, risques, licences
- **Extensions (niveau 3)** : test hors distribution sur faux titres SONICS (Suno, Udio) ou sur AI-OpenBMAT si l'accès est accordé ; extraits courts de 5 s ; estimation du facteur de vitesse comme dans l'article de robustesse.
- **Budget** : ≈ 15–25 h de GPU T4 (ré-encodage, petits modèles) + CPU pour les attaques. *Estimation personnelle à valider en semaine 1.* Colab ne garantit pas le GPU attribué ; checkpoint sur Drive toutes les 15 min ; scripts reprenables.
- **Risque principal — fuite de format** : FMA est en MP3 ; le format de FakeMusicCaps est à vérifier. Si les formats diffèrent, tout passer par la même chaîne MP3 + même rééchantillonnage avant d'entraîner ; M0 sert à le prouver.
- **Risque — décalage de genres** : FakeMusicCaps ne fournit pas forcément de contreparties réelles ; le réel vient de FMA ; à discuter explicitement.
- **Licences** : les deux dépôts Deezer sont CC BY-NC ; celui d'ISMIR 2025 mentionne deux demandes de brevet. **Réimplémenter depuis l'article, usage pédagogique uniquement, pas de réutilisation commerciale.**

### 1.11 Livrables et ligne de CV
Livrables : dépôt public (code propre, README), notebook, figure de la matrice de généralisation, rapport avec hypothèses datées, ablations, analyse d'erreurs, compte rendu du test d'écoute, rapport de calcul (heures de GPU par expérience).

Ligne de CV à adapter : « AI-music detection: reimplemented a spectral-fingerprint detector, built codec-resynthesis training data (EnCodec, DAC, Mimi), measured cross-generator generalization on 5 open text-to-music models, quantified dataset shortcuts with a metadata-only baseline, and evaluated robustness to speed, pitch and MP3 attacks. »

---

## 2. Prérequis supposés et auto-diagnostic

On suppose acquis (M2 IA) : CNN, Transformers, rétropropagation, PyTorch, classification binaire, courbes ROC de base, NumPy/pandas. Test en 10 questions (à faire sans notes ; ce qui coince indique ce qu'il faut revoir) :

1. Que calcule une convolution 2D ? Quel est le champ réceptif après 3 couches 3×3 ?
2. Que fait un max-pooling et à quelle transformation donne-t-il une invariance (approximative) ?
3. Pourquoi une convolution transposée peut-elle produire des motifs périodiques ?
4. Définis AUC, TPR, FPR. Que signifie « AUC = 0,5 » ?
5. Écris la perte d'un GAN standard.
6. Pourquoi sépare-t-on train/test « par groupe » (ici artiste) et pas au hasard ?
7. Qu'est-ce qu'un intervalle de confiance bootstrap ? Que ne dit-il pas ?
8. Que signifie « classifieur calibré » ?
9. Qu'est-ce qu'un auto-encodeur ? Qu'ajoute un VAE ?
10. Qu'est-ce qu'une quantification vectorielle (K centroïdes) ?

Si > 3 réponses vacillent sur les points 1–2, 5, 9–10 : revoir d'abord CS231n (http://cs231n.stanford.edu **[V]**) ou les notes CS236 sur VAE/GAN (https://deepgenerativemodels.github.io/notes/index.html **[V]**).

---

## 3. Carte des notions à maîtriser

Chaque notion a un **identifiant** (N1…N24) que la nouvelle session utilise pour écrire un cours dédié. « Pourquoi » relie la notion à une hypothèse ou une ablation.

### A. Signal audio

| ID | Notion | Pourquoi (lien projet) | Niveau | Ressources principales (détails §4) |
|---|---|---|---|---|
| **N1** | Échantillonnage, Nyquist, repliement (aliasing), quantification | Sample rate natif = raccourci (H3) ; réplicas lors du rééchantillonnage | Comprendre + calculer | Velardo ; MVA index §2 ; HF audio course unité 1 |
| **N2** | DFT/FFT, fenêtrage, STFT, compromis temps-fréquence | Base de tous les détecteurs | Comprendre + implémenter | MVA index §2 ; FMP ; JOS |
| **N3** | Spectrogramme (magnitude/phase), dB, échelle **mel**, axe **log-fréquence**, **CQT** | Choix de représentation (ablation « linéaire vs log ») ; M3 | Comprendre + implémenter | MVA index §2–3 ; FMP ; Müller et al. 2022 |
| **N4** | Rééchantillonnage ; **changement de vitesse** vs pitch-shift vs time-stretch ; effet sur le spectre | Attaques de H4 ; pourquoi un décalage de vitesse = **mise à l'échelle multiplicative** de l'axe fréquence | Comprendre + dériver | Lab L3 ; papier de robustesse |
| **N5** | **Compression MP3** : MDCT, masquage psychoacoustique, coupure passe-bas, débit | FMA en MP3 → raccourci possible ; attaque MP3 | Comprendre (haut niveau) + mesurer | Sources web [R] (§8) ; Lab L2 |

### B. Apprentissage et évaluation pour l'audio

| ID | Notion | Pourquoi | Niveau | Ressources |
|---|---|---|---|---|
| **N6** | **CNN 2D sur spectrogramme** : champ réceptif, pooling, invariances ; choix de représentation d'entrée | M2 ; comprendre pourquoi la translation en fréquence ≠ translation en log-fréquence | Comprendre + implémenter | MVA index §7–8 ; Choi tutorial |
| **N7** | **Transformer sur patches de spectrogramme** (AST) ; SpecTTTra en inférence | M4 (référence externe) | Survoler / comprendre | Papier AST ; dépôt SONICS |
| **N8** | **Métriques de détection** : ROC, AUC, EER, TPR@1 % FPR, seuil ; **calibration** (ECE, temperature scaling) | Toutes les mesures du projet | Implémenter (de zéro) | Guo et al. ; Lab L7 |
| **N9** | **Incertitude** : bootstrap vs variance entre seeds vs variabilité des données ; IC | Rapporter honnêtement des écarts d'AUC | Comprendre + implémenter | Lab L7 |

### C. Génération audio neuronale (pour comprendre d'où viennent les artefacts)

| ID | Notion | Pourquoi | Niveau | Ressources |
|---|---|---|---|---|
| **N10** | **Auto-encodeur**, **VAE** (ELBO, reparamétrisation) | Stable Audio Open (VAE), AudioLDM/MusicLDM (VAE sur mel) | Comprendre | CS236 notes VAE ; Weng VAE |
| **N11** | **GAN** : pertes adversariales, discriminateurs multi-échelles | Codecs (EnCodec, DAC) et vocodeurs (HiFi-GAN) sont entraînés ainsi | Comprendre | CS236 notes GAN ; papiers codecs |
| **N12** | **Couches de suréchantillonnage** : conv transposée, interpolation + conv, sous-pixel ; **artefacts tonaux** et **réplicas spectraux** | Cœur de H1 : les « pics spectraux » viennent d'ici | Comprendre + reproduire (Lab L4) | Pons et al. ; Odena et al. (Distill) |
| **N13** | **Quantification vectorielle**, VQ-VAE, **RVQ** (quantification vectorielle résiduelle) | Codecs neuronaux | Comprendre + dériver | VQ-VAE ; blog AssemblyAI RVQ ; SoundStream |
| **N14** | **Codecs audio neuronaux** : SoundStream, EnCodec, DAC, Mimi ; débit ↔ nombre de codebooks | Fabrication des faux de substitution | Comprendre + utiliser | Papiers ; Lab L5 |
| **N15** | **LM sur jetons de codec** (MusicGen) ; entrelacement des codebooks | MusicGen = générateur de test lié à EnCodec | Comprendre | MusicGen |
| **N16** | **Diffusion latente audio** (Stable Audio Open, AudioLDM2, MusicLDM, Mustango) ; **vocodeur** (HiFi-GAN) | 4 des 5 générateurs de test | Survoler + documenter le décodeur | Papiers (§6) ; Weng diffusion |

### D. Détection de musique IA

| ID | Notion | Pourquoi | Niveau | Ressources |
|---|---|---|---|---|
| **N17** | **Empreintes spectrales (fakeprints)** et explication de Fourier des artefacts | M1 ; H1 | Comprendre + implémenter | https://arxiv.org/abs/2506.19108 |
| **N18** | **Généralisation hors distribution** : matrice croisée détecteur × générateur ; leave-one-generator-out | H1, H2 | Comprendre + implémenter | Afchar ; FakeMusicCaps ; Müller 2022 |
| **N19** | **Raccourcis** (shortcut learning) et **fuites** : format, sr, durée, volume, silence ; split par artiste | H3 | Comprendre + détecter | Geirhos et al. ; Müller (silence) |
| **N20** | **Baseline métadonnées (M0)** comme contrôle négatif | H3 | Implémenter | Lab L6 |
| **N21** | **Robustesse** aux attaques ; invariance d'échelle : log-fréquence + corrélation croisée + max-pooling | H4 ; M3 | Comprendre + implémenter (mini) | Papier de robustesse |
| **N22** | **Interprétabilité** et analyse d'erreurs ; test d'écoute | Restitution | Implémenter | Afchar ; Lab L8 |

### E. Méthode

| ID | Notion | Pourquoi | Niveau | Ressources |
|---|---|---|---|---|
| **N23** | Hypothèse falsifiable, ablation (un facteur à la fois), contrôle positif/négatif, pré-enregistrement | Cadre du projet | Appliquer | §1.4 ; Geirhos et al. |
| **N24** | Licences (CC BY-NC, brevets), usage pédagogique ; éthique de la détection | §1.10 | Comprendre | Dépôts Deezer |

---

## 4. Programme d'apprentissage

### 4.1 Ressources de référence

| Ressource | Ce qu'on y prend | Lien |
|---|---|---|
| **MVA — G. Richard, R. Badeau** (Télécom Paris) | Slides « Audio indexing » : 10 sections — 1 machine listening, **2 traitement du signal**, **3 hauteur/harmonie/CQT**, 4 empreintes audio, 5 rythme, **6 timbre/MFCC**, **7 deep learning pour l'audio** (CNN, représentations d'entrée, convolutions dilatées/strides, RNN), **8 exemples DL** (CREPE, U-Nets multipitch, auto-encodeurs/GAN/attention/Transformers), 9 DCASE, 10 synthèse. *Pas de contenu auto-supervisé ni d'évaluation.* | Dossier : https://perso.telecom-paristech.fr/grichard/Enseignements/MVA/ **[V]** ; slides : https://perso.telecom-paristech.fr/grichard/Enseignements/MVA/2026-MVA_indexation-english.pdf **[V]** |
| **Stanford CS224S** (Spoken Language Processing) | Utile pour la **parole** : vocodeurs, TTS (WaveNet, Tacotron, FastSpeech), wav2vec 2.0, HuBERT. Le syllabus ne donne pas de liens de slides ni de lectures en URL. Option, pas central. | https://web.stanford.edu/class/cs224s/index.html **[V]** ; syllabus : https://web.stanford.edu/class/cs224s/semesters/2024-spring/syllabus **[V]** ; slides cours 1 : https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec1.pdf **[V]** |
| **Hugging Face Audio Course** | Unités 0–6. Prendre l'unité sur les **données audio** (échantillonnage, spectrogrammes), puis celle sur les **architectures Transformers pour l'audio** et celle du **classifieur de genre**. Titres d'unités de mémoire **[M]** : vérifier sur la page. | https://huggingface.co/learn/audio-course **[V]** ; https://huggingface.co/learn/audio-course/chapter3 **[V]** |
| **Fundamentals of Music Processing (M. Müller)** | Référence « classique » du domaine : représentations, Fourier, STFT, etc. Chapitres 1–2 (représentation, Fourier) **[M pour le détail]**. | https://www.audiolabs-erlangen.de/FMP **[V]** |
| **Spectral Audio Signal Processing (J.O. Smith)** | Référence de fond sur DFT/STFT/fenêtrage, à consulter à la demande. | https://ccrma.stanford.edu/~jos/sasp/ **[V]** |
| **V. Velardo — The Sound of AI** | Vidéos très pédagogiques : « Audio Signal Processing for ML » (échantillonnage, Fourier, STFT, mel) et « PyTorch for Audio ». Épisodes à repérer par titre. | https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 **[R]** ; https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm **[R]** |
| **K. Choi et al. — DL pour la MIR (tutoriel)** | Lecture de référence sur les représentations d'entrée et les CNN pour la musique. | https://arxiv.org/abs/1709.04396 **[V]** ; code : https://github.com/keunwoochoi/dl4mir **[V]** |
| **Stanford CS236 — notes** | VAE, GAN, modèles de diffusion (rappels pour C). | https://deepgenerativemodels.github.io **[V]** ; notes : https://deepgenerativemodels.github.io/notes/index.html **[V]** |
| **L. Weng — blog** | VAE et diffusion, très lisibles. | https://lilianweng.github.io/posts/2018-08-12-vae/ **[V]** ; https://lilianweng.github.io/posts/2021-07-11-diffusion-models **[V]** |
| **AssemblyAI — RVQ** | Explication courte de la quantification vectorielle résiduelle. | https://assemblyai.com/blog/what-is-residual-vector-quantization **[V]** |

### 4.2 Deux parcours
- **Parcours minimal ≈ 25–28 h** : uniquement les éléments marqués **★** ci-dessous.
- **Parcours approfondi ≈ 40–45 h** : avec les éléments **☆**.

Ces durées sont **mes estimations**, non mesurées. À ajuster dès la semaine 1. L'apprentissage s'entrelace avec les semaines du projet : on n'apprend pas tout avant de commencer, on apprend ce qui est utile pour la semaine suivante.

### 4.3 Planning

#### Phase 0 — Socle signal (≈ 10–12 h, avant la semaine 1 du projet)
Objectif : savoir expliquer un spectrogramme et ce qui se passe quand on rééchantillonne.
- ★ **Velardo** : épisodes sur l'échantillonnage/Nyquist, la transformée de Fourier, la STFT, le spectrogramme et le mel (≈ 3 h). **[R]**
- ★ **MVA index §2** (traitement du signal) (≈ 2 h). **[V]**
- ★ **HF Audio Course**, unité sur les données audio (≈ 2 h). **[V pour le cours, M pour le titre de l'unité]**
- ★ **Labs L1 et L3** (≈ 3 h) — voir §7.
- ☆ FMP, chapitre sur Fourier (≈ 2–3 h) ; JOS à la demande.
- Notions : N1, N2, N3 (début), N4 (début).
- Checkpoint : voir §10, bloc P0.

#### Semaine 1 (projet : banc d'essai, M1, prédictions) — apprentissage ≈ 8–10 h
Objectif : comprendre **d'où viennent les pics spectraux** et reproduire un fakeprint.
- ★ **Papier** : Fourier explanation (https://arxiv.org/abs/2506.19108 **[V]**) — lire 2 fois ; première lecture : résumé, figures, conclusion ; seconde : méthode.
- ★ **Papier** : Pons et al., *Upsampling artifacts in neural audio synthesis* (https://arxiv.org/abs/2010.14356 **[R]**) ; code : https://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio **[R]**.
- ★ **Article Distill** (Odena et al.) : https://distill.pub/2016/deconv-checkerboard/ **[R]** (lecture rapide, 30 min).
- ★ **MVA index §3** (CQT, hauteur/harmonie) (≈ 1,5 h) **[V]**.
- ★ **Papier** : Müller et al. 2022, *Does Audio Deepfake Detection Generalize?* (https://arxiv.org/abs/2203.16263 **[R]**) — pour le **cqtspec/logspec vs melspec** (selon son résumé : environ 37 % d'EER en moins en moyenne, toutes choses égales) et le jeu « in the wild ».
- ★ **N5 (MP3)** : un article de vulgarisation + **Lab L2** (§8 pour les sources).
- ★ **Labs L2 et L4**.
- ☆ Papier : Afchar et al. 2024 (https://arxiv.org/abs/2405.04181 **[V]**) — première lecture seulement.
- Notions : N3, N4, N5, N12, N17.
- Livrable d'apprentissage : une page « Pourquoi une convolution transposée laisse des pics » avec la figure de L4.

#### Semaine 2 (projet : ré-encodage par codecs) — apprentissage ≈ 8–10 h
Objectif : savoir ce qu'est un codec neuronal et ce qui le distingue d'un générateur.
- ★ **Auto-encodeur/VAE/GAN** : CS236 notes (sections VAE et GAN) **[V]** + Weng VAE **[V]** (≈ 3 h ; sauter si acquis).
- ★ **RVQ** : blog AssemblyAI **[V]** (30 min), puis **SoundStream** (https://arxiv.org/abs/2107.03312 **[V]**) : lecture partielle (architecture, RVQ, pertes).
- ★ **EnCodec** (https://arxiv.org/abs/2210.13438 **[V]**) : lecture complète ; dépôt https://github.com/facebookresearch/encodec **[V]** (cité dans tes listes).
- ★ **DAC** (https://arxiv.org/abs/2306.06546 **[V]**) : résumé, méthode, ablations ; dépôt https://github.com/descriptinc/descript-audio-codec **[V]**.
- ★ **Lab L5** (ré-encoder un extrait avec EnCodec, comparer les spectres moyens).
- ☆ **Mimi** : décrit dans l'article Moshi (Kyutai) — https://arxiv.org/pdf/2410.00037v1 **[V]** (identité de l'article à confirmer à l'ouverture) ; surveys codecs : https://arxiv.org/abs/2402.13071 , https://arxiv.org/abs/2402.13236 , https://arxiv.org/abs/2406.00146 **[V]** (titres non relus).
- ☆ **MusicGen** (https://arxiv.org/abs/2306.05284 **[R]**) : sections sur la tokenisation et l'entrelacement.
- Notions : N10, N11, N13, N14, N15.
- Livrable d'apprentissage : tableau « codec × débit × nombre de codebooks × fréquence d'échantillonnage » rempli à partir des papiers.

#### Semaine 3 (projet : M1/M2, matrice 5×5, contrôles de raccourci) — apprentissage ≈ 8–10 h
Objectif : savoir **détecter un raccourci** et **mesurer sans se mentir**.
- ★ **MVA index §7–8** (DL pour l'audio, exemples) (≈ 2 h) **[V]**.
- ★ **Choi et al.** tutoriel (sections sur les représentations d'entrée et les CNN) **[V]**.
- ★ **Geirhos et al., *Shortcut Learning in Deep Neural Networks*** (https://arxiv.org/abs/2004.07780 **[V]**) : sections d'introduction et d'exemples ; retenir la grille de lecture « stratégie décisionnelle ≠ intention ».
- ★ **Müller et al., silence** (https://arxiv.org/abs/2106.12914 **[R]**) : d'après son résumé, un modèle qui ne voit que la durée du silence initial atteint environ 85 % de précision (EER 15,1 %), et tailler le silence dégrade des modèles établis (EER de ~3,6 % à ~15,5 % pour RawNet2). **C'est le précédent exact de H3/M0.**
- ★ **Afchar et al. 2024** (https://arxiv.org/abs/2405.04181 **[V]**) : lecture complète ; sujets à extraire : calibration, interprétabilité, robustesse (titre exact à confirmer à l'ouverture).
- ★ **FakeMusicCaps** (https://arxiv.org/abs/2409.10684 **[V]**).
- ★ **Guo et al., calibration** (https://arxiv.org/abs/1706.04599 **[R]**) : définition de l'ECE, temperature scaling.
- ★ **Labs L6 et L7**.
- ☆ **Papiers des 5 générateurs** — lecture ciblée « décodeur / vocodeur » (§6).
- Notions : N6, N7, N8, N9, N16, N18, N19, N20.
- Livrable d'apprentissage : tableau §6 rempli.

#### Semaine 4 (projet : robustesse, M3, erreurs, test d'écoute) — apprentissage ≈ 6–8 h
Objectif : comprendre **pourquoi la log-fréquence rend l'échelle invariante**, et conclure proprement.
- ★ **Papier de robustesse** (https://arxiv.org/abs/2607.27454 **[V]**) : lecture complète ; dépôt https://github.com/thomas0barand/robust-deepfake-detector **[V]**.
- ★ **Lab L3 (relu)** + dérivation sur papier : si f' = a·f (changement de vitesse), alors log f' = log f + log a (translation) → une corrélation croisée suivie d'un max-pooling est invariante à la translation.
- ★ **SONICS** (https://arxiv.org/abs/2408.14080 **[V]**) : lecture ciblée (architecture SpecTTTra, cadre d'évaluation).
- ★ **Lab L8** (protocole de test d'écoute).
- ☆ **AST** (https://arxiv.org/abs/2104.01778 **[V]**) ; **AI-OpenBMAT** (https://arxiv.org/abs/2602.06823 **[V]**) pour l'extension ; **Musika** (https://arxiv.org/abs/2208.08706 **[R]**) comme exemple de codec/GAN cité par le dépôt Deezer.
- Notions : N21, N22, N23, N24.
- Livrable d'apprentissage : une page « Ce que j'aurais prédit / ce qui est arrivé ».

---

## 5. Papiers : ordre de lecture

Légende : **P1** = à lire intégralement ; **P2** = lecture ciblée ; **P3** = culture / extension. « Quoi chercher » = les questions à garder en tête pendant la lecture. Pour quelqu'un qui n'a pas eu de cours : **lire dans cet ordre**, une première passe courte (résumé, figures, conclusion) avant la lecture complète.

| # | Priorité | Papier | Lien | Quoi chercher |
|---|---|---|---|---|
| 1 | P1 | *A Fourier Explanation of AI-music Artifacts* (ISMIR 2025) | https://arxiv.org/abs/2506.19108 **[V]** | Quelle est la **cause** des pics ? Quel est le **critère de détection** ? Que montrent les tests sur Suno/Udio ? Quelles limites l'article avoue-t-il ? |
| 2 | P1 | Müller et al. — *Does Audio Deepfake Detection Generalize?* | https://arxiv.org/abs/2203.16263 **[R]** | Quelle représentation gagne (cqt/log vs mel) ? Écart entre jeu de benchmark et jeu « in the wild » ? |
| 3 | P1 | Afchar et al. (Deezer), 2024 | https://arxiv.org/abs/2405.04181 **[V]** | Comment mesurent-ils calibration et robustesse ? Que révèle l'interprétabilité ? |
| 4 | P1 | FakeMusicCaps | https://arxiv.org/abs/2409.10684 **[V]** | Comment les 5 modèles sont-ils échantillonnés ? Quel format audio ? Quels détecteurs de base ? |
| 5 | P1 | EnCodec | https://arxiv.org/abs/2210.13438 **[V]** | Architecture encodeur/décodeur, RVQ, pertes (reconstruction + adversariale), débits. |
| 6 | P1 | *Improved Robustness in AI-Generated Music Detection* (juillet 2026) | https://arxiv.org/abs/2607.27454 **[V]** | Pourquoi les détecteurs actuels échouent sous vitesse ? Comment la log-fréquence + corrélation + max-pool règle-t-elle le problème ? Résiste-t-elle au MP3 ? |
| 7 | P1 | Geirhos et al. — *Shortcut Learning in DNNs* | https://arxiv.org/abs/2004.07780 **[V]** | Comment distinguer une stratégie de décision intentionnelle d'un raccourci ? |
| 8 | P2 | Pons et al. — *Upsampling artifacts in neural audio synthesis* | https://arxiv.org/abs/2010.14356 **[R]** | Sources d'artefacts (opérateurs, réplicas) ; quel suréchantillonneur est sain ? |
| 9 | P2 | Odena et al. — *Deconvolution and Checkerboard Artifacts* | https://distill.pub/2016/deconv-checkerboard/ **[R]** | Pourquoi un recouvrement inégal laisse un motif ; resize-convolution. |
| 10 | P2 | SoundStream | https://arxiv.org/abs/2107.03312 **[V]** | RVQ et dropout de quantificateurs ; pertes adversariales. |
| 11 | P2 | DAC (*High-Fidelity Audio Compression with Improved RVQGAN*) | https://arxiv.org/abs/2306.06546 **[V]** | Différences avec EnCodec (activation, pertes, codebooks) ; débits testés. |
| 12 | P2 | Müller et al. — silence | https://arxiv.org/abs/2106.12914 **[R]** | Comment construisent-ils le contrôle négatif ? Que change l'ablation « sans silence » ? |
| 13 | P2 | SONICS | https://arxiv.org/abs/2408.14080 **[V]** | Architecture SpecTTTra ; protocole ; ce que l'article annonce de la généralisation. |
| 14 | P2 | Choi et al. — tutoriel DL pour la MIR | https://arxiv.org/abs/1709.04396 **[V]** | Représentations d'entrée ; CNN ; pièges d'évaluation. |
| 15 | P2 | Guo et al. — *On Calibration of Modern Neural Networks* | https://arxiv.org/abs/1706.04599 **[R]** | ECE, diagramme de fiabilité, temperature scaling. |
| 16 | P2 | MusicGen | https://arxiv.org/abs/2306.05284 **[R]** | Tokenizer EnCodec ; entrelacement ; fréquence d'échantillonnage. |
| 17 | P2 | Stable Audio Open | https://arxiv.org/abs/2407.14358 **[R]** | Auto-encodeur (VAE, 156 M paramètres d'après l'article), DiT, 44,1 kHz stéréo. |
| 18 | P2 | AudioLDM 2 | https://arxiv.org/abs/2308.05734 **[R]** | « Language of audio » (AudioMAE + GPT-2) puis diffusion latente ; quel vocodeur ? |
| 19 | P2 | MusicLDM | https://arxiv.org/abs/2308.01546 **[R]** | Adapte Stable Diffusion/AudioLDM, ré-entraîne CLAP et **HiFi-GAN** sur la musique. |
| 20 | P2 | Mustango | https://arxiv.org/abs/2311.08355 **[R]** | Étend Tango ; MuNet ; quel vocodeur ? |
| 21 | P3 | AST (*Audio Spectrogram Transformer*) | https://arxiv.org/abs/2104.01778 **[V]** | Comment tokeniser un spectrogramme en patches. |
| 22 | P3 | Musika | https://arxiv.org/abs/2208.08706 **[R]** | Auto-encodeur adversarial + GAN ; cité par le dépôt Deezer comme codec. |
| 23 | P3 | AI-OpenBMAT | https://arxiv.org/abs/2602.06823 **[V]** | Conditions réalistes (musique en arrière-plan) ; accès au jeu. |
| 24 | P3 | Mimi (article Moshi) ; surveys codecs | https://arxiv.org/pdf/2410.00037v1 **[V]** ; https://arxiv.org/abs/2402.13071 ; https://arxiv.org/abs/2402.13236 ; https://arxiv.org/abs/2406.00146 | Vue d'ensemble des codecs ; débit/qualité. |
| 25 | P3 | Article listé dans les ressources du projet | https://arxiv.org/abs/2603.09936 **[M pour le titre]** | Titre et pertinence **non relus** : ouvrir avant d'en dépendre. |

Conseil de lecture pour un public sans cours : pour chaque papier, remplir 5 lignes — *problème, idée, méthode en 3 phrases, résultat clé, limite avouée* — avant de passer au suivant.

---

## 6. Fiche à remplir : les 5 générateurs de FakeMusicCaps

Objectif : documenter le **décodeur** de chacun (pour H1). Ce qui est connu est marqué ; le reste est **à vérifier dans le papier** (ne pas le supposer).

| Générateur | Famille (ce qui est connu) | Décodeur / vocodeur (à vérifier) | Lien probable avec EnCodec/DAC | Prédiction à écrire |
|---|---|---|---|---|
| **MusicGen** | LM Transformer sur jetons **EnCodec** (RVQ), 32 kHz [R] | Décodeur d'EnCodec (type de suréchantillonnage ?) | **Fort** | Détecteur « EnCodec » → meilleure AUC |
| **Stable Audio Open** | Diffusion latente (DiT 1 057 M) sur un VAE 156 M (latent 64), 44,1 kHz stéréo, T5 [R] | Décodeur de son propre VAE (convolutions strided/Snake côté encodeur ; décodeur à confirmer) | Moyen/faible | À écrire |
| **AudioLDM 2** | AudioMAE « language of audio » + GPT-2 + diffusion latente [R] | **Vocodeur ? VAE sur mel ?** | À déterminer | À écrire |
| **MusicLDM** | Adapte Stable Diffusion/AudioLDM ; **ré-entraîne CLAP et HiFi-GAN** [R] | HiFi-GAN | À déterminer | À écrire |
| **Mustango** | Étend Tango (text-to-audio) ; MuNet [R] | Vocodeur ? | À déterminer | À écrire |

Utilisation : regrouper les générateurs en familles de décodeurs (par exemple « codec-décodeur », « VAE-waveform », « VAE-mel + vocodeur »), puis calculer la corrélation de rang entre **similarité de famille** et **AUC croisée** (test de H1).

---

## 7. Mini-labs d'ancrage

Chaque lab dure 0,5–2 h, se fait sur CPU (sauf L5) et sert de matière au cours écrit dans la nouvelle session. Les durées sont des estimations.

| Lab | Contenu | Notion(s) | Ce qu'on doit voir |
|---|---|---|---|
| **L1** Aliasing | Synthétiser un sinus à fréquence proche de Nyquist ; rééchantillonner sans filtre puis avec filtre ; tracer la STFT. | N1 | Réplicas/repliement visibles sans filtre. |
| **L2** MP3 aller-retour | Encoder un extrait FMA en MP3 à plusieurs débits (par exemple avec `ffmpeg`), décoder, comparer le spectre moyen. | N5 | La **coupure passe-bas** et ses variations avec le débit ; mesurer, ne pas supposer. |
| **L3** Vitesse et axe log | Prendre un son harmonique ; changer la vitesse (×0,9, ×1,1) ; tracer le spectre sur axe linéaire puis logarithmique. | N3, N4, N21 | Sur l'axe linéaire, l'écart entre harmoniques change ; sur l'axe log, le spectre **se translate**. |
| **L4** Pics de suréchantillonnage | Réseau jouet : conv1D transposée (stride 3, noyau 5) vs interpolation « plus proche voisin » + conv ; poids aléatoires ; entrée bruit blanc ; moyenne de spectres. | N12, N17 | Pics périodiques avec la conv transposée ; spectre plat (ou lisse) avec l'autre. |
| **L5** Aller-retour codec | Passer un extrait dans EnCodec (puis DAC) ; comparer spectre moyen original vs reconstruit ; débits différents. *(GPU ou CPU, Colab).* | N14, N17 | Des pics/une coupure apparaissent selon le débit. |
| **L6** M0 | Extraire métadonnées (format, sr, durée, volume, bande passante) ; entraîner un classifieur ; AUC avec IC bootstrap. | N19, N20 | AUC > 0,5 *avant* même de regarder l'audio : c'est le raccourci. |
| **L7** Métriques | Implémenter ROC, AUC, EER, TPR@1 % FPR en NumPy ; vérifier avec scikit-learn ; bootstrap ; comparer variance bootstrap et variance entre seeds. | N8, N9 | Les trois mesures d'incertitude ne se valent pas. |
| **L8** Test d'écoute | Écrire le protocole : 20 extraits, ordre aléatoire, réponse binaire + confiance ; script d'analyse ; calcul de taux de réussite et d'accord. | N22 | Protocole prêt avant de recruter. |

---

## 8. Outils, données, dépôts, licences

| Élément | Usage | Lien / remarque |
|---|---|---|
| FMA | Réel | https://github.com/mdeff/fma **[V]** ; `fma_small` (8 000 × 30 s MP3) |
| FakeMusicCaps | Faux de test | https://zenodo.org/record/13732524/latest **[V]** ; 12,9 Go |
| EnCodec | Ré-encodage | https://github.com/facebookresearch/encodec **[V]** |
| DAC | Ré-encodage | https://github.com/descriptinc/descript-audio-codec **[V]** |
| SONICS | M4 (inférence) | https://github.com/awsaf49/sonics **[V]** (MIT pour le code et les modèles) |
| Dépôt ISMIR 2025 Deezer | Référence de M1 (à réimplémenter, pas à copier) | https://github.com/deezer/ismir25-ai-music-detector **[V]** (CC BY-NC 4.0 ; brevets mentionnés) |
| Dépôt robustesse | Référence de M3 | https://github.com/thomas0barand/robust-deepfake-detector **[V]** (licence illisible) |
| Dépôt deepfake-detector | Référence | https://github.com/deezer/deepfake-detector **[V]** (CC BY-NC 4.0) |
| Sources web sur le MP3 | N5 (vulgarisation) | https://www.forasoft.com/learn/audio-for-video/articles-audio/how-audio-compression-works **[R]** (article de blog, pas académique) ; https://typeset.io/pdf/the-theory-behind-mp3-4a7suq9ncw.pdf **[R]** (document de type mémoire, qualité non évaluée) |
| Calcul | Colab T4 | T4 ≈ 1,19 unité/h (mesure de mars 2026, fluctuante) ; nombre d'unités Pro **non confirmé** ; fp16 sur T4 |

Colab : sauvegarder un checkpoint toutes les 15 min sur Drive, scripts reprenables, ne pas compter sur un GPU garanti.

---

## 9. Glossaire (FR / EN)

| Terme | Définition courte |
|---|---|
| Fréquence d'échantillonnage (*sample rate*) | Nombre d'échantillons par seconde. Fixe la fréquence maximale représentable (Nyquist = sr/2). |
| Repliement (*aliasing*) | Fréquences au-dessus de Nyquist repliées dans la bande utile. |
| STFT | Transformée de Fourier à court terme : spectre sur des fenêtres glissantes. |
| Spectrogramme mel | Spectrogramme dont l'axe fréquence suit une échelle perceptive (mel). |
| CQT | Transformée à Q constant : résolution logarithmique en fréquence (une octave = nombre fixe de bins). |
| Rééchantillonnage (*resampling*) | Changer la fréquence d'échantillonnage. |
| Changement de vitesse | Lecture plus rapide/lente : hauteur **et** durée changent ensemble (mise à l'échelle de l'axe fréquence). |
| MDCT | Transformée en cosinus discrète modifiée, base de la compression MP3. |
| Masquage psychoacoustique | Un son fort cache des sons faibles voisins ; le codec les quantifie grossièrement. |
| Codec neuronal | Encodeur-décodeur appris, avec bottleneck quantifié, pour compresser l'audio. |
| RVQ | Quantification vectorielle résiduelle : plusieurs codebooks appliqués successivement à ce qui reste. |
| Vocodeur | Réseau qui transforme une représentation (mel) en onde sonore. |
| Conv transposée | Suréchantillonnage appris ; peut laisser des motifs périodiques. |
| Fakeprint | Empreinte spectrale moyenne caractéristique d'un générateur. |
| AUC / EER | Aire sous la courbe ROC / taux d'erreur où FPR = FNR. |
| TPR@1 % FPR | Taux de détection quand on accepte 1 % de faux positifs. |
| Calibration | Les probabilités prédites correspondent aux fréquences observées. |
| Raccourci (*shortcut*) | Signal corrélé à l'étiquette mais sans lien avec la tâche visée. |
| Fuite (*leakage*) | Information de test présente dans l'entraînement ou corrélée au split. |
| Contrôle négatif | Expérience censée donner un résultat nul si l'instrument est sain. |
| Hors distribution (*OOD*) | Données qui diffèrent de celles d'entraînement. |

---

## 10. Checkpoints d'auto-évaluation

À faire sans notes ; si deux réponses coincent dans un bloc, relire avant de passer au suivant.

**P0 — Signal**
1. Pourquoi un son échantillonné à 22,05 kHz ne contient-il rien au-dessus de 11,025 kHz ?
2. Que représente un bin de STFT ? Comment le compromis temps-fréquence dépend-il de la fenêtre ?
3. Pourquoi l'échelle mel et pourquoi des dB ?
4. Quelle est la différence entre changer la vitesse et décaler la hauteur seule ?

**S1 — Artefacts**
1. Pourquoi une convolution transposée de noyau non divisible par le stride laisse-t-elle un motif périodique ?
2. Que dit l'article de Fourier sur l'origine des pics, et pourquoi dépendent-ils de l'architecture ?
3. Quelle coupure de fréquence observes-tu sur un MP3 à 128 kbit/s, mesurée dans L2 ?
4. Qu'est-ce qu'un fakeprint et comment le calcule-t-on ?

**S2 — Codecs**
1. Dessine encodeur → RVQ → décodeur ; que change le nombre de codebooks ?
2. Qu'est-ce qui distingue un codec d'un générateur de musique ?
3. Pourquoi un morceau passé dans EnCodec est-il « faux » aux yeux d'un détecteur ?
4. Quelle perte entraîne EnCodec en plus de la reconstruction ?

**S3 — Mesures et raccourcis**
1. Définis AUC, EER, TPR@1 % FPR ; calcule-les sur un exemple de 6 scores.
2. Que fait M0 et que prouve un AUC > 0,5 pour M0 ?
3. Pourquoi découper par artiste ?
4. Quelle différence entre IC bootstrap et variance entre seeds ?
5. Cite un précédent de raccourci dans la détection de deepfakes audio.

**S4 — Robustesse**
1. Pourquoi une translation sur l'axe log correspond-elle à un changement de vitesse ?
2. Pourquoi une corrélation croisée + max-pooling est-elle invariante à la translation ?
3. Quelle nuance H4 porte-t-elle sur MP3 et bruit ?
4. Comment sélectionnes-tu les 20 extraits mal classés sans biais ?

---

## 11. Limites et liens non vérifiés

- **Slides AI for Sound (Peeters)** : introuvables en ligne ; non utilisés.
- **Liens [R]** : vus dans des résultats de recherche (titres/auteurs cohérents), pages non ouvertes par moi.
- **Détails de cours** (numéros d'épisodes Velardo, titres des unités HF Audio, chapitres FMP) : **[M]**, à confirmer en ouvrant les pages.
- **CS224S** : le syllabus n'a pas de liens de slides ni de lectures ; seul le lien de la première séance est confirmé.
- **Papiers 2025–2026** (Fourier, robustesse, AI-OpenBMAT, etc.) : connus par leur résumé ou la page du dépôt ; relire avant de citer des chiffres.
- **AI-OpenBMAT** : accès et licence inconnus.
- **Dépôt de robustesse** : licence non lisible.
- **Titre exact d'Afchar et al. 2024** et **papier 2603.09936** : à confirmer à l'ouverture.
- **Budget GPU** et **durées d'étude** : estimations, pas des mesures.
- **Colab** : nombre d'unités Pro non confirmé (100 ou 500 selon les sources).

---

## 12. Mode d'emploi pour la nouvelle session

### 12.1 Rôle de la session
La session sert à **produire des cours** (un par notion de la carte §3) et à **préparer les labs**, pas à refaire le projet. L'étudiant gardera ses cours comme base de révision et de rapport.

### 12.2 Règles
1. Ce fichier est la **source de vérité**. En cas de doute, le relire.
2. **Ne citer que les liens de ce fichier** avec leur étiquette. Ne jamais inventer d'URL, de numéro de page ou de chiffre.
3. Tout ce qui est **[R]** ou **[M]** est signalé comme tel dans le cours.
4. **Paraphraser** les papiers ; pas de longues citations.
5. Chaque cours finit par **5 questions d'auto-évaluation** avec corrigé.
6. Dire clairement ce qui est établi (papier) et ce qui est **hypothèse du projet**.

### 12.3 Format d'un cours par notion
1. Intuition en 5 lignes.
2. Formalisation (formules, définitions) — niveau M2.
3. Exemple numérique ou schéma.
4. Code minimal (NumPy/PyTorch) quand le niveau le demande.
5. Pièges fréquents.
6. Lien avec le projet (hypothèse, ablation, détecteur concernés).
7. Lectures (liens de ce fichier).
8. Questions d'auto-évaluation + corrigé.

### 12.4 Prompt à coller pour démarrer
> Tu es mon tuteur pour le projet décrit dans le fichier joint. Je suis en M2 IA, je n'ai aucun cours d'audio ni de génératif. Lis le fichier en entier. Puis écris le cours de la notion **N__** en suivant le format du §12.3, en n'utilisant que les liens du fichier, avec leurs étiquettes. Signale ce qui est incertain. Termine par cinq questions avec corrigé, et propose-moi le lab associé.

### 12.5 Ordre conseillé pour écrire les cours
N1 → N2 → N3 → N4 → N5 → N12 → N17 (semaine 1) ; N10 → N11 → N13 → N14 → N15 (semaine 2) ; N8 → N9 → N19 → N20 → N6 → N18 → N16 (semaine 3) ; N21 → N22 → N23 (semaine 4). N7 et N24 en dernier.
