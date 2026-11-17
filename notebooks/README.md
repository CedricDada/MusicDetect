# Exécution sur Kaggle

`musicdetect_kaggle.ipynb` termine la phase 1 sur Kaggle sans refaire ce qui est déjà calculé.
Il lit les modèles présents et la colonne `test` des fichiers de métriques, puis n'exécute que ce qui manque.

## Pourquoi passer sur Kaggle

Une session lancée par **Save & Run All (Commit)** tourne en arrière-plan, fenêtre fermée, et n'est pas
tuée par un redémarrage de Windows Update — ce qui a interrompu deux exécutions locales.

Le T4 n'apporte qu'un gain modeste : les goulets sont l'augmentation de vitesse de M3 et le calcul des
empreintes à l'évaluation, tous deux sur CPU et parallélisés par `--jobs`. Le vrai gain vient d'ailleurs :
`--jobs` vaut **1** par défaut (`cli.py`), et les entraînements locaux ont donc tourné sur un seul cœur.
Avec `--jobs 4` l'augmentation passe d'environ 1 h 39 à une vingtaine de minutes.

Si tu butes sur la limite de 9 h des sessions GPU, repasse l'accélérateur à *None* : tu gagnes 3 h de
session pour quelques minutes de GPU perdues.

## Les deux Datasets

| Dataset | Contenu | Fréquence |
|---|---|---|
| `musicdetect-segments` | 2 index CSV + `segments/fma_phase1/` + `segments/fmc_phase1/` (~13 Go) | **une seule fois** |
| `musicdetect-etat` | `models/` et `results/` (~20 Mo) | nouvelle version à chaque session |

Les segments ne sont **pas** régénérés sur Kaggle. Les résultats des graines 1 et 2 ont été calculés sur
ces segments-là ; une reconstruction introduirait un risque de divergence impossible à vérifier, alors
que les monter en lecture seule garantit que les trois graines portent sur les mêmes données.

## 1. Préparer le dossier à téléverser

> **Le piège :** le CLI Kaggle parcourt le dossier avec `os.walk(followlinks=False)`, et Python voit une
> jonction Windows comme un lien. Les jonctions `fma_phase1_s1`, `_s2`, `_s3` seraient donc **ignorées** —
> mais si tu téléversais `data/work` tel quel, tu risquerais au mieux un Dataset incomplet. On déplace donc
> les deux vrais dossiers dans un dossier d'envoi, puis on les remet.

Le déplacement est instantané : même volume, aucune copie de données.

Les deux dossiers de segments sont mis **à la racine** du dossier d'envoi, et non sous `segments/` :
le CLI produit alors deux archives séparées de 4,8 Go et 8,2 Go au lieu d'une seule de 13 Go, ce qui
limite la casse si un envoi échoue. Le notebook cherche chaque élément par son nom, la mise en page
du Dataset n'a donc pas d'importance.

```powershell
$P = "D:\3A\Stages\Préparation entretiens - projets  audio, World models\musicdetect"
$U = "D:\kaggle_upload_segments"

New-Item -ItemType Directory -Force $U | Out-Null
Copy-Item "$P\data\work\index_fma_phase1.csv" $U
Copy-Item "$P\data\work\index_fmc_phase1.csv" $U
Move-Item  "$P\data\work\segments\fma_phase1" "$U\fma_phase1"
Move-Item  "$P\data\work\segments\fmc_phase1" "$U\fmc_phase1"
```

Pendant l'envoi, les jonctions locales `fma_phase1_s*` pointent dans le vide. C'est sans conséquence
tant que tu ne lances pas `run_reste_phase1.ps1`, et tout redevient normal à l'étape 3.

## 2. Téléverser

```powershell
pip install --user kaggle
# jeton d'API : Kaggle > Settings > Create New Token  ->  %USERPROFILE%\.kaggle\kaggle.json

@'
{
  "title": "musicdetect segments phase1",
  "id": "cedricdada/musicdetect-segments",
  "licenses": [{"name": "CC0-1.0"}]
}
'@ | Set-Content -Encoding utf8 "$U\dataset-metadata.json"

kaggle datasets create -p $U --dir-mode zip
```

`--dir-mode zip` envoie chaque sous-dossier comme une archive, ce qui est bien plus rapide que
27 000 fichiers un par un. Kaggle les décompresse à l'arrivée ; le notebook retrouve les segments par
leur index, où qu'ils soient montés.

Compte plusieurs heures selon ta liaison montante : 13 Go à 10 Mb/s font environ 3 h.

## 3. Remettre les segments en place

```powershell
Move-Item "$U\fma_phase1" "$P\data\work\segments\fma_phase1"
Move-Item "$U\fmc_phase1" "$P\data\work\segments\fmc_phase1"
Remove-Item -Recurse $U
```

## 4. Dataset d'état et secret

```powershell
$E = "D:\kaggle_upload_etat"
New-Item -ItemType Directory -Force $E | Out-Null
Copy-Item -Recurse "$P\data\work\models" "$E\models"
Copy-Item -Recurse "$P\results"          "$E\results"

@'
{
  "title": "musicdetect etat",
  "id": "cedricdada/musicdetect-etat",
  "licenses": [{"name": "CC0-1.0"}]
}
'@ | Set-Content -Encoding utf8 "$E\dataset-metadata.json"

kaggle datasets create -p $E
```

Puis, dans le notebook Kaggle : **Add-ons → Secrets**, un secret nommé `musicdetect` contenant le JSON
du jeton, `{"username": "...", "key": "..."}`. C'est ce qui permet à `save_state()` de publier une
nouvelle version du Dataset d'état à la fin de chaque session.

## 5. Avant chaque session

Panneau de droite : accélérateur **GPU T4**, **Internet On**, les deux Datasets en entrée avec
`musicdetect-etat` sur sa **dernière** version. Puis **Save & Run All (Commit)**.

La cellule 0 est la seule à modifier entre deux sessions :

| Réglage | Rôle |
|---|---|
| `STAGE` | `train_eval`, `attacks`, `report` ou `all` |
| `SEEDS` | graines traitées ; celles qui sont finies sont sautées de toute façon |
| `ATTACK_PRESET` | `prereg` = les 3 attaques pré-enregistrées ; `full` = les 19 de `default.yaml` |
| `MAX_HOURS` | budget ; une étape qui dépasserait est reportée proprement à la session suivante |
| `JOBS` | 4 sur Kaggle |

## Ce que le notebook corrige par rapport aux commandes locales

**`--attack-on` au lieu d'un seul `--test`.** `evaluate` réécrit `results/<tag>/<stem>.csv` avec les
seuls tests de l'invocation (`pipeline.py`). Lancer les attaques avec `--test fmc:musicgen` seul ferait
passer ce fichier de 7 lignes à 1, et le rapport perdrait quatre générateurs plus le contrôle `identity`.
Le notebook passe la liste complète et désigne la cible séparément.

**Le contrôle en-domaine 12 kbps.** Les évaluations des modèles 12 kbps testaient `fma:encodec_24k_6kbps`,
donc mesuraient un transfert entre débits et non le contrôle en-domaine. `tests_for()` ajoute
`fma:<codec>` quand il diffère de 6 kbps.

**La grille d'attaques réduite.** `default.yaml` déclare 19 attaques ; H4 n'en teste que trois
(vitesse 0,9 et 1,1, MP3 128). Le preset `prereg` s'y tient. Le preset `full` retire automatiquement
`pitch` si `librubberband` manque dans le ffmpeg de l'image Kaggle, plutôt que d'échouer en cours de route.

**L'agrégation multi-graines**, absente du dépôt : moyenne ± écart-type sur les graines, écrite dans
`results/agregat_graines.csv`, séparément des intervalles bootstrap. Ce sont deux quantités distinctes,
et pour M2 l'écart entre graines dépasse la largeur de l'intervalle bootstrap.
