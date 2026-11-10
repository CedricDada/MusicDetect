# Documents du projet

| Dossier / fichier | Contenu | Pages |
|---|---|---|
| `cours/cours_detection_musique_IA.{tex,pdf}` + `cours/fig/` | **Cours** : tout ce qu'il faut maîtriser pour réaliser le projet (signal, STFT, codecs neuronaux, fakeprint, robustesse, évaluation) | 19 |
| `projet/proposition_projet_longue.{tex,pdf}` | **Proposition de projet complète** (version encadrant) : question, hypothèses pré-enregistrées, données et chaîne, détecteurs, protocole, budget et risques, livrables, bibliographie annotée, plan de formation, glossaire | 32 |
| `projet/proposition_projet_2_pages.{tex,pdf}` | **Proposition courte** avec figures (chaîne, matrice de tests H1–H2, pourquoi la log-fréquence aide) | 2 |
| `projet/archive_2_pages_sans_figures.{tex,pdf}` | version précédente de la proposition courte, sans figures (archive) | 2 |
| `projet/dossier_projet_et_programme_v2.{tex,pdf}` | **Dossier de projet + programme d'apprentissage**, version 2 corrigée (reprise détaillée de la description initiale) | 30 |
| `projet/description_initiale.md` | description d'origine fournie au départ (Markdown) | — |
| `DEPENDENCIES.md` | outils système, bibliothèques et versions, poids, dépôts consultés et licences | — |

## Recompiler

Les `.tex` utilisent `babel` en français (`\usepackage[french]{babel}`). Les PDF fournis sont des **aperçus compilés
sans le paquet `babel-french`** (absent de l'environnement de rédaction) : les guillemets et espaces insécables
français y sont approximés. Avec une installation TeX Live complète :

```bash
cd docs/cours  && latexmk -pdf cours_detection_musique_IA.tex
cd docs/projet && for f in proposition_projet_longue proposition_projet_2_pages dossier_projet_et_programme_v2; do latexmk -pdf $f.tex; done
```

Le cours réclame le dossier `fig/` à côté du `.tex` (cinq figures PDF).

## Ce que ces documents contiennent de non vérifié

La proposition longue et le dossier v2 portent chacun une section « Limites et éléments non vérifiés » ; le dossier v2
étiquette en plus chaque lien (**[V]** vérifié, **[R]** vu en résultat de recherche, **[M]** non vérifié). Les chiffres de papiers récents lus par
résumé seulement sont signalés comme tels. Le code du dépôt suit la proposition, avec un écart connu :
la ligne « VAE de Stable Audio Open (oracle) » de la matrice H1/H2 n'est pas implémentée (README, § 10).
