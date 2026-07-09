# Multilayer Perceptron

Implémentation from scratch d'un perceptron multicouche (MLP) en Python/NumPy, pour prédire si une tumeur est **maligne (M)** ou **bénigne (B)** à partir du Wisconsin Breast Cancer Dataset.

Projet réalisé dans le cadre du cursus Data Science de l'école 42, sans librairie de deep learning (pas de TensorFlow/PyTorch/Keras) — feedforward, backpropagation et descente de gradient sont codés manuellement.

## Sommaire

- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [1. Split du dataset](#1-split-du-dataset)
  - [2. Entraînement](#2-entraînement)
  - [3. Prédiction](#3-prédiction)
- [Fonctionnement](#fonctionnement)
- [Bonus](#bonus)
- [Résultats](#résultats)

## Architecture du projet

```
Multilayer_Perceptron/
├── mlp.py                  # point d'entrée (CLI, argparse)
├── SRC/
│   ├── split.py             # découpage du dataset (train/valid)
│   ├── train.py              # entraînement du réseau
│   ├── predict.py            # chargement du modèle + évaluation
│   └── utils.py               # fonctions partagées (activations, forward pass, sauvegarde...)
├── data/
│   ├── data.csv               # dataset original (fourni)
│   ├── data_train.csv         # généré par `split`
│   └── data_valid.csv         # généré par `split`
├── models/
│   ├── saved_model.npy        # généré par `train` (poids, biais, topologie, normalisation)
│   └── metrics.csv            # généré par `train` (historique des métriques par epoch)
├── graph/
│   ├── loss.png                # généré par `train`
│   └── accuracy.png            # généré par `train`
└── README.md
```

## Installation

```bash
pip install numpy pandas matplotlib --break-system-packages
```

## Utilisation

Le programme s'utilise via un point d'entrée unique (`mlp.py`) avec trois sous-commandes : `split`, `train`, `predict`.

### 1. Split du dataset

Sépare `data.csv` en un jeu d'entraînement et un jeu de validation (80/20), de façon **stratifiée** (même proportion de M/B dans les deux sous-ensembles) et reproductible via une seed.

```bash
python mlp.py split --dataset data/data.csv --seed 42
```

| Argument | Type | Défaut | Description |
|---|---|---|---|
| `--dataset` | str | `data/data.csv` | Chemin vers le CSV source |
| `--seed` | int | `42` | Graine aléatoire pour la reproductibilité |

Génère `data/data_train.csv` et `data/data_valid.csv`.

### 2. Entraînement

Entraîne un réseau de neurones feedforward par mini-batch gradient descent.

```bash
python mlp.py train --layer 24 24 --epochs 100 --batch_size 8 --learning_rate 0.01
```

| Argument | Type | Défaut | Description |
|---|---|---|---|
| `--layer` | int (liste) | `[24, 24]` | Tailles des couches cachées |
| `--epochs` | int | `100` | Nombre d'epochs |
| `--loss` | str | `categoricalCrossentropy` | Fonction de coût (seule valeur supportée actuellement) |
| `--batch_size` | int | `8` | Taille des mini-batchs |
| `--learning_rate` | float | `0.01` | Taux d'apprentissage |
| `--early_stopping` | flag | désactivé | Active l'arrêt anticipé (voir [Bonus](#bonus)) |
| `--patience` | int | `10` | Nombre d'epochs sans amélioration avant arrêt (si `--early_stopping`) |

À chaque epoch, affiche la loss et l'accuracy sur train et validation. À la fin :
- sauvegarde le modèle (topologie, poids, biais, paramètres de normalisation) dans `models/saved_model.npy`
- sauvegarde l'historique des métriques dans `models/metrics.csv`
- génère les courbes d'apprentissage dans `graph/loss.png` et `graph/accuracy.png`

### 3. Prédiction

Charge un modèle entraîné et évalue ses performances sur un dataset.

```bash
python mlp.py predict --dataset data/data_valid.csv --model models/saved_model.npy
```

| Argument | Type | Défaut | Description |
|---|---|---|---|
| `--dataset` | str | `data/data_valid.csv` | Dataset à évaluer |
| `--model` | str | `models/saved_model.npy` | Modèle à charger |

Affiche l'architecture du réseau, l'accuracy, et la binary cross-entropy (formule imposée par le sujet).

## Fonctionnement

Le réseau est un perceptron multicouche entièrement connecté (dense), entraîné par backpropagation :

- **Feedforward** : les données traversent chaque couche via `Z = A·W + b`, suivi d'une activation (`sigmoid` pour les couches cachées, `softmax` en sortie pour obtenir une distribution de probabilités sur les 2 classes).
- **Initialisation des poids** : méthode `heUniform`, tirage uniforme dans `[-√(6/n_in), +√(6/n_in)]`, pour éviter les problèmes de vanishing/exploding gradient.
- **Backpropagation** : calcul des gradients couche par couche en partant de la sortie (`dZ = A - y_onehot` pour la combinaison softmax + cross-entropy), remontée via la chain rule jusqu'à la première couche.
- **Mise à jour des poids** : descente de gradient classique, `w -= learning_rate * dW`.
- **Normalisation** : les features sont standardisées (moyenne 0, écart-type 1) à partir des statistiques du train set ; les mêmes statistiques sont réutilisées pour la validation et la prédiction.

## Bonus

- **Historique des métriques** : sauvegarde de `train_loss`, `train_accuracy`, `valid_loss`, `valid_accuracy` par epoch dans `models/metrics.csv`.
- **Early stopping** : surveille `val_loss` et arrête l'entraînement après `--patience` epochs sans amélioration, en conservant une copie des poids au meilleur point observé (pas les poids de la dernière epoch). Permet d'éviter l'overfitting et de réduire le temps d'entraînement.

## Résultats

Sur ce dataset, avec l'architecture par défaut (`[24, 24]`), le modèle atteint typiquement :

- **~99% d'accuracy** sur le train set
- **~95-97% d'accuracy** sur le jeu de validation
- Une binary cross-entropy de validation autour de **0.08-0.10**

Un écart entre les performances train/validation est observable dès une trentaine d'epochs (overfitting), ce que corrige en partie le bonus early stopping.