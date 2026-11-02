# Projet 12 : Détection automatique de faux billets (Machine Learning)

### Contexte
L'**ONCFM** (organisation de lutte contre le faux-monnayage) dispose d'une machine qui mesure six caractéristiques géométriques de chaque billet : longueur, hauteurs, marges, diagonale.

**Mon objectif :** construire un algorithme qui distingue automatiquement les vrais des faux billets à partir de ces seules mesures. Le critère prioritaire : **détecter un maximum de faux billets**, car un faux non détecté coûte bien plus cher qu'un vrai contrôlé par erreur.

### Missions réalisées
* **Analyse exploratoire et ACP :** identification des variables discriminantes (la longueur et les marges séparent nettement vrais et faux, la diagonale presque pas).
* **Prétraitement sans fuite de données :** séparation train/test **avant** toute transformation, imputation des valeurs manquantes par KNNImputer calibrée sur le train seul, standardisation.
* **Comparaison de 4 algorithmes :** régression logistique, KNN, Random Forest et K-means (non supervisé, interprété a posteriori).
* **Choix du modèle :** à performance égale, la **régression logistique** a été retenue pour son interprétabilité, un argument décisif face à des équipes non techniques.
* **Mise en production :** script en ligne de commande qui traite un billet ou un fichier complet et renvoie la probabilité associée à chaque prédiction.

### Résultats
* **99 %** d'accuracy sur le jeu de test.
* **98 %** de recall sur la classe « faux billet ».

### Utilisation du script de production
Le script `scripts/predict_billets.py` charge le modèle (`scripts/model_faux_billets.joblib`) et prédit si un billet est vrai ou faux, avec la probabilité associée.

```bash
# Un fichier complet de billets
python predict_billets.py --csv billets_production.csv

# Un seul billet
python predict_billets.py --diagonal 171.8 --height_left 104.0 --height_right 104.2 --margin_low 4.3 --margin_up 3.1 --length 113.1
```

Sans argument, le script passe en mode interactif.

### Compétences & Outils
* **Langage :** Python
* **Librairies :** Pandas, Scikit-learn, joblib
* **Techniques :** classification supervisée, clustering, ACP, pipeline de prétraitement, métriques (recall, matrice de confusion)

### Reproduire l'analyse
Les données brutes sont dans `data-raw/`. Pour exécuter les notebooks, placez ces fichiers dans le même dossier que le notebook.

### Limites
Trois billets proches de la frontière vrai/faux restent mal classés par tous les modèles supervisés : la limite vient des données, pas du choix d'algorithme. Le seuil de décision pourrait être ajusté au coût réel d'une erreur.

<img src="images/Capture d'écran 2026-08-11 103001.png" width="500px"><img src="images/Capture d'écran 2026-08-11 103054.png" width="500px">
<img src="images/Capture d'écran 2026-08-11 103113.png" width="500px"><img src="images/Capture d'écran 2026-08-11 103133.png" width="500px">

---
📂 Dossier complet sur mon portfolio : [remydaura.github.io/p12.html](https://remydaura.github.io/p12.html)

*Ce projet fait partie de la formation Data Analyst d'OpenClassrooms.*
