"""
Script de production - Detection de faux billets (ONCFM)
============================================================

Ce script charge le modele entraine (pipeline complet : imputation
KNNImputer + standardisation + regression logistique) et permet de predire si un ou plusieurs
billets sont vrais ou faux, a partir de leurs caracteristiques geometriques.

PREREQUIS (2 fichiers doivent etre dans le meme dossier) :
    - predict_billets.py       (ce script)
    - model_faux_billets.joblib (le modele entraine)

UTILISATION
-----------

1) Predire un fichier CSV contenant plusieurs billets :
    python predict_billets.py --csv billets_production.csv

   Le fichier CSV doit contenir les colonnes :
   diagonal, height_left, height_right, margin_low, margin_up, length
   (separateur ";", comme le fichier d'exemple billets.csv)

   Un fichier de resultats est cree a cote (par defaut : resultats_predictions.csv),
   et un resume est affiche dans le terminal.

2) Predire un seul billet en donnant ses valeurs directement :
    python predict_billets.py --diagonal 171.8 --height_left 104.0 \\
        --height_right 104.2 --margin_low 4.3 --margin_up 3.1 --length 113.1

3) Mode interactif (si aucun argument n'est fourni) :
    python predict_billets.py
   Le script demande alors a l'utilisateur ce qu'il souhaite faire.
"""

import argparse
import os
import sys

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_faux_billets.joblib")
COLONNES_ATTENDUES = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]


def charger_modele():
    """Charge le pipeline entraine. Arrete le programme proprement si le fichier est introuvable."""
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : le fichier modele '{MODEL_PATH}' est introuvable.")
        print("Verifiez qu'il est bien dans le meme dossier que ce script.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predire_dataframe(pipeline, df):
    """Ajoute les colonnes prediction et probabilites a un DataFrame de billets."""
    colonnes_manquantes = [c for c in COLONNES_ATTENDUES if c not in df.columns]
    if colonnes_manquantes:
        print(f"Erreur : colonnes manquantes dans le fichier : {colonnes_manquantes}")
        print(f"Colonnes attendues : {COLONNES_ATTENDUES}")
        sys.exit(1)

    X = df[COLONNES_ATTENDUES]
    predictions = pipeline.predict(X)
    probabilites = pipeline.predict_proba(X)

    resultat = df.copy()
    resultat["prediction"] = ["Vrai" if p else "Faux" for p in predictions]
    resultat["probabilite_faux"] = probabilites[:, 0].round(4)
    resultat["probabilite_vrai"] = probabilites[:, 1].round(4)
    return resultat


def traiter_csv(pipeline, chemin_csv, chemin_sortie):
    if not os.path.exists(chemin_csv):
        print(f"Erreur : le fichier '{chemin_csv}' est introuvable.")
        sys.exit(1)

    # Detection automatique du separateur (";" ou ",") pour eviter tout
    # decalage si le fichier fourni le jour J n'est pas au meme format
    # que les donnees d'entrainement.
    df = pd.read_csv(chemin_csv, sep=None, engine="python")
    resultat = predire_dataframe(pipeline, df)
    resultat.to_csv(chemin_sortie, sep=";", index=False)

    nb_total = len(resultat)
    nb_vrais = (resultat["prediction"] == "Vrai").sum()
    nb_faux = (resultat["prediction"] == "Faux").sum()

    print(f"\n{nb_total} billets analyses :")
    print(f"  - {nb_vrais} vrais billets")
    print(f"  - {nb_faux} faux billets")
    print(f"\nResultats detailles ecrits dans : {chemin_sortie}")


def traiter_billet_unique(pipeline, valeurs):
    df = pd.DataFrame([valeurs])
    resultat = predire_dataframe(pipeline, df)

    prediction = resultat["prediction"].iloc[0]
    proba_vrai = resultat["probabilite_vrai"].iloc[0]
    proba_faux = resultat["probabilite_faux"].iloc[0]

    print(f"\nPrediction : {prediction.upper()}")
    print(f"Probabilite vrai billet : {proba_vrai * 100:.2f}%")
    print(f"Probabilite faux billet : {proba_faux * 100:.2f}%")


def demander_valeur(nom, obligatoire=True):
    while True:
        saisie = input(f"  {nom}" + (" (laisser vide si inconnu) " if not obligatoire else "") + " : ").strip()
        if saisie == "" and not obligatoire:
            return None
        if saisie == "" and obligatoire:
            print("    -> cette valeur est obligatoire.")
            continue
        try:
            return float(saisie.replace(",", "."))
        except ValueError:
            print("    -> merci d'entrer un nombre valide (ex: 171.8).")


def mode_interactif(pipeline):
    print("=== Mode interactif - Detection de faux billets ===\n")
    print("Que souhaitez-vous faire ?")
    print("  1) Analyser un fichier CSV contenant plusieurs billets")
    print("  2) Analyser un seul billet en saisissant ses mesures")
    choix = input("Votre choix (1 ou 2) : ").strip()

    if choix == "1":
        chemin_csv = input("Chemin vers le fichier CSV : ").strip()
        chemin_sortie = input(
            "Nom du fichier de resultats (laisser vide pour 'resultats_predictions.csv') : "
        ).strip() or "resultats_predictions.csv"
        traiter_csv(pipeline, chemin_csv, chemin_sortie)

    elif choix == "2":
        print("\nSaisissez les mesures du billet (en mm) :")
        valeurs = {
            "diagonal": demander_valeur("diagonal"),
            "height_left": demander_valeur("height_left"),
            "height_right": demander_valeur("height_right"),
            "margin_low": demander_valeur("margin_low", obligatoire=False),
            "margin_up": demander_valeur("margin_up"),
            "length": demander_valeur("length"),
        }
        traiter_billet_unique(pipeline, valeurs)

    else:
        print("Choix non reconnu. Veuillez relancer le script.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Predit si un ou plusieurs billets sont vrais ou faux a partir de leurs mesures geometriques."
    )
    parser.add_argument("--csv", type=str, help="Chemin vers un fichier CSV contenant plusieurs billets.")
    parser.add_argument(
        "--output", type=str, default="resultats_predictions.csv",
        help="Nom du fichier de resultats (utilise uniquement avec --csv). Par defaut : resultats_predictions.csv"
    )
    parser.add_argument("--diagonal", type=float, help="Diagonale du billet (mm).")
    parser.add_argument("--height_left", type=float, help="Hauteur du billet, cote gauche (mm).")
    parser.add_argument("--height_right", type=float, help="Hauteur du billet, cote droit (mm).")
    parser.add_argument("--margin_low", type=float, default=None, help="Marge basse du billet (mm). Facultatif.")
    parser.add_argument("--margin_up", type=float, help="Marge haute du billet (mm).")
    parser.add_argument("--length", type=float, help="Longueur du billet (mm).")

    args = parser.parse_args()
    pipeline = charger_modele()

    if args.csv:
        traiter_csv(pipeline, args.csv, args.output)

    elif args.diagonal is not None or args.height_left is not None or args.length is not None:
        valeurs_obligatoires = {
            "diagonal": args.diagonal,
            "height_left": args.height_left,
            "height_right": args.height_right,
            "margin_up": args.margin_up,
            "length": args.length,
        }
        manquantes = [k for k, v in valeurs_obligatoires.items() if v is None]
        if manquantes:
            print(f"Erreur : valeurs manquantes en ligne de commande : {manquantes}")
            print("(margin_low est la seule valeur facultative, elle peut etre imputee automatiquement)")
            sys.exit(1)

        valeurs = {**valeurs_obligatoires, "margin_low": args.margin_low}
        traiter_billet_unique(pipeline, valeurs)

    else:
        # Aucun argument fourni -> mode interactif
        mode_interactif(pipeline)


if __name__ == "__main__":
    main()
