import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import linregress
#  Charger le fichier CSV
df = pd.read_csv("data/Extreme_climate_events.csv", index_col=0)
#  Afficher les 5 premières lignes
print("Aperçu des données :")
print(df.head())
# Afficher les noms des des colonnes
print("\nColonnes disponibles :")
print(df.columns)
# Afficher les informations générales du fichier
print("\nInformations sur le fichier :")
print(df.info())
# Vérifier les valeurs manquantes
print("\nVérification des valeurs manquantes :")
print(df.isnull().sum())
# Supprimer les lignes sans date
df_clean = df.dropna(subset=["Date"]).copy()
# Extraire l'année de début
df_clean["annee_debut"] = df_clean["Date"].str.extract(r'(\d{4})')
# Créer une colonne mois vide car le fichier ne contient pas le mois
df_clean["mois"] = None
# Renommer le type d'événement
df_clean["type_evenement"] = df_clean["Event Type"]
# Créer une colonne région approximative depuis le titre
df_clean["region"] = df_clean["Title"].str.replace(r",?\s*\d{4}.*", "", regex=True)
# Créer les colonnes absentes dans le fichier original
df_clean["blesses"] = None
df_clean["morts"] = None
df_clean["degats_materiels"] = None
print("\nDonnées nettoyées :")
print(df_clean[[
    "annee_debut",
    "mois",
    "region",
    "type_evenement",
    "blesses",
    "morts",
    "degats_materiels"
]].head())
# 3. Agrégation temporelle 
# Supprimer les lignes sans date, car on ne peur pas le placer dans le temps
df_clean = df.dropna(subset=["Date"]).copy()
# Agréger les données par année la colonne Date
df_clean["annee"] = df_clean["Date"].str.extract(r"(\d{4})")
# Convertir l'année en nombre
df_clean["annee"] = df_clean["annee"].astype(int)
# Le fichier contient pas le mois, donc on crée une colonne mois inconnue
df_clean["mois"] = "inconnu"
# Renommer le type d'événement pour avoir un nom plus simple
df_clean["type_evenement"] = df_clean["Event Type"]
# Compter les événements par année mois et catégorie
events_par_temps = (
    df_clean
    .groupby(["annee", "mois", "type_evenement"])
    .size()
    .reset_index(name="nombre_evenements")
)

print("\nNombre d'événements par année, mois et catégorie :")
print(events_par_temps.head(20))
# Vérifier les années présentes
annees_presentes = sorted(df_clean["annee"].unique())

print("\nAnnées présentes dans le dataset :")
print(annees_presentes)

# Créer la liste complète des années entre la première et la dernière année
annee_min = df_clean["annee"].min()
annee_max = df_clean["annee"].max()

annees_completes = set(range(annee_min, annee_max + 1))
annees_existantes = set(annees_presentes)

annees_manquantes = sorted(annees_completes - annees_existantes)

print("\nAnnées manquantes :")
print(annees_manquantes)
## 4. Analyse de tendance par catégorie
resultats_tendance = []
for categorie in df_clean["type_evenement"].unique():
    data_categorie = events_par_temps[events_par_temps["type_evenement"] == categorie]
    series = (
        data_categorie
        .set_index("annee")["nombre_evenements"]
        .reindex(range(annee_min, annee_max + 1), fill_value=0)
    )
    test_mk = mk.original_test(series)
    regression = linregress(list(range(annee_min, annee_max + 1)), series)
    resultats_tendance.append({
        "type_evenement": categorie,
        "tendance_mann_kendall": test_mk.trend,
        "p_value": test_mk.p,
        "pente_mann_kendall": test_mk.slope,
        "pente_regression": regression.slope
    })
print("\nAnalyse de tendance par catégorie :")
print(resultats_tendance)
