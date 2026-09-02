import pandas as pd
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