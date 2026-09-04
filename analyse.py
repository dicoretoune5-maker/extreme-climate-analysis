import pandas as pd
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, spearmanr
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
# 5. Corrélation entre cotégories
from scipy.stats import spearmanr
# Créer un tableau avec les années en ligne et les catégories en colennes
table_categories = (
    events_par_temps
    .groupby(["annee", "type_evenement"])["nombre_evenements"]
    .sum()
    .reset_index()
    .pivot_table(index="annee", columns="type_evenement", values="nombre_evenements")
)
# Ajouter les années manquantes et remplacer les valeurs vides par 0
table_categories = table_categories.reindex(range(annee_min, annee_max + 1), fill_value=0)
table_categories = table_categories.fillna(0)

print("\nTableau des comptages par année et catégorie :")
print(table_categories.head())
# Retirer la tendance commune de chaque catégorie
residus = pd.DataFrame(index=table_categories.index)
x = np.arange(len(table_categories.index))
for categorie in table_categories.columns:
    y = table_categories[categorie].values
    regression = linregress(x, y)
    trendance = regression.intercept + regression.slope * x
    residus[categorie] = y - trendance
print("\nDonnées après retrait de la tendance :")
print(residus.head()) 
# Calculer la matrice de corrélation de Spearman
correlation_spearman = residus.corr(method="spearman")

print("\nCorrélation de Spearman après retrait de la tendance :")
print(correlation_spearman)
# Calculer les p-values des corrélations
categories = residus.columns

p_values = pd.DataFrame(index=categories, columns=categories)

for cat1 in categories:
    for cat2 in categories:
        coef, p_value = spearmanr(residus[cat1], residus[cat2])
        p_values.loc[cat1, cat2] = p_value

print("\nP-values des corrélations :")
print(p_values)

# 6. Analyse attendue : repondre aux questions du projet

print("\n--- Analyse attendue ---")

# On prend les 10 dernieres annees disponibles dans le dataset.
# On utilise la derniere annee du fichier, pas l'annee actuelle.
derniere_annee = annee_max
premiere_annee_10_ans = derniere_annee - 9

donnees_10_ans = df_clean[
    (df_clean["annee"] >= premiere_annee_10_ans)
    & (df_clean["annee"] <= derniere_annee)
].copy()

print(f"\nPeriode analysee pour les 10 dernieres annees : {premiere_annee_10_ans}-{derniere_annee}")


def analyser_tendance_categorie(nom_categorie):
    """Afficher une tendance simple pour une categorie."""
    serie = (
        events_par_temps[events_par_temps["type_evenement"] == nom_categorie]
        .set_index("annee")["nombre_evenements"]
        .reindex(range(premiere_annee_10_ans, derniere_annee + 1), fill_value=0)
    )

    if serie.sum() == 0:
        print(f"\n{nom_categorie} : aucune donnee sur les 10 dernieres annees.")
        return

    regression = linregress(list(serie.index), serie.values)

    print(f"\n{nom_categorie} sur les 10 dernieres annees :")
    print(serie)
    print(f"Pente de tendance : {regression.slope:.3f} evenement(s) par an")

    if regression.slope > 0:
        print("Interpretation simple : tendance a la hausse.")
    elif regression.slope < 0:
        print("Interpretation simple : tendance a la baisse.")
    else:
        print("Interpretation simple : tendance stable.")


# Question 1 : tornades ou tempetes sur les 10 dernieres annees
print("\nQuestion 1 : Le nombre de tornades ou de tempetes evolue-t-il sur les 10 dernieres annees ?")
analyser_tendance_categorie("Storm")

tornades = donnees_10_ans[
    donnees_10_ans["Title"].str.contains("tornado", case=False, na=False)
]
print(f"\nNombre de lignes mentionnant une tornade sur les 10 dernieres annees : {len(tornades)}")
print("Remarque : le dataset n'a pas une categorie separee 'Tornado'. Les tornades semblent incluses dans 'Storm'.")

# Question 2 : chaleur extreme
print("\nQuestion 2 : Les episodes de chaleur extreme deviennent-ils plus frequents ?")
analyser_tendance_categorie("Heat")

# Question 3 : froid extreme
print("\nQuestion 3 : Les episodes de froid extreme deviennent-ils plus frequents ?")
analyser_tendance_categorie("Cold, snow & ice")

# Question 4 : annees avec davantage d'evenements
print("\nQuestion 4 : Certaines annees presentent-elles davantage d'evenements extremes que d'autres ?")
evenements_par_annee = (
    df_clean
    .groupby("annee")
    .size()
    .reset_index(name="nombre_evenements")
    .sort_values("nombre_evenements", ascending=False)
)

print("\nTop 10 des annees avec le plus d'evenements :")
print(evenements_par_annee.head(10))

# Question 5 : correlation entre tempetes, chaleur et froid
print("\nQuestion 5 : Correlation entre tempetes, chaleur et froid")

categories_cibles = ["Storm", "Heat", "Cold, snow & ice"]
categories_disponibles = [
    categorie for categorie in categories_cibles
    if categorie in residus.columns
]

correlation_cible = residus[categories_disponibles].corr(method="spearman")

print("\nCorrelation de Spearman apres retrait de tendance :")
print(correlation_cible)

plt.figure(figsize=(10, 6))
sns.heatmap(
            correlation_spearman, 
            annot=True, 
            map="coolwarm", 
            center=0
)

plt.title("Une corrélation entre deux catégories d'événements ne prouve pas l'existence d'un lien causal. Deux catégories peuvent évoluer ensemble parce qu'elles sont influencées par une variable commune, comme l'amélioration des systèmes d'observation, l'augmentation du nombre d'études disponibles, ou une tendance climatique globale. Pour limiter cet effet, la tendance temporelle commune a été retirée avant de calculer les corrélations de Spearman.")
plt.tight_layout()
plt.show()