import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymcdm.methods import TOPSIS
from pymcdm.helpers import rrankdata

# affichage de figures
plot = 0

if plot == 1:

    # Données RÉELLES basées sur sources fiables (2025)
    data_vendee = {
        'Localisation': ['Bouin (Production)', 'Maché (Station)', 'La Roche-sur-Yon (Station)',
                         'Les Sables d\'Olonne (Station)', 'Saint-Gilles-Croix-de-Vie (Station)',
                         'Challans (Station - projet)', 'Les Achards (Station)'],
        'Type': ['Production H2', 'Station H2', 'Station H2', 'Station H2', 'Station H2',
                 'Station H2 (projet)', 'Station H2'],
        'Capacité': [300, 800, 300, 300, 300, 300, 800],  # kg/jour
        'Statut': ['Active', 'Active', 'Active', 'Active', 'Active', 'En projet', 'Active'],
        'Latitude': [46.85, 46.70, 46.67, 46.49, 46.70, 46.84, 46.65],  # Coordonnées approx
        'Longitude': [-1.78, -1.92, -1.43, -1.78, -1.93, -1.85, -1.50]
    }

    df_vendee = pd.DataFrame(data_vendee)
    print("🗺️ Installations H2 Vendée (données réelles 2025) :\n", df_vendee)

    # 1. Carte schématique Vendée (positions relatives)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Carte simplifiée Vendée
    x_coords = df_vendee['Longitude'].values
    y_coords = df_vendee['Latitude'].values

    prod_mask = df_vendee['Type'] == 'Production H2'
    stations_mask = df_vendee['Type'] == 'Station H2'

    ax1.scatter(x_coords[prod_mask], y_coords[prod_mask], s=800, c='cyan', marker='s',
                label='Production (Bouin)', edgecolor='darkblue', linewidth=2)
    ax1.scatter(x_coords[stations_mask], y_coords[stations_mask], s=400, c='orange', marker='o',
                label='Stations actives', edgecolor='red', linewidth=2)
    ax1.scatter(x_coords[-1], y_coords[-1], s=400, c='lightgreen', marker='^',
                label='Les Achards', edgecolor='darkgreen')

    ax1.set_title('🗺️ Stations & Production H2 - Vendée (2025)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for i, row in df_vendee.iterrows():
        ax1.annotate(row['Localisation'][:12], (row['Longitude'], row['Latitude']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 2. Capacité production/distribution
    ax2.bar(df_vendee['Localisation'][:5], df_vendee['Capacité'][:5],
            color=['cyan'] + ['orange'] * 4, alpha=0.8)
    ax2.set_title('Capacité journalière (kg/jour)')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df_vendee['Capacité'][:5]):
        ax2.text(i, v + 20, f'{v}', ha='center')

    # 3. Statut des installations
    statut_counts = df_vendee['Statut'].value_counts()
    ax3.pie(statut_counts.values, labels=statut_counts.index, autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral'])
    ax3.set_title('Statut installations H2 Vendée')

    # 4. Réseau H2 Vendée (flux Bouin → Stations)
    ax4.barh(['Production Bouin', 'Stations Total'],
             [df_vendee['Capacité'][0], df_vendee['Capacité'][1:].sum()],
             color=['cyan', 'orange'])
    ax4.set_title('Flux H2 : Production vs Distribution')
    for i, v in enumerate([df_vendee['Capacité'][0], df_vendee['Capacité'][1:].sum()]):
        ax4.text(v + 20, i, f'{v} kg/jour', va='center')

    plt.tight_layout()
    plt.show()

    # Tableau récapitulatif
    print("\n📋 RÉCAPITULATIF VENDÉE H2 :")
    print(df_vendee[['Localisation', 'Type', 'Capacité', 'Statut']].to_string(index=False))

    print(
        f"\n🎯 TOTAL : {df_vendee['Capacité'].sum():.0f} kg/j = {df_vendee['Capacité'].sum() * 365 / 1000:.0f} tonnes/an")
    print(f"   • Production : {df_vendee.loc[0, 'Capacité']} kg/j (Bouin)")
    print(f"   • Stations actives : 5/6 ({df_vendee['Capacité'][1:6].sum():.0f} kg/j)")


else:

    # =========================
    # 5. Analyse multicritère MCDA (TOPSIS)
    # =========================

    # a) Définir les alternatives (scénarios énergétiques H2)
    alternatives = [
        "Scénario 1 : H2 100% Vendée",
        "Scénario 2 : H2 Vendée + import régional",
        "Scénario 3 : Import H2 national/UE"
    ]

    # b) Définir les critères (exemple : coût, GES, emploi, acceptabilité)
    # IMPORTANT : les valeurs ci-dessous sont des valeurs fictives pour illustrer la méthode.
    # Adapte-les avec tes données réelles plus tard.
    # Matrice de décision : lignes = alternatives, colonnes = critères
    # Critères:
    #   C1 = Coût actualisé du H2 (€/kg) -> à MINIMISER
    #   C2 = Émissions de GES sur le cycle de vie (kgCO2e/kg H2) -> à MINIMISER
    #   C3 = Emplois locaux créés (équivalents temps plein) -> à MAXIMISER
    #   C4 = Acceptabilité locale (note 1–5) -> à MAXIMISER

    decision_matrix = np.array([
        [6.0, 1.0, 120, 4.5],   # Scénario 1
        [5.0, 1.5,  80, 4.0],   # Scénario 2
        [4.0, 3.0,  30, 3.0],   # Scénario 3
    ], dtype=float)

    # c) Poids des critères (somme = 1)
    # Exemple : coût 30 %, GES 35 %, emploi 20 %, acceptabilité 15 %
    weights = np.array([0.30, 0.35, 0.20, 0.15], dtype=float)

    # d) Type de critère : 1 = à maximiser, -1 = à minimiser
    types = np.array([-1, -1, 1, 1], dtype=int)

    # e) Application de TOPSIS avec PyMCDM
    topsis = TOPSIS()
    preferences = topsis(decision_matrix, weights, types)
    ranking = rrankdata(preferences)  # 1 = meilleure option

    # f) Affichage des résultats
    print("\n🔎 Résultats MCDA (TOPSIS) – Scénarios H2 :")
    for alt, pref, rank in zip(alternatives, preferences, ranking):
        print(f"  Rang {int(rank)} | Score = {pref:.3f} | {alt}")