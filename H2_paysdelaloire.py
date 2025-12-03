import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Données fictives par région (échelles normalisées 0-100)
data = {
    'Région': ['Hauts-de-France', 'Grand Est', 'Nouvelle-Aquitaine', 'Occitanie',
               'PACA', 'Pays de la Loire', 'Bretagne', 'Normandie'],
    'Potentiel_ENR': [85, 78, 92, 88, 76, 82, 79, 74],  # Éolien + solaire
    'Acceptabilité': [65, 72, 68, 70, 62, 75, 71, 69],   # Soutien local %
    'Infrastructures': [80, 75, 82, 78, 85, 70, 73, 77], # Réseaux existants
    'Population': [6000, 5500, 6000, 5800, 5100, 3800, 3300, 3300], # milliers
    'Vulnérabilité_env': [45, 52, 38, 42, 55, 40, 48, 50] # Impact écologique
}

df = pd.DataFrame(data)
print("📊 Données d'indicateurs régionaux :\n", df.round(1))

# 2. Production H2 potentielle réelle (référence fictive TWh/an)
df['Production_H2_reelle'] = [12.5, 11.2, 14.8, 13.1, 10.9, 9.8, 9.2, 8.7]

# 3. Modèle de prédiction
X = df[['Potentiel_ENR', 'Acceptabilité', 'Infrastructures', 'Population', 'Vulnérabilité_env']]
y = df['Production_H2_reelle']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Prédictions
df['Production_H2_predite'] = model.predict(X_scaled)
df['Classement'] = df['Production_H2_predite'].rank(ascending=False).astype(int)

print("\n🔮 Prédictions par région :\n", df[['Région', 'Production_H2_reelle', 'Production_H2_predite', 'Classement']].round(2))

# 4. Focus Vendée (Pays de la Loire)
vendee_idx = df[df['Région'] == 'Pays de la Loire'].index[0]
print(f"\n🎯 VENDÉE (Pays de la Loire) :")
print(f"   Production réelle prédite : {df.loc[vendee_idx, 'Production_H2_predite']:.1f} TWh/an")
print(f"   Classement national : {df.loc[vendee_idx, 'Classement']}e / {len(df)}")
print(f"   Score composite : {df.loc[vendee_idx, ['Potentiel_ENR', 'Acceptabilité']].mean():.1f}/100")

# 5. Visualisations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Classement régions
colors = ['gold' if r == 'Pays de la Loire' else 'lightblue' for r in df['Région']]
bars = ax1.barh(df['Région'], df['Production_H2_predite'], color=colors, alpha=0.8)
ax1.set_title('🏆 Classement Production H₂ Prédite par Région')
ax1.set_xlabel('Production potentielle (TWh/an)')
for i, v in enumerate(df['Production_H2_predite']):
    ax1.text(v + 0.1, i, f'{v:.1f}', va='center')

# Facteurs clés Vendée
ax2.bar(df.columns[1:6], df.iloc[vendee_idx, 1:6], color='orange', alpha=0.7)
ax2.set_title('📈 Profil Vendée - Facteurs clés')
ax2.set_ylabel('Score (0-100)')
ax2.tick_params(axis='x', rotation=45)

# Corrélation indicateurs vs production
corr = df.corr(numeric_only=True)['Production_H2_predite'].sort_values(ascending=False)
ax3.barh(corr.index[1:], corr.values[1:], color='green', alpha=0.7)
ax3.set_title('📊 Influence des facteurs sur la production H₂')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)

# Radar Vendée vs Moyenne nationale
categories = ['ENR', 'Accept.', 'Infra.', 'Pop.', 'Vulnér.']
values_vendee = df.iloc[vendee_idx, 1:6].values
values_moy = df.iloc[:, 1:6].mean().values

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
values_vendee = np.concatenate((values_vendee, [values_vendee[0]]))
values_moy = np.concatenate((values_moy, [values_moy[0]]))
angles += angles[:1]

ax4.plot(angles, values_vendee, 'o-', linewidth=2, label='Vendée', color='orange')
ax4.fill(angles, values_vendee, alpha=0.25, color='orange')
ax4.plot(angles, values_moy, 'o-', linewidth=2, label='Moyenne France', color='gray')
ax4.fill(angles, values_moy, alpha=0.25, color='gray')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_title('🎯 Comparaison Vendée vs Moyenne Nationale')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

print("\n✅ Modèle créé ! Vendée : 6e position avec bon potentiel ENR + acceptabilité.")
print("Coefficients du modèle :", dict(zip(['ENR','Accept','Infra','Pop','Vuln'], model.coef_.round(2))))
