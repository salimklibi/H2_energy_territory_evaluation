# Analyse MCDA et Prédiction Hydrogène Pays de la Loire / Vendée

Ce repo implémente une prédiction de production H2 régionale via régression linéaire (H2_paysdelaloire.py) et une analyse MCDA (TOPSIS, PROMETHEE, SAW) pour scénarios H2 en Vendée (H2_vendee.py), avec focus sur facteurs comme potentiel ENR, acceptabilité et infrastructures.


## Installation

Clonez le repo : git clone https://github.com/votreusername/h2-paysdelaloire-vendee.git

Créez un environnement Conda : conda create -n h2-analysis python=3.10

Installez les dépendances : pip install pandas numpy scikit-learn matplotlib seaborn pymcdm.
​

# Utilisation

H2_paysdelaloire.py : Exécutez python H2_paysdelaloire.py pour générer prédictions H2, classements régionaux, visualisations (barres, radar) et focus Vendée.
​
H2_vendee.py : Lancez python H2_vendee.py (définissez METHOD=TOPSIS ou autre) pour rankings MCDA de scénarios H2 locaux vs imports.
​
