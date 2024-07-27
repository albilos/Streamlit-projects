import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Fonction pour charger les données
@st.cache
def load_data():
    data = pd.read_csv('sales_data.csv')
    return data

# Chargement des données
data = load_data()

# Titre de l'application
st.title("Tableau de Bord des Ventes")

# Sous-titre de l'application
st.markdown("## Visualisation des Ventes de Produits")

# Section des données brutes
st.markdown("### Données Brutes")
st.write(data)

# Section de sélection de produit
st.markdown("### Sélection de Produit")
product = st.selectbox('Sélectionnez un produit:', data['Product'].unique())

# Filtrer les données en fonction du produit sélectionné
filtered_data = data[data['Product'] == product]

# Section des visualisations
st.markdown(f"### Visualisation des Ventes pour {product}")
plt.figure(figsize=(10, 5))
plt.plot(filtered_data['Date'], filtered_data['Sales'], marker='o')
plt.title(f'Ventes mensuelles de {product}')
plt.xlabel('Date')
plt.ylabel('Ventes')
plt.grid(True)
st.pyplot(plt)

# Conclusion
st.markdown("### Conclusion")
st.write("Cette application permet de visualiser les ventes mensuelles de chaque produit. Sélectionnez un produit dans le menu déroulant pour voir le graphique des ventes correspondantes.")

# Conclusion
st.markdown("### Conclusion")
st.write("Cette application permet de visualiser les ventes mensuelles de chaque produit. Sélectionnez un produit dans le menu déroulant pour voir le graphique des ventes correspondantes.")
