import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Charger les données Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Développer le modèle
clf = RandomForestClassifier()
clf.fit(X, y)

# Titre de l'application
st.title("Classification des Fleurs Iris")

# Sidebar pour les paramètres de l'utilisateur
st.sidebar.header("Paramètres d'entrée")

def user_input_features():
    sepal_length = st.sidebar.slider('Longueur du sépale', float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Largeur du sépale', float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Longueur du pétale', float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Largeur du pétale', float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))
    
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Afficher les paramètres d'entrée
st.subheader("Paramètres d'entrée")
st.write(input_df)
# Prédictions
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Afficher les résultats
st.subheader("Résultats de la Prédiction")
st.write(f"Classe prédite : {iris.target_names[prediction][0]}")
st.write("Probabilités de la Prédiction :")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

# Visualiser les données
st.subheader("Visualisation des Données")

# Sélectionner les classes à visualiser
options = st.multiselect(
    'Choisissez les classes à visualiser',
    iris.target_names,
    iris.target_names)

# Filtrer les données
filtered_data = X[y.isin([iris.target_names.tolist().index(option) for option in options])]
filtered_target = y[y.isin([iris.target_names.tolist().index(option) for option in options])]

# Afficher le scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(filtered_data.iloc[:, 0], filtered_data.iloc[:, 1], c=filtered_target, cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
st.pyplot(fig)