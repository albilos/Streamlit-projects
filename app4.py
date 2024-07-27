import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Prediction des prix des voitures en fonction des caracteristiques")
st.subheader("realisé par billo")
st.markdown("*** cette application uitilise le modèle de machine learning pour predire le prix de la voiture***")
# chargement du modèle
model = joblib.load(filename="model_final.joblib")

# function d'inference 
def inference(symboling,wheel_base,length,width,height,curb_weight,engine_size,compression_ratio,city_mpg,highway_mpg):
    new_data = np.array([
        symboling,wheel_base,length,width,height,
        curb_weight,engine_size,compression_ratio,
        city_mpg,highway_mpg
    ])
    pred = model.predict(new_data.reshape(1,-1))
    return pred
    

# saisir les caracteristique de chaque voiture
symboling = st.number_input(label="symboling:", min_value=0,value=3)
wheel_base = st.number_input("wheel-base:", value=97)
length     = st.number_input("length:", value=173)
width      = st.number_input("width:", value=65)
height     = st.number_input("height:",value=54)
curb_weight = st.number_input("curb-weight:",value=2414)
engine_size = st.number_input("engine-size:",value=120)
compression_ratio = st.number_input("compression-ratio",value=9)
city_mpg          = st.number_input("city-mpg:",value=24)
highway_mpg       = st.number_input("highway-mpg",value=30)

# creation du button de prediction

if st.button("predict"):
    prediction = inference(
        symboling,wheel_base,length,width,
        height,curb_weight,engine_size,compression_ratio,city_mpg,highway_mpg
        
    )
    resultat = " le prix de la voiture en dollar est: " +str(prediction[0])
    st.success(resultat)





