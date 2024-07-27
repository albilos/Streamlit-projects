import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Charger le modèle YOLO pré-entraîné
model = YOLO('yolov8n.pt')  # Utilise le modèle YOLOv8 nano

# Titre de l'application
st.title("Détection d'Objets dans les Images")

# Section de téléchargement d'image
st.subheader("Téléchargez une Image")
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir le fichier téléchargé en une image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    # Convertir l'image en format compatible avec OpenCV
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Faire la détection d'objets
    results = model(image)

    # Dessiner les boîtes de détection sur l'image
    detected_image = image.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            
            # Dessiner la boîte
            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Ajouter le label
            cv2.putText(detected_image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convertir l'image détectée en format RGB
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    st.image(detected_image, caption='Image avec Détection', use_column_width=True)

    # Afficher les résultats de la détection
    st.subheader("Résultats de la Détection")
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            st.write(f"Objet: {label}, Confiance: {confidence:.2f}")
