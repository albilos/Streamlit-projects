import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Charger le modèle YOLO pré-entraîné
model = YOLO('yolov8n.pt')  # Utilise le modèle YOLOv8 nano

# Titre de l'application
st.title("Détection d'Objets dans les Images et Vidéos")

# Sidebar pour sélectionner le mode
st.sidebar.header("Mode de détection")
mode = st.sidebar.selectbox("Choisissez le mode", ["Image", "Webcam", "Batch"])

# Détection d'objets dans une image
if mode == "Image":
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

# Détection en temps réel via la webcam
elif mode == "Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            
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

            return detected_image

    st.subheader("Détection en temps réel via Webcam")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Traitement par lots d'images
elif mode == "Batch":
    st.subheader("Téléchargez un Dossier d'Images")
    uploaded_files = st.file_uploader("Choisissez des images...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files is not None and len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            # Convertir le fichier téléchargé en une image
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Image téléchargée: {uploaded_file.name}', use_column_width=True)
            
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
            st.image(detected_image, caption=f'Image avec Détection: {uploaded_file.name}', use_column_width=True)

            # Afficher les résultats de la détection
            st.subheader(f"Résultats de la Détection pour {uploaded_file.name}")
            for result in results:
                for box in result.boxes:
                    label = model.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    st.write(f"Objet: {label}, Confiance: {confidence:.2f}")
