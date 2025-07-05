import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications import ConvNeXtLarge, MobileNetV3Large
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# === Mapping label ke nama kelas ===
class_labels = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Back",
    4: "Gall Midge",
    5: "Healthy",
    6: "Powdery Mildew",
    7: "Sooty Mould"
}

# === Load model CNN pretrained tanpa top ===
@st.cache_resource
def load_feature_models():
    convnext_base = ConvNeXtLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    mobilenetv3_base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return (
        Model(inputs=convnext_base.input, outputs=convnext_base.output),
        Model(inputs=mobilenetv3_base.input, outputs=mobilenetv3_base.output)
    )

# === Load model Random Forest ===
@st.cache_resource
def load_rf_model():
    return joblib.load("random_forest_model.pkl")  # ganti dengan path modelmu

# === Ekstraksi fitur dari satu gambar ===
def extract_features_from_image(img_array, conv_model, mob_model):
    img_array_expanded = np.expand_dims(img_array, axis=0)
    features_conv = conv_model.predict(convnext_preprocess(img_array_expanded))
    features_mob = mob_model.predict(mobilenetv3_preprocess(img_array_expanded))
    
    flat_conv = features_conv.reshape((1, -1))
    flat_mob = features_mob.reshape((1, -1))
    
    return np.concatenate([flat_conv, flat_mob], axis=1)

# === UI Streamlit ===
st.title("🌿 Deteksi Penyakit Daun Mangga")
st.write("Gunakan model CNN + Random Forest untuk memprediksi jenis penyakit tanaman mangga berdasarkan gambar daun.")

uploaded_file = st.file_uploader("📤 Unggah gambar daun...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='📷 Gambar yang Diunggah', use_column_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang memproses..."):
            # Preprocessing
            image_resized = image.resize((224, 224))
            img_array = np.array(image_resized).astype(np.float32)

            # Load models
            conv_model, mob_model = load_feature_models()
            rf_model = load_rf_model()

            # Extract features and predict
            features = extract_features_from_image(img_array, conv_model, mob_model)
            prediction = rf_model.predict(features)
            proba = rf_model.predict_proba(features)[0]

            predicted_class = prediction[0]
            predicted_label = class_labels[predicted_class]

            # Output
            st.success(f"Hasil Prediksi: **{predicted_label}** (Kelas {predicted_class})")

            # Show class probabilities
            st.subheader("📊 Probabilitas Tiap Kelas:")
            fig, ax = plt.subplots()
            class_names = list(class_labels.values())
            ax.barh(class_names, proba, color="skyblue")
            ax.set_xlabel("Probabilitas")
            ax.set_ylabel("Kelas")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
