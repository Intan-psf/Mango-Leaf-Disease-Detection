import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# Label kelas
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

# Load model CNN dari file .h5
@st.cache_resource
def load_feature_models():
    convnext_model = load_model("convnext_feature_extractor.h5", compile=False)
    mobilenet_model = load_model("mobilenetv3_feature_extractor.h5", compile=False)
    return convnext_model, mobilenet_model

# Load model Random Forest dari .pkl
@st.cache_resource
def load_rf_model():
    return joblib.load("rf_model.pkl")

# Ekstraksi fitur
def extract_features(img_array, conv_model, mob_model):
    img_array_expanded = np.expand_dims(img_array, axis=0)
    features_conv = conv_model.predict(convnext_preprocess(img_array_expanded))
    features_mob = mob_model.predict(mobilenetv3_preprocess(img_array_expanded))
    return np.concatenate([features_conv.reshape(1, -1), features_mob.reshape(1, -1)], axis=1)

# ============================== DASHBOARD UI ============================== #
st.set_page_config(page_title="Deteksi Penyakit Daun Mangga", layout="wide")
st.sidebar.title("🍃 Deteksi Daun Mangga")
st.sidebar.write("Unggah gambar daun untuk dideteksi jenis penyakitnya.")

# Upload Gambar
uploaded_file = st.sidebar.file_uploader("📤 Unggah gambar daun", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized).astype(np.float32)

    # Load models dari file
    conv_model, mob_model = load_feature_models()
    rf_model = load_rf_model()

    # Ekstrak fitur dan prediksi
    features = extract_features(img_array, conv_model, mob_model)
    prediction = rf_model.predict(features)
    proba = rf_model.predict_proba(features)[0]

    predicted_class = prediction[0]
    predicted_label = class_labels[predicted_class]

    # ===================== LAYOUT TAMPILAN ===================== #
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="📷 Gambar Daun", use_column_width=True)

    with col2:
        st.markdown("### 🧠 Hasil Prediksi")
        st.success(f"**{predicted_label}** (Kelas {predicted_class})")
        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        fig, ax = plt.subplots()
        ax.barh(list(class_labels.values()), proba, color="lightgreen")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.invert_yaxis()
        st.pyplot(fig)

        # (Opsional) Info penyakit
        st.markdown("#### ℹ️ Informasi Singkat")
        st.info(f"Jenis penyakit **{predicted_label}** dapat mempengaruhi pertumbuhan daun mangga. Perhatikan gejala visual dan lakukan penanganan sesuai saran ahli tanaman.")

else:
    st.markdown("## Selamat datang di Dashboard Deteksi Penyakit Daun Mangga 🍃")
    st.markdown("Silakan unggah gambar daun melalui sidebar untuk memulai prediksi penyakit.")
