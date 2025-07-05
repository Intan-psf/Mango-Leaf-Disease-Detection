import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ConvNeXtLarge, MobileNetV3Large
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from tensorflow.keras.models import Model
import joblib
import matplotlib.pyplot as plt

# Label kelas (ganti sesuai datasetmu)
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

@st.cache_resource
def load_feature_models():
    convnext_base = ConvNeXtLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    mobilenetv3_base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return (
        Model(inputs=convnext_base.input, outputs=convnext_base.output),
        Model(inputs=mobilenetv3_base.input, outputs=mobilenetv3_base.output)
    )

@st.cache_resource
def load_rf_model():
    return joblib.load("rf_model.pkl")

def extract_features(img_array, conv_model, mob_model):
    img_array_expanded = np.expand_dims(img_array, axis=0)
    features_conv = conv_model.predict(convnext_preprocess(img_array_expanded))
    features_mob = mob_model.predict(mobilenetv3_preprocess(img_array_expanded))
    return np.concatenate([features_conv.reshape(1, -1), features_mob.reshape(1, -1)], axis=1)

st.set_page_config(page_title="Deteksi Penyakit Daun Mangga", layout="wide")
st.sidebar.title("🍃 Deteksi Daun Mangga")
st.sidebar.write("Unggah gambar daun untuk dideteksi jenis penyakitnya.")

uploaded_file = st.sidebar.file_uploader("📤 Unggah gambar daun", type=["jpg", "jpeg", "png"])


if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized).astype(np.float32)

    conv_model, mob_model = load_feature_models()
    rf_model = load_rf_model()

    features = extract_features(img_array, conv_model, mob_model)
    st.write("Feature shape:", features.shape)
    st.write("Feature sample (first 10 values):", features[0][:10])

    prediction = rf_model.predict(features)
    proba = rf_model.predict_proba(features)[0]

    st.write("Predicted class index:", prediction[0])
    st.write("Probabilities:", proba)

    predicted_class = prediction[0]
    predicted_label = class_labels.get(predicted_class, "Unknown")

    # ... lanjut tampilan hasil ...



    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="📷 Gambar Daun", use_container_width=True)


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

        st.markdown("#### ℹ️ Informasi Singkat")
        st.info(f"Jenis penyakit **{predicted_label}** dapat mempengaruhi pertumbuhan daun mangga. Perhatikan gejala visual dan lakukan penanganan sesuai saran ahli tanaman.")
else:
    st.markdown("## Selamat datang di Dashboard Deteksi Penyakit Daun Mangga 🍃")
    st.markdown("Silakan unggah gambar daun melalui sidebar untuk memulai prediksi penyakit.")
