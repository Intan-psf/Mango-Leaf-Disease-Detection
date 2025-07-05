import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ConvNeXtLarge, MobileNetV3Large
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Load Model CNN untuk Ekstraksi Fitur ===
convnext_base = ConvNeXtLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
convnext_model = Model(inputs=convnext_base.input, outputs=convnext_base.output)

mobilenetv3_base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenetv3_model = Model(inputs=mobilenetv3_base.input, outputs=mobilenetv3_base.output)

# === Load Model Random Forest ===
rf = joblib.load('random_forest_model.pkl')  # pastikan file ini sudah kamu simpan

# === Label Kelas Penyakit Daun Mangga ===
label_map = {
    0: "Anthracnose",
    1: "Bacterial Canker",
    2: "Cutting Weevil",
    3: "Die Back",
    4: "Gall Midge",
    5: "Healthy",
    6: "Powdery Mildew",
    7: "Sooty Mould"
}

# === Streamlit App ===
st.title("Prediksi Penyakit Daun Mangga")
st.markdown("Unggah gambar daun mangga untuk mendeteksi jenis penyakit menggunakan model gabungan ConvNeXt + MobileNetV3 + Random Forest.")

uploaded_file = st.file_uploader("Unggah gambar daun mangga (format JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # === Preprocess Gambar ===
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # === Ekstraksi Fitur dari Dua CNN ===
    conv_feat = convnext_model.predict(convnext_preprocess(img_array))
    mobv3_feat = mobilenetv3_model.predict(mobilenetv3_preprocess(img_array))

    # === Gabungkan Fitur ===
    conv_feat_flat = conv_feat.reshape(conv_feat.shape[0], -1)
    mobv3_feat_flat = mobv3_feat.reshape(mobv3_feat.shape[0], -1)
    combined_feat = np.concatenate([conv_feat_flat, mobv3_feat_flat], axis=1)

    # === Prediksi dengan Random Forest ===
    prediction = rf.predict(combined_feat)
    predicted_label = label_map.get(prediction[0], "Tidak diketahui")

    st.success(f"**Hasil Prediksi:** {predicted_label}")
