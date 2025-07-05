# 🍃 Mango-Leaf-Detection

This is a simple Streamlit web app for detecting mango leaf diseases using an ensemble of feature extractors (ConvNeXtLarge, MobileNetV3Large) and a Random Forest Classifier.

🔍 The app predicts the type of disease on an uploaded leaf image and displays the prediction along with its probability score.

---

## 📂 Project Structure

```
├── main.py         # Streamlit app main script
├── app.py          # (Optional) Additional app script
├── dashboard.py    # (Optional) Dashboard or additional visualization
├── rf_model.pkl    # Pre-trained Random Forest model
├── .gitignore      # Ignore large model files (.h5)
├── README.md       # Project description
```

⚡ **Note:**

- The feature extractor files (.h5) are large and not tracked by Git.
- You may download them separately if needed and place them in this project folder.

---

## 🧩 How It Works

- The app utilizes pre-trained ConvNeXtLarge and MobileNetV3Large models as feature extractors on resized input images (224x224).
- Extracted features from both networks are concatenated and then classified using a Random Forest model (`rf_model.pkl`).
- Class prediction indices are mapped to disease labels using an inline Python dictionary (`class_labels`) inside `main.py`.
- 👉 No external `labels.txt` file is needed since all label mappings are defined directly in the script for easier maintenance.

---

## 👥 Collaborators

**Muhammad Rizky Albani**  
Practicum: SVM classification experiment (initial trial, not used in final model)  
Report: Introduction  

**Innayatul Laili Husnaini**  
Practicum: Feature combination using ConvNeXt extractor  
Report: Problem formulation, methodology explanation (tools and techniques); also served as report coordinator  

**Intan Permata Sari Fauziah**  
Practicum: Feature combination using MobileNetV3 and combined ConvNeXt + MobileNetV3 extractors  
Report: Method section (data sources, data splitting, feature extraction process, model evaluation)  

**Anita Hasna Zahira Safa**  
Practicum: Preprocessing steps (rescaling, normalization, rotation augmentations)  
Report: Introduction  

**Raihan Muhammad Nafi’**  
Practicum: Additional augmentations (zoom, horizontal flip, vertical flip)  
Report: Results and discussion section  

**Nafisa Salsabila**  
Practicum: Feature extraction and model training with Random Forest  
Report: Method section (data preprocessing, feature extraction, model training process, and tools used)

---

✨ **We welcome feedback and contributions to improve this project further!**
