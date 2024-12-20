import streamlit as st
import joblib
import pdfplumber

# Cargar modelo y vectorizador
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Crear interfaz de usuario
st.title("Predicción de Montos en Actas PDF")

# Subida de archivo
uploaded_file = st.file_uploader("Sube un archivo PDF con el acta", type="pdf")
if uploaded_file is not None:
    acta_text = extract_text_from_pdf(uploaded_file)
    st.write("Texto extraído del acta:")
    st.text_area("", acta_text, height=200)

    # Convertir texto a representación TF-IDF
    acta_tfidf = vectorizer.transform([acta_text])
    predicted_value = model.predict(acta_tfidf)[0]

    # Mostrar el monto predicho
    st.write(f"**Monto predicho:** {predicted_value:.2f}")
    