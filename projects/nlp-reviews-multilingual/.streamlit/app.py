"""Streamlit demo for the multilingual review classifier."""

import streamlit as st

from src.predict import Predictor


st.set_page_config(
    page_title="Multilingual Review Classifier",
    page_icon=None,
    layout="centered",
)

st.title("Multilingual Review Sentiment Classifier")
st.caption("DistilBERT multilingue fine-tuned on Amazon Reviews Multi (ES + EN)")

with st.sidebar:
    st.markdown("### Acerca de / About")
    st.markdown(
        "Modelo entrenado en resenas reales. Clasifica sentimiento "
        "(positivo, neutral, negativo) en espanol o ingles."
    )
    st.markdown(
        "Trained on real reviews. Classifies sentiment (positive, neutral, "
        "negative) in Spanish or English."
    )
    st.markdown("Ver `theory/` en el repo para detalles tecnicos.")


@st.cache_resource
def load_predictor():
    return Predictor()


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "Model not found. Run `python -m src.train` first to fine-tune the "
        "transformer, then reload this page."
    )
    st.stop()


col_es, col_en = st.columns(2)
with col_es:
    st.markdown("**Ejemplo (ES):**")
    if st.button("Cargar ejemplo en espanol"):
        st.session_state["input_text"] = (
            "Me encanto el producto, llego rapido y funciona perfecto."
        )
with col_en:
    st.markdown("**Example (EN):**")
    if st.button("Load English example"):
        st.session_state["input_text"] = (
            "The phone is great, but the charger died after a week."
        )

text = st.text_area(
    "Pega una resena / Paste a review",
    value=st.session_state.get("input_text", ""),
    height=150,
)

if text.strip():
    with st.spinner("Clasificando..."):
        result = predictor.predict(text)

    emoji = {
        "positive": "[+]",
        "neutral": "[=]",
        "negative": "[-]",
    }[result["label"]]
    st.markdown(f"### {emoji} {result['label'].upper()}")
    st.markdown(f"**Confianza / Confidence:** {result['confidence']:.3f}")

    st.markdown("**Distribucion de probabilidades:**")
    st.bar_chart(result["probs"])
else:
    st.info("Ingresa una resena arriba / Enter a review above to classify.")
