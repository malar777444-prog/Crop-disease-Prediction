
import streamlit as st
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Crop Disease Prediction AI",
    page_icon="🌱",
    layout="centered"
)

# ---------------- LANGUAGE SUPPORT ----------------

LANGUAGES = {
    "English": {
        "title": "Crop Disease Prediction AI",
        "subtitle": "Upload a leaf photo to identify a possible crop condition.",
        "upload": "Upload a crop leaf image",
        "analyze": "Analyze Leaf",
        "result": "Prediction Result",
        "confidence": "Model confidence",
        "recommendations": "Recommendations",
        "healthy": "The model predicts this leaf may be healthy.",
        "disease": "The model detected a possible disease.",
        "uncertain": "Low confidence. Please try a clearer photo.",
        "disclaimer": "AI screening only. This is not a confirmed diagnosis.",
        "loading": "Loading AI model. First run may take several minutes.",
        "analyzing": "Analyzing leaf image...",
        "top": "Other possible predictions",
        "error": "Prediction failed. Technical details are shown below."
    },
    "தமிழ்": {
        "title": "பயிர் நோய் கண்டறியும் AI",
        "subtitle": "பயிர் இலையின் புகைப்படத்தைப் பதிவேற்றவும்.",
        "upload": "இலைப் புகைப்படத்தைப் பதிவேற்றவும்",
        "analyze": "இலையை ஆய்வு செய்",
        "result": "கணிப்பு முடிவு",
        "confidence": "மாதிரியின் நம்பிக்கை",
        "recommendations": "பரிந்துரைகள்",
        "healthy": "இந்த இலை ஆரோக்கியமாக இருக்கலாம்.",
        "disease": "சாத்தியமான நோய் கண்டறியப்பட்டுள்ளது.",
        "uncertain": "நம்பிக்கை குறைவாக உள்ளது. தெளிவான படத்தை முயற்சிக்கவும்.",
        "disclaimer": "இது ஆரம்ப AI கணிப்பு மட்டுமே; உறுதிப்படுத்தப்பட்ட நோயறிதல் அல்ல.",
        "loading": "AI மாதிரி ஏற்றப்படுகிறது. சில நிமிடங்கள் ஆகலாம்.",
        "analyzing": "இலைப் படம் ஆய்வு செய்யப்படுகிறது...",
        "top": "மற்ற சாத்தியமான கணிப்புகள்",
        "error": "கணிப்பு தோல்வியடைந்தது. தொழில்நுட்ப விவரங்கள் கீழே உள்ளன."
    },
    "हिन्दी": {
        "title": "फसल रोग पहचान AI",
        "subtitle": "पत्ती की तस्वीर अपलोड करें।",
        "upload": "पत्ती की तस्वीर अपलोड करें",
        "analyze": "पत्ती का विश्लेषण करें",
        "result": "अनुमान का परिणाम",
        "confidence": "मॉडल का विश्वास",
        "recommendations": "सुझाव",
        "healthy": "यह पत्ती स्वस्थ हो सकती है।",
        "disease": "संभावित रोग पहचाना गया है।",
        "uncertain": "विश्वास कम है। एक स्पष्ट तस्वीर आज़माएँ।",
        "disclaimer": "यह केवल प्रारंभिक AI जाँच है, पक्की पुष्टि नहीं।",
        "loading": "AI मॉडल लोड हो रहा है। कुछ मिनट लग सकते हैं।",
        "analyzing": "पत्ती की तस्वीर का विश्लेषण हो रहा है...",
        "top": "अन्य संभावित अनुमान",
        "error": "अनुमान विफल हुआ। तकनीकी विवरण नीचे हैं।"
    }
}

language = st.selectbox(
    "Choose Language / மொழி / भाषा",
    list(LANGUAGES.keys())
)

t = LANGUAGES[language]

st.title("🌱 " + t["title"])
st.write(t["subtitle"])

st.info(
    "This prototype classifies visible leaf conditions. "
    "It cannot reliably predict disease before symptoms appear."
)

# ---------------- LOAD AI MODEL ----------------

@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "image-classification",
        model="Kathir56/plant-disease-tamilnadu"
    )

# ---------------- RECOMMENDATIONS ----------------

def get_recommendations(label):
    label = label.lower()

    if "healthy" in label:
        return [
            "Continue regular crop monitoring.",
            "Water according to the crop's needs.",
            "Inspect leaves regularly for new symptoms."
        ]

    return [
        "Inspect both sides of the affected leaf.",
        "Check nearby plants for similar symptoms.",
        "Avoid spreading potentially infected plant material.",
        "Do not apply pesticides based only on this AI result.",
        "Consult a local agricultural expert for confirmation."
    ]

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    t["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        st.success("Image uploaded successfully!")

        if st.button(t["analyze"], type="primary"):
            try:
                with st.spinner(t["loading"]):
                    model = load_model()

                with st.spinner(t["analyzing"]):
                    predictions = model(image, top_k=3)

                if not predictions:
                    st.error("The model returned no predictions.")
                else:
                    best = predictions[0]
                    label = best["label"]
                    confidence = float(best["score"])

                    readable_label = (
                        label.replace("___", " - ")
                        .replace("_", " ")
                    )

                    st.divider()
                    st.header("🔍 " + t["result"])

                    st.subheader(readable_label)
                    st.write(
                        f"**{t['confidence']}: {confidence:.1%}**"
                    )
                    st.progress(
                        max(0.0, min(1.0, confidence))
                    )

                    if confidence < 0.60:
                        st.warning(t["uncertain"])
                    elif "healthy" in label.lower():
                        st.success(t["healthy"])
                    else:
                        st.warning(t["disease"])

                    st.subheader("💡 " + t["recommendations"])

                    for recommendation in get_recommendations(label):
                        st.write("• " + recommendation)

                    with st.expander(t["top"]):
                        for prediction in predictions:
                            name = (
                                prediction["label"]
                                .replace("___", " - ")
                                .replace("_", " ")
                            )
                            score = float(prediction["score"])
                            st.write(f"{name}: {score:.1%}")

                    st.warning(t["disclaimer"])

            except Exception as error:
                st.error(t["error"])
                st.error(
                    f"{type(error).__name__}: {error}"
                )
                st.exception(error)

    except (UnidentifiedImageError, OSError) as error:
        st.error("Please upload a valid JPG or PNG image.")
        st.exception(error)

st.divider()
st.caption("Educational prototype | AI-assisted crop screening")
