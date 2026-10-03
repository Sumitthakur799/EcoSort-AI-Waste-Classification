import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(16,185,129,0.08), transparent 25%),
        radial-gradient(circle at 95% 10%, rgba(34,197,94,0.06), transparent 25%),
        #f6faf8;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* HERO */

.hero {
    background: linear-gradient(
        135deg,
        #052e1b 0%,
        #064e3b 55%,
        #087f5b 100%
    );

    border-radius: 28px;
    padding: 45px 25px;
    text-align: center;
    margin-bottom: 35px;

    box-shadow: 0 18px 45px rgba(6,78,59,0.20);
}

.hero-icon {
    font-size: 58px;
    margin-bottom: 8px;
}

.hero-title {
    color: white;
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #c7f9d4;
    font-size: 18px;
    margin-top: 8px;
}


/* HEADINGS */

h2, h3 {
    color: #12372a !important;
    font-weight: 750 !important;
}


/* CATEGORY CARDS */

.category-box {
    background: white;
    border: 1px solid #d9eee1;
    border-radius: 15px;
    padding: 16px 10px;
    text-align: center;
    color: #17633f;
    font-weight: 650;

    box-shadow: 0 5px 18px rgba(15,23,42,0.05);

    transition: 0.2s ease;
}

.category-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(15,23,42,0.08);
}


/* UPLOAD */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #b8ddc7;
    border-radius: 20px;
    padding: 15px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #10b981;
    background: #f8fffa;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 13px 20px;

    font-size: 16px;
    font-weight: 700;

    color: white;

    background: linear-gradient(
        135deg,
        #059669,
        #047857
    );

    box-shadow: 0 8px 20px rgba(5,150,105,0.25);

    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    background: linear-gradient(
        135deg,
        #047857,
        #065f46
    );

    box-shadow: 0 12px 25px rgba(5,150,105,0.35);
}


/* IMAGE */

[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;

    box-shadow: 0 8px 25px rgba(15,23,42,0.08);
}


/* RESULT */

.result-card {
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #f0fdf4
    );

    border: 1px solid #bbf7d0;

    border-radius: 22px;

    padding: 30px;

    text-align: center;

    box-shadow: 0 10px 30px rgba(16,185,129,0.10);

    margin-bottom: 18px;
}

.result-label {
    color: #527064;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

.result-value {
    color: #047857;
    font-size: 40px;
    font-weight: 850;
    margin: 8px 0;
}

.result-confidence {
    color: #42564c;
    font-size: 16px;
}


/* TOP 3 */

.prediction-card {
    background: white;

    border: 1px solid #e3eee7;

    border-radius: 14px;

    padding: 14px 18px;

    margin-bottom: 8px;

    box-shadow: 0 4px 15px rgba(15,23,42,0.04);
}

.prediction-name {
    color: #173c2d;
    font-weight: 700;
}

.prediction-score {
    float: right;
    color: #047857;
    font-weight: 700;
}


/* METRICS */

[data-testid="stMetric"] {
    background: white;

    border: 1px solid #e3eee7;

    border-radius: 18px;

    padding: 20px;

    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
}

[data-testid="stMetricLabel"] {
    color: #668075 !important;
}

[data-testid="stMetricValue"] {
    color: #12372a !important;
    font-weight: 750 !important;
}


/* PROGRESS */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #10b981,
        #059669
    );
}


/* MOBILE */

@media (max-width: 768px) {

    .hero-title {
        font-size: 36px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    .hero {
        padding: 35px 18px;
    }

    .result-value {
        font-size: 34px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(
        6,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=output
    )

    model.load_weights("waste_mobilenetv2.h5")

    return model


# =========================================================
# LOAD CLASSES
# =========================================================

@st.cache_data
def load_classes():

    with open("class_names.json", "r") as f:
        return json.load(f)


model = load_model()
classes = load_classes()


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = classes[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top_predictions = []

    for index in top_indices:

        top_predictions.append(
            (
                classes[index],
                float(predictions[index] * 100)
            )
        )

    return (
        predicted_class,
        confidence,
        top_predictions
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero"><div class="hero-icon">♻️</div><div class="hero-title">EcoSort AI</div><div class="hero-subtitle">AI-Based Waste Classification using MobileNetV2</div></div>',
    unsafe_allow_html=True
)


# =========================================================
# CATEGORIES
# =========================================================

st.subheader("♻️ Supported Waste Categories")

category_columns = st.columns(6)

for i, category in enumerate(classes):

    with category_columns[i]:

        st.markdown(
            f'<div class="category-box">{category.capitalize()}</div>',
            unsafe_allow_html=True
        )


# =========================================================
# MAIN AREA
# =========================================================

st.write("")

left, right = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT - UPLOAD
# =========================================================

with left:

    st.subheader("📤 Upload Waste Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear waste image."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        classify = st.button(
            "🔍  Classify Waste",
            use_container_width=True
        )

    else:

        st.info(
            "Upload a JPG, JPEG or PNG image to start classification."
        )

        classify = False


# =========================================================
# RIGHT - RESULT
# =========================================================

with right:

    st.subheader("🎯 Classification Result")

    if uploaded_file is None:

        st.info(
            "Your prediction will appear here after uploading an image."
        )

    elif classify:

        (
            predicted_class,
            confidence,
            top_predictions
        ) = predict_image(image)

        st.markdown(
            f'<div class="result-card"><div class="result-label">Predicted Waste Type</div><div class="result-value">{predicted_class.upper()}</div><div class="result-confidence">Confidence: <b>{confidence:.2f}%</b></div></div>',
            unsafe_allow_html=True
        )

        st.write("**Model Confidence**")

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.write("")

        st.subheader("🏆 Top 3 Predictions")

        for i, (label, score) in enumerate(
            top_predictions,
            start=1
        ):

            st.markdown(
                f'<div class="prediction-card"><span class="prediction-name">{i}. {label.capitalize()}</span><span class="prediction-score">{score:.2f}%</span></div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(score / 100, 1.0)
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.write("")
st.write("")

st.subheader("📊 Model Information")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Architecture",
        "MobileNetV2"
    )

with info2:
    st.metric(
        "Input Size",
        "224 × 224"
    )

with info3:
    st.metric(
        "Classes",
        "6"
    )

with info4:
    st.metric(
        "Test Accuracy",
        "88.68%"
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.write("")

with st.expander("ℹ️ About this project"):

    st.write(
        """
        **EcoSort AI** is an AI-based waste classification
        application using MobileNetV2 transfer learning.

        The model classifies waste into six categories:

        • Cardboard  
        • Glass  
        • Metal  
        • Paper  
        • Plastic  
        • Trash

        The model uses 224 × 224 RGB images and MobileNetV2
        preprocessing.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "♻️ EcoSort AI  •  Waste Classification using Deep Learning  •  MobileNetV2 Transfer Learning"
)