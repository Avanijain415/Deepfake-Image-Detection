''' import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------------
# LOAD MODEL (SAFE)
# ----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_bestest_model.h5")

model = load_model()

# ----------------------------------
# IMAGE PREPROCESSING
# ----------------------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------------
# UI HEADER
# ----------------------------------
st.markdown(
    """
    <h2 style='text-align: center;'>Deepfake Image Detection</h2>
    <p style='text-align: center; color: gray;'>
    Upload a human face image to verify whether it is real or deepfake
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# FILE UPLOADER
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png","jfif"]
)

if uploaded_file is not None:

    # -------- IMAGE LOADING (FIXED) --------
    image = Image.open(uploaded_file)

    # Force RGB (VERY IMPORTANT)
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)

    # Validate image shape
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        st.error("Please upload a valid RGB image.")
    else:
        # Display image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # -------- PREDICTION --------
        processed_img = preprocess_image(img_array)
        prediction = model.predict(processed_img, verbose=0)[0][0]

        if prediction > 0.5:
            label = "FAKE"
            confidence = prediction * 100
            color = "#e74c3c"
        else:
            label = "REAL"
            confidence = (1 - prediction) * 100
            color = "#2ecc71"

        confidence = round(confidence, 2)

        # -------- RESULT UI --------
        confidence = round(confidence, 2)

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:10px;
            text-align:center;
            background-color:#f8f9fa;
            ">
            <h3 style="color:{color};">{label}</h3>
            <p style="font-size:18px;">
                Confidence: <b>{confidence}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.progress(float(confidence) / 100)

    st.caption(
        "Prediction is generated using a deep learning model trained on real and deepfake facial images."
    )
'''








import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_bestest_model.h5")

model = load_model()

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# HEADER TOP EDGE (no margin)
# -------------------------------
st.markdown(
    """
    <style>
    .header-top {margin:0; padding:0;}
    </style>
    <div class="header-top" style="text-align:center;">
        <h1 style="color:#1f77b4; font-size:30px; margin:0; padding:0;">🧠 Deepfake Detection</h1>
        <p style="color:#555; font-size:12px; margin:0; padding:0;">Upload a human face image to check if it is real or fake</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDE-BY-SIDE LAYOUT
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png","jfif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image)

        if img_array.ndim != 3 or img_array.shape[2] != 3:
            st.error("Please upload a valid RGB image.")
        else:
            # smaller uploaded image
            max_width = 160
            aspect_ratio = image.height / image.width
            display_height = int(max_width * aspect_ratio)
            display_image = image.resize((max_width, display_height))

            st.markdown(
                """
                <div style="
                    border-radius:10px;
                    border:1px solid #ccc;
                    padding:4px;
                    background-color:#f7f7f7;
                    text-align:center;
                    margin-bottom:5px;">
                    <p style="margin:0; font-size:12px; color:#555;">Uploaded Image</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(display_image, use_container_width=False, width=max_width)

if uploaded_file is not None:
    processed_img = preprocess_image(img_array)
    with st.spinner("Analyzing image..."):
        prediction = model.predict(processed_img, verbose=0)[0][0]

    if prediction > 0.5:
        label = "FAKE"
        confidence = prediction * 100
        color = "#e74c3c"
    else:
        label = "REAL"
        confidence = (1 - prediction) * 100
        color = "#2ecc71"

    confidence = round(confidence, 2)

    with col2:
        st.markdown(
            f"""
            <div style="
                padding:12px;
                border-radius:12px;
                text-align:center;
                background: #f0f4ff;
                border:1px solid #ccc;
                box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
                max-width:200px;
                margin-left:auto;
                margin-right:auto;
                margin-top:20px;">
                <h2 style="color:{color}; font-size:20px; margin-bottom:2px;">{label}</h2>
                <p style="font-size:12px; color:#333; margin:0;">
                    Confidence: <b>{confidence}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(confidence)/100)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#777; font-size:11px; margin-top:5px;">
        Prediction is generated using a deep learning model trained on real and deepfake facial images.
    </div>
    """,
    unsafe_allow_html=True
)




