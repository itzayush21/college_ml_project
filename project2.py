import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas

# --------------------------------
# UI CONFIGURATION
# --------------------------------
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.markdown("<h1 style='text-align: center;'>🔢 MNIST Digit Classification</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size:18px;'>"
    "Classify handwritten digits using a trained machine learning model.<br>"
    "Upload a 28x28 grayscale image or draw a digit below."
    "</div><br>", unsafe_allow_html=True)

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return joblib.load("knn_mnist_model1.sav")  # Ensure this file exists

model = load_model()

# --------------------------------
# SIDEBAR MODE SELECTION
# --------------------------------
st.sidebar.title("🧭 Mode Selection")
mode = st.sidebar.radio("Choose input method:", ["Upload Image", "Draw Digit"])

# --------------------------------
# IMAGE PREPROCESSING FUNCTION
# --------------------------------
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Invert the image (digit becomes white, background becomes black)
    image = ImageOps.invert(image)

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Normalize the image (scale pixel values to [0, 1])
    image_np = np.array(image) / 255.0

    return image_np.reshape(1, -1), image_np

# --------------------------------
# DISPLAY IMAGE + PREDICTION
# --------------------------------
def display_results(original, processed, prediction, probabilities):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Original Input**")
        st.image(original, width=150)
    with col2:
        st.markdown("**Processed for Model**")
        fig, ax = plt.subplots()
        ax.imshow(processed.reshape(28, 28), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)


    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>🧠 Predicted Digit: <span style='color:#4CAF50;'>{prediction}</span></h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center;'>Confidence: <strong>{np.max(probabilities)*100:.2f}%</strong></div>", unsafe_allow_html=True)

# --------------------------------
# UPLOAD IMAGE
# --------------------------------
if mode == "Upload Image":
    st.subheader("📤 Upload a Digit Image (28x28, Grayscale Preferred)")
    uploaded_file = st.file_uploader("Upload PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        #processed, norm_img = preprocess_image(img)
        img= img.convert("L")

        img = img.resize((28, 28))
        norm_img=np.array(img) / 255.0
        processed=norm_img.reshape(1, -1)
        if st.button("submit"):
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0]
            display_results(img, norm_img, pred, prob)

# --------------------------------
# DRAW DIGIT
# --------------------------------
elif mode == "Draw Digit":
    st.subheader("✏️ Draw a Digit Below")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=24,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert drawn canvas to image
        if st.button("submit"):
            canvas_img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
            processed, norm_img = preprocess_image(canvas_img)

            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0]
            print(prob)
            display_results(canvas_img, norm_img, pred, prob)

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:14px;'>"
    "This project uses a pre-trained classifier on the MNIST dataset.<br>"
    "Ensure your digit is clearly visible on a white background for best results."
    "</div>", unsafe_allow_html=True)
