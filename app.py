import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load MobileNetV2 model for image feature extraction
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load the trained LSTM caption generator model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Set custom web page settings
st.set_page_config(page_title="Arideep's Image Caption Generator", page_icon="📸")

# App title and instructions
st.title("🧠 Arideep's AI Image Caption Generator")
st.markdown("Upload an image below, and this AI model will generate a caption for it!")

# Upload image
uploaded_image = st.file_uploader("📁 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.subheader("🖼️ Uploaded Image")
    st.image(uploaded_image, caption="Selected Image", use_column_width=True)

    st.subheader("📝 Generated Caption")

    # Spinner while generating caption
    with st.spinner("Thinking..."):
        # Preprocess image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract image features
        image_features = mobilenet_model.predict(image, verbose=0)

        max_caption_length = 34

        # Helper: map index back to word
        def get_word_from_index(index, tokenizer):
            return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

        # Generate caption
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                if predicted_word is None:
                    break
                caption += " " + predicted_word
                if predicted_word == "endseq":
                    break
            return caption

        # Run prediction
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Clean up the caption text
        final_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

        # Display caption
        st.markdown(
            f'<div style="border-left: 6px solid #00A5FF; background-color: #f9f9f9; padding: 15px; margin-top: 20px;">'
            f'<p style="font-size: 18px; font-style: italic; color: #333;">“{final_caption}”</p>'
            f'</div>',
            unsafe_allow_html=True
        )
