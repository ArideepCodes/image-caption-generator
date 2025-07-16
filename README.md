# 🧠 Arideep's AI Image Caption Generator

![License](https://img.shields.io/github/license/ArideepCodes/image-caption-generator)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

---



## 📌 Overview

This project is an AI-powered web app that generates captions for images using a combination of:

- 🔍 **MobileNetV2** for feature extraction (lightweight and fast)
- 🧠 **LSTM-based** model for sequence generation
- 🖼️ A clean UI built with **Streamlit**

Users can upload `.jpg`, `.jpeg`, or `.png` files and see real-time caption predictions.

---

## 🧠 Model Architecture

- **Feature Extractor**: Pretrained MobileNetV2 (used instead of VGG16 for better performance)
- **Caption Generator**: LSTM model trained on Flickr8k
- **Tokenizer**: Fitted on training captions and saved as `tokenizer.pkl`

---

## 📁 Folder Structure

📁 arideep-image-caption-generator
├── app.py
├── mymodel.h5
├── tokenizer.pkl
├── requirements.txt
├── README.md
└── resource/
    └── demo.gif
⚙️ Installation
Clone or download this repo:

bash
Copy
Edit
git clone https://github.com/ArideepCodes/image-caption-generator.git
cd image-caption-generator
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
☁️ Deployment (Streamlit Cloud)
Push this repo to your GitHub

Go to streamlit.io/cloud

Create a new app and select your repo

Set app.py as the entry point

Click Deploy 🎉

🗃 Dataset
This model was trained on the Flickr8k dataset, which contains over 8,000 images, each paired with five descriptive captions.

🚀 Future Improvements
Add beam search decoding

Display caption confidence scores

Deploy as a mobile or desktop app

Add multilingual caption generation

Upgrade to a Transformer-based captioning model

🐛 Issues or Suggestions?
Open an issue or submit a pull request on GitHub

📜 License
This project is licensed under the MIT License.
Built with ❤️ by Arideep Kanshabanik

