import os
os.environ["STREAMLIT_WATCHDOG_SKIP"]="true"


import streamlit as st
from models.text_embedder import TextEmbedder
from models.image_embedder import ImageEmbedder
from models.audio_embedder import AudioEmbedder

st.title(" NeuroRecX-KG Multimodal Embedder")

text_model = TextEmbedder()
image_model = ImageEmbedder()
audio_model = AudioEmbedder()

text_input = st.text_input("Enter some text")
image_input = st.file_uploader("Upload an image", type=["jpg", "png"])
audio_input = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if text_input:
    vec = text_model.embed(text_input)
    st.success("Text embedding generated!")
    st.write(vec[:10])  # preview

if image_input:
    with open("temp_image.png", "wb") as f:
        f.write(image_input.read())
    vec = image_model.embed("temp_image.png")
    st.success("Image embedding generated!")
    st.write(vec[:10])  # preview

if audio_input:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_input.read())
    vec = audio_model.embed("temp_audio.wav")
    st.success("Audio embedding generated!")
    st.write(vec[:10])  # preview
if audio_input is not None:
    embedding = audio_model.embed(audio_input)
    st.write(embedding)
