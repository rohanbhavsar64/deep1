import streamlit as st
import numpy as np
# Load the MNIST dataset
model = load_model('my_model.h5') 

st.title('Deep Learning Model Prediction')
st.write('Upload an image to get predictions from the model.')
uploaded_file = st.file_uploader("Choose an video...")
import os
import time
import warnings
import torch
from faster_whisper import WhisperModel

def initialize_model(model_size="large"):
    """
    Initialize the Whisper model.
    Returns the loaded model and the device used.
    """
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Measure the time taken to load the model
    start_time = time.time()
    print("Loading Faster Whisper model...")
    
    # Load the model with mixed precision (float16)
    model = WhisperModel(model_size, device=device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")
    
    return model, device

def transcribe_and_translate(model, audio_file):
    """
    Transcribe and translate the audio file using the provided model.
    Returns the transcription text and detected language.
    """
    start_time = time.time()
    print("Transcribing and translating audio...")

    # Perform transcription and translation0
    segments, info = model.transcribe(audio_file, beam_size=2, task="translate")
    transcription = "".join([segment.text for segment in segments])
    
    # Measure the time taken
    process_time = time.time() - start_time
    print(f"Process completed in {process_time:.2f} seconds.")
    
    return {"transcription": transcription, "language": info.language}

if __name__ == "__main__":
    # Specify the audio file
    audio_file = "/content/WIN_20240730_19_25_14_Pro.mp4"
    model_size = "large"

    # Initialize the model
    model, device = initialize_model(model_size=model_size)
    
    # Measure total processing time
    total_start_time = time.time()
    result = transcribe_and_translate(model, audio_file)
    total_time = time.time() - total_start_time
    
    # Display the results
    print("\n--- Speech-to-Text Results ---")
    print(f"Transcription: {result['transcription']}")
    print(f"Detected Language: {result['language']}")
    print(f"Total time for processing: {total_time:.2f} seconds.")
