import streamlit as st
# Import individual components from transformers
from streamlit_mic_recorder import mic_recorder
from transformers import AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, AutoTokenizer
import os
import tempfile
import io
import librosa # For audio loading and resampling
import soundfile as sf # For reading/writing audio files
import numpy as np # For numerical operations on audio data

# --- Meta Tags and Favicon (Go at the very top of your script) ---
st.set_page_config(
    page_title="Automatic Speech Recognition App | EmeditWeb",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Header, Footer, AND Elastic Textarea & Mobile Responsiveness ---
st.markdown(
    """
    <style>
    /* Global App Padding for Responsiveness */
    .stApp {
        padding-bottom: 70px;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Header Fonts */
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        text-align: center;
        color: #69F0AE;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
    }
    .small-italic-font {
        font-size:18px !important;
        font-style: italic;
        text-align: center;
        color: #B0B0B0;
        margin-top: -20px;
        margin-bottom: 30px;
    }

    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2D2D2D;
        color: #B0B0B0;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.3);
    }

    /* --- ELASTIC TEXTAREA CSS --- */
    .stTextArea textarea {
        height: auto !important;
        overflow-y: hidden !important;
        resize: none !important;
    }
    /* --- END ELASTIC TEXTAREA CSS --- */

    /* --- MEDIA QUERIES for Mobile Responsiveness --- */
    @media (max-width: 768px) {
        .big-font {
            font-size: 36px !important;
        }
        .small-italic-font {
            font-size: 14px !important;
            margin-top: -10px;
            margin-bottom: 20px;
        }
        .stApp {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            margin-bottom: 10px;
        }
    }
    /* --- END MEDIA QUERIES --- */

    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown('<p class="big-font">Automatic Speech Recognition App</p>', unsafe_allow_html=True)
st.markdown('<p class="small-italic-font">Powered by OpenAI Whisper Model</p>', unsafe_allow_html=True)

# --- End Header Section ---


# Load the ASR model components directly
@st.cache_resource
def load_asr_model_components():
    model_name = st.secrets.get("ASR_MODEL_NAME", "distil-whisper/distil-small.en")

    # Load the specific model, feature extractor, and tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Optional: Move model to GPU if available for faster inference
    # import torch
    # if torch.cuda.is_available():
    #     model.to("cuda")

    return model, feature_extractor, tokenizer

# Unpack the components once the function is called
asr_model, asr_feature_extractor, asr_tokenizer = load_asr_model_components()


def transcribe_long_form(audio_input_bytes, file_format="wav"):
    """
    Transcribes long audio input using the ASR model's generate method for higher accuracy.
    Handles bytesIO objects from mic_recorder or file_uploader.
    """
    if audio_input_bytes is None:
        st.warning("No audio found, please retry.")
        return ""

    filepath = None
    try:
        audio_io = io.BytesIO(audio_input_bytes)

        # Create a temporary file to save the audio bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp_file:
            tmp_file.write(audio_io.getvalue())
            filepath = tmp_file.name

        # Load audio using librosa, ensuring it's resampled to 16kHz (Whisper's expected sample rate)
        # librosa.load returns (audio_array, sample_rate)
        # Suppress any future librosa warnings about default sample rate if they appear
        audio_array, current_sr = librosa.load(filepath, sr=16000)

        # Prepare input features using the feature extractor
        # The .input_features extracts the relevant tensor from the batch
        input_features = asr_feature_extractor(
            audio_array,
            sampling_rate=current_sr,
            return_tensors="pt" # Return PyTorch tensors
        ).input_features

        # Generate transcription using the model's generate method.
        # This method handles its own internal chunking for long audio.
        predicted_ids = asr_model.generate(input_features)

        # Decode the predicted token IDs to text
        transcription = asr_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription
    finally:
        # Clean up the temporary file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

# --- Main App Content ---
tab1, tab2 = st.tabs(["Transcribe Microphone", "Transcribe Audio File"])

with tab1:
    st.header("Transcribe from Microphone")
    st.write("Click 'Start recording' to capture your voice.")

    mic_recorder_audio_data = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        format="wav",
        just_once=True,
        use_container_width=True,
        key="mic_recorder_widget"
    )

    transcription_mic = ""
    if mic_recorder_audio_data:
        audio_bytes = mic_recorder_audio_data.get("bytes")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("Transcribing from microphone..."):
                transcription_mic = transcribe_long_form(audio_bytes, file_format="wav")

            st.text_area(
                "Transcription:",
                value=transcription_mic,
                height=None,
                key="transcription_mic_output",
                help="This box automatically adjusts its size to show the full transcription."
            )

            if transcription_mic:
                st.download_button(
                    label="Download Transcription",
                    data=transcription_mic,
                    file_name="microphone_transcription.txt",
                    mime="text/plain",
                    key="download_mic_transcription_button",
                    help="Click to download the full transcribed text as a .txt file."
                )
        else:
            st.warning("No audio bytes captured from microphone.")

with tab2:
    st.header("Transcribe from Audio File")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg", "aac"])

    transcription_file = ""
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        with st.spinner("Transcribing..."):
            file_extension = uploaded_file.name.split('.')[-1]
            transcription_file = transcribe_long_form(uploaded_file.getvalue(), file_format=file_extension)

        st.text_area(
            "Transcription:",
            value=transcription_file,
            height=None,
            key="transcription_file_output",
            help="This box automatically adjusts its size to show the full transcription."
        )

        if transcription_file:
            st.download_button(
                label="Download Transcription",
                data=transcription_file,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt",
                mime="text/plain",
                key="download_file_transcription_button",
                help="Click to download the full transcribed text as a .txt file."
            )

# --- Footer Section ---
st.markdown(
    """
    <div class="footer">
        Built by Emedit | For 3MTT July Project Showcase 🚀
    </div>
    """,
    unsafe_allow_html=True
)
# --- End Footer Section ---

# Codes written by Emmanuel Itighise 
