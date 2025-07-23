
import streamlit as st
from transformers import pipeline
import os
import tempfile
import io
from streamlit_mic_recorder import mic_recorder

# --- Meta Tags and Favicon (Go at the very top of your script) ---
st.set_page_config(
    page_title="Automatic Speech Recognition App | EmeditWeb",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Header and Footer ---
st.markdown(
    """
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        text-align: center;
        color: #69F0AE; /* Match primaryColor for a sleek look */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4); /* Darker shadow for dark theme */
    }
    .small-italic-font {
        font-size:18px !important;
        font-style: italic;
        text-align: center;
        color: #B0B0B0; /* Slightly muted text for subtitle */
        margin-top: -20px;
        margin-bottom: 30px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2D2D2D; /* Match secondaryBackgroundColor or slightly darker */
        color: #B0B0B0; /* Muted text color */
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100; /* Ensure footer is on top of other elements */
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.3); /* Subtle shadow at the top */
    }
    .stApp {
        padding-bottom: 70px; /* Add padding to the bottom of the main content to prevent footer overlap */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown('<p class="big-font">Automatic Speech Recognition App</p>', unsafe_allow_html=True)
st.markdown('<p class="small-italic-font">Powered by OpenAI Whisper Model</p>', unsafe_allow_html=True)

# --- End Header Section ---


# Load the ASR model
@st.cache_resource
def load_asr_model():
    return pipeline(task="automatic-speech-recognition",
                    model="distil-whisper/distil-small.en")

asr = load_asr_model()

def transcribe_long_form(audio_input_bytes, file_format="wav"):
    """
    Transcribes audio input using the ASR model.
    Handles bytesIO objects from mic_recorder or file_uploader.
    """
    if audio_input_bytes is None:
        st.warning("No audio found, please retry.")
        return ""

    filepath = None
    try:
        # Create a BytesIO object from the audio bytes
        audio_io = io.BytesIO(audio_input_bytes)

        # Save BytesIO content to a temporary file for the ASR pipeline to process
        # Use the provided file_format for the suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp_file:
            tmp_file.write(audio_io.getvalue())
            filepath = tmp_file.name

        # --- CRITICAL CHANGE HERE: REMOVE problematic arguments ---
        output = asr(
            filepath,
            chunk_length_s=30,  # For handling long audio files (e.g., audio more than 30 secs)
            batch_size=8, 

        )
        # --- END CRITICAL CHANGE ---

        return output["text"]
    finally:
        # Clean up the temporary file if it was created
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

# --- Main App Content ---
tab1, tab2 = st.tabs(["Transcribe Microphone", "Transcribe Audio File"])

with tab1:
    st.header("Transcribe from Microphone")
    st.write("Click 'Start recording' to capture your voice.")

    audio_data = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        format="wav",
        just_once=True,
        use_container_width=True,
        key="mic_recorder_widget"
    )

    if audio_data:
        audio_bytes = audio_data.get("bytes")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.write("Transcribing from microphone...")
            transcription_mic = transcribe_long_form(audio_bytes, file_format="wav")
            st.text_area("Transcription", value=transcription_mic, height=150)
        else:
            st.warning("No audio bytes captured from microphone.")

with tab2:
    st.header("Transcribe from Audio File")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg", "aac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        st.write("Transcribing...")

        file_extension = uploaded_file.name.split('.')[-1]
        transcription_file = transcribe_long_form(uploaded_file.getvalue(), file_format=file_extension)
        st.text_area("Transcription", value=transcription_file, height=150)

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
