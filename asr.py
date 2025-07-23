import streamlit as st
from transformers import pipeline
import os
import tempfile
import io

# --- Meta Tags and Favicon (Go at the very top of your script) ---
st.set_page_config(
    page_title="Automatic Speech Recognition App | EmeditWeb",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
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

def transcribe_long_form(audio_input):
    """
    Transcribes audio input using the ASR model.
    Handles both file paths and bytesIO objects for audio.
    """
    if audio_input is None:
        st.warning("No audio found, please retry.")
        return ""

    filepath = None
    if isinstance(audio_input, str):
        filepath = audio_input
    elif isinstance(audio_input, io.BytesIO):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_input.getvalue())
            filepath = tmp_file.name
    else:
        st.error("Unsupported audio input type.")
        return ""

    try:
        output = asr(
            filepath,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=8,
        )
        return output["text"]
    finally:
        if filepath and filepath.startswith(tempfile.gettempdir()):
            os.remove(filepath)

# Tabs for functionality
tab1, tab2 = st.tabs(["Transcribe Microphone", "Transcribe Audio File"])

with tab1:
    st.header("Transcribe from Microphone")
    st.write("Please use an external library for microphone input as Streamlit doesn't have a native one yet.")
    st.info("For microphone input in Streamlit, you would typically use a third-party component like `streamlit-webrtc` or `streamlit_mic_recorder`. This example will show a placeholder for it.")

    st.markdown("---")
    st.markdown("**Note:** Streamlit does not have a native microphone input widget. For live microphone input, you would need to integrate a custom component like `streamlit-webrtc` or `streamlit_mic_recorder`.")


with tab2:
    st.header("Transcribe from Audio File")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        st.write("Transcribing...")

        transcription_file = transcribe_long_form(io.BytesIO(uploaded_file.getvalue()))
        st.text_area("Transcription", value=transcription_file, height=150)

# --- Footer Section ---
# Streamlit does not have a native footer widget. We use markdown with HTML/CSS.
st.markdown(
    """
    <div class="footer">
        Built by Emmanuel Itighise | For 3MTT July Project Showcase 🚀
    </div>
    """,
    unsafe_allow_html=True
)
# --- End Footer Section ---
