import streamlit as st
from transformers import pipeline
import os
import tempfile
import io
from streamlit_mic_recorder import mic_recorder

# --- Meta Tags and Favicon 
st.set_page_config(
    page_title="Automatic Speech Recognition App | EmeditWeb",
    page_icon="🎙️",
    # --- CHANGE FOR MOBILE RESPONSIVENESS ---
    # Changed from "wide" to "centered" for better mobile fitting.
    # "centered" layouts adapt more gracefully to smaller screens.
    layout="centered", 
    # --- END CHANGE ---
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
        /* --- ADDED FOR MOBILE RESPONSIVENESS --- */
        /* Reduce horizontal padding on very small screens for better use of space */
        padding-left: 1rem; 
        padding-right: 1rem;
    }

    /* --- MEDIA QUERIES for Mobile Responsiveness --- */
    @media (max-width: 768px) { /* Styles for screens up to 768px wide (e.g., tablets and phones) */
        .big-font {
            font-size: 36px !important; /* Smaller font for mobile header */
        }
        .small-italic-font {
            font-size: 14px !important; /* Smaller font for mobile subtitle */
            margin-top: -10px; /* Adjust spacing */
            margin-bottom: 20px;
        }
        .stApp {
            padding-left: 0.5rem; /* Even smaller side padding on very narrow screens */
            padding-right: 0.5rem;
        }
        /* Further adjustments if needed for specific elements, e.g., button sizes */
        .stButton>button {
            width: 100%; /* Make buttons full width on small screens if desired */
            margin-bottom: 10px; /* Add space between stacked buttons */
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

        output = asr(
            filepath,
            chunk_length_s=30,  # For handling long audio files (e.g., audio more than 30 secs)
            batch_size=8, 
        )

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

    transcription_mic = "" # Initialize to empty string
    if audio_data:
        audio_bytes = audio_data.get("bytes")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with st.spinner("Transcribing from microphone..."):
                transcription_mic = transcribe_long_form(audio_bytes, file_format="wav")
            
            # Elastic Textbox
            st.text_area(
                "Transcription:",
                value=transcription_mic,
                height=None, # Makes the textbox elastic
                key="transcription_mic_output",
                help="This box automatically adjusts its size to show the full transcription."
            )
            
            # Download Function
            if transcription_mic: # Only show download button if transcription exists
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

    transcription_file = "" # Initialize to empty string
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        with st.spinner("Transcribing..."):
            file_extension = uploaded_file.name.split('.')[-1]
            transcription_file = transcribe_long_form(uploaded_file.getvalue(), file_format=file_extension)
        
        # Elastic Textbox
        st.text_area(
            "Transcription:",
            value=transcription_file,
            height=None, # Makes the textbox elastic
            key="transcription_file_output",
            help="This box automatically adjusts its size to show the full transcription."
        )
        
        # Download Function
        if transcription_file: # Only show download button if transcription exists
            st.download_button(
                label="Download Transcription",
                data=transcription_file,
                file_name=f"{uploaded_file.name.split('.')[0]}_transcription.txt", # Uses original file name
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
