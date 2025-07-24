import streamlit as st
from transformers import pipeline
import os
import tempfile
import io
from streamlit_mic_recorder import mic_recorder
# No need for HfFolder as the current model is public and accessed via secrets for obscurity, not private access.

# --- Meta Tags and Favicon (Go at the very top of your script) ---
st.set_page_config(
    page_title="Automatic Speech Recognition App | EmeditWeb",
    page_icon="🎙️",
    # --- MOBILE RESPONSIVENESS: Centered layout is generally better for mobile ---
    layout="centered", 
    # --- END MOBILE RESPONSIVENESS CHANGE ---
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Header, Footer & Mobile Responsiveness ---
st.markdown(
    """
    <style>
    /* Global App Padding for Responsiveness */
    .stApp {
        padding-bottom: 70px; /* Add padding to the bottom of the main content to prevent footer overlap */
        padding-left: 1rem; 
        padding-right: 1rem;
    }

    /* Header Fonts */
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
    
    /* Footer Styling */
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

    /* --- REMOVED ELASTIC TEXTAREA CSS --- */
    /* The previous CSS to force height: auto and hide overflow-y is removed */
    /* to restore fixed height with internal scrollbar functionality. */
    /* If you ever want elastic behavior again, you can re-add that CSS. */

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
        /* Make buttons full width on small screens if desired for better touch targets */
        .stButton>button {
            width: 100%; 
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
    # --- Retrieving Model Name from Streamlit Settings (Secrets) -
    model_name = st.secrets.get("model")
    
    # Load the ASR pipeline with the model
    return pipeline(
        task="automatic-speech-recognition",
        model=model_name
    )

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
# Using tabs helps organize content and naturally stacks on mobile.
tab1, tab2 = st.tabs(["Transcribe Microphone", "Transcribe Audio File"])

with tab1:
    st.header("Transcribe from Microphone")
    st.write("Click 'Start recording' to capture your voice.")

    # Microphone Recorder Widget
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
            
            # --- Fixed heights  ---
            st.text_area(
                "Transcription:",
                value=transcription_mic,
                height=300, # Set to 300px (double of original 150px)
                key="transcription_mic_output",
                help="The transcription text. Scroll within the box to see the full content."
            )
            
            # --- Download Function ---
            # The download button only appears if there's actual transcription text.
            if transcription_mic: 
                st.download_button(
                    label="Download Transcription",
                    data=transcription_mic,
                    file_name="microphone_text.txt", # Corrected file extension to .txt
                    mime="text/plain",
                    key="download_mic_transcription_button",
                    help="Click to download the full transcribed text as a .txt file."
                )
        else:
            st.warning("No audio bytes captured from microphone.")

with tab2:
    st.header("Transcribe from Audio File")
    # File Uploader Widget
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg", "aac"])

    transcription_file = "" # Initialize to empty string
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        with st.spinner("Transcribing..."):
            file_extension = uploaded_file.name.split('.')[-1]
            transcription_file = transcribe_long_form(uploaded_file.getvalue(), file_format=file_extension)
        
        # --- FIXED HEIGHT TEXTBOX (300px = 150px * 2) ---
        st.text_area(
            "Transcription:",
            value=transcription_file,
            height=300, # Set to 300px (double of original 150px)
            key="transcription_file_output",
            help="The transcription text. Scroll within the box to see the full content."
        )
        
        # --- Download Function ---
        # The download button only appears if there's actual transcription text.
        if transcription_file: 
            st.download_button(
                label="Download Transcription",
                data=transcription_file,
                # Dynamically set file name based on uploaded file's original name
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcription.txt", 
                mime="text/plain",
                key="download_file_transcription_button",
                help="Click to download the full transcribed text as a .txt file."
            )

# --- Footer Section ---
st.markdown(
    """
    <div class="footer">
        Built by Emmanuel Itighise | For 3MTT July Project Showcase 🖥️
    </div>
    """,
    unsafe_allow_html=True
)
# --- End Footer Section ---

# Codes written by Emmanuel Itighise
