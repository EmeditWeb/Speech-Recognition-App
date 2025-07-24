# Speech-Recognition-App

This repository hosts the code for my Streamlit-powered speech application. This app leverages advanced speech models to understand and provide speech-to-text transcription with stunning accuracy.

---

## App Preview

Here's a quick look at what the application looks like:

![App Screenshot](https://github.com/EmeditWeb/Speech-Recognition-App/blob/main/Screenshot_20250724-073626.jpg)
 
---

## Features

* ### Real-time audio processing.
* ### Supports audio file of more than 1 minute.
* ### Intuitive user interface for easy interaction.

---

## Supported Audio Formats & File Size

The application is designed to handle various popular audio formats for your convenience:

* **Supported Formats:** WAV, MP3, FLAC, OGG, AAC
* **File Size Limit:** Please ensure individual audio files do not exceed **200MB**.

---

## How to Run Locally

Follow these steps to set up and run the application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EmeditWeb/Speech-Recognition-App.git
    cd [YourRepoName]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note:** Make sure you have a `requirements.txt` file in your repository listing all necessary Python libraries, like `streamlit`, `transformers`, `torch`, `librosa`, etc.)*

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    *(**Note:** Replace `app.py` with the actual name of your main Streamlit application file if it's different.)*

    Your browser should automatically open to the application (usually `http://localhost:8501`).

---

## Deployment (Streamlit)

This application is deployed on Streamlit for easy access. You can find the live application here:

* [ASR App](https://speech-recognition-app25.streamlit.app/)

---

## Model Information

This application utilizes a custom-trained/fine-tuned OpenAI Whisper speech model. The model file (approx. 300-350MB) is managed securely and not directly included in this public repository to ensure optimal performance and protect intellectual property. It is sourced from Hugging Face Hub private repository

---

## Contributing

We welcome contributions! If you have suggestions or want to improve the app, please feel free to:

* Open an issue
* Submit a pull request

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or inquiries, please reach out to [Emmanuel Itighise](https://github.com/EmeditWeb)
