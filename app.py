import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from streamlit_chat import message
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import speech_recognition as sr
from io import BytesIO

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def transcribe_audio_with_speech_recognition(audio_bytes):
    # Convert audio_bytes (WAV) to a format compatible with SpeechRecognition (e.g., WAV)
    audio = AudioSegment.from_wav(BytesIO(audio_bytes))
    wav_file = BytesIO()
    audio.export(wav_file, format="wav")
    wav_file.seek(0)

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Transcribe using SpeechRecognition
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Initialize the Streamlit app
st.set_page_config(page_title="Project: Q&A Chatbot")

# Add custom CSS to set border and font
st.markdown(
    """
    <style>
    .main {
        height:auto;
        width:50%;
        margin:auto;
        border: 2px solid #007BFF; /* Blue border */
        padding: 10px;
        border-radius: 8px;
        max-height: 80vh; /* Maximum height to allow scrolling */
        overflow-y: auto; /* Enable vertical scrolling */
        font-family: 'Arial', sans-serif; /* Change font family */
    }
    .streamlit-expanderHeader {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("👽SkyChat(voice) 2.1.0 : Gemini Version (History)")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Add the audio recording feature
st.subheader("Record and Upload Audio")
audio_bytes = audio_recorder(
    text="Click to Record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="6x",
    pause_threshold=2.0,  # Adjust as needed
    sample_rate=41_000    # Adjust as needed
)

if audio_bytes is not None:
    st.audio(audio_bytes, format='audio/wav')
    
    # Transcribe the recorded WAV audio
    transcript = transcribe_audio_with_speech_recognition(audio_bytes)
    st.write("Transcribed Text: ", transcript)
    
    # Use the transcribed text as a prompt
    if transcript:
        with st.spinner("Generating response from audio..."):
            response = get_gemini_response(transcript)
            # Add the transcribed text and response to session state chat history
            st.session_state['chat_history'].append(("You", transcript))
            for chunk in response:
                st.session_state['chat_history'].append(("Bot", chunk.text))
            st.success("Response generated successfully from audio!", icon="✅")

# Display the chat history using the streamlit_chat message function
for role, text in reversed(st.session_state['chat_history']):
    if role == "You":
        message(text, is_user=True)
    else:
        message(text)
