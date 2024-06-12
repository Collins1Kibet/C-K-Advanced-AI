from PIL import Image
from dotenv import load_dotenv
import google.generativeai as gen_ai
from google.cloud import speech
from gtts import gTTS
import os
import io
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av 

# Checking if the .env file exists and loading it
if os.path.exists('.env'):
    load_dotenv('.env')

# Function for getting the API Key from Streamlit Secrets
def get_env_var(key, default_value=None):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default_value)

# Setting up the Streamlit Web app Page
st.set_page_config(
    page_title='Cobest AI',
    page_icon='🔍',
    layout='centered'
)

# Creating a Sidebar for Navigating to Various Pages
with st.sidebar:
    selected = option_menu(menu_title='Cobest AI',
                           options=[
                            'ChatBot',
                            'Voice Chat',
                            'Image Captioning',
                            'Embed Text',
                            'Ask me anything'],
                            menu_icon='robot', icons=[
                                'chat-left-dots-fill',
                                'mic-fill',
                                'image-fill',
                                'textarea-t',
                                'question-square-fill'],
                            default_index=0)

# Getting and configuring the API Key
GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")

gen_ai.configure(api_key=GOOGLE_API_KEY)

# AI Chatbot Function
def load_gemini_pro_model():
    gemini_pro_model = gen_ai.GenerativeModel('gemini-pro')
    return gemini_pro_model

# Speech Recognition and Text to Speech Functions
def recognize_input_speech(audio_data):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript

def text_to_speech(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    st.audio("response.mp3")

# Function for Image Recognition and Captioning
def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = gen_ai.GenerativeModel('gemini-pro-vision')
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result

# Funtion for Text Embedding
def embedding_model_response(input_text):
    embedding_gemini_pro_model = "models/embedding-001"
    embedding_model = gen_ai.embed_content(model=embedding_gemini_pro_model, content=input_text, task_type='retrieval_document')
    return embedding_model

# AI Response Function
def gemini_pro_response(user_input):
    gemini_pro_model = gen_ai.GenerativeModel('gemini-pro')
    feedback = gemini_pro_model.generate_content(user_input)
    output = feedback.text
    return output

# Role (Model to Assistant) translation Function for Streamlit
def tranlates_role_for_streamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role
        
# Modelling the Chatbot Page
if selected == 'ChatBot':
    model = load_gemini_pro_model()
    st.title('🤖 Cobest ChatBot')
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
    for message in st.session_state.chat_session.history:
        with st.chat_message(tranlates_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)
    user_promt = st.chat_input('Ask ai...')
    if user_promt:
        st.chat_message('user').markdown(user_promt)
        cobest_response = st.session_state.chat_session.send_message(user_promt)
        with st.chat_message('assistant'):
            st.markdown(cobest_response.text)
            

# Initializing session state for voice response
if 'voice_response' not in st.session_state:
    st.session_state.voice_response = None
    
# Modelling the Voice Chat Page
if selected == 'Voice Chat':
    st.title('🎙️ Voice Chat')

    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.client = speech.SpeechClient()
            self.buffer = io.BytesIO()

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio_data = frame.to_ndarray()
            self.buffer.write(audio_data.tobytes())
            return frame

        def transcribe(self) -> str:
            audio_data = self.buffer.getvalue()
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )
            response = self.client.recognize(config=config, audio=audio)
            return response.results[0].alternatives[0].transcript

    webrtc_ctx = webrtc_streamer(
        key="speech-recognition",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.audio_processor:
        if st.button('Recognize Speech'):
            transcription = webrtc_ctx.audio_processor.transcribe()
            st.write('You said: ' + transcription)
            feedback = gemini_pro_response(transcription)
            st.session_state.voice_response = feedback
            st.write('AI said: ' + feedback)
        if st.session_state.voice_response:
            if st.button('Play Response'):
                text_to_speech(st.session_state.voice_response)
                
# Creating a functioning Image Captioning Page
if selected == 'Image Captioning':
    st.title('📷 Cobest Snap Caption')
    uploaded_image = st.file_uploader('Upload an Image...', type=['jpg', 'jpeg', 'png'])
    if st.button('Generate Caption'):
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            resized_image = image.resize((800, 500))
            st.image(resized_image)
        default_prompt = 'write a short caption for this image'
        caption = gemini_pro_vision_response(default_prompt, image)
        with col2:
            st.info(caption)

# Modelling the Text Embedding Page
if selected == 'Embed Text':
    st.title('🖹 Embed Text')
    input_txt = st.text_area(label="", placeholder='Enter the text to get embeddings...')
    if st.button('Embed'):
        response = embedding_model_response(input_txt)
        st.markdown(response)

# Creating a Page for AI Response to any Question
if selected == 'Ask me anything':
    st.title('❓ Ask Me a Question')
    text_area = st.text_area(label='', placeholder='Ask me anything...')
    if st.button('Get an Answer'):
        feedback = gemini_pro_response(text_area)
        st.markdown(feedback)
