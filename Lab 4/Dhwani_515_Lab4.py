import streamlit as st
import pdfplumber
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
import torch
import io
import base64
import time
from streamlit_mic_recorder import mic_recorder

# Initialize models and components
@st.cache_resource(show_spinner="Loading language models...")
def load_models():
    # English model (FLAN-T5)
    en_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    en_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    # Switched to MuRIL for better compatibility
    kn_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", use_fast=False)
    kn_model = AutoModelForQuestionAnswering.from_pretrained("google/muril-base-cased")
    
    # French model (CamemBERT)
    fr_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    fr_model = AutoModelForQuestionAnswering.from_pretrained("camembert-base")
    
    return {
        "en": {"model": en_model, "tokenizer": en_tokenizer, "lang": "en", "model_name": "FLAN-T5"},
        "kn": {"model": kn_model, "tokenizer": kn_tokenizer, "lang": "kn", "model_name": "MuRIL"},
        "fr": {"model": fr_model, "tokenizer": fr_tokenizer, "lang": "fr", "model_name": "CamemBERT"}
    }

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    filename = f"response_{lang}.mp3"
    tts.save(filename)
    return filename

def get_answer(context, question, model_info):
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]
    lang = model_info["lang"]
    
    if lang == "en":
        # Generative approach for English
        input_text = f"question: {question} context: {context[:1000]}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:  
        # Extractive approach for other languages
        inputs = tokenizer(question, context[:1000], return_tensors="pt", 
                          max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        return tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )

# Streamlit UI
st.set_page_config(page_title="Multilingual Document QA", layout="wide")
st.title("📄🗣️ Multilingual Document QA System")
st.subheader("Comparative Study of Foundation, Indic & International Language Models")

# Load models once
models = load_models()

# Language selection
st.sidebar.header("Configuration")
selected_lang = st.sidebar.radio("Select Language", ["English", "Kannada", "French"])
lang_code = {"English": "en", "Kannada": "kn", "French": "fr"}[selected_lang]
model_info = models[lang_code]

# Display model information in sidebar
st.sidebar.divider()
st.sidebar.subheader("Model Information")
st.sidebar.markdown(f"**{selected_lang} Model:** {model_info['model_name']}")

# Add model descriptions
st.sidebar.caption("**Model Details:**")
if lang_code == "en":
    st.sidebar.markdown("- **Type:** Foundation Model")
    st.sidebar.markdown("- **Architecture:** Seq2Seq Transformer")
    st.sidebar.markdown("- **Specialization:** General English QA")
elif lang_code == "kn":
    st.sidebar.markdown("- **Type:** Indic Language Model")
    st.sidebar.markdown("- **Architecture:** BERT-based")
    st.sidebar.markdown("- **Specialization:** Multilingual Indian Languages")
else:
    st.sidebar.markdown("- **Type:** International Language Model")
    st.sidebar.markdown("- **Architecture:** RoBERTa-based")
    st.sidebar.markdown("- **Specialization:** French Language Understanding")

# PDF Upload
st.header(f"{selected_lang} Model: {model_info['model_name']}")
uploaded_file = st.file_uploader(f"Upload {selected_lang} PDF", type="pdf")

context = ""
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        context = extract_text_from_pdf(uploaded_file)
    st.success("PDF processed successfully!")
    st.caption(f"Extracted text length: {len(context)} characters")
    
    # Show preview
    with st.expander("View extracted text"):
        st.text(context[:1000] + "..." if len(context) > 1000 else context)

# Question Input - Dual Mode
st.subheader("Ask a Question")
input_method = st.radio("Input Method:", ["Text", "Speech"], horizontal=True)

question = ""
if input_method == "Text":
    question = st.text_input("Type your question here:", key=f"question_{lang_code}")
else:
    st.write("Click the microphone button and speak your question:")
    
    # Initialize session state for recording
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'last_recording' not in st.session_state:
        st.session_state.last_recording = None
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    
    # Create two columns for microphone and status
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Microphone recorder
        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key=f"mic_recorder_{lang_code}"
        )
    
    # Process audio recording
    if audio and st.session_state.last_recording != audio:
        st.session_state.last_recording = audio
        st.session_state.recording = True
        st.rerun()
    
    if st.session_state.recording:
        with st.spinner("Processing your speech..."):
            # Simulate processing time
            time.sleep(2)
            
            # In a real application, you would send the audio to a speech-to-text service
            # For demo purposes, we'll use a placeholder
            st.session_state.speech_text = "This is a placeholder for the speech recognition result. In a real implementation, this would be the transcribed text of your question."
            st.session_state.recording = False
    
    # Display the recognized speech
    if st.session_state.speech_text:
        st.info("Recognized speech:")
        st.write(st.session_state.speech_text)
        question = st.session_state.speech_text

# Process question
if st.button("Get Answer") and context and question:
    with st.spinner("Processing your question..."):
        answer = get_answer(context, question, model_info)
    
    st.subheader("Answer:")
    st.success(answer)
    
    # Text-to-speech
    with st.spinner("Generating audio response..."):
        audio_file = text_to_speech(answer, lang_code)
    
    st.audio(audio_file, format='audio/mp3')
    

# Instructions
st.divider()
st.subheader("Implementation Notes:")
st.write("""
1. **Models Used**:
   - English: FLAN-T5 (Foundation model)
   - Kannada: MuRIL (Indic language model)
   - French: CamemBERT (International model)

2. **Input Methods**:
   - Text: Type your question directly
   - Speech: Click the microphone button and speak your question

3. **Workflow**:
   - Upload PDF documents
   - Choose input method (text or speech)
   - Get text and audio answers

4. **Technical Notes**:
   - PDF text extraction with pdfplumber
   - Answers generated using Hugging Face Transformers
   - Audio responses using Google's Text-to-Speech
   - Context limited to first 1000 characters for demo
   - Speech recognition is simulated for the demo
""")

st.caption("Note: This implementation uses MuRIL instead of IndicBERT for better compatibility and stability.\n For a production system, integrate a real speech-to-text API like Google Cloud Speech-to-Text.\n")