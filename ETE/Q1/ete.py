import streamlit as st
import pdfplumber
from transformers import (
    pipeline, AutoTokenizer, AutoModelForQuestionAnswering, 
    AutoModelForSeq2SeqLM
)
import torch
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Multilingual Transformer NLP",
    page_icon="🤖",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading language models...")
def load_models():
    models = {}
    
    # English models (FLAN-T5)
    models["en"] = {
        "ner": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple"),
        "summarization": pipeline("summarization", model="google/flan-t5-base"),
        "translation": pipeline("translation_en_to_fr", model="google/flan-t5-base"),
        "qa": {"model": AutoModelForQuestionAnswering.from_pretrained("google/flan-t5-base"),
               "tokenizer": AutoTokenizer.from_pretrained("google/flan-t5-base")},
        "lang": "en",
        "model_name": "FLAN-T5 (English)"
    }
    
    # Kannada models (MuRIL)
    models["kn"] = {
        "ner": pipeline("ner", model="google/muril-base-cased", aggregation_strategy="simple"),
        "summarization": pipeline("summarization", model="google/muril-base-cased"),
        "translation": pipeline("translation", model="google/muril-base-cased"),
        "qa": {"model": AutoModelForQuestionAnswering.from_pretrained("google/muril-base-cased"),
               "tokenizer": AutoTokenizer.from_pretrained("google/muril-base-cased")},
        "lang": "kn",
        "model_name": "MuRIL (Kannada)"
    }
    
    # French models (CamemBERT)
    models["fr"] = {
        "ner": pipeline("ner", model="camembert-base", aggregation_strategy="simple"),
        "summarization": pipeline("summarization", model="camembert-base"),
        "translation": pipeline("translation_en_to_fr", model="camembert-base"),
        "qa": {"model": AutoModelForQuestionAnswering.from_pretrained("camembert-base"),
               "tokenizer": AutoTokenizer.from_pretrained("camembert-base")},
        "lang": "fr",
        "model_name": "CamemBERT (French)"
    }
    
    return models

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def get_answer(context, question, model_info):
    tokenizer = model_info["qa"]["tokenizer"]
    model = model_info["qa"]["model"]
    
    # Format input for FLAN-T5 style models
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def perform_ner(text, model_info):
    return model_info["ner"](text)

def perform_summarization(text, model_info):
    # Adjust max_length based on text length
    max_length = min(len(text) // 4, 1024)
    return model_info["summarization"](text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']

def perform_translation(text, model_info, target_lang):
    # For translation to French using available models
    if target_lang == "fr":
        result = model_info["translation"](f"translate English to French: {text}")
        return result[0]['translation_text']
    else:
        # For other languages, use a generic approach
        return f"Translation to {target_lang} would be performed here with the appropriate model."

# --- Streamlit UI ---
st.title("🌍 Multilingual Transformer NLP Tasks")
st.subheader("Named Entity Recognition, Text Summarization, and Machine Translation")

# Load models
models = load_models()

# Sidebar configuration
st.sidebar.header("Configuration")
task = st.sidebar.selectbox(
    "Select NLP Task",
    ["Named Entity Recognition", "Text Summarization", "Machine Translation", "Question Answering"]
)

selected_lang = st.sidebar.radio("Select Language", ["English", "Kannada", "French"])
lang_code = {"English": "en", "Kannada": "kn", "French": "fr"}[selected_lang]
model_info = models[lang_code]

# Display model information in sidebar
st.sidebar.divider()
st.sidebar.subheader("Model Information")
st.sidebar.markdown(f"**{selected_lang} Model:** {model_info['model_name']}")

# Main content area
st.header(f"{task} - {selected_lang}")

# Input method selection
input_method = st.radio("Input Method:", ["Text", "PDF Upload"], horizontal=True)

input_text = ""
if input_method == "Text":
    input_text = st.text_area("Enter text:", height=200, 
                             placeholder=f"Enter text in {selected_lang} for processing...")
else:  # PDF Upload
    uploaded_file = st.file_uploader(f"Upload {selected_lang} PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            input_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF processed successfully!")
        st.caption(f"Extracted text length: {len(input_text)} characters")
        
        # Show preview
        with st.expander("View extracted text"):
            st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)

# Task-specific UI elements
if task == "Named Entity Recognition":
    if st.button("Extract Entities") and input_text:
        with st.spinner("Identifying entities..."):
            try:
                entities = perform_ner(input_text, model_info)
                
                if entities:
                    st.subheader("Identified Entities")
                    entity_df = pd.DataFrame({
                        "Entity": [ent.get("word", "") for ent in entities],
                        "Type": [ent.get("entity_group", "") for ent in entities],
                        "Confidence": [f"{ent.get('score', 0):.2%}" for ent in entities]
                    })
                    st.dataframe(entity_df)
                    
                    # Highlight entities in text
                    highlighted_text = input_text
                    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
                        start, end = entity['start'], entity['end']
                        entity_text = highlighted_text[start:end]
                        highlighted_text = (
                            highlighted_text[:start] +
                            f'<mark style="background-color: #6A0572; color: white;">{entity_text}</mark>' +
                            highlighted_text[end:]
                        )
                    
                    st.subheader("Text with Entities Highlighted")
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    st.info("No entities found in the text.")
                    
            except Exception as e:
                st.error(f"Entity extraction failed: {str(e)}")

elif task == "Text Summarization":
    if st.button("Generate Summary") and input_text:
        with st.spinner("Generating summary..."):
            try:
                summary = perform_summarization(input_text, model_info)
                st.subheader("Summary")
                st.success(summary)
                
            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")

elif task == "Machine Translation":
    target_lang = st.selectbox("Translate to:", ["English", "Kannada", "French"])
    target_lang_code = {"English": "en", "Kannada": "kn", "French": "fr"}[target_lang]
    
    if st.button("Translate") and input_text:
        with st.spinner("Translating..."):
            try:
                translation = perform_translation(input_text, model_info, target_lang_code)
                st.subheader(f"Translation to {target_lang}")
                st.success(translation)
                
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")

elif task == "Question Answering":
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and input_text and question:
        with st.spinner("Finding answer..."):
            try:
                answer = get_answer(input_text, question, model_info)
                st.subheader("Answer")
                st.success(answer)
                
            except Exception as e:
                st.error(f"Question answering failed: {str(e)}")

# Footer and information
st.divider()
st.subheader("Implementation Details:")
st.write("""
This application demonstrates three key NLP tasks using transformer models for multiple languages:

1. **Named Entity Recognition**: Identifies and classifies named entities in text
2. **Text Summarization**: Generates concise summaries of longer texts
3. **Machine Translation**: Translates text between different languages
4. **Question Answering**: Answers questions based on provided context

**Supported Languages and Models**:
- English: FLAN-T5 model
- Kannada: MuRIL model
- French: CamemBERT model

**Technical Stack**:
- Hugging Face Transformers for model inference
- Streamlit for the web interface
- pdfplumber for PDF text extraction
""")