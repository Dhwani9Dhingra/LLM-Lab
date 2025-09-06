# Multilingual Transformer NLP Application

A Streamlit-based web application that performs various Natural Language Processing tasks using transformer models for English, Kannada, and French languages.


## Features
- **Named Entity Recognition (NER)**: Identify and classify entities in text
- **Text Summarization**: Generate concise summaries of longer texts
- **Machine Translation**: Translate text between supported languages
- **Question Answering**: Answer questions based on provided context
- **Multilingual Support**: Process text in English, Kannada, and French
- **Multiple Input Methods**: Text input or PDF file upload


## Supported Languages and Models
- **English**: FLAN-T5 model (`google/flan-t5-base`)
- **Kannada**: MuRIL model (`google/muril-base-cased`)
- **French**: CamemBERT model (`camembert-base`)


### Architecture
The application follows a modular architecture:
1. **User Interface Layer**: Streamlit-based web interface
2. **Model Management Layer**: Handles loading and caching of transformer models
3. **Processing Layer**: Performs the actual NLP tasks using the loaded models
4. **Input/Output Layer**: Manages text extraction from PDFs and result presentation


### NLP Tasks Implementation
1. **Named Entity Recognition**: Uses each language's base model to identify entities
2. **Text Summarization**: Leverages the sequence-to-sequence capabilities of the models
3. **Machine Translation**: Utilizes the cross-lingual understanding of the models
4. **Question Answering**: Implements context-based answer extraction


## Model Information
### FLAN-T5 (English)
- A instruction-tuned version of T5 that excels at following natural language instructions
- Strong performance on various NLP tasks without task-specific fine-tuning
- Good at zero-shot and few-shot learning scenarios

### MuRIL (Kannada)
- Multilingual Representations for Indian Languages
- Specifically designed for Indian languages including Kannada
- Based on BERT architecture with support for 17 Indian languages

### CamemBERT (French)
- A French version of the BERT model
- Pretrained on a large French corpus
- Excellent performance on French NLP tasks

