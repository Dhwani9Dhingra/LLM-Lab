import streamlit as st
import torch
import pandas as pd
import os
import numpy as np
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    BertForTokenClassification, BertForQuestionAnswering,
    pipeline
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="BERT NLP Lab",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---
def safe_get(df, index, column, default=""):
    """Safely get value from DataFrame with comprehensive null checks"""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return default
        if index >= len(df):
            return default
        row = df.iloc[index]
        if column not in row:
            return default
        return str(row[column]) if pd.notna(row[column]) else default
    except Exception as e:
        st.warning(f"Data access warning: {str(e)}")
        return default

# --- Model Loading ---
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_seq_model():
    try:
        model_name = "textattack/bert-base-uncased-imdb"
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load sequence classification model: {str(e)}")
        return None, None

@st.cache_resource(show_spinner="Loading NER model...")
def load_ner_model():
    try:
        model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
        tokenizer = BertTokenizer.from_pretrained("dslim/bert-base-NER")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load NER model: {str(e)}")
        return None, None

@st.cache_resource(show_spinner="Loading QA model...")
def load_qa_model():
    try:
        model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load QA model: {str(e)}")
        return None, None

# --- Data Loading ---
@st.cache_data(show_spinner="Loading sample data...")
def load_sample_data():
    """Load sample datasets with comprehensive error handling"""
    data_files = {
        "imdb": "synthetic_sentiment.csv",
        "conll": "synthetic_ner.csv",
        "squad": "synthetic_qa.csv"
    }
    
    loaded_data = {}
    for key, fname in data_files.items():
        try:
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                loaded_data[key] = df if not df.empty else pd.DataFrame()
                if df.empty:
                    st.warning(f"Empty dataset: {fname}")
            else:
                st.warning(f"File not found: {fname}")
                loaded_data[key] = pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading {fname}: {str(e)}")
            loaded_data[key] = pd.DataFrame()
    
    return loaded_data

# --- Evaluation Functions ---
@st.cache_data(show_spinner="Evaluating sentiment model...")
def evaluate_sentiment(_model, _tokenizer, texts, labels):
    """Evaluate sentiment analysis model"""
    predictions = []
    for text in texts:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _model(**inputs)
        predictions.append(torch.argmax(outputs.logits).item())
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "report": classification_report(labels, predictions)
    }

@st.cache_data(show_spinner="Evaluating NER model...")
def evaluate_ner(_model, _tokenizer, texts):
    """Evaluate NER model (simplified version)"""
    nlp = pipeline("ner", model=_model, tokenizer=_tokenizer, aggregation_strategy="simple")
    entities = []
    for text in texts:
        entities.append(nlp(text))
    return entities  # For demo - real evaluation needs ground truth

@st.cache_data(show_spinner="Evaluating QA model...")
def evaluate_qa(_model, _tokenizer, contexts, questions, answers):
    """Evaluate QA model"""
    exact_matches = 0
    f1_scores = []
    
    for context, question, true_answer in zip(contexts, questions, answers):
        inputs = _tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        pred_answer = _tokenizer.convert_tokens_to_string(
            _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        
        # Calculate metrics
        exact_matches += int(pred_answer.strip().lower() == true_answer.strip().lower())
        
        # Simple F1 calculation
        pred_tokens = set(pred_answer.lower().split())
        true_tokens = set(true_answer.lower().split())
        common = pred_tokens & true_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(true_tokens) if true_tokens else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return {
        "exact_match": exact_matches / len(answers),
        "f1": np.mean(f1_scores),
        "samples": len(answers)
    }

# --- Initialize Models and Data ---
seq_model, seq_tokenizer = load_seq_model()
ner_model, ner_tokenizer = load_ner_model()
qa_model, qa_tokenizer = load_qa_model()
sample_data = load_sample_data() or {}

# --- UI Components ---
st.title("BERT Model Applications in NLP")
st.markdown("""
Implementing BERT for Sequence Classification, Token Classification & Question Answering
""")

# Sidebar navigation
task = st.sidebar.selectbox(
    "Select NLP Task",
    ["Sequence Classification", "Token Classification", "Question Answering"],
)

# --- Model Information ---
st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Information")

with st.sidebar.expander("ℹ️ Technical Specifications"):
    st.markdown("""
    ### Sequence Classification
    **Model**: `textattack / bert-base-uncased-imdb`  
    - **Architecture**: BERT-Base (12 layers)  
    - **Max Sequence Length**: 512 tokens  

    ### Named Entity Recognition
    **Model**: `dslim / bert-base-NER`  
    - **Entity Types**: PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS  

    ### Question Answering  
    **Model**: `bert-large-uncased-whole-word-masking-finetuned-squad`  
    - **Training Data**: SQuAD 2.0  
    """)

# --- Task Implementations ---
if task == "Sequence Classification":
    st.header("📝 Squence Classification : Sentiment Analysis")
    
    # Get sample data
    df = sample_data.get("imdb", pd.DataFrame())
    sample_options = ["Custom Input"] + [f"Sample {i+1}" for i in range(min(3, len(df)))]
    
    selected_sample = st.selectbox(
        "Select Sample Review",
        sample_options,
        index=0
    )
    
    if selected_sample == "Custom Input":
        default_text = ""
    else:
        sample_idx = int(selected_sample.split()[1]) - 1
        default_text = safe_get(df, sample_idx, "text", "")
    
    input_text = st.text_area(
        "Input Text", 
        value=default_text,
        height=200,
        help="Enter a sentence or paragraph to analyze its sentiment"
    )
    
    if st.button("Analyze Sentiment"):
        if not seq_model or not seq_tokenizer:
            st.error("Model not loaded properly")
        elif not input_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    inputs = seq_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = seq_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    sentiment = "Positive" if torch.argmax(probs).item() == 1 else "Negative"
                    confidence = probs[0][torch.argmax(probs)].item()
                    
                    st.success(f"**Prediction**: {sentiment} (Confidence: {confidence:.2%})")
                    
                    fig, ax = plt.subplots(figsize=(2, 2))
                    sns.barplot(
                        x=["Negative", "Positive"], 
                        y=probs.detach().numpy()[0],
                        palette="viridis"
                    )
                    ax.set_ylabel("Confidence Score", fontsize=5)
                    ax.set_title("Sentiment Probability Distribution" , fontsize=6)
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    st.pyplot(fig,use_container_width=False) 
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # Evaluation Section
    if "imdb" in sample_data and not sample_data["imdb"].empty:
        with st.expander("Model Evaluation Metrics"):
            df = sample_data["imdb"]
            if "label" not in df.columns:
                st.warning("No labels found for evaluation")
            else:
                # Use first 100 samples for faster evaluation
                eval_texts = df["text"].tolist()[:100]
                eval_labels = df["label"].tolist()[:100]
                
                results = evaluate_sentiment(
                    seq_model, seq_tokenizer,
                    eval_texts, eval_labels
                )
                
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{results['accuracy']:.2%}")
                cols[1].metric("F1 Score", f"{results['f1']:.2%}")
                cols[2].metric("Precision", f"{results['precision']:.2%}")
                cols[3].metric("Recall", f"{results['recall']:.2%}")
                
                st.text("Classification Report:")
                st.text(results["report"])

elif task == "Token Classification":
    st.header("🔍 Token Classification : Named Entity Recognition")
    
    df = sample_data.get("conll", pd.DataFrame())
    sample_options = ["Custom Input"] + [f"Sample {i+1}" for i in range(min(3, len(df)))]
    
    selected_sample = st.selectbox(
        "Select Sample Text",
        sample_options,
        index=0
    )
    
    if selected_sample == "Custom Input":
        default_text = ""
    else:
        sample_idx = int(selected_sample.split()[1]) - 1
        default_text = safe_get(df, sample_idx, "text", "")
    
    input_text = st.text_area(
        "Input Text", 
        value=default_text,
        height=150,
        help="Enter text containing names, organizations, or locations"
    )
    
    if st.button("Extract Entities"):
        if not ner_model or not ner_tokenizer:
            st.error("Model not loaded properly")
        elif not input_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Identifying entities..."):
                try:
                    ner_pipeline = pipeline(
                        "ner", 
                        model=ner_model, 
                        tokenizer=ner_tokenizer,
                        aggregation_strategy="simple"
                    )
                    entities = ner_pipeline(input_text)
                    
                    if entities:
                        st.subheader("Identified Entities")
                        entity_df = pd.DataFrame({
                            "Entity": [ent.get("word", "") for ent in entities],
                            "Type": [ent.get("entity_group", "") for ent in entities],
                            "Confidence": [f"{ent.get('score', 0):.2%}" for ent in entities]
                        })
                        st.dataframe(entity_df)
                        
                except Exception as e:
                    st.error(f"Entity extraction failed: {str(e)}")

elif task == "Question Answering":
    st.header("❓ Question Answering System")
    
    df = sample_data.get("squad", pd.DataFrame())
    sample_options = ["Custom Input"] + [f"Sample {i+1}" for i in range(min(3, len(df)))]
    
    selected_sample = st.selectbox(
        "Select Sample QA Pair",
        sample_options,
        index=0
    )
    
    if selected_sample == "Custom Input":
        default_context = ""
        default_question = ""
    else:
        sample_idx = int(selected_sample.split()[1]) - 1
        default_context = safe_get(df, sample_idx, "context", "")
        default_question = safe_get(df, sample_idx, "question", "")
    
    context = st.text_area(
        "Context Paragraph", 
        value=default_context,
        height=150,
        help="Enter the text that contains the answer"
    )
    
    question = st.text_input(
        "Question",
        value=default_question,
        help="Ask a question about the context"
    )
    
    
    if st.button("Get Answer"):
        if not qa_model or not qa_tokenizer:
            st.error("Model not loaded properly")
        elif not context.strip() or not question.strip():
            st.warning("Please provide both context and question")
        else:
            with st.spinner("Searching for answer..."):
                try:
                    inputs = qa_tokenizer(
                        question, 
                        context, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = qa_model(**inputs)
                    
                    answer_start = torch.argmax(outputs.start_logits)
                    answer_end = torch.argmax(outputs.end_logits) + 1
                    
                    answer = qa_tokenizer.convert_tokens_to_string(
                        qa_tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0][answer_start:answer_end]
                        )
                    )
                    
                    if answer.strip():
                        st.success(f"**Answer**: {answer}")
                        
                        start_idx = context.lower().find(answer.lower())
                        
                        if start_idx != -1:
                            end_idx = start_idx + len(answer)
                            highlighted = (
                                context[:start_idx] +
                                '<mark style="background-color: #6A0572; color: white;">' +
                                context[start_idx:end_idx] +
                                '</mark>' +
                                context[end_idx:]
                            )
                            st.markdown(highlighted, unsafe_allow_html=True)
                        else:
                            st.info("Answer position not found in original context")
                    else:
                        st.warning("No answer found in the given context")
                except Exception as e:
                    st.error(f"Question answering failed: {str(e)}")

    # Evaluation Section
    if "squad" in sample_data and not sample_data["squad"].empty:
        with st.expander("Model Evaluation Metrics"):
            df = sample_data["squad"]
            if "answer" not in df.columns:
                st.warning("No answers found for evaluation")
            else:
                # Use first 20 samples for faster evaluation
                eval_contexts = df["context"].tolist()[:20]
                eval_questions = df["question"].tolist()[:20]
                eval_answers = df["answer"].tolist()[:20]
                
                results = evaluate_qa(
                    qa_model, qa_tokenizer,
                    eval_contexts, eval_questions, eval_answers
                )
                
                cols = st.columns(2)
                cols[0].metric("Exact Match", f"{results['exact_match']:.2%}")
                cols[1].metric("F1 Score", f"{results['f1']:.2%}")
                st.caption(f"Evaluated on {results['samples']} samples")

# Footer
st.markdown("---")
st.caption("""
**BERT NLP Application  
Models provided by Hugging Face Transformers | UI built with Streamlit
""")