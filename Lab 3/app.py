import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import streamlit as st
import model_utils as mu
import pandas as pd
import matplotlib.pyplot as plt

# Precompute and cache results
@st.cache_data(show_spinner=False)
def load_results():
    return mu.precompute_results()

all_results = load_results()

# App title
st.title("🏥 Medical QA Model Comparison")
st.markdown("Compare foundation models with domain-specific medical AI models")

# Sidebar with model info
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    - **FLAN-T5**: General-purpose foundation model
    - **BioGPT**: Medical generative model
    - **ClinicalBERT**: Medical extractive model
    """)
    st.markdown("**Evaluation Metric**: ROUGE Scores")
    st.markdown("""
    **ROUGE Metrics**:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    """)
    st.markdown("**Note**: Results precomputed for faster loading")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Q&A Demo", "Example Explorer", "Performance Analysis"])

with tab1:
    st.header("Medical Q&A Demo")
    st.markdown("Ask medical questions to compare model responses")
    
    # Custom Q&A interface
    context = st.text_area("Medical Context (or leave blank for examples):",
                         "Type 2 diabetes is characterized by insulin resistance...",
                         height=150)
    
    question = st.text_input("Question:", 
                           "What is the first-line treatment for type 2 diabetes?")
    
    if st.button("Get Answers", type="primary"):
        with st.spinner("Generating answers..."):
            # Get pipelines
            qa_pipelines = mu.initialize_models()
            # Generate predictions
            predictions = mu.generate_answers(qa_pipelines, context, question)
            # Evaluate
            reference = "metformin"  # Simplified for demo
            rouge_scores = mu.evaluate_rouge(predictions, reference)
            
            # Display results
            st.subheader("Model Responses")
            cols = st.columns(3)
            models = [
                ("FLAN-T5 (Foundation)", "flan_t5", "#FF6B6B"),
                ("BioGPT (Medical)", "biogpt", "#4ECDC4"),
                ("ClinicalBERT (Medical)", "clinicalbert", "#556270")
            ]
            
            for col, (name, key, color) in zip(cols, models):
                with col:
                    st.markdown(f"### {name}")
                    st.markdown(f"<div style='background-color:{color}20; padding:10px; border-radius:5px;'>{predictions[key]}</div>", 
                               unsafe_allow_html=True)
                    st.metric("ROUGE-L Score", f"{rouge_scores[key]['rougeL']:.3f}")
            
            # Show reference
            st.subheader("Reference Answer")
            st.info(reference)

with tab2:
    st.header("Example Explorer")
    st.markdown("Explore precomputed medical QA examples")
    
    # Example selector
    example_id = st.selectbox("Select example:", 
                            options=[1, 2, 3, 4, 5],
                            format_func=lambda x: f"Example {x}")
    
    # Get selected example
    result = next((r for r in all_results if r["id"] == example_id), None)
    
    if result:
        st.subheader("Medical Context")
        st.info(result["context"])
        st.subheader("Question")
        st.success(result["question"])
        
        # Display model answers
        st.subheader("Model Answers")
        cols = st.columns(3)
        models = [
            ("FLAN-T5", "flan_t5", "#FF6B6B"),
            ("BioGPT", "biogpt", "#4ECDC4"),
            ("ClinicalBERT", "clinicalbert", "#556270")
        ]
        
        for col, (name, key, color) in zip(cols, models):
            with col:
                st.markdown(f"##### {name}")
                st.markdown(f"<div style='background-color:{color}20; padding:10px; border-radius:5px;'>{result['predictions'][key]}</div>", 
                           unsafe_allow_html=True)
                st.metric("ROUGE-L", f"{result['scores'][key]['rougeL']:.3f}")
        
        # Visualization
        st.subheader("Performance Comparison")
        fig = mu.plot_example_comparison(result)
        st.pyplot(fig)
        
        # Reference answer
        st.subheader("Reference Answer")
        st.success(result["reference"])

with tab3:
    st.header("Performance Analysis")
    st.markdown("Overall model performance across all examples")
    
    # Calculate average scores
    scores_data = []
    for result in all_results:
        for model in ["flan_t5", "biogpt", "clinicalbert"]:
            scores_data.append({
                "model": model,
                "example_id": result["id"],
                "rouge1": result["scores"][model]["rouge1"],
                "rouge2": result["scores"][model]["rouge2"],
                "rougeL": result["scores"][model]["rougeL"]
            })
    
    scores_df = pd.DataFrame(scores_data)
    avg_scores = scores_df.groupby('model').mean().reset_index()
    
    # Show summary
    st.subheader("Average Performance")
    avg_df = avg_scores.copy()
    avg_df['model'] = avg_df['model'].map({
        'flan_t5': 'FLAN-T5',
        'biogpt': 'BioGPT',
        'clinicalbert': 'ClinicalBERT'
    })
    
    # Format numeric columns
    numeric_cols = ['rouge1', 'rouge2', 'rougeL']
    formatted_avg_df = avg_df.copy()
    formatted_avg_df[numeric_cols] = formatted_avg_df[numeric_cols].applymap(lambda x: f"{x:.3f}")
    
    # Display table
    st.dataframe(formatted_avg_df.style.background_gradient(
        cmap='Blues', 
        subset=numeric_cols
    ))
    
    # Show charts
    st.subheader("Performance Charts")
    comparison_fig = mu.plot_performance_comparison(all_results)
    st.pyplot(comparison_fig)
    
    # Detailed examples
    with st.expander("View All Examples"):
        for result in all_results:
            st.markdown(f"### Example {result['id']}: {result['question']}")
            st.write(f"**Context:** {result['context']}")
            st.write(f"**Reference:** {result['reference']}")
            
            cols = st.columns(3)
            for col, model in zip(cols, ["flan_t5", "biogpt", "clinicalbert"]):
                model_name = {
                    "flan_t5": "FLAN-T5",
                    "biogpt": "BioGPT",
                    "clinicalbert": "ClinicalBERT"
                }[model]
                with col:
                    st.markdown(f"**{model_name}**")
                    st.info(result['predictions'][model])
                    st.metric("ROUGE-L", f"{result['scores'][model]['rougeL']:.3f}")
            st.divider()