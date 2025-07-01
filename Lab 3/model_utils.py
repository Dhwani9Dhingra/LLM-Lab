from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Configuration
MODELS = {
    "flan_t5": "google/flan-t5-base",
    "biogpt": "microsoft/BioGPT-Large",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT"
}

# Synthetic medical QA dataset
MEDICAL_QA_EXAMPLES = [
    {
        "id": 1,
        "context": "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency. First-line treatment is typically metformin, followed by sulfonylureas or SGLT2 inhibitors. Lifestyle modifications are crucial for management.",
        "question": "What is the first-line treatment for type 2 diabetes?",
        "answer": "metformin"
    },
    {
        "id": 2,
        "context": "Myocardial infarction occurs when blood flow decreases to the coronary artery, causing heart muscle damage. Primary symptoms include chest pain, shortness of breath, and nausea.",
        "question": "What are the primary symptoms of myocardial infarction?",
        "answer": "chest pain, shortness of breath, nausea"
    },
    {
        "id": 3,
        "context": "ACE inhibitors like lisinopril are used for hypertension management. They work by inhibiting angiotensin-converting enzyme, reducing vasoconstriction.",
        "question": "How do ACE inhibitors work?",
        "answer": "inhibiting angiotensin-converting enzyme, reducing vasoconstriction"
    },
    {
        "id": 4,
        "context": "Asthma is a chronic inflammatory disease of the airways characterized by bronchospasm, wheezing, and shortness of breath. Rescue inhalers contain short-acting beta-agonists like albuterol.",
        "question": "What medication is in rescue inhalers for asthma?",
        "answer": "albuterol"
    },
    {
        "id": 5,
        "context": "Antibiotics are used to treat bacterial infections. Overuse can lead to antibiotic resistance. Penicillin was the first widely used antibiotic, discovered by Alexander Fleming in 1928.",
        "question": "Who discovered penicillin?",
        "answer": "Alexander Fleming"
    }
]

def initialize_models():
    """Initialize models and cache them in memory"""
    if not hasattr(initialize_models, 'pipelines'):
        initialize_models.pipelines = {}
        
        # FLAN-T5 (foundation model)
        initialize_models.pipelines["flan_t5"] = pipeline(
            "text2text-generation",
            model=MODELS["flan_t5"],
            tokenizer=MODELS["flan_t5"]
        )
        
        # BioGPT (domain-specific generative)
        biogpt_tokenizer = AutoTokenizer.from_pretrained(MODELS["biogpt"])
        biogpt_model = AutoModelForCausalLM.from_pretrained(MODELS["biogpt"])
        initialize_models.pipelines["biogpt"] = pipeline(
            "text-generation",
            model=biogpt_model,
            tokenizer=biogpt_tokenizer
        )
        
        # ClinicalBERT (domain-specific extractive)
        initialize_models.pipelines["clinicalbert"] = pipeline(
            "question-answering",
            model=MODELS["clinicalbert"],
            tokenizer=MODELS["clinicalbert"]
        )
    return initialize_models.pipelines

def generate_answers(qa_pipelines, context, question):
    """Generate answers from all models"""
    results = {}
    
    # FLAN-T5
    results["flan_t5"] = qa_pipelines["flan_t5"](
        f"question: {question} context: {context}",
        max_length=100
    )[0]['generated_text']
    
    # BioGPT
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    results["biogpt"] = qa_pipelines["biogpt"](
        prompt,
        max_new_tokens=50,
        num_return_sequences=1
    )[0]['generated_text'].split("Answer:")[-1].strip()
    
    # ClinicalBERT
    results["clinicalbert"] = qa_pipelines["clinicalbert"](
        question=question,
        context=context
    )['answer']
    
    return results

def evaluate_rouge(predictions, reference):
    """Evaluate predictions using ROUGE metrics"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {}
    for model, pred in predictions.items():
        raw_scores = scorer.score(reference, pred)
        scores[model] = {
            "rouge1": raw_scores["rouge1"].fmeasure,
            "rouge2": raw_scores["rouge2"].fmeasure,
            "rougeL": raw_scores["rougeL"].fmeasure
        }
    return scores

def precompute_results():
    """Precompute and cache results for all examples"""
    cache_file = "model_results_cache.json"
    
    # Use cached results if available
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Compute results if cache doesn't exist
    qa_pipelines = initialize_models()
    all_results = []
    
    for example in MEDICAL_QA_EXAMPLES:
        predictions = generate_answers(qa_pipelines, example["context"], example["question"])
        scores = evaluate_rouge(predictions, example["answer"])
        
        all_results.append({
            "id": example["id"],
            "context": example["context"],
            "question": example["question"],
            "reference": example["answer"],
            "predictions": predictions,
            "scores": scores
        })
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def get_results_by_id(results, example_id):
    """Get results for specific example ID"""
    for result in results:
        if result["id"] == example_id:
            return result
    return None

def plot_performance_comparison(results):
    """Create performance comparison charts"""
    # Prepare data
    scores_data = []
    for result in results:
        for model in ["flan_t5", "biogpt", "clinicalbert"]:
            scores_data.append({
                "model": model,
                "example_id": result["id"],
                "rouge1": result["scores"][model]["rouge1"],
                "rouge2": result["scores"][model]["rouge2"],
                "rougeL": result["scores"][model]["rougeL"]
            })
    
    df = pd.DataFrame(scores_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average scores by model
    avg_scores = df.groupby('model').mean().reset_index()
    colors = {'flan_t5': '#FF6B6B', 'biogpt': '#4ECDC4', 'clinicalbert': '#556270'}
    
    # Bar plot - Average ROUGE Scores
    ax1 = axes[0, 0]
    bar_width = 0.25
    index = np.arange(len(avg_scores))
    
    ax1.bar(index, avg_scores['rouge1'], bar_width, label='ROUGE-1', color=[colors[m] for m in avg_scores['model']])
    ax1.bar(index + bar_width, avg_scores['rouge2'], bar_width, label='ROUGE-2', color=[colors[m] for m in avg_scores['model']], alpha=0.8)
    ax1.bar(index + 2*bar_width, avg_scores['rougeL'], bar_width, label='ROUGE-L', color=[colors[m] for m in avg_scores['model']], alpha=0.6)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Average ROUGE Scores by Model')
    ax1.set_xticks(index + bar_width)
    ax1.set_xticklabels(['FLAN-T5', 'BioGPT', 'ClinicalBERT'])
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Line plot - Performance across examples
    ax2 = axes[0, 1]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax2.plot(model_data['example_id'], model_data['rougeL'], 
                label=model, marker='o', color=colors[model])
    
    ax2.set_xlabel('Example ID')
    ax2.set_ylabel('ROUGE-L Score')
    ax2.set_title('Model Performance Across Examples')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.set_xticks(range(1, len(MEDICAL_QA_EXAMPLES)+1))
    
    # Box plot - Score distribution
    ax3 = axes[1, 0]
    box_data = [df[df['model'] == model]['rougeL'] for model in df['model'].unique()]
    ax3.boxplot(box_data, labels=['FLAN-T5', 'BioGPT', 'ClinicalBERT'])
    ax3.set_ylabel('ROUGE-L Score')
    ax3.set_title('Score Distribution Comparison')
    ax3.set_ylim(0, 1)
    
    # Radar plot - Score profile
    ax4 = axes[1, 1]
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for model in avg_scores['model']:
        values = avg_scores[avg_scores['model'] == model][['rouge1', 'rouge2', 'rougeL']].values[0].tolist()
        values += values[:1]
        ax4.plot(angles, values, linewidth=2, linestyle='solid', 
                label=model, color=colors[model])
        ax4.fill(angles, values, alpha=0.1, color=colors[model])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Score Profile Comparison', size=12)
    ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    return fig

def plot_example_comparison(example_data):
    """Create example-specific comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ["flan_t5", "biogpt", "clinicalbert"]
    model_names = ["FLAN-T5", "BioGPT", "ClinicalBERT"]
    colors = ['#FF6B6B', '#4ECDC4', '#556270']
    
    # Extract scores
    rouge1 = [example_data['scores'][m]["rouge1"] for m in models]
    rouge2 = [example_data['scores'][m]["rouge2"] for m in models]
    rougeL = [example_data['scores'][m]["rougeL"] for m in models]
    
    # Bar positions
    x = np.arange(len(model_names))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, rouge1, width, label='ROUGE-1', color=colors, alpha=0.9)
    ax.bar(x, rouge2, width, label='ROUGE-2', color=colors, alpha=0.7)
    ax.bar(x + width, rougeL, width, label='ROUGE-L', color=colors, alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Performance on Example {example_data["id"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add score labels
    for i, model in enumerate(models):
        ax.text(i - width, rouge1[i] + 0.02, f'{rouge1[i]:.2f}', ha='center')
        ax.text(i, rouge2[i] + 0.02, f'{rouge2[i]:.2f}', ha='center')
        ax.text(i + width, rougeL[i] + 0.02, f'{rougeL[i]:.2f}', ha='center')
    
    return fig