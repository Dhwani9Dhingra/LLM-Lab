import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import requests

# --- Streamlit App Title ---
st.title("Story Generation System")

# --- Sidebar Controls ---
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
max_length = st.sidebar.slider("Max Output Length (tokens)", min_value=100, max_value=2000, value=300, step=50)

# --- Get Available Ollama Models ---
@st.cache_data
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = [model["name"] for model in response.json().get("models", [])]
        return sorted(models)
    except Exception as e:
        st.error(f"Error fetching models from Ollama: {e}")
        return ["gemma:2b"]  # fallback

available_models = get_ollama_models()
selected_model = st.sidebar.radio("Ollama Model", options=available_models, index=0)

# --- User Input ---
st.header("Ask for a Story")
user_query = st.text_input("Enter your story topic or prompt:", value="The Death of King Arthur")

# --- Load Data and Initialize ChromaDB ---
@st.cache_resource
def load_chromadb_collection():
    client = chromadb.PersistentClient(r"C:\Users\dhwan\Desktop\clg\LLM\Stroy-Generation-using-multiple-models")
    collection = client.get_or_create_collection("folk_stories")
    return client, collection

client, collection = load_chromadb_collection()

@st.cache_data
def load_text_data():
    df = pd.read_csv(r"C:\Users\dhwan\Desktop\clg\LLM\Stroy-Generation-using-multiple-models\data\1000Folk_Story_around_the_Globe.csv")
    story_seq = ['genre', 'title', 'full_text']
    
    def preprocess_text(text):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        return text

    df['full_text'] = df['full_text'].apply(preprocess_text)
    df = df[story_seq]
    return df

text_data = load_text_data()

# --- Helper Functions ---
def rag_prompt(topic, length=500):
    return f"Write a {length}-word story on the topic of {topic}. "

def query_ollama(prompt, model="gemma:2b", temperature=0.7, max_tokens=500):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTPError: {e}\nIs the Ollama server running and is the model '{model}' pulled?")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def retrieve_stories(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    stories = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        stories.append({
            "title": meta.get('title', 'Unknown'),
            "genre": meta.get('genre', 'Unknown'),
            "text": doc
        })
    return stories

# --- Main Logic ---
if st.button("Get Stories"):
    with st.spinner("Retrieving similar stories..."):
        retrieved_stories = retrieve_stories(user_query, n_results=3)
        st.subheader("Similar Stories from Database")
        for idx, story in enumerate(retrieved_stories, 1):
            with st.expander(f"{idx}. {story['title']} ({story['genre']})"):
                st.write(story['text'][:1000] + ("..." if len(story['text']) > 1000 else ""))

    with st.spinner(f"Generating a new story using {selected_model}..."):
        prompt = rag_prompt(user_query, length=max_length)
        ollama_response = query_ollama(prompt, model=selected_model, temperature=temperature, max_tokens=max_length)
        if ollama_response:
            st.subheader("Generated Story")
            st.write(ollama_response)
        else:
            st.warning("No response from Ollama.")

# --- Footer ---
st.markdown("---")
st.markdown("**Note:** Ensure the Ollama server is running and the selected model is pulled to use it at `http://localhost:11434`.")
