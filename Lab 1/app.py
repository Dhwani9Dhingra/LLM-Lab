import streamlit as st
import requests

# Set your Groq API key here
GROQ_API_KEY = ""

# Define the models available from Groq
MODELS = {
    "Llama 3": "llama-3.3-70b-versatile",
    "Gemini 2.5 flash": "gemma2-9b-it",
    # Add more models as needed
}

# Function to call Groq API
def call_groq_api(prompt, model, max_tokens, temperature):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    print(f"Calling API with data: {data}")  # Debug print

    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"API response status: {response.status_code}")  # Debug print
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")  # Debug print
        return {"error": str(e)}

# Streamlit UI
def main():
    st.title("Chatbot")

    # Sidebar for model selection and parameter adjustments
    st.sidebar.header("Model Settings")
    selected_model = st.sidebar.selectbox("Select a model", options=list(MODELS.keys()))
    max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=5000, value=150)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7)

    # Text input for the user's question
    user_input = st.text_input("Ask me for a recipe or any cooking tips:")

    if user_input:
        # Display loading message while waiting for response
        with st.spinner('Thinking...'):
            # Call Groq API with the selected model and parameters
            response = call_groq_api(user_input, MODELS[selected_model], max_tokens, temperature)
            print(f"Full API response: {response}")  # Debug print

            # Display the response or error
            if "error" in response:
                st.error(f"Error: {response['error']}")
                st.write("Please check the terminal/console for more details.")
            elif "choices" in response and len(response["choices"]) > 0:
                st.text("Answer: " + response["choices"][0]["message"]["content"])
            else:
                st.text("Sorry, I couldn't generate an answer for that.")
                st.write("Full response:", response)  # Show full response for debugging

if __name__ == "__main__":
    main()