import streamlit as st
import os
import time
import sys, asyncio
from pathlib import Path
# ---------------------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

try:
    import groq as groq_sdk  
except Exception:
    groq_sdk = None 
# ---------------------------------------------------------------
# Streamlit runs user code in a thread that may not have an asyncio loop.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# ------------------------------------------------------------------------------
# Set up environment variables (robust loading)
load_dotenv(override=True)
local_env = Path(__file__).with_name(".env")
if local_env.exists():
    load_dotenv(local_env, override=True)

# Read from env or Streamlit secrets (if set)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

# Streamlit UI setup
st.set_page_config(page_title="Domain Specific Q/A: Legal Domain")
col1, col2, col3 = st.columns([1, 4, 1])
st.title("⚖️Domain Specific Model Question Answering and ChatBot : Legal")
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #ffd0d0; }
    div.stButton > button:active { background-color: #ff6262; }
    div[data-testid="stStatusWidget"] div button { display: none; }
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

#---------------------------------------------
#Streamlit sidebar
with st.sidebar:
    st.subheader("🛠️ Tools & Technologies")
    st.markdown(
        """
--**Domain**: Legal: Indian Laws
-- **Laws**: Criminal, Labour, Companies and Copyright laws 
- **LLM Model**: llama-3.3-70b 
- **Framework**: Langchain
- **Embeddings**: Google Generative AI 
- **Database**: FAISS 
- **Conversation history**:5-ConversationBufferWindowMemory
        """
    )

# --- [ Hard checks before we proceed (prevent 401 at runtime)
errors = []
if not os.environ.get("GOOGLE_API_KEY"):
    errors.append("Missing GOOGLE_API_KEY")
if not groq_api_key:
    errors.append("Missing GROQ_API_KEY")
elif not groq_api_key.startswith("gsk_"):
    errors.append("GROQ_API_KEY format looks wrong (should start with 'gsk_').")

if errors:
    st.error(" / ".join(errors) + ". Set them in the sidebar and rerun.")
    st.stop()


if groq_sdk is not None:
    try:
        groq_client = groq_sdk.Groq(api_key=groq_api_key)
        _ = groq_client.models.list()  
    except Exception as e:
        if e.__class__.__name__ == "AuthenticationError" or "invalid_api_key" in str(e).lower():
            st.error("Groq rejected your API key (invalid/expired). Double-check and paste a valid key in the sidebar.")
            st.stop()
        st.error(f"Groq key check failed: {e}")
        st.stop()
# ------------------------------------------------------------------------------

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Initialize embeddings and vector store
# NOTE: If you ever hit event-loop issues again, you can force REST:
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")  # <-- optional fallback
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot , your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the LLM
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
except Exception as e:
    msg = str(e)
    if "invalid_api_key" in msg.lower() or "authenticationerror" in e.__class__.__name__.lower():
        st.error("GROQ_API_KEY is invalid. Paste a valid key in the sidebar (should start with 'gsk_').")
        st.stop()
    raise  
# ------------------------------------------------------------------------------

# Set up the QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("💡Thinking...", expanded=True):
            result = qa.invoke({"question": input_prompt})
            message_placeholder = st.empty()
            full_response = "\n\n\n"

            # for debugging:
            # st.write(result)

            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
