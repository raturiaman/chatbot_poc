import streamlit as st
from rag_model import HRPolicyRAG

st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")
st.title("HR Policy Chatbot")

# Retrieve sensitive data from Streamlit Cloud secrets
pdf_path = st.secrets["PDF_PATH"]         # e.g., "pdfs"
openai_api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_model():
    return HRPolicyRAG(pdf_path, openai_api_key)

model = load_model()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Define a list of suggested questions.
suggestions = [
    "What is the leave policy?",
    "How can I update my personal details?",
    "What benefits do employees receive?"
]

# Display suggestion buttons.
st.markdown("#### Suggested Questions:")
cols = st.columns(len(suggestions))
for idx, sug in enumerate(suggestions):
    if cols[idx].button(sug, key=f"sug_{idx}"):
        st.session_state["messages"].append({"role": "user", "content": sug})
        with st.spinner("Generating answer..."):
            answer = model.get_answer(sug)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.experimental_rerun()

# Display conversation history.
st.markdown("### Conversation")
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input at the footer.
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("Generating answer..."):
        answer = model.get_answer(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.experimental_rerun()