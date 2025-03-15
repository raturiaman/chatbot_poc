import streamlit as st
from rag_model import HRPolicyRAG

st.title("HR Policy Chatbot")

# Retrieve the PDF path and OpenAI API key from Streamlit secrets
pdf_path = st.secrets["PDF_PATH"]         # e.g., "pdfs"
openai_api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_model():
    return HRPolicyRAG(pdf_path, openai_api_key)

model = load_model()

st.markdown("### Ask a question about our HR policies:")
user_question = st.text_input("Enter your question here:")

if st.button("Get Answer") and user_question:
    answer = model.get_answer(user_question)
    st.markdown("#### Answer:")
    st.write(answer)