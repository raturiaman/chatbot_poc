import streamlit as st
from rag_model import HRPolicyRAG

st.title("HR Policy Chatbot")

# Retrieve the PDF path and OpenAI API key from Streamlit secrets
pdf_path = st.secrets["PDF_PATH"]         # e.g., "pdfs"
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Debug: Display the secrets (remove sensitive info in production!)
st.write("PDF Path:", pdf_path)
st.write("API Key provided:", bool(openai_api_key))  # Show True/False for API key

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = HRPolicyRAG(pdf_path, openai_api_key)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("Model did not load. Please check your configuration.")
else:
    st.markdown("### Ask a question about our HR policies:")
    user_question = st.text_input("Enter your question here:")

    if st.button("Get Answer") and user_question:
        try:
            answer = model.get_answer(user_question)
            st.markdown("#### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")