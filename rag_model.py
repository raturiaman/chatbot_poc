import os
import openai
import PyPDF2
import numpy as np

class HRPolicyRAG:
    def __init__(self, pdf_path, openai_api_key, embedding_model="text-embedding-ada-002"):
        # Set the OpenAI API key.
        openai.api_key = openai_api_key
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.documents = self._extract_text_from_pdf(pdf_path)
        # Pre-compute embeddings only if documents were extracted.
        if self.documents:
            self.doc_embeddings = [self.get_embedding(doc) for doc in self.documents]
        else:
            self.doc_embeddings = []

    def _extract_text_from_pdf(self, pdf_path):
        texts = []
        # If the given path is a directory, iterate over all PDF files.
        if os.path.isdir(pdf_path):
            for file in os.listdir(pdf_path):
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(pdf_path, file)
                    try:
                        with open(full_path, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                text = page.extract_text()
                                if text:
                                    texts.append(text)
                    except Exception as e:
                        texts.append(f"Error reading PDF {file}: {e}")
        else:
            # If a single file is provided.
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            texts.append(text)
            except Exception as e:
                texts.append("Error reading PDF: " + str(e))
        return texts

    def get_embedding(self, text):
        # Use OpenAI API to get an embedding.
        response = openai.Embedding.create(
            input=text,
            model=self.embedding_model
        )
        embedding = response['data'][0]['embedding']
        return np.array(embedding)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    def get_answer(self, question, threshold=0.75):
        # Check if there are any document embeddings.
        if not self.doc_embeddings:
            return "No HR policy documents available. Please contact HR department."
        
        # Compute embedding for the question.
        question_embedding = self.get_embedding(question)
        similarities = [self.cosine_similarity(question_embedding, doc_emb) for doc_emb in self.doc_embeddings]
        
        # Guard against an empty similarities list.
        if not similarities:
            return "No HR policy documents available. Please contact HR department."
        
        max_sim = max(similarities)
        best_index = np.argmax(similarities)

        # If similarity is below threshold, fallback.
        if max_sim < threshold:
            return "Please contact HR department"
        else:
            context = self.documents[best_index]
            prompt = (
                f"Using the following HR policy context, answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an HR policy assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer