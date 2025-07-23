import streamlit as st
import pdfplumber
from openai import OpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

# Set title
st.set_page_config(page_title="PDF Chatbot Assistant", layout="wide")
st.title("📄 AiRa PDF Chat-Assist")

# Load OpenAI API key
client = OpenAI(api_key=st.secrets["openai"]["api_key"])
openai_api_key = st.secrets["openai"]["api_key"]

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    st.session_state.pdf_text = text
    st.success("✅ Document uploaded and processed!")
    st.text_area("📝 Extracted Text Preview", text[:1000])

    # Summarize the document
    if st.button("🔍 Summarize Document"):
        with st.spinner("Generating summary..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Summarize this document in one line:\n{text[:3000]}"}
                    ],
                    temperature=0.3,
                    max_tokens=50
                )
                summary = response.choices[0].message.content
                st.session_state.summary = summary
                st.success("📌 Summary:")
                st.write(summary)

            except Exception as e:
                st.error(f"❌ Error during summarization: {e}")

    # Vector-based Q&A setup
    if st.button("⚡️ Enable Smart Q&A"):
        with st.spinner("Indexing document and preparing chatbot..."):
            try:
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.create_documents([st.session_state.pdf_text])

                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = FAISS.from_documents(docs, embeddings)

                retriever = vectorstore.as_retriever()
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_api_key)

                st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
                st.success("✅ Q&A enabled! Ask questions below.")
            except Exception as e:
                st.error(f"❌ Error while setting up Q&A: {e}")

# Ask a question
if st.session_state.qa_chain:
    st.markdown("---")
    st.subheader("💬 Ask Questions About the Document")

    user_question = st.text_input("Ask a question", key="input")

    if user_question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
            st.session_state.chat_history.append((user_question, result["answer"]))

# Chat history display
if st.session_state.chat_history:
    st.markdown("### 🗨️ Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        message(q, is_user=True, key=f"user_{i}")
        message(a, is_user=False, key=f"bot_{i}")
