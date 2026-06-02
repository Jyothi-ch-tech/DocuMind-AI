import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind AI")
st.caption("Multi-PDF RAG Assistant powered by Gemini")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:

    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write("Uploaded Files:")

        for pdf in uploaded_files:
            st.write(f"📄 {pdf.name}")

    st.divider()

    st.write(f"Pages: {st.session_state.page_count}")
    st.write(f"Chunks: {st.session_state.chunk_count}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Knowledge Base"):
        st.session_state.vectorstore = None
        st.session_state.processed = False
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.messages = []
        st.rerun()

if uploaded_files and not st.session_state.processed:

    all_documents = []

    with st.spinner("Loading PDFs..."):

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            all_documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_documents)

    st.session_state.page_count = len(all_documents)
    st.session_state.chunk_count = len(chunks)

    with st.spinner("Creating Embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.processed = True

    st.success("Knowledge Base Ready!")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your PDFs")

if query and st.session_state.vectorstore:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching Documents..."):

            results = st.session_state.vectorstore.similarity_search_with_score(
                query,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc, score in results]
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        prompt = f"""
You are an expert document assistant.

Rules:
1. Answer only from the provided context.
2. Do not hallucinate.
3. If answer is unavailable, say:
   "I could not find this information in the uploaded documents."
4. Use bullet points when useful.

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        st.markdown(response.content)

        st.download_button(
            "Download Answer",
            response.content,
            file_name="answer.txt"
        )

        with st.expander("Sources Used"):

            for idx, (doc, score) in enumerate(results, 1):

                st.write(
                    f"Source {idx} | Similarity Score: {score:.4f}"
                )

                st.write(doc.page_content)

                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )