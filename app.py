import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.pipeline import build_rag_pipeline
from src.retriever import retrieve_documents
from src.image_processor import process_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Multimodal RAG")
st.write("Upload documents or images and ask questions using RAG and Gemini.")


# ============================================================
# DOCUMENT RAG SECTION
# ============================================================

st.header("📚 Document Q&A")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key="pdf_uploader"
)


if uploaded_file is not None:

    # Save uploaded PDF
    with open("data/uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully.")

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            vectorstore, embeddings = build_rag_pipeline(
                "data/uploaded.pdf"
            )

            st.session_state["vectorstore"] = vectorstore

        st.success("Document processed successfully!")

st.subheader("💬 Prompt")

question = st.text_input(
    "Ask a question about the document:",
    key="document_question"
)


if st.button("Search Document"):

    if "vectorstore" not in st.session_state:

        st.warning("Please upload and process the document first.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        vectorstore = st.session_state["vectorstore"]

        with st.spinner("Searching the document..."):

            results_with_scores = (
                vectorstore.similarity_search_with_relevance_scores(
                    question,
                    k=3
                )
            )

        # Minimum relevance required
        RELEVANCE_THRESHOLD = 0.40

        relevant_results = [
            (doc, score)
            for doc, score in results_with_scores
            if score >= RELEVANCE_THRESHOLD
        ]

        # No sufficiently relevant information
        if not relevant_results:

            st.warning(
                "I don't know based on the uploaded document."
            )

        else:

            results = [
                doc
                for doc, score in relevant_results
            ]

            context = "\n\n".join(
                doc.page_content
                for doc in results
            )

            st.subheader("🔎 Retrieved Context")

            st.write(context)

# ============================================================
# DIVIDER
# ============================================================

st.divider()


# ============================================================
# MULTIMODAL IMAGE Q&A SECTION
# ============================================================

st.header("🖼️ Multimodal Image Q&A")


uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)


if uploaded_image is not None:

    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_question = st.text_input(
        "Ask something about the image:",
        key="image_question"
    )

    if st.button("Ask Question"):

        # Check API key internally
        if not GEMINI_API_KEY:

            st.error(
                "Gemini API key is not configured. "
                "Please check your .env file."
            )

        elif not image_question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Analyzing image..."):

                answer = process_image(
                    image,
                    image_question,
                    GEMINI_API_KEY
                )

            st.subheader("💡 Answer")

            st.write(answer)