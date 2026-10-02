from src.loader import load_pdf
from src.chunker import split_documents
from src.embeddings import create_embeddings
from src.retriever import create_vectorstore


def build_rag_pipeline(pdf_path):

    # Step 1: Load PDF
    documents = load_pdf(pdf_path)

    print("Documents loaded:", len(documents))


    # Step 2: Split documents
    chunks = split_documents(documents)

    print("Chunks created:", len(chunks))


    # Step 3: Create embeddings
    embeddings = create_embeddings()

    print("Embedding model loaded")


    # Step 4: Create vector database
    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print("Vector database created")


    return vectorstore, embeddings