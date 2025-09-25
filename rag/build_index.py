import os
from tqdm import tqdm
import faiss
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .load_data import load_movie_data

def build_faiss_index(data_path: str, save_path: str = "faiss_index", chunk_size: int = 500, chunk_overlap: int = 50):
    """Build FAISS index from movie plots with chunking and efficient embedding."""

    print("Loading movie data...")
    movies = load_movie_data(data_path)
    print(f"Loaded {len(movies)} movies")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create documents with chunked plot
    docs = []
    for movie in tqdm(movies, desc="Chunking plots"):
        text = movie["plot"]
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": movie["title"],
                    "year": movie["release_year"],
                    "director": movie["director"],
                    "cast": movie["cast"],
                    "origin": movie["origin_ethnicity"],
                    "wiki": movie["wiki_page"],
                    "genre": movie["genre"]
                }
            ))

    print(f"Total chunks created: {len(docs)}")

    # Embeddings
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    print("Embedding documents...")
    vectors = embedding_model.embed_documents([doc.page_content for doc in docs])

    # Create FAISS vectorstore
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip([doc.page_content for doc in docs], vectors)),
        embedding=embedding_model,
        metadatas=[doc.metadata for doc in docs]
    )

    # Save FAISS index
    print("Saving FAISS index...")
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

if __name__ == "__main__":
    build_faiss_index("data/wiki_movie_plots_deduped.csv")
