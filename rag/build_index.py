import os
import faiss
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from tqdm import tqdm
from .load_data import load_movie_data
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_faiss_index(data_path: str, save_path: str = "faiss_index") -> None:
    print("Loading movie data...")
    movies = load_movie_data(data_path)
    print(f"Loaded {len(movies)} movies")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    docs = []
    for movie in tqdm(movies, desc="Creating documents"):
        splits = text_splitter.split_text(movie["plot"])
        for chunk in splits:
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

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    print("Saving index...")
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

if __name__ == "__main__":
    build_faiss_index("data/wiki_movie_plots_deduped.csv")
