import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_rag_pipeline(index_path: str = "faiss_index"):
    """Load RAG pipeline with retriever."""
    
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Cleaner prompt template
    template = """You are a knowledgeable movie expert. Using the provided context from movie plots, answer the question concisely in 2-3 sentences.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    - Provide a clear, summarized answer in paragraph form
    - Base your answer only on the information in the context
    - Keep the answer focused and relevant to the question
    - If the answer cannot be determined from the context, say "I don't have enough information to answer this question based on the available movie plots."

    ANSWER:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Load model
    model_name = "unsloth/gemma-3-270m-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,         # generate only a short answer
        temperature=0.0,            # deterministic
        do_sample=False,            # avoid repetition
        return_full_text=False,     # don't echo the prompt
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa