import streamlit as st
import os
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings
from dotenv import load_dotenv

load_dotenv()

def read_doc(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunks(docs, chunksize=800, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=chunksize, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)
    return docs

def initialize_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = PineconeClient(api_key=pinecone_api_key)
    if "langqa1" not in pc.list_indexes().names():
        pc.create_index(
            index_name="langqa1",
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc

def main():
    st.title("PDF Question Answering App")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        st.write("Processing PDF...")
        doc = read_doc(file_path)
        data = chunks(doc)
        
        st.write("Initializing Pinecone...")
        pc = initialize_pinecone()
        embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
        index = Pinecone.from_documents(data, embeddings, index_name="langqa1")
        
        key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(temperature=0.5, model="llama3-8b-8192")
        chain = load_qa_chain(llm, chain_type="stuff")
        
        def retrieve_query(query, k=2):
            return index.similarity_search(query, k=k)
        
        def retrieve_ans(query):
            docsearch = retrieve_query(query)
            response = chain.run(input_documents=docsearch, question=query)
            return response
        
        query = st.text_input("Ask a question about the PDF:")
        if st.button("Get Answer") and query:
            answer = retrieve_ans(query)
            st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
