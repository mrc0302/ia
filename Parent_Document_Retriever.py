import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.schema import Document
import io
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import tempfile
import os
import time
import shutil
import pickle
import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Verificar e criar pasta para armazenar vectorstores
VECTOR_STORE_DIR = "vectorstores"
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

def get_pdf_text(docs_pdf):
    text = ""   
    for doc in docs_pdf:
        text += doc.page_content
    return text
 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_retriever(vector_store, docs): 
    # The storage layer for the parent documents
    store = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
    )
        
    retriever.add_documents(docs, ids=None)
    return retriever

def save_vector_store(vector_store, name=None):
    if name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"vectorstore_{timestamp}"
    
    filepath = os.path.join(VECTOR_STORE_DIR, f"{name}.pkl")
    
    with open(filepath, "wb") as f:
        pickle.dump(vector_store, f)
    
    return filepath

def load_vector_store(filepath):
    with open(filepath, "rb") as f:
        vector_store = pickle.load(f)
    
    return vector_store

def list_saved_vector_stores():
    files = [f for f in os.listdir(VECTOR_STORE_DIR) if f.endswith('.pkl')]
    return files

def get_conversational_chain(model):
    prompt_template = """
    Answer the question in portuguese Brazil \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, retriever):
    if not retriever:
        st.error("Por favor, carregue e processe os arquivos primeiro antes de fazer perguntas.")
        return
        
    try:
        docs = retriever.invoke(user_question)

        model = ChatGoogleGenerativeAI(
            
            model="gemini-2.0-flash-thinking-exp-01-21",
            temperature=0.3,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=10,
            candidate_count=3,
            system_instruction="Você é um assistente especializado em análise jurídica e direito processual, capaz de responder perguntas complexas com precisão e clareza.",
            user_instruction="Você é um assistente especializado em análise jurídica e direito processual, capaz de responder " \
            "perguntas complexas na área do direito, além de possuir capacidade de elaborar decisões judiciais com precisão e clareza.",
            safety_settings=None,
            stream=True,
            convert_system_message_to_human=True,
            tools=[{"type": "calculator"}, {"type": "web_search"}],
            client=None,
            credentials=None,
            )
        chain = get_conversational_chain(model)
        
        response = chain(
            {"input_documents": docs, "question": user_question}
             , return_only_outputs=True)

        st.markdown(f"""
            <div style="height: 600px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 10px; border-radius: 5px;">
                <div>**Reply:** {response['output_text']}</div>
            </div>
            """, 
            unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Erro ao processar sua pergunta: {str(e)}")

def process_document(file, file_type):
    docs = []
    temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
    
    try:
        # Salvar temporariamente o arquivo
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Processar baseado no tipo de arquivo
        if file_type == "pdf":
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
        elif file_type == "csv":
            loader = CSVLoader(temp_file_path)
            docs = loader.load()
    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo {file.name}: {str(e)}")
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return docs

def main():
    st.set_page_config("Chat com Documentos")
    st.header("Chat with Documents using Gemini💁")
   
    # Initialize session state to store the retriever
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
        st.session_state.vector_store = None
        st.session_state.docs_processed = False
   
    with st.sidebar:
        st.title("Menu:")
        
        # Aba de opções
        tab1, tab2 = st.tabs(["Carregar Arquivos", "Gerenciar Vectorstores"])
        
        with tab1:
            # Seleção de tipo de arquivo
            file_type = st.radio(
                "Selecione o tipo de arquivo",
                ["PDF", "CSV"],
                captions=["Documentos em PDF", "Dados em CSV"]
            )
             
            # Upload dos arquivos
            uploaded_files = st.file_uploader(
                f"Upload seus arquivos {file_type} e clique em 'Processar'",
                type=[file_type.lower()],
                accept_multiple_files=True,
                key="content_files"
            )

            # Processamento de arquivos
            if uploaded_files:
                st.info(f"{len(uploaded_files)} arquivo(s) enviado(s). Clique em 'Processar'.")

                if st.button("Processar Arquivos"):
                    with st.spinner("Processando arquivos..."):
                        all_docs = []
                        for uploaded_file in uploaded_files:
                            docs = process_document(uploaded_file, file_type.lower())
                            all_docs.extend(docs)
                        
                        if all_docs:
                            text = ""
                            for doc in all_docs:
                                text += doc.page_content
                            
                            text_chunks = get_text_chunks(text)     
                            vector_store = create_vector_store(text_chunks) 
                            st.session_state.vector_store = vector_store
                            st.session_state.retriever = get_retriever(vector_store, all_docs)
                            st.session_state.docs_processed = True
                        
                            # Salvar vectorstore
                            save_option = st.checkbox("Salvar Vector Store")
                            if save_option:
                                vs_name = st.text_input("Nome do Vector Store (opcional)")
                                if st.button("Salvar"):
                                    filepath = save_vector_store(vector_store, vs_name)
                                    st.success(f"Vector Store salvo em: {filepath}")
                        
                            st.success(f"Processamento concluído! {len(all_docs)} documentos extraídos no total.")
        
        with tab2:
            st.subheader("Vector Stores Salvos")
            saved_vs_files = list_saved_vector_stores()
            
            if not saved_vs_files:
                st.info("Nenhum Vector Store salvo encontrado.")
            else:
                selected_vs = st.selectbox("Selecione um Vector Store para carregar", saved_vs_files)
                
                if st.button("Carregar Vector Store"):
                    with st.spinner("Carregando Vector Store..."):
                        filepath = os.path.join(VECTOR_STORE_DIR, selected_vs)
                        vector_store = load_vector_store(filepath)
                        
                        # Criação de documentos vazios para o retriever
                        # Apenas para inicializar o retriever - os documentos estão no vector store
                        dummy_docs = [Document(page_content="")]
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = get_retriever(vector_store, dummy_docs)
                        st.session_state.docs_processed = True
                        
                        st.success(f"Vector Store '{selected_vs}' carregado com sucesso!")

    # Main area for questions
    st.subheader("Faça sua pergunta")
    user_question = st.text_input("Pergunte algo sobre os documentos carregados:")
    
    if user_question:
        user_input(user_question, st.session_state.retriever)


if __name__ == "__main__":
    main()