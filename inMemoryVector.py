import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss


# Configurar sua API key
genai.configure(api_key="AIzaSyBJ9ifJikSVNqF7njMoc-wlcIrdjLWcvY4")

# Configurar embeddings do Google
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key="AIzaSyBJ9ifJikSVNqF7njMoc-wlcIrdjLWcvY4"
)

# Criar o vector store
vector_store = InMemoryVectorStore(embeddings)

# Seu modelo Gemini
llm = genai.GenerativeModel( 
    model_name="gemini-2.0-flash-exp",
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }       
)

def carregar_faiss_existente(caminho_pasta_faiss):
    """
    Carrega um índice FAISS já salvo
    """
    print(f"Carregando FAISS de: {caminho_pasta_faiss}")
       
    
    # Carregar o vector store FAISS
    vector_store = FAISS.load_local(
        caminho_pasta_faiss, 
        embeddings,
        allow_dangerous_deserialization=True  # Necessário para carregar pickle
    )
    
    print(f"✅ FAISS carregado com sucesso!")
    print(f"   Número de vetores: {vector_store.index.ntotal}")
    print(f"   Dimensão: {vector_store.index.d}")
    
    return vector_store

# Carregar seu FAISS
vector_store = carregar_faiss_existente("faiss_legal_store_gemini_faiss_legal_store_gemini")
def chat_com_faiss_interativo(vector_store, k=3):
    """
    Versão interativa da sua função original
    """
    print("Chat iniciado! Digite 'sair' para parar")
    
    while True:
        pergunta = input("\nSua pergunta: ")
        
        if pergunta.lower() == 'sair':
            break
            
        if pergunta.strip():
            try:
                # Buscar documentos
                docs = vector_store.similarity_search(pergunta, k=k)
                contexto = "\n".join([doc.page_content for doc in docs])
                
                # Prompt
                prompt = f"Contexto: {contexto}\n\nPergunta: {pergunta}\n\nResposta:"
                
                # Gemini
                response = llm.generate_content(prompt)
                print(f"\nResposta: {response.text}")
                
            except Exception as e:
                print(f"Erro: {e}")

chat_com_faiss_interativo(vector_store, k=3)