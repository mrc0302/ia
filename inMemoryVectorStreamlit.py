import streamlit as st
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
import os

# Configuração da página
st.set_page_config(
    page_title="Chat com FAISS e Gemini",
    page_icon="🤖",
    layout="wide"
)

# Título da aplicação
st.title("🤖 Chat com FAISS e Gemini")
st.markdown("---")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Campo para API Key
    api_key = st.text_input(
        "Google API Key:", 
        type="password",
        value="AIzaSyBJ9ifJikSVNqF7njMoc-wlcIrdjLWcvY4",
        help="Insira sua chave da API do Google"
    )
    
    # Campo para caminho do FAISS
    caminho_faiss = st.text_input(
        "Caminho do FAISS:",
        value="faiss_legal_store_gemini_faiss_legal_store_gemini",
        help="Caminho para o diretório do índice FAISS"
    )
    
    # Número de documentos para busca
    k_docs = st.slider(
        "Número de documentos para busca:",
        min_value=1,
        max_value=10,
        value=3,
        help="Quantos documentos similares buscar"
    )
    
    # Botão para carregar FAISS
    if st.button("🔄 Carregar FAISS", type="primary"):
        if api_key and caminho_faiss:
            with st.spinner("Carregando FAISS..."):
                try:
                    # Configurar API key
                    genai.configure(api_key=api_key)
                    
                    # Configurar embeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
                    
                    # Carregar FAISS
                    vector_store = FAISS.load_local(
                        caminho_faiss,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Salvar no session state
                    st.session_state.vector_store = vector_store
                    st.session_state.embeddings = embeddings
                    st.session_state.api_key = api_key
                    
                    st.success("✅ FAISS carregado com sucesso!")
                    st.info(f"📊 Número de vetores: {vector_store.index.ntotal}")
                    st.info(f"📐 Dimensão: {vector_store.index.d}")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao carregar FAISS: {str(e)}")
        else:
            st.warning("⚠️ Preencha a API Key e o caminho do FAISS")

# Área principal do chat
if 'vector_store' in st.session_state:
    st.header("💬 Chat")
    
    # Inicializar histórico de mensagens
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Adicionar mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Processar pergunta
        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                try:
                    # Configurar modelo Gemini
                    llm = genai.GenerativeModel(
                        model_name="gemini-2.0-flash-exp",
                        generation_config={
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "top_k": 4,
                            "max_output_tokens": 8192,
                            "response_mime_type": "text/plain",
                        }
                    )
                    
                    # Buscar documentos similares
                    docs = st.session_state.vector_store.similarity_search(prompt, k=k_docs)
                    contexto = "\n".join([doc.page_content for doc in docs])
                    
                    # Criar prompt
                    full_prompt = f"Contexto: {contexto}\n\nPergunta: {prompt}\n\nResposta:"
                    
                    # Gerar resposta
                    response = llm.generate_content(full_prompt)
                    resposta = response.text
                    
                    # Exibir resposta
                    st.markdown(resposta)
                    
                    # Adicionar resposta ao histórico
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                    
                except Exception as e:
                    st.error(f"❌ Erro ao processar pergunta: {str(e)}")
    
    # Botão para limpar histórico
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()
    
    # Seção de informações dos documentos encontrados
    if st.session_state.messages:
        with st.expander("📄 Documentos Utilizados na Última Consulta"):
            if 'vector_store' in st.session_state and len(st.session_state.messages) > 0:
                try:
                    ultima_pergunta = None
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            ultima_pergunta = msg["content"]
                            break
                    
                    if ultima_pergunta:
                        docs = st.session_state.vector_store.similarity_search(ultima_pergunta, k=k_docs)
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Documento {i}:**")
                            st.text_area(
                                f"Conteúdo {i}",
                                doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                height=100,
                                key=f"doc_{i}"
                            )
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.json(doc.metadata)
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Erro ao exibir documentos: {str(e)}")

else:
    st.info("👈 Configure a API Key e carregue o índice FAISS na barra lateral para começar o chat.")
    
    # Seção de ajuda
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        ### Passos para usar a aplicação:
        
        1. **Configure a API Key**: Insira sua chave da API do Google Gemini na barra lateral
        2. **Defina o caminho do FAISS**: Especifique o diretório onde está salvo seu índice FAISS
        3. **Ajuste o número de documentos**: Escolha quantos documentos similares buscar (padrão: 3)
        4. **Carregue o FAISS**: Clique no botão "Carregar FAISS"
        5. **Inicie o chat**: Digite suas perguntas na área de chat
        
        ### Recursos disponíveis:
        - 💬 **Chat interativo** com histórico de mensagens
        - 📄 **Visualização dos documentos** utilizados em cada consulta
        - ⚙️ **Configurações ajustáveis** na barra lateral
        - 🗑️ **Limpeza do histórico** quando necessário
        """)

# Footer
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit, LangChain e Google Gemini")