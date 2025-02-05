import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

# Configuração da página Streamlit
st.set_page_config(page_title="Analisador de CSV com IA", layout="wide")

load_dotenv()
google_api_key = os.getenv("google_api_key")
#-------------------------------------------------------------------------------
genai.configure(api_key=os.getenv("google_api_key"))

# Título do aplicativo
st.title("📊 Analisador de CSV com IA")

# Configurações no sidebar
with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Google API Key", type="password")
    chunk_size = st.number_input("Tamanho do Chunk", value=1000, min_value=100)
    chunk_overlap = st.number_input("Overlap do Chunk", value=200, min_value=0)

def initialize_genai():
    """Inicializa a API do Google e os embeddings"""
    genai.configure(api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=google_api_key
    )
    return embeddings

def csv_to_documents(df):
    """Converte DataFrame em documentos para processamento"""
    documents = []
    for index, row in df.iterrows():
        # Concatenar todas as colunas em um único texto
        content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
        
        # Criar documento
        doc = Document(
            page_content=content,
            metadata={"row": index}
        )
        documents.append(doc)
    
    return documents

def process_csv(df, embeddings, chunk_size=1000, chunk_overlap=200):
    """Processa o DataFrame e prepara para análise"""
    # Converter DataFrame em documentos
    documents = csv_to_documents(df)
    
    # Dividir texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    
    # Criar banco de vetores
    with st.spinner("Criando banco de vetores..."):
        vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def analyze_csv(vectorstore, query):
    """Realiza análise do CSV com base na query"""
    # Buscar documentos relevantes
    with st.spinner("Buscando documentos relevantes..."):
        docs = vectorstore.similarity_search(query)
    
    # Configurar modelo e gerar resposta
    with st.spinner("Gerando análise..."):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"Contexto: {docs[0].page_content}\nPergunta: {query}"
        )
    
    return response.text

def saveFaissIndex(vectorstore, save_path):
    """
    Salva o índice FAISS em uma pasta específica
    
    Args:
        vectorstore: O objeto vectorstore do FAISS
        save_path: Caminho onde o índice será salvo
    """
    try:
        # Cria o diretório se não existir
        os.makedirs(save_path, exist_ok=True)
        # Salva o índice diretamente na pasta especificada
        vectorstore.save_local(save_path)
        return True
    except Exception as e:
        print(f"Erro ao salvar índice: {e}")
        return False

def loadFaissIndex(load_path, embeddings):
    """
    Carrega um índice FAISS salvo
    
    Args:
        load_path: Caminho onde o índice está salvo
        embeddings: Objeto de embeddings inicializado
    """
    try:
        vectorstore = FAISS.load_local(load_path, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Erro ao carregar índice: {e}")
        return None

def main():
    # Upload do arquivo CSV
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file:
        try:
            # Inicializar API e embeddings
            embeddings = initialize_genai()
            
            # Ler o CSV
            df = pd.read_csv(uploaded_file)
            
            # Mostrar preview dos dados
            st.subheader("Preview dos dados")
            st.dataframe(df.head())
            
            # Informações sobre o dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linhas", df.shape[0])
            with col2:
                st.metric("Colunas", df.shape[1])
            with col3:
                st.metric("Células", df.size)
            
            # Processar CSV
            if st.button("Processar Dados"):
                vectorstore = process_csv(df, embeddings, chunk_size, chunk_overlap)
                st.session_state['vectorstore'] = vectorstore
                
                # Adiciona campos para salvar o índice
                save_path = st.text_input("Digite o caminho para salvar o índice:", "indices/meu_indice")
                if st.button("Salvar Índice"):
                    if saveFaissIndex(vectorstore, save_path):
                        st.success(f"Índice salvo com sucesso em {save_path}")
                    else:
                        st.error("Erro ao salvar o índice")
                
                st.success("Dados processados com sucesso!")
            
            # Carregamento de índice existente
            st.subheader("Carregar Índice Existente")
            load_path = st.text_input("Digite o caminho do índice a ser carregado:")
            if st.button("Carregar Índice") and load_path:
                vectorstore = loadFaissIndex(load_path, embeddings)
                if vectorstore:
                    st.session_state['vectorstore'] = vectorstore
                    st.success("Índice carregado com sucesso!")
                else:
                    st.error("Erro ao carregar o índice")
            
            # Área de consulta
            if 'vectorstore' in st.session_state:
                st.subheader("Faça sua consulta")
                query = st.text_area("Digite sua pergunta sobre os dados")
                
                if st.button("Analisar") and query:
                    analysis = analyze_csv(st.session_state['vectorstore'], query)
                    
                    st.subheader("Resultado da Análise")
                    st.write(analysis)
                    
                    # Opção para download da análise
                    st.download_button(
                        label="Download da Análise",
                        data=analysis,
                        file_name="analise.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")
    
    # Instruções de uso
    if not uploaded_file:
        st.info("""
        ### Como usar:
        1. Insira sua Google API Key no menu lateral
        2. Faça upload de um arquivo CSV
        3. Clique em 'Processar Dados'
        4. Digite sua pergunta e clique em 'Analisar'
        
        Você também pode:
        - Salvar o índice processado para uso futuro
        - Carregar um índice previamente salvo
        """)

if __name__ == "__main__":
    main()