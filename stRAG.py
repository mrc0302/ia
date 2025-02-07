import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document

def main():

    # Configuração da página Streamlit
    #st.set_page_config(page_title="RAG App com Gemini", layout="wide")
    st.title("Aplicação RAG com Google Gemini")

    # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")

    # Configurar Gemini
    genai.configure(api_key=google_api_key)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create the model
    generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
    )
    # Inicializar o modelo Gemini
    #model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

    # Função para processar arquivo CSV e criar documentos
    def process_csv(file_path):
        
        df = pd.read_csv(file_path)
        documents = []
        
        for _, row in df.iterrows():
            content = f"ID: {row['id']}\nAssunto: {row['assunto']}\nClasse: {row['classe']}\nTexto: {row['texto']}"
            doc = Document(
                page_content=content,
                metadata={
                    'id': row['id'],
                    'assunto': row['assunto'],
                    'classe': row['classe']
                }
            )
            documents.append(doc)
        
        return documents

    # Função para carregar e processar documentos
    @st.cache_resource
    def create_vector_store(_documents, _vector_db_path):
        try:
            # Dividir documentos
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(_documents)
            
            # Criar Vector Store
            vector_store = FAISS.from_documents(chunks, embedding_model)
            
            # Salvar Vector Store
            vector_store.save_local(_vector_db_path)
            
            return vector_store
        except Exception as e:
            st.error(f"Erro ao processar documentos: {str(e)}")
            return None

    # Função para carregar Vector Store existente
    @st.cache_resource
    def load_vector_store(_vector_db_path):
        try:
            return FAISS.load_local(_vector_db_path, embedding_model)
        except Exception as e:
            st.error(f"Erro ao carregar Vector Store: {str(e)}")
            return None

    # Função para recuperar chunks relevantes
    def retrieve_relevant_chunks(vector_store, query, top_k=5):
        return vector_store.similarity_search(query, k=top_k)

    # Função para gerar resposta
    def generate_response(query, context):
        prompt = f"""
        Contexto: {context}
        
        Pergunta: {query}
        
        Por favor, responda à pergunta acima usando apenas as informações fornecidas no contexto.
        Se a informação não estiver disponível no contexto, diga que não pode responder.
        
        Resposta:
        """
        
        response = model.generate_content(prompt)
        return response.text

    # Interface Streamlit
    st.sidebar.header("Configurações")

    # Seleção do modo de operação
    mode = st.sidebar.radio(
        "Escolha o modo de operação:",
        ["Criar nova base de conhecimento", "Usar base existente"]
    )

    if mode == "Criar nova base de conhecimento":
        # Upload do arquivo CSV
        uploaded_file = st.sidebar.file_uploader(
            "Faça upload do arquivo CSV",
            type=['csv']
        )
        
        # Campo para digitar o caminho do diretório
        vector_db_path = st.sidebar.text_input(
            "Digite o caminho completo para salvar a base vetorial:",
            value="./vector_db",
            help="Exemplo: C:/MeuProjeto/vector_db"
        )
        
        if uploaded_file and vector_db_path:
            if st.sidebar.button("Criar Base de Conhecimento"):
                try:
                    # Criar diretório se não existir
                    os.makedirs(vector_db_path, exist_ok=True)
                    
                    # Salvar arquivo CSV temporariamente
                    csv_path = os.path.join(".", uploaded_file.name)
                    with open(csv_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Processar CSV e criar documentos
                    documents = process_csv(csv_path)
                    
                    # Criar Vector Store
                    vector_store = create_vector_store(documents, vector_db_path)
                    
                    # Limpar arquivo CSV temporário
                    os.remove(csv_path)
                    
                    if vector_store:
                        st.success("Base de conhecimento criada com sucesso!")
                        st.session_state['vector_store'] = vector_store
                        st.session_state['vector_db_path'] = vector_db_path
                except Exception as e:
                    st.error(f"Erro ao criar base de conhecimento: {str(e)}")

    else:
        # Campo para digitar o caminho do diretório existente
        vector_db_path = st.sidebar.text_input(
            "Digite o caminho da base vetorial existente:",
            help="Exemplo: C:/MeuProjeto/vector_db"
        )
        
        if vector_db_path:
            if st.sidebar.button("Carregar Base de Conhecimento"):
                vector_store = load_vector_store(vector_db_path)
                if vector_store:
                    st.success("Base de conhecimento carregada com sucesso!")
                    st.session_state['vector_store'] = vector_store
                    st.session_state['vector_db_path'] = vector_db_path

    # Interface de consulta
    if 'vector_store' in st.session_state:
        query = st.text_input("Digite sua pergunta:")
        
        if query:
            with st.spinner("Processando sua pergunta..."):
                try:
                    # Recuperar chunks relevantes
                    relevant_chunks = retrieve_relevant_chunks(st.session_state['vector_store'], query)
                    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
                    
                    # Gerar resposta
                    response = generate_response(query, context)
                    
                    # Exibir resultados
                    st.subheader("Resposta:")
                    st.write(response)
                    
                    # Exibir contexto usado (opcional)
                    with st.expander("Ver contexto utilizado"):
                        st.write(context)
                except Exception as e:
                    st.error(f"Erro ao processar a pergunta: {str(e)}")
    else:
        st.info("Por favor, crie uma nova base de conhecimento ou selecione uma existente para começar.")

if __name__ == "__main__":
    main()
