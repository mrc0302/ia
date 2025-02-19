import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from typing import List
import docx
import PyPDF2
from bs4 import BeautifulSoup
import io
import shutil
import logging
from time import sleep

# Configuração de página como primeiro comando
st.set_page_config(
    page_title="RAG App",
    layout="wide"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Iniciando aplicação")

# Inicializações globais
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'base_dir' not in st.session_state:
    st.session_state['base_dir'] = ""

# Funções auxiliares
def get_available_bases(base_dir):
    """Lista todas as bases de conhecimento disponíveis no diretório escolhido"""
    bases = {}
    if os.path.exists(base_dir):
        for base_name in os.listdir(base_dir):
            base_path = os.path.join(base_dir, base_name)
            if os.path.isdir(base_path) and os.path.exists(os.path.join(base_path, "index.faiss")):
                bases[base_name] = base_path
    return bases

def process_text_file(file_content: str) -> List[Document]:
    """Process plain text content"""
    if not file_content.strip():
        raise ValueError("O arquivo de texto está vazio")
    return [Document(page_content=file_content)]

def process_docx(file_content: bytes) -> List[Document]:
    """Process DOCX file content"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            raise ValueError("O documento Word está vazio")
        return [Document(page_content=text)]
    except Exception as e:
        raise ValueError(f"Erro ao processar arquivo DOCX: {str(e)}")

def process_pdf(file_content: bytes) -> List[Document]:
    """Process PDF file content"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        if not text.strip():
            raise ValueError("O arquivo PDF está vazio ou não contém texto extraível")
        return [Document(page_content=text)]
    except Exception as e:
        raise ValueError(f"Erro ao processar arquivo PDF: {str(e)}")

def process_html(file_content: str) -> List[Document]:
    """Process HTML file content"""
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        text = soup.get_text(separator="\n", strip=True)
        if not text:
            raise ValueError("O arquivo HTML está vazio ou não contém texto")
        return [Document(page_content=text)]
    except Exception as e:
        raise ValueError(f"Erro ao processar arquivo HTML: {str(e)}")

def process_csv(file_content: bytes) -> List[Document]:
    """Process CSV file content"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        documents = []
        
        required_columns = ['id', 'assunto', 'classe', 'texto']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing_columns)}")
        
        df = df.fillna('')
        
        for _, row in df.iterrows():
            content = (
                f"ID: {row['id']}\n"
                f"Assunto: {row['assunto']}\n"
                f"Classe: {row['classe']}\n"
                f"Texto: {row['texto']}"
            )
            
            doc = Document(
                page_content=content,
                metadata={
                    'id': str(row['id']),
                    'assunto': str(row['assunto']),
                    'classe': str(row['classe'])
                }
            )
            documents.append(doc)
            
        if not documents:
            raise ValueError("Nenhum documento válido foi encontrado no CSV")
            
        return documents
        
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo CSV está vazio")
    except Exception as e:
        raise ValueError(f"Erro ao processar o CSV: {str(e)}")

@st.cache_data(show_spinner=False)
def process_uploaded_file(uploaded_file) -> List[Document]:
    """Process uploaded file based on its type"""
    try:
        file_type = uploaded_file.type
        file_content = uploaded_file.getvalue()
        
        if not file_content:
            raise ValueError("O arquivo está vazio")
        
        if file_type == "text/csv":
            return process_csv(file_content)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return process_docx(file_content)
        elif file_type == "application/pdf":
            return process_pdf(file_content)
        elif file_type == "text/html":
            return process_html(file_content.decode('utf-8'))
        elif file_type == "text/plain":
            return process_text_file(file_content.decode('utf-8'))
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {file_type}")
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        raise

@st.cache_resource(show_spinner=False)
def create_vector_store(documents, vector_db_path):
    """Create and save vector store"""
    try:
        os.makedirs(vector_db_path, exist_ok=True)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        with st.spinner("Criando base de conhecimento..."):
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(chunks, embedding_model)
            vector_store.save_local(vector_db_path)
            
        # Verificação final
        if not os.path.exists(os.path.join(vector_db_path, "index.faiss")):
            raise FileNotFoundError("Falha ao salvar vector store")
            
        return vector_store
        
    except Exception as e:
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
        raise e

@st.cache_resource(show_spinner=False)
def load_vector_store(vector_db_path):
    """Load vector store from disk"""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.load_local(
            vector_db_path, 
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Erro ao carregar vector store: {str(e)}")
        return None

def main():
    """Função principal do aplicativo"""
    # Inicialização da API Google
    try:
        load_dotenv()
        google_api_key = os.getenv("google_api_key")
        
        if not google_api_key:
            st.error("Chave de API do Google não encontrada. Verifique seu arquivo .env")
            return
            
        genai.configure(api_key=google_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        )
    except Exception as e:
        st.error(f"Erro ao configurar API Google: {str(e)}")
        logger.exception("Falha na inicialização da API:")
        return
    
    # Layout principal
    st.title("RAG App com Google Gemini")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Seletor de modo
        mode = st.radio(
            "Modo:",
            ["Criar base", "Usar base existente", "Deletar base"]
        )
        
        if mode == "Criar base":
            # Interface para criar base
            folder_path = st.text_input(
                "Pasta para salvar:",
                help="Caminho para a pasta onde a base será salva"
            )
            
            if folder_path and not os.path.exists(folder_path):
                st.warning(f"A pasta {folder_path} será criada ao salvar")
            
            uploaded_file = st.file_uploader(
                "Upload do arquivo",
                type=['csv', 'docx', 'pdf', 'txt', 'html']
            )
            
            if uploaded_file and folder_path:
                base_name = st.text_input(
                    "Nome da base:",
                    help="Nome descritivo para a base"
                )
                
                if st.button("Criar Base", disabled=not base_name):
                    try:
                        with st.spinner("Processando arquivo..."):
                            base_name = "".join(c for c in base_name.strip() if c.isalnum() or c in ('-', '_'))
                            os.makedirs(folder_path, exist_ok=True)
                            base_path = os.path.join(folder_path, base_name)
                            
                            if os.path.exists(base_path):
                                st.error("Uma base com este nome já existe")
                            else:
                                documents = process_uploaded_file(uploaded_file)
                                if documents:
                                    vector_store = create_vector_store(documents, base_path)
                                    st.session_state['vector_store'] = vector_store
                                    st.session_state['base_dir'] = folder_path
                                    st.success(f"Base '{base_name}' criada com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao criar base: {str(e)}")
                        logger.exception("Erro na criação da base:")
                        
        elif mode == "Usar base existente":
            # Interface para usar base existente
            folder_path = st.text_input(
                "Pasta da base:",
                help="Caminho para a pasta que contém suas bases"
            )
            
            if folder_path:
                if not os.path.exists(folder_path):
                    st.error(f"A pasta {folder_path} não existe")
                else:
                    bases = get_available_bases(folder_path)
                    
                    if not bases:
                        st.warning(f"Nenhuma base encontrada em {folder_path}")
                    else:
                        selected_base = st.selectbox(
                            "Selecione a base:",
                            options=list(bases.keys())
                        )
                        
                        if st.button("Carregar Base"):
                            try:
                                with st.spinner("Carregando base..."):
                                    vector_store = load_vector_store(bases[selected_base])
                                    if vector_store:
                                        st.session_state['vector_store'] = vector_store
                                        st.session_state['base_dir'] = folder_path
                                        st.success(f"Base '{selected_base}' carregada!")
                                    else:
                                        st.error("Falha ao carregar a base")
                            except Exception as e:
                                st.error(f"Erro ao carregar: {str(e)}")
                                logger.exception("Erro no carregamento:")
                                
        else:  # Deletar base
            # Interface para deletar base
            folder_path = st.text_input(
                "Pasta da base:",
                help="Caminho para a pasta que contém suas bases"
            )
            
            if folder_path:
                if not os.path.exists(folder_path):
                    st.error(f"A pasta {folder_path} não existe")
                else:
                    bases = get_available_bases(folder_path)
                    
                    if not bases:
                        st.warning(f"Nenhuma base para deletar em {folder_path}")
                    else:
                        base_to_delete = st.selectbox(
                            "Selecione para excluir:",
                            options=list(bases.keys())
                        )
                        
                        if st.button("Deletar Base", type="primary", use_container_width=True):
                            try:
                                base_path = bases[base_to_delete]
                                # Confirmação adicional
                                if st.checkbox("Confirmar exclusão", value=False):
                                    shutil.rmtree(base_path)
                                    st.success(f"Base '{base_to_delete}' deletada")
                                else:
                                    st.info("Marque a confirmação para excluir")
                            except Exception as e:
                                st.error(f"Erro ao deletar: {str(e)}")
    
    # Coluna para consultas
    with col2:
        st.subheader("Consultar Base de Conhecimento")
        
        if st.session_state.get('vector_store'):
            query = st.text_input("Sua pergunta:")
            
            if query:
                try:
                    with st.spinner("Processando consulta..."):
                        results = st.session_state['vector_store'].similarity_search(query, k=5)
                        context = "\n".join([doc.page_content for doc in results])
                        
                        prompt = f"""
                        Contexto: {context}
                        
                        Pergunta: {query}
                        
                        Por favor, responda usando apenas as informações do contexto.
                        Se a informação não estiver disponível, diga que não pode responder.
                        
                        Resposta:
                        """
                        
                        response = model.generate_content(prompt)
                        
                        st.markdown("### Resposta:")
                        st.write(response.text)
                        
                        with st.expander("Ver contexto"):
                            st.write(context)
                except Exception as e:
                    st.error(f"Erro na consulta: {str(e)}")
                    logger.exception("Erro durante consulta:")
        else:
            st.info("Selecione ou crie uma base primeiro")
            st.divider()
            st.write("Status: Aguardando seleção de base")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro não tratado: {str(e)}")
        logger.exception("Erro fatal na aplicação:")
