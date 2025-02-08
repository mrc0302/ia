import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
# import tkinter as tk
# from tkinter import filedialog
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
import json
from typing import List
import docx
import PyPDF2
from bs4 import BeautifulSoup
import io
import shutil
from typing import Dict, Any


# Diretório base para as bases de conhecimento
BASE_DIR = os.path.normpath(r"C:/Users/mcres/Documents/base de conhecimento/")
os.makedirs(BASE_DIR, exist_ok=True)
# Verificar permissões
if not os.access(BASE_DIR, os.W_OK):
    st.error(f"Sem permissão de escrita no diretório: {BASE_DIR}")

def initialize_session_state():
    if 'knowledge_bases' not in st.session_state:
        st.session_state['knowledge_bases'] = load_knowledge_bases()
    if 'selected_bases' not in st.session_state:
        st.session_state['selected_bases'] = {}
    if 'vector_db_path' not in st.session_state:
        st.session_state['vector_db_path'] = None
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'confirm_delete' not in st.session_state:
        st.session_state['confirm_delete'] = False

def load_knowledge_bases():
    json_path = os.path.join(BASE_DIR, 'knowledge_bases.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_knowledge_bases(bases):
    json_path = os.path.join(BASE_DIR, 'knowledge_bases.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(bases, f, ensure_ascii=False, indent=4)

# def select_save_directory():
#     """Open a file dialog to select save directory"""
#     root = tk.Tk()
#     root.withdraw()  # Hide the main window
#     directory = filedialog.askdirectory(
#         title="Selecione o diretório para salvar a base de conhecimento",
#         initialdir=BASE_DIR
#     )
#     return directory

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
    """Process CSV file content with error handling and validation"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        documents = []
        
        # Verificar se as colunas necessárias existem
        required_columns = ['id', 'assunto', 'classe', 'texto']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing_columns)}")
        
        # Preencher valores nulos com strings vazias
        df = df.fillna('')
        
        for index, row in df.iterrows():
            try:
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
            except Exception as row_error:
                st.warning(f"Erro ao processar linha {index + 1}: {str(row_error)}")
                continue
        
        if not documents:
            raise ValueError("Nenhum documento válido foi encontrado no CSV")
            
        return documents
        
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo CSV está vazio")
    except Exception as e:
        raise ValueError(f"Erro ao processar o CSV: {str(e)}")

def process_uploaded_file(uploaded_file) -> List[Document]:
    """Process uploaded file based on its type with enhanced error handling"""
    try:
        st.write(f"Iniciando processamento do arquivo: {uploaded_file.name}")
        file_type = uploaded_file.type
        file_content = uploaded_file.getvalue()
        
        if not file_content:
            raise ValueError("O arquivo está vazio")
        
        st.write(f"Tamanho do arquivo: {len(file_content)} bytes")
        
        documents = None
        if file_type == "text/csv":
            documents = process_csv(file_content)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            documents = process_docx(file_content)
        elif file_type == "application/pdf":
            documents = process_pdf(file_content)
        elif file_type == "text/html":
            documents = process_html(file_content.decode('utf-8'))
        elif file_type == "text/plain":
            documents = process_text_file(file_content.decode('utf-8'))
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {file_type}")
        
        if not documents:
            raise ValueError("Nenhum documento foi extraído do arquivo")
            
        st.write(f"Arquivo processado com sucesso: {len(documents)} documentos extraídos")
        return documents
            
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
        st.error(f"Tipo do erro: {type(e).__name__}")
        raise

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _vector_db_path):
    try:
        # Normalizar o caminho do diretório
        _vector_db_path = os.path.normpath(_vector_db_path.strip())
        st.write(f"Iniciando criação do vector store em: {_vector_db_path}")
        
        # Garantir que o diretório existe
        os.makedirs(_vector_db_path, exist_ok=True)
        
        st.write("Diretórios criados com sucesso")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        st.write("Dividindo documentos em chunks...")
        chunks = text_splitter.split_documents(_documents)
        st.write(f"Documentos divididos em {len(chunks)} chunks")
        
        # Criar o vector store
        st.write("Criando vector store...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        st.write("Salvando vector store...")
        
        # Usar o método integrado do FAISS para salvar
        vector_store.save_local(_vector_db_path)
        
        # Verificar se os arquivos necessários foram criados
        index_path = os.path.join(_vector_db_path, "index.faiss")
        json_path = os.path.join(_vector_db_path, "index.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Arquivo index.faiss não foi criado em: {index_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo index.json não foi criado em: {json_path}")
            
        st.write("Vector store criado e salvo com sucesso!")
        return vector_store
        
        pass

    except Exception as e:
        st.error(f"Erro ao processar documentos: {str(e)}")
        st.error(f"Tipo do erro: {type(e).__name__}")
        st.error(f"Detalhes adicionais: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def create_vector_store(_documents, _vector_db_path):
    try:
        # Normalizar o caminho do diretório
        _vector_db_path = os.path.normpath(_vector_db_path.strip())
        st.write(f"Iniciando criação do vector store em: {_vector_db_path}")
        
        # Garantir que o diretório existe
        os.makedirs(_vector_db_path, exist_ok=True)
        
        st.write("Diretórios criados com sucesso")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        st.write("Dividindo documentos em chunks...")
        chunks = text_splitter.split_documents(_documents)
        st.write(f"Documentos divididos em {len(chunks)} chunks")
        
        # Criar o vector store
        st.write("Criando vector store...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        st.write("Salvando vector store...")
        
        # Salvar docstore separadamente
        docstore = {
            "docstore": {
                doc.page_content: {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in chunks
            }
        }
        
        # Salvar o índice FAISS
        vector_store.save_local(_vector_db_path)
        
        # Salvar docstore em JSON
        docstore_path = os.path.join(_vector_db_path, "docstore.json")
        with open(docstore_path, 'w', encoding='utf-8') as f:
            json.dump(docstore, f, ensure_ascii=False, indent=2)
        
        # Verificar se os arquivos necessários foram criados
        index_path = os.path.join(_vector_db_path, "index.faiss")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Arquivo index.faiss não foi criado em: {index_path}")
        if not os.path.exists(docstore_path):
            raise FileNotFoundError(f"Arquivo docstore.json não foi criado em: {docstore_path}")
            
        st.write("Vector store criado e salvo com sucesso!")
        return vector_store
        
    except Exception as e:
        st.error(f"Erro ao processar documentos: {str(e)}")
        st.error(f"Tipo do erro: {type(e).__name__}")
        st.error(f"Detalhes adicionais: {str(e)}")
        # Adicionar mais informações de debug
        st.error(f"Caminho do diretório: {_vector_db_path}")
        st.error(f"Número de documentos processados: {len(chunks) if 'chunks' in locals() else 'N/A'}")
        return None

@st.cache_resource
def load_vector_store(_vector_db_path):
    try:
        # Verificar se os arquivos necessários existem
        index_path = os.path.join(_vector_db_path, "index.faiss")
        docstore_path = os.path.join(_vector_db_path, "docstore.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Arquivo de índice não encontrado em: {index_path}")
        if not os.path.exists(docstore_path):
            raise FileNotFoundError(f"Arquivo docstore.json não encontrado em: {docstore_path}")
        
        # Carregar o vector store
        vector_store = FAISS.load_local(_vector_db_path, embedding_model, allow_dangerous_deserialization= True )
        
        # Carregar docstore se necessário
        if os.path.exists(docstore_path):
            with open(docstore_path, 'r', encoding='utf-8') as f:
                docstore_data = json.load(f)
                # Aqui você pode usar o docstore_data se necessário
        
        return vector_store
    except Exception as e:
        st.error(f"Erro ao carregar Vector Store: {str(e)}")
        return None
    
def retrieve_relevant_chunks(vector_store, query, top_k=5):
    return vector_store.similarity_search(query, k=top_k)

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

def delete_knowledge_base(base_name: str):
    """
    Delete a knowledge base and its associated files
    """
    try:
        # Obter o caminho da base
        base_path = st.session_state['knowledge_bases'].get(base_name)
        if not base_path:
            st.error(f"Base de conhecimento '{base_name}' não encontrada.")
            return False
        
        # Limpar cache antes de excluir
        st.cache_resource.clear()
        st.cache_data.clear()
        
        # Remover arquivos da base
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
            st.write(f"Arquivos removidos de: {base_path}")
        
        # Remover do dicionário de bases
        del st.session_state['knowledge_bases'][base_name]
        
        # Atualizar o arquivo JSON
        save_knowledge_bases(st.session_state['knowledge_bases'])
        
        # Limpar o vector store se a base excluída estiver carregada
        if st.session_state.get('vector_db_path') == base_path:
            st.session_state['vector_store'] = None
            st.session_state['vector_db_path'] = None
        
        return True
    
    except Exception as e:
        st.error(f"Erro ao excluir base de conhecimento: {str(e)}")
        return False

def create_new_knowledge_base():
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo",
        type=['csv', 'docx', 'pdf', 'txt', 'html']
    )
    
    if uploaded_file:
        base_name = st.sidebar.text_input(
            "Nome da base de conhecimento:",
            help="Digite um nome descritivo para esta base"
        )
        
        if base_name and st.sidebar.button("Criar Base de Conhecimento"):
            try:
                # Remover espaços extras e caracteres especiais do nome da base
                base_name = "".join(c for c in base_name.strip() if c.isalnum() or c in ('-', '_'))
                
                # Criar o caminho completo e normalizado
                full_path = os.path.normpath(os.path.join(BASE_DIR, base_name))
                
                # Garantir que o diretório BASE_DIR existe
                os.makedirs(BASE_DIR, exist_ok=True)
                
                # Criar o diretório específico da base
                os.makedirs(full_path, exist_ok=True)
                
                st.write(f"Diretório criado em: {full_path}")
                
                # Processar o arquivo
                documents = process_uploaded_file(uploaded_file)
                if not documents:
                    raise ValueError("Nenhum documento foi processado.")
                    
                vector_store = create_vector_store(documents, full_path)
                
                if vector_store:
                    # Atualiza o JSON de bases de conhecimento
                    st.session_state['knowledge_bases'][base_name] = full_path
                    save_knowledge_bases(st.session_state['knowledge_bases'])
                    
                    st.success(f"Base de conhecimento '{base_name}' criada com sucesso!")
                    st.session_state['vector_store'] = vector_store
                    st.session_state['vector_db_path'] = full_path
                else:
                    raise ValueError("Falha ao criar vector store")
                    
            except Exception as e:
                st.error(f"Erro ao criar base de conhecimento: {str(e)}")

def use_existing_bases():
    st.sidebar.subheader("Selecione as bases para consulta:")
    
    if not st.session_state['knowledge_bases']:
        st.sidebar.warning("Nenhuma base de conhecimento encontrada.")
        return
    
    # Limpa seleções anteriores
    st.session_state['selected_bases'] = {}
    
    # Verifica e filtra bases válidas
    valid_bases = {}
    for base_name, path in st.session_state['knowledge_bases'].items():
        if os.path.exists(os.path.join(path, "index.faiss")):
            valid_bases[base_name] = path
        else:
            st.sidebar.warning(f"Base '{base_name}' está corrompida ou incompleta.")
    
    # Atualiza as bases de conhecimento se necessário
    if len(valid_bases) != len(st.session_state['knowledge_bases']):
        st.session_state['knowledge_bases'] = valid_bases
        save_knowledge_bases(valid_bases)
    
    # Cria checkboxes para cada base de conhecimento válida
    selected_any = False
    for base_name, path in valid_bases.items():
        st.session_state['selected_bases'][base_name] = st.sidebar.checkbox(
            f"Base: {base_name}",
            key=f"checkbox_{base_name}"
        )
        if st.session_state['selected_bases'][base_name]:
            selected_any = True
    
    if selected_any:
        if st.sidebar.button("Carregar Bases Selecionadas"):
            # Limpar cache explicitamente
            st.cache_resource.clear()
            st.cache_data.clear()
            
            # Limpar estado atual
            st.session_state['vector_store'] = None
            
            combined_store = None
            with st.spinner("Carregando bases de conhecimento..."):
                
                for base_name, selected in st.session_state['selected_bases'].items():
                    if selected:
                        st.write(f"Carregando base: {base_name}")
                        st.warning(base_name)

                        vector_store = load_vector_store(valid_bases[base_name])
                        
                        if vector_store:
                            if combined_store is None:
                                combined_store = vector_store
                            else:
                                combined_store.merge_from(vector_store)
                
                if combined_store:
                    st.session_state['vector_store'] = combined_store
                    st.success("Bases de conhecimento carregadas com sucesso!")
                    st.rerun()
                else:
                    st.error("Nenhuma base foi carregada com sucesso")
    else:
        st.sidebar.info("Selecione pelo menos uma base de conhecimento")

def manage_knowledge_bases():
    """
    Interface para gerenciar bases de conhecimento existentes
    """
    st.sidebar.subheader("Gerenciar Bases de Conhecimento")
    
    if not st.session_state['knowledge_bases']:
        st.sidebar.warning("Nenhuma base de conhecimento encontrada.")
        return
    
    # Lista todas as bases disponíveis
    base_to_delete = st.sidebar.selectbox(
        "Selecione a base para excluir:",
        options=list(st.session_state['knowledge_bases'].keys()),
        key="delete_base_select"
    )
    
    # Criar colunas para os botões
    col1, col2 = st.sidebar.columns(2)
    
    # Botão inicial de exclusão
    if col1.button("Excluir Base", key="delete_initial"):
        st.session_state['confirm_delete'] = True
        
    # Botão de confirmação
    if st.session_state.get('confirm_delete', False):
        st.sidebar.warning(f"Tem certeza que deseja excluir a base '{base_to_delete}'?")
        if col1.button("Sim, Excluir", key="confirm_yes"):
            if delete_knowledge_base(base_to_delete):
                st.sidebar.success(f"Base de conhecimento '{base_to_delete}' excluída com sucesso!")
                st.session_state['confirm_delete'] = False
                # Força atualização da interface
                st.rerun()
        
        if col2.button("Cancelar", key="confirm_no"):
            st.session_state['confirm_delete'] = False
            st.rerun()

def handle_query():
    if 'vector_store' in st.session_state and st.session_state['vector_store']:
        query = st.text_input("Digite sua pergunta:")
        
        if query:
            with st.spinner("Processando sua pergunta..."):
                try:
                    relevant_chunks = retrieve_relevant_chunks(st.session_state['vector_store'], query)
                    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
                    response = generate_response(query, context)
                    
                    st.subheader("Resposta:")
                    st.write(response)
                    
                    with st.expander("Ver contexto utilizado"):
                        st.write(context)
                except Exception as e:
                    st.error(f"Erro ao processar a pergunta: {str(e)}")
    else:
        st.info("Por favor, crie uma nova base de conhecimento ou selecione uma existente para começar.")

def main():
    

    initialize_session_state()

    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    
    if not google_api_key:
        st.error("API key não encontrada. Por favor, configure a variável de ambiente 'google_api_key'.")
        return
    
    # Configure Google API
    genai.configure(api_key=google_api_key)
    
    # Configure generation parameters
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
    }

    # Define global models
    global embedding_model, model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config,
    )

    # UI Components
    st.sidebar.header("Configurações")
    mode = st.sidebar.radio(
        "Escolha o modo de operação:",
        ["Criar nova base de conhecimento", "Usar base existente", "Gerenciar bases"]
    )

    if mode == "Criar nova base de conhecimento":
        create_new_knowledge_base()
    elif mode == "Usar base existente":
        use_existing_bases()
    else:
        manage_knowledge_bases()

    handle_query()

if __name__ == "__main__":
    main()
