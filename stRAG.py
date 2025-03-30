import streamlit as st

# IMPORTANTE: set_page_config deve ser o primeiro comando Streamlit
st.set_page_config(page_title="RAG App", layout="wide")

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import tempfile
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from typing import List
import io
import shutil
import pickle
import base64


def main():
    st.title("RAG App com Modelos Gemma/Gemini")
    
    # Importar bibliotecas necessárias para armazenamento temporário
    import tempfile
    
    # Inicializar diretório temporário na primeira execução
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()
        st.sidebar.success(f"Diretório temporário criado para esta sessão")
    
    # Mostrar informações sobre o ambiente para ajudar no debug
    st.sidebar.subheader("Informações do Sistema")
    st.sidebar.info(f"Diretório de trabalho atual: {os.getcwd()}")
    st.sidebar.info(f"Diretório temporário: {st.session_state['temp_dir']}")
    st.sidebar.info(f"Ambiente: {'Streamlit Cloud' if os.getenv('STREAMLIT_SHARING') else 'Local'}")
    
    # Informações sobre armazenamento no Streamlit Cloud
    st.sidebar.subheader("Sobre Armazenamento")
    st.sidebar.warning("""
    **Importante**: No Streamlit Cloud, os arquivos são temporários e podem ser excluídos quando a sessão terminar.
    
    Para uso em produção, considere usar um serviço de armazenamento como S3, GCS ou um banco de dados em nuvem.
    """)
    
    # Inicializar estado da sessão
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'base_dir' not in st.session_state:
        st.session_state['base_dir'] = ""
    
    # Configuração da API
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    
    # Permitir entrada direta da chave API caso não esteja no .env
    if not google_api_key:
        google_api_key = st.text_input("Insira sua chave da API Google:", type="password")
        if not google_api_key:
            st.error("Chave da API Google não encontrada. Insira a chave acima ou configure a variável 'google_api_key' no arquivo .env")
            return
    
    genai.configure(api_key=google_api_key)
    
    # Configuração do modelo Gemma/Gemini
    st.sidebar.subheader("Configuração do Modelo")
    modelo_selecionado = st.sidebar.selectbox(
        "Selecione o modelo:",
        ["gemma-3-27b-it", "gemma-2-27b-it", "gemma-2-9b-it", "gemma-2-2b-it", "gemini-1.5-flash-latest"]
    )
    
    temperatura = st.sidebar.slider("Temperatura:", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Tokens máximos:", 1024, 8192, 4096, 512)
    
    generation_config = {
        "temperature": temperatura,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": max_tokens,
    }
    
    # Modelos globais
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        model = genai.GenerativeModel(
            model_name=modelo_selecionado,
            generation_config=generation_config,
        )
    except Exception as e:
        st.error(f"Erro ao inicializar modelos: {str(e)}")
        st.info("Dica: Verifique se a chave da API Google está correta e tem acesso aos modelos necessários.")
        return
    
    # Interface principal com duas colunas
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Gerenciamento de Base")
        mode = st.radio(
            "Modo:",
            ["Criar base", "Usar base existente"]
        )
        
        if mode == "Criar base":
            create_knowledge_base(embedding_model)
        else:
            use_existing_base(embedding_model)
    
    with col2:
        st.subheader("Consulta")
        handle_query(model)


def process_csv(file_content: bytes) -> List[Document]:
    """Processa arquivo CSV e converte em documentos"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        st.write(f"Colunas detectadas: {', '.join(df.columns)}")
        
        documents = []
        df = df.fillna('')
        
        # Detectar automaticamente se o CSV tem um formato específico ou é genérico
        if all(col in df.columns for col in ['id', 'texto']):
            # Formato específico com id e texto
            for _, row in df.iterrows():
                metadata = {k: str(v) for k, v in row.items() if k != 'texto'}
                doc = Document(
                    page_content=str(row['texto']),
                    metadata=metadata
                )
                documents.append(doc)
        else:
            # Formato genérico - converte todas as linhas em documentos
            for idx, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={'row_id': str(idx)}
                )
                documents.append(doc)
        
        if not documents:
            raise ValueError("Nenhum documento válido foi encontrado no CSV")
        
        st.success(f"{len(documents)} documentos processados com sucesso")
        return documents
    
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo CSV está vazio")
    except Exception as e:
        raise ValueError(f"Erro ao processar o CSV: {str(e)}")


def create_vector_store(documents, vector_db_path, embedding_model):
    """Cria e salva a base de conhecimento vetorial"""
    try:
        # Garantir que o caminho é absoluto
        vector_db_path = os.path.abspath(vector_db_path)
        st.info(f"Salvando base em: {vector_db_path}")
        
        # Criar diretório se não existir
        os.makedirs(vector_db_path, exist_ok=True)
        
        with st.spinner("Dividindo documentos em chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(documents)
            st.info(f"Total de chunks: {len(chunks)}")
        
        with st.spinner("Criando embeddings e salvando vector store..."):
            vector_store = FAISS.from_documents(chunks, embedding_model)
            
            # Salvar localmente
            vector_store.save_local(vector_db_path)
            
            # Também oferecer a opção de download para backup
            if st.checkbox("Gerar arquivo para download?", value=False):
                # Serializar o vetor store para um arquivo temporário
                temp_file = os.path.join(vector_db_path, "vector_store.pkl")
                with open(temp_file, "wb") as f:
                    pickle.dump(vector_store, f)
                
                # Ler o arquivo para disponibilizá-lo para download
                with open(temp_file, "rb") as f:
                    bytes_data = f.read()
                
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:file/pickle;base64,{b64}" download="vector_store.pkl">Baixar arquivo da base vetorial</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Mostrar arquivos criados para debug
            files_created = os.listdir(vector_db_path)
            st.success(f"Arquivos criados no diretório: {', '.join(files_created)}")
        
        # Verificação final
        if not os.path.exists(os.path.join(vector_db_path, "index.faiss")):
            raise FileNotFoundError("Falha ao salvar vector store - arquivo index.faiss não encontrado")
        
        return vector_store
    
    except Exception as e:
        st.error(f"Erro ao criar vector store: {str(e)}")
        st.error(f"Caminho tentado: {vector_db_path}")
        if os.path.exists(vector_db_path):
            try:
                shutil.rmtree(vector_db_path)
            except:
                st.warning(f"Não foi possível limpar o diretório {vector_db_path}")
        return None


def load_vector_store(vector_db_path, embedding_model):
    """Carrega uma base de conhecimento vetorial existente"""
    try:
        # Garantir que o caminho é absoluto
        vector_db_path = os.path.abspath(vector_db_path)
        st.info(f"Tentando carregar base de: {vector_db_path}")
        
        # Verificar se os arquivos necessários existem
        if not os.path.exists(os.path.join(vector_db_path, "index.faiss")):
            st.error(f"Arquivo index.faiss não encontrado em {vector_db_path}")
            files_in_dir = os.listdir(vector_db_path) if os.path.exists(vector_db_path) else []
            st.info(f"Arquivos encontrados no diretório: {', '.join(files_in_dir)}")
            return None
            
        with st.spinner("Carregando base de conhecimento..."):
            vector_store = FAISS.load_local(
                vector_db_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            st.success("Base carregada com sucesso!")
        return vector_store
    except Exception as e:
        st.error(f"Erro ao carregar vector store: {str(e)}")
        st.error(f"Caminho tentado: {vector_db_path}")
        return None


def get_available_bases(base_dir):
    """Lista todas as bases de conhecimento disponíveis no diretório escolhido"""
    bases = {}
    if os.path.exists(base_dir):
        # Procurar bases diretamente no diretório especificado
        for base_name in os.listdir(base_dir):
            base_path = os.path.join(base_dir, base_name)
            if os.path.isdir(base_path) and os.path.exists(os.path.join(base_path, "index.faiss")):
                bases[base_name] = base_path
        
        # Se nenhuma base for encontrada, procurar recursivamente em até 2 níveis
        if not bases:
            st.warning("Nenhuma base encontrada no diretório especificado. Procurando em subdiretórios...")
            for root, dirs, _ in os.walk(base_dir):
                # Limitar a profundidade da busca
                if root.count(os.sep) > base_dir.count(os.sep) + 2:
                    continue
                
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if os.path.exists(os.path.join(dir_path, "index.faiss")):
                        bases[dir_name] = dir_path
                        st.info(f"Base encontrada em: {dir_path}")
    
    return bases


def create_knowledge_base(embedding_model):
    """Interface para criar nova base de conhecimento"""
    # Usar diretório temporário para ambiente Streamlit Cloud
    import tempfile
    
    # Criar um diretório persistente dentro da sessão Streamlit
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()
    
    # Oferecer opções de local para salvar base
    storage_option = st.radio(
        "Onde salvar a base:",
        ["Diretório temporário (recomendado para Streamlit Cloud)", "Caminho personalizado"]
    )
    
    if storage_option == "Diretório temporário (recomendado para Streamlit Cloud)":
        folder_path = st.session_state['temp_dir']
        st.info(f"Usando diretório temporário: {folder_path}")
    else:
        folder_path = st.text_input(
            "Caminho para salvar a base:",
            help="Digite o caminho completo onde deseja salvar a base vetorial"
        )
    
    if folder_path:
        st.info(f"Caminho completo: {os.path.abspath(folder_path)}")
    
    if folder_path and not os.path.exists(folder_path):
        st.warning(f"A pasta {folder_path} não existe. Ela será criada ao salvar.")
    
    uploaded_file = st.file_uploader(
        "Upload do arquivo CSV",
        type=['csv']
    )
    
    if uploaded_file and folder_path:
        st.write(f"Arquivo selecionado: {uploaded_file.name}")
        
        base_name = st.text_input(
            "Nome da base:",
            help="Nome descritivo para identificar esta base"
        )
        
        if base_name and st.button("Criar Base"):
            try:
                # Sanitizar nome da base
                base_name = "".join(c for c in base_name.strip() if c.isalnum() or c in ('-', '_'))
                
                # Garantir que o diretório existe
                os.makedirs(folder_path, exist_ok=True)
                base_path = os.path.join(folder_path, base_name)
                
                if os.path.exists(base_path):
                    st.error("Uma base com este nome já existe nesta pasta")
                    return
                
                # Processar o arquivo
                file_content = uploaded_file.getvalue()
                documents = process_csv(file_content)
                
                if not documents:
                    raise ValueError("Nenhum documento foi processado")
                
                # Criar e salvar a base vetorial
                vector_store = create_vector_store(documents, base_path, embedding_model)
                
                if vector_store:
                    st.session_state['vector_store'] = vector_store
                    st.session_state['base_dir'] = folder_path
                    st.success(f"Base '{base_name}' criada com sucesso em {folder_path}!")
                    st.balloons()
            
            except Exception as e:
                st.error(f"Erro ao criar base: {str(e)}")


def use_existing_base(embedding_model):
    """Interface para usar bases existentes"""
    # Usar diretório temporário para ambiente Streamlit Cloud
    if 'temp_dir' not in st.session_state:
        import tempfile
        st.session_state['temp_dir'] = tempfile.mkdtemp()
    
    # Oferecer opções para carregar base
    load_option = st.radio(
        "Como carregar a base:",
        ["Usar base criada nesta sessão", "Carregar de um diretório", "Fazer upload de arquivo"]
    )
    
    folder_path = None
    
    if load_option == "Usar base criada nesta sessão":
        folder_path = st.session_state['temp_dir']
        st.info(f"Usando diretório temporário: {folder_path}")
        
    elif load_option == "Carregar de um diretório":
        folder_path = st.text_input(
            "Caminho da pasta com bases existentes:",
            help="Digite o caminho completo onde estão suas bases"
        )
        if folder_path:
            st.info(f"Caminho completo: {os.path.abspath(folder_path)}")
    
    elif load_option == "Fazer upload de arquivo":
        uploaded_file = st.file_uploader(
            "Faça upload do arquivo da base vetorial",
            type=["pkl"]
        )
        
        if uploaded_file:
            try:
                with st.spinner("Carregando base do arquivo..."):
                    # Carregar o vetor store do arquivo
                    vector_store = pickle.loads(uploaded_file.getvalue())
                    
                    # Verificar se é um vetor store válido
                    if not isinstance(vector_store, FAISS):
                        st.error("O arquivo não contém uma base vetorial válida.")
                        return
                    
                    st.session_state['vector_store'] = vector_store
                    st.success("Base carregada com sucesso do arquivo!")
                    return
            except Exception as e:
                st.error(f"Erro ao carregar base do arquivo: {str(e)}")
                return
    
    if not folder_path:
        st.info("Por favor, informe o caminho da pasta que contém suas bases")
        return
    
    if not os.path.exists(folder_path):
        st.error(f"A pasta {folder_path} não existe")
        return
    
    bases = get_available_bases(folder_path)
    
    if not bases:
        st.warning(f"Nenhuma base encontrada em {folder_path}")
        return
    
    selected_base = st.selectbox(
        "Selecione a base:",
        options=list(bases.keys())
    )
    
    if selected_base and st.button("Carregar Base"):
        try:
            vector_store = load_vector_store(bases[selected_base], embedding_model)
            
            if vector_store:
                st.session_state['vector_store'] = vector_store
                st.session_state['base_dir'] = folder_path
                st.success(f"Base '{selected_base}' carregada com sucesso!")
        
        except Exception as e:
            st.error(f"Erro ao carregar base: {str(e)}")


def handle_query(model):
    """Interface para consultas à base de conhecimento"""
    if st.session_state.get('vector_store'):
        st.write("Base carregada e pronta para consultas")
        
        query = st.text_area("Digite sua pergunta:")
        
        if query and st.button("Consultar"):
            try:
                with st.spinner("Buscando informações relevantes..."):
                    results = st.session_state['vector_store'].similarity_search(query, k=5)
                    context = "\n\n".join([doc.page_content for doc in results])
                
                with st.spinner("Gerando resposta com Gemma-3-27b-it..."):
                    prompt = f"""
                    Contexto: {context}
                    
                    Pergunta: {query}
                    
                    Com base apenas nas informações do contexto acima, responda à pergunta.
                    Se a informação não estiver disponível no contexto, diga que não pode responder com os dados fornecidos.
                    """
                    
                    response = model.generate_content(prompt)
                
                st.subheader("Resposta:")
                st.markdown(response.text)
                
                with st.expander("Ver contexto utilizado"):
                    st.markdown(context)
            
            except Exception as e:
                st.error(f"Erro na consulta: {str(e)}")
    else:
        st.info("Nenhuma base carregada. Crie uma nova base ou carregue uma existente.")


if __name__ == "__main__":
    main()
