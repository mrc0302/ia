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
import io
import shutil


def main():
    st.title("RAG Legal App com Gemma-3-27b-it")
    
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
    
    # Configuração do modelo Gemma
    generation_config = {
        "temperature": 0.3,  # Temperatura reduzida para respostas mais precisas em contexto legal
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 4096,
    }
    
    # Modelos globais
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        model = genai.GenerativeModel(
            model_name="gemma-3-27b-it",
            generation_config=generation_config,
        )
    except Exception as e:
        st.error(f"Erro ao inicializar modelos: {str(e)}")
        st.info("Dica: Verifique se a chave da API Google está correta e tem acesso aos modelos necessários.")
        return
    
    # Interface principal com duas colunas
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Gerenciamento de Base Legal")
        mode = st.radio(
            "Modo:",
            ["Criar base legal", "Usar base legal existente"]
        )
        
        if mode == "Criar base legal":
            create_legal_knowledge_base(embedding_model)
        else:
            use_existing_legal_base(embedding_model)
    
    with col2:
        st.subheader("Consulta Jurídica")
        handle_legal_query(model)


def process_legal_csv(file_content: bytes) -> List[Document]:
    """Processa arquivo CSV com conteúdo legal e converte em documentos"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        st.write(f"Colunas detectadas: {', '.join(df.columns)}")
        
        documents = []
        df = df.fillna('')
        
        # Verificar formato para documentos legais
        legal_columns = ['id', 'texto_legal', 'tipo_documento', 'data', 'jurisdicao']
        
        # Verificar se o CSV tem pelo menos id e texto_legal
        if all(col in df.columns for col in ['id', 'texto_legal']):
            # Formato específico para documentos legais
            for _, row in df.iterrows():
                metadata = {k: str(v) for k, v in row.items() if k != 'texto_legal'}
                doc = Document(
                    page_content=str(row['texto_legal']),
                    metadata=metadata
                )
                documents.append(doc)
                
        # Formato alternativo com 'id' e 'texto'
        elif all(col in df.columns for col in ['id', 'texto']):
            for _, row in df.iterrows():
                metadata = {k: str(v) for k, v in row.items() if k != 'texto'}
                doc = Document(
                    page_content=str(row['texto']),
                    metadata=metadata
                )
                documents.append(doc)
                
        else:
            # Formato genérico - converte todas as linhas em documentos
            st.warning("Formato CSV não otimizado para documentos legais. Processando no modo genérico.")
            for idx, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=content,
                    metadata={'row_id': str(idx)}
                )
                documents.append(doc)
        
        if not documents:
            raise ValueError("Nenhum documento válido foi encontrado no CSV")
        
        st.success(f"{len(documents)} documentos legais processados com sucesso")
        return documents
    
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo CSV está vazio")
    except Exception as e:
        raise ValueError(f"Erro ao processar o CSV: {str(e)}")


def create_legal_vector_store(documents, vector_db_path, embedding_model):
    """Cria e salva a base de conhecimento vetorial otimizada para textos legais"""
    try:
        os.makedirs(vector_db_path, exist_ok=True)
        
        with st.spinner("Dividindo documentos legais em chunks..."):
            # Tamanho de chunk maior para capturar contexto legal adequadamente
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Chunks maiores para documentos legais
                chunk_overlap=300,  # Overlap maior para manter contexto
                separators=["\n\n", "\n", ".", " ", ""],  # Separadores específicos para documentos legais
            )
            chunks = text_splitter.split_documents(documents)
            st.info(f"Total de chunks: {len(chunks)}")
        
        with st.spinner("Criando embeddings e salvando vector store..."):
            # Criar diretório para o faiss_legal_store_gemini
            vector_store = FAISS.from_documents(chunks, embedding_model)
            vector_store.save_local(vector_db_path)
        
        # Verificação final
        if not os.path.exists(os.path.join(vector_db_path, "index.faiss")):
            raise FileNotFoundError("Falha ao salvar vector store")
        
        # Salvar informações adicionais sobre o índice
        index_info = {
            "type": "faiss_legal_store_gemini",
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "embedding_model": "models/embedding-001"
        }
        
        with open(os.path.join(vector_db_path, "index_info.txt"), "w") as f:
            for key, value in index_info.items():
                f.write(f"{key}: {value}\n")
        
        return vector_store
    
    except Exception as e:
        st.error(f"Erro ao criar vector store legal: {str(e)}")
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
        return None


def load_legal_vector_store(vector_db_path, embedding_model):
    """Carrega uma base de conhecimento vetorial legal existente"""
    try:
        with st.spinner("Carregando base de conhecimento legal..."):
            vector_store = FAISS.load_local(
                vector_db_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Verificar se é uma base legal
            info_path = os.path.join(vector_db_path, "index_info.txt")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = f.read()
                    if "faiss_legal_store_gemini" not in info:
                        st.warning("Esta base não foi otimizada para documentos legais")
        
        return vector_store
    except Exception as e:
        st.error(f"Erro ao carregar vector store legal: {str(e)}")
        return None


def get_available_legal_bases(base_dir):
    """Lista todas as bases de conhecimento legal disponíveis no diretório escolhido"""
    bases = {}
    if os.path.exists(base_dir):
        for base_name in os.listdir(base_dir):
            base_path = os.path.join(base_dir, base_name)
            if os.path.isdir(base_path) and os.path.exists(os.path.join(base_path, "index.faiss")):
                # Verificar se é uma base legal
                info_path = os.path.join(base_path, "index_info.txt")
                is_legal = False
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        is_legal = "faiss_legal_store_gemini" in f.read()
                
                # Adicionar à lista, marcando se é legal ou não
                name_display = f"{base_name} [LEGAL]" if is_legal else base_name
                bases[name_display] = base_path
    return bases


def create_legal_knowledge_base(embedding_model):
    """Interface para criar nova base de conhecimento legal"""
    folder_path = st.text_input(
        "Caminho para salvar a base legal:",
        help="Digite o caminho completo onde deseja salvar a base vetorial"
    )
    
    if folder_path and not os.path.exists(folder_path):
        st.warning(f"A pasta {folder_path} não existe. Ela será criada ao salvar.")
    
    uploaded_file = st.file_uploader(
        "Upload do arquivo CSV com documentos legais",
        type=['csv']
    )
    
    if uploaded_file and folder_path:
        st.write(f"Arquivo selecionado: {uploaded_file.name}")
        
        base_name = st.text_input(
            "Nome da base legal:",
            help="Nome descritivo para identificar esta base de documentos legais"
        )
        
        if base_name and st.button("Criar Base Legal"):
            try:
                # Sanitizar nome da base
                base_name = "".join(c for c in base_name.strip() if c.isalnum() or c in ('-', '_'))
                
                # Garantir que o diretório existe
                os.makedirs(folder_path, exist_ok=True)
                base_path = os.path.join(folder_path, f"faiss_legal_store_gemini_{base_name}")
                
                if os.path.exists(base_path):
                    st.error("Uma base legal com este nome já existe nesta pasta")
                    return
                
                # Processar o arquivo
                file_content = uploaded_file.getvalue()
                documents = process_legal_csv(file_content)
                
                if not documents:
                    raise ValueError("Nenhum documento legal foi processado")
                
                # Criar e salvar a base vetorial
                vector_store = create_legal_vector_store(documents, base_path, embedding_model)
                
                if vector_store:
                    st.session_state['vector_store'] = vector_store
                    st.session_state['base_dir'] = folder_path
                    st.success(f"Base legal '{base_name}' criada com sucesso em {folder_path}!")
                    st.balloons()
            
            except Exception as e:
                st.error(f"Erro ao criar base legal: {str(e)}")


def use_existing_legal_base(embedding_model):
    """Interface para usar bases legais existentes"""
    folder_path = st.text_input(
        "Caminho da pasta com bases legais existentes:",
        help="Digite o caminho completo onde estão suas bases"
    )
    
    if not folder_path:
        st.info("Por favor, informe o caminho da pasta que contém suas bases legais")
        return
    
    if not os.path.exists(folder_path):
        st.error(f"A pasta {folder_path} não existe")
        return
    
    bases = get_available_legal_bases(folder_path)
    
    if not bases:
        st.warning(f"Nenhuma base encontrada em {folder_path}")
        return
    
    selected_base = st.selectbox(
        "Selecione a base legal:",
        options=list(bases.keys())
    )
    
    if selected_base and st.button("Carregar Base Legal"):
        try:
            vector_store = load_legal_vector_store(bases[selected_base], embedding_model)
            
            if vector_store:
                st.session_state['vector_store'] = vector_store
                st.session_state['base_dir'] = folder_path
                st.success(f"Base '{selected_base}' carregada com sucesso!")
        
        except Exception as e:
            st.error(f"Erro ao carregar base legal: {str(e)}")


def handle_legal_query(model):
    """Interface para consultas à base de conhecimento legal"""
    if st.session_state.get('vector_store'):
        st.write("Base legal carregada e pronta para consultas")
        
        query = st.text_area("Digite sua consulta jurídica:")
        
        if query and st.button("Consultar"):
            try:
                with st.spinner("Buscando informações legais relevantes..."):
                    # Aumentar número de resultados para melhor contexto legal
                    results = st.session_state['vector_store'].similarity_search(query, k=7)
                    context = "\n\n".join([doc.page_content for doc in results])
                
                with st.spinner("Gerando resposta jurídica com Gemma-3-27b-it..."):
                    prompt = f"""
                    Contexto Legal: {context}
                    
                    Consulta Jurídica: {query}
                    
                    Você é um assistente jurídico especializado. Com base apenas nas informações do contexto legal acima, 
                    responda à consulta jurídica de forma clara e objetiva.
                    
                    Se a informação não estiver disponível no contexto, diga que não pode responder com os dados fornecidos
                    e sugira que o usuário consulte um advogado para orientação específica.
                    
                    Estruture sua resposta em:
                    1. Entendimento da consulta
                    2. Fundamentação legal (apenas com base no contexto fornecido)
                    3. Conclusão
                    """
                    
                    response = model.generate_content(prompt)
                
                st.subheader("Resposta Jurídica:")
                st.markdown(response.text)
                
                with st.expander("Ver contexto legal utilizado"):
                    st.markdown(context)
                
                # Mostrar metadados dos documentos utilizados em um expander separado
                with st.expander("Metadados dos documentos consultados"):
                    for i, doc in enumerate(results):
                        st.subheader(f"Documento {i+1}")
                        st.json(doc.metadata)
                        st.divider()
            
            except Exception as e:
                st.error(f"Erro na consulta jurídica: {str(e)}")
    else:
        st.info("Nenhuma base legal carregada. Crie uma nova base legal ou carregue uma existente.")


if __name__ == "__main__":
    main()
