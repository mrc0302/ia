import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
from io import StringIO

# LangChain e integrações
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.Youtubeing import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Carregadores de Documentos
from PyPDF2 import PdfReader
import docx

# APIs e Serviços
import google.generativeai as genai
from serpapi import GoogleSearch

# --- CARREGAMENTO DE CONFIGURAÇÕES E CSS ---

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

# Configura a API do Google Generative AI
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("A chave da API do Google não foi encontrada. Verifique seu arquivo .env.")

def load_css(file_name):
    """Função para carregar um arquivo CSS."""
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Arquivo CSS '{file_name}' não encontrado.")

# --- FUNÇÕES DE PROCESSAMENTO DE DADOS E MODELOS ---

@st.cache_resource
def get_llm():
    """Retorna uma instância do modelo de linguagem da LangChain."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", # Modelo mais recente e otimizado
        temperature=0.4,
        max_tokens=8192,
        top_p=0.95,
        top_k=50,
        convert_system_message_to_human=True
    )

@st.cache_resource
def get_embeddings_model():
    """Retorna uma instância do modelo de embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    """Extrai texto de um arquivo DOCX."""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def process_uploaded_file(uploaded_file):
    """Processa um arquivo carregado e retorna um dicionário com seu conteúdo."""
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        return {'name': uploaded_file.name, 'content': extract_text_from_pdf(uploaded_file), 'type': 'pdf'}
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return {'name': uploaded_file.name, 'content': extract_text_from_docx(uploaded_file), 'type': 'docx'}
    elif file_type in ["text/plain", "text/csv"]:
         return {'name': uploaded_file.name, 'content': StringIO(uploaded_file.getvalue().decode("utf-8")).read(), 'type': 'txt/csv'}
    else:
        st.warning(f"Formato de arquivo '{file_type}' não suportado diretamente. Tratando como texto.")
        return {'name': uploaded_file.name, 'content': StringIO(uploaded_file.getvalue().decode("utf-8")).read(), 'type': 'text'}


@st.cache_resource
def carregar_vector_store():
    """Carrega o vector store FAISS do disco."""
    try:
        embeddings = get_embeddings_model()
        return FAISS.load_local(
            "faiss_legal_store_gemini",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Erro ao carregar o vector store: {e}")
        return None

@st.cache_resource(show_spinner="Extraindo filtros do banco de dados...")
def extrair_campos_unicos(_vector_store):
    """Extrai valores únicos de metadados para os filtros (versão eficiente)."""
    try:
        classes = set()
        assuntos = set()
        # Itera sobre os metadados diretamente, que é muito mais rápido
        for i in range(_vector_store.index.ntotal):
            doc = _vector_store.docstore.search(_vector_store.index_to_docstore_id[i])
            if doc and hasattr(doc, 'metadata'):
                if doc.metadata.get('classe'):
                    classes.add(doc.metadata['classe'])
                if doc.metadata.get('assunto'):
                    assuntos.add(doc.metadata['assunto'])
        return sorted(list(classes)), sorted(list(assuntos))
    except Exception as e:
        st.error(f"Erro ao extrair campos para filtros: {e}")
        return [], []

def perform_web_search(query, num_results=5):
    """Realiza uma busca na web usando a API da SerpApi."""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": num_results,
            "hl": "pt-br",
            "gl": "br"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results:
            formatted_results = []
            for result in results["organic_results"][:num_results]:
                formatted_results.append(
                    f"Título: {result.get('title', '')}\n"
                    f"Resumo: {result.get('snippet', '')}\n"
                    f"Link: {result.get('link', '')}"
                )
            return "\n\n".join(formatted_results)
        else:
            return "Nenhum resultado encontrado na web."
    except Exception as e:
        return f"Não foi possível realizar a busca na web: {e}"

def gerar_resposta(query, llm, vector_store, uploaded_files, campo, valor_campo):
    """Gera uma resposta usando RAG (Retrieval-Augmented Generation) de forma eficiente."""
    try:
        # 1. Configurar o Retriever com base nos filtros
        metadata_filter = {}
        if campo in ["classe", "assunto"] and valor_campo and valor_campo.strip():
            metadata_filter[campo] = valor_campo

        retriever = vector_store.as_retriever(
            search_kwargs={'k': 10, 'filter': metadata_filter}
        )

        # 2. Criar documentos a partir dos arquivos de upload
        docs_from_upload = []
        if uploaded_files:
            for doc_data in uploaded_files:
                docs_from_upload.append(Document(
                    page_content=doc_data['content'],
                    metadata={'fonte': doc_data['name'], 'tipo': doc_data['type']}
                ))

        # 3. Recuperar documentos do vector store
        retrieved_docs = retriever.invoke(query)
        
        # 4. Combinar todos os documentos de contexto
        final_context_docs = retrieved_docs + docs_from_upload

        # 5. Definir o Prompt Template
        prompt_template = """Você é um assistente especialista em direito.
        Responda à pergunta do usuário de forma clara, objetiva e em português, com base exclusivamente no contexto fornecido.
        Se a resposta não estiver no contexto, informe que não encontrou a informação nos documentos disponíveis.
        Ao citar uma informação, se possível, mencione a fonte (ex: [Classe: Ação Civil, Assunto: Meio Ambiente] ou [fonte: nome_do_arquivo.pdf]).

        <contexto>
        {context}
        </contexto>
        
        Pergunta: {input}
        
        Resposta:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 6. Criar e invocar a Chain de RAG
        Youtube_chain = create_stuff_documents_chain(llm, prompt)
        
        # Passamos todos os documentos (recuperados + upload) para a chain final
        response = Youtube_chain.invoke({
            "input": query,
            "context": final_context_docs
        })

        return response['answer']

    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao processar sua solicitação."

def stream_data(resposta):
    """Gera a resposta palavra por palavra para um efeito de streaming."""
    for word in resposta.split(" "):
        yield word + " "
        time.sleep(0.02)
        
# --- INICIALIZAÇÃO DA SESSÃO E DA PÁGINA ---

def init_session():
    """Inicializa o estado da sessão do Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'documentos_contexto' not in st.session_state:
        st.session_state.documentos_contexto = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False
    if 'use_juris_search' not in st.session_state:
        st.session_state.use_juris_search = False

def main():
    st.set_page_config(
        page_title="Sistema de Modelos Judiciais",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Assistente Jurídico com IA")
    load_css("static/styles.css")
    
    # Inicializa a sessão
    init_session()

    # Carrega os modelos e o vector store
    llm = get_llm()
    vector_store = carregar_vector_store()
    
    if not vector_store:
        st.stop() # Interrompe a execução se o vector store não puder ser carregado

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Painel de Controle")
        tab1, tab2, tab3 = st.tabs(["Pesquisa", "Arquivos", "Configurações"])

        with tab1:
            with st.expander("🔍 Filtros de Busca", expanded=True):
                texto_livre = st.text_input("Buscar por termos:", placeholder="Ex: dano moral, consumidor")
                
                classes, assuntos = extrair_campos_unicos(vector_store)
                
                campo = st.selectbox("Filtrar por:", ["Nenhum", "classe", "assunto"])
                
                valor_campo = ""
                if campo == "classe":
                    valor_campo = st.selectbox("Selecione a Classe:", classes)
                elif campo == "assunto":
                    valor_campo = st.selectbox("Selecione o Assunto:", assuntos)

                num_results = st.slider("Nº de Resultados da Busca:", 1, 20, 5)

                if st.button("Buscar Documentos", use_container_width=True):
                    if not texto_livre and campo == "Nenhum":
                        st.warning("Especifique um termo de busca ou um filtro.")
                    else:
                        with st.spinner("Buscando..."):
                            # Usa a busca por similaridade do retriever, que é eficiente
                            retriever = vector_store.as_retriever(search_kwargs={'k': num_results})
                            docs = retriever.invoke(texto_livre)
                            st.session_state.documentos_contexto = docs
                            if not docs:
                                st.warning("Nenhum resultado encontrado.")
                            else:
                                st.success(f"{len(docs)} documentos encontrados.")
        
            # Exibição dos documentos encontrados
            if st.session_state.documentos_contexto:
                with st.expander("📚 Documentos Encontrados", expanded=True):
                    for i, doc in enumerate(st.session_state.documentos_contexto):
                        # Tenta extrair um título significativo dos metadados
                        title = doc.metadata.get('assunto', f"Documento {i+1}")
                        st.info(f"**{title}**")
                        st.text_area(
                            label="Conteúdo:",
                            value=doc.page_content[:1000] + "...", # Mostra um trecho
                            height=150,
                            key=f"doc_text_{i}",
                            disabled=True
                        )


        with tab2:
            with st.expander("🔗 Upload de Arquivos", expanded=True):
                uploaded_files = st.file_uploader(
                    label="Arraste seus arquivos para usar no chat",
                    label_visibility="collapsed", # Esconde o label, mas mantém para acessibilidade
                    type=["pdf", "docx", "txt", "csv"],
                    accept_multiple_files=True,
                    help="Use estes arquivos como contexto para suas perguntas no chat.",
                )
                
                if st.button("Processar Arquivos", use_container_width=True):
                    if uploaded_files:
                        processed_docs = []
                        with st.spinner("Processando arquivos..."):
                            for up_file in uploaded_files:
                                try:
                                    processed_docs.append(process_uploaded_file(up_file))
                                except Exception as e:
                                    st.error(f"Erro ao processar {up_file.name}: {e}")
                        st.session_state.uploaded_files = processed_docs
                        st.success(f"{len(processed_docs)} arquivos prontos para uso.")

        with tab3:
            with st.expander("⚙️ Configurações", expanded=True):
                # A chave (key) vincula o estado do widget diretamente ao st.session_state
                st.toggle(
                    "Habilitar busca na internet (SerpApi)",
                    key='use_web_search',
                    help="Permite que o assistente busque informações na internet se não encontrar nos documentos.",
                )
                if st.button("🗑️ Limpar Chat e Arquivos", use_container_width=True, type="primary"):
                    init_session()
                    st.rerun()

    # --- ÁREA PRINCIPAL DO CHAT ---
    
    # Container para exibir o chat
    chat_container = st.container()
    with chat_container:
        # Exibir mensagens do histórico
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input do usuário fixo no final da página
    if prompt := st.chat_input("Faça sua pergunta aqui..."):
        # Adiciona e exibe a mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Gera e exibe a resposta do assistente
        with st.spinner("Analisando documentos e pensando..."):
            
            final_query = prompt
            web_context = ""
            # Verifica se a busca na web está habilitada
            if st.session_state.use_web_search:
                web_context = perform_web_search(prompt)
                final_query = f"{prompt}\n\nContexto adicional da web:\n{web_context}"

            resposta = gerar_resposta(
                query=final_query,
                llm=llm,
                vector_store=vector_store,
                uploaded_files=st.session_state.uploaded_files,
                campo=campo if 'campo' in locals() else "Nenhum",
                valor_campo=valor_campo if 'valor_campo' in locals() else ""
            )
        
        with chat_container:
            with st.chat_message("assistant"):
                st.write_stream(stream_data(resposta))
        
        # Adiciona a resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()
