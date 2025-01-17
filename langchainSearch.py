import streamlit as st
import faiss
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import openai
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
google_api_key = os.getenv("google_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração da página
st.set_page_config(page_title="Sistema Legal", layout="wide")

# Carregar arquivo CSS
def carregar_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

carregar_css("style.css")

# Função para limpar o chat, histórico e resultados
def limpar_tudo():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.documentos_contexto = []
    st.rerun()

# Inicializar estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'documentos_contexto' not in st.session_state:
    st.session_state.documentos_contexto = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def carregar_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        return FAISS.load_local(
            "faiss_legal_store_gemini", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Erro ao carregar vector store: {str(e)}")
        return None

def get_llm():
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key,
        temperature=0.7
    )

def gerar_resposta(pergunta, contexto_docs, historico):
    try:
        llm = get_llm()
        contexto_texto = "\n\n".join([
            f"Documento {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(contexto_docs)
        ])
        historico_texto = "\n".join([
            f"Humano: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
            for msg in historico[-3:]
        ])
        prompt = f"""Baseado nos seguintes documentos jurídicos e no histórico da conversa, responda à pergunta de forma clara e objetiva.
        
        Documentos de Referência:
        {contexto_texto}

        Histórico da Conversa:
        {historico_texto}

        Pergunta Atual: {pergunta}

        Responda usando uma linguagem formal e técnica apropriada para o contexto jurídico."""
        resposta = llm.invoke(prompt)
        return str(resposta)
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

def busca_combinada(vector_store, query, campo, valor_campo, texto_livre, num_results):
    try:
        todos_docs = vector_store.similarity_search("", k=100000)
        resultados = []
        termos_busca = []
        if texto_livre:
            termos_busca = [termo.strip().upper() for termo in texto_livre.split(',') if termo.strip()]
        for doc in todos_docs:
            match = True
            if campo and valor_campo:
                if doc.metadata.get(campo) != valor_campo:
                    match = False
            if termos_busca and match:
                conteudo = doc.page_content.upper()
                if not all(termo in conteudo for termo in termos_busca):
                    match = False
            if match:
                resultados.append(doc)
        if query and resultados:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
            temp_store = FAISS.from_documents(resultados, embeddings)
            resultados = temp_store.similarity_search(
                query,
                k=min(num_results, len(resultados))
            )
        return resultados[:num_results]
    except Exception as e:
        st.error(f"Erro na busca: {str(e)}")
        return []

def extrair_campos_unicos(vector_store):
    try:
        resultados = vector_store.similarity_search("", k=100000)
        classes = set()
        assuntos = set()
        for doc in resultados:
            metadata = getattr(doc, 'metadata', {})
            if metadata.get('classe'):
                classes.add(metadata['classe'])
            if metadata.get('assunto'):
                assuntos.add(metadata['assunto'])
        return sorted(list(classes)), sorted(list(assuntos))
    except Exception as e:
        st.error(f"Erro ao extrair campos: {str(e)}")
        return [], []

vector_store = carregar_vector_store()
if vector_store:
    # Sidebar
    with st.sidebar:
        expander_pesq = st.expander("🔍 Filtros de Busca", expanded=True)
        with expander_pesq:            
            query = st.text_input("Busca Semântica:", placeholder="Digite sua consulta")
            campo = st.selectbox(
                "Campo:", ["", "classe", "assunto"], help="Filtrar por campo"
            )
            valores = []
            if campo:
                classes, assuntos = extrair_campos_unicos(vector_store)
                valores = classes if campo == "classe" else assuntos
            valor_campo = st.selectbox(
                "Valor do Campo:", [""] + valores, help="Selecione o valor do campo"
            )
            texto_livre = st.text_input(
                "Texto Livre:", placeholder="Ex: termo1, termo2"
            )
            num_results = st.slider("Nº de Resultados:", 1, 20, 4)

            col1, col2 = st.columns([1,1])
            with col1:
                buscar = st.button("Buscar")
                if buscar:
                    if not any([query, campo and valor_campo, texto_livre]):
                        st.warning("Especifique pelo menos um critério de busca.")
                    else:
                        with st.spinner("Buscando..."):
                            docs = busca_combinada(
                                vector_store, query, campo, valor_campo, texto_livre, num_results
                            )
                            st.session_state.documentos_contexto = docs
                            if not docs:
                                st.warning("Nenhum resultado encontrado.")
            with col2:
                if st.button("🗑️ Limpar Tudo", key="limpar_tudo"):
                    limpar_tudo()

        # Documentos no sidebar
        if st.session_state.documentos_contexto:
            with st.expander("📚 Documentos Encontrados", expanded=False):
                st.markdown(f"Total: {len(st.session_state.documentos_contexto)}")
                doc_titles = [f"Assunto : {doc.metadata.get('assunto', 'Sem assunto')}" 
                            for i, doc in enumerate(st.session_state.documentos_contexto)]
                selected_doc = st.selectbox("Selecione um documento:", doc_titles)
                
                if selected_doc:
                    doc_index = doc_titles.index(selected_doc)
                    doc = st.session_state.documentos_contexto[doc_index]
                    st.text_area(
                        "Conteúdo:",
                        doc.page_content,
                        height=300,
                        key=f"text_{doc_index}"
                    )

    # Área principal do chat
    st.markdown("#### 💬 Chat Assistente Jurídico")

    # Exibir mensagens do chat na área principal
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("O que você gostaria de perguntar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("🤔 Pensando..."):
            resposta = gerar_resposta(prompt, st.session_state.documentos_contexto, st.session_state.chat_history)
            st.session_state.messages.append({"role": "assistant", "content": resposta})
            st.session_state.chat_history.extend([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': resposta}
            ])
            
        with st.chat_message("assistant"):
            st.markdown(resposta)

else:
    st.error("Não foi possível carregar o vector store")
