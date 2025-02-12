import cloudscraper
from bs4 import BeautifulSoup
import time
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from PyPDF2 import PdfReader
import csv
import docx
from googlesearch import search
from serpapi import GoogleSearch
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import pandas as pd
from io import StringIO
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



    # Configuração da página


    # Carregar variáveis de ambiente
load_dotenv()
google_api_key = os.getenv("google_api_key")
#-------------------------------------------------------------------------------
genai.configure(api_key=os.getenv("google_api_key"))

#def main():

def get_model():  # retorna o modelo para pergunta, diferente do embedding   
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config
    )
    return model
    #-------------------------------------------------------------------------------

def get_mq_retriever(vector_store, llm):
    
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever= vector_store.as_retriever(),
        llm=llm
        )
    return mq_retriever

def get_embeddings():

    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    
    return embeddings

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def limpar_tudo():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.documentos_contexto = []
    st.session_state.uploaded_files = []
    st.session_state.use_web_search = True
    st.session_state.chat_session=[] # importante
    st.session_state.documentos_contexto=[]
    
    st.rerun()


def extract_text_from_csv(uploaded_file):

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()

    #reader = csv.DictReader(f)
       
    # for row in reader:
    #     textCSV += f"{row.get('id', '')} {row.get('titulo', '')} {row.get('classe', '')} {row.get('conteúdo', '')} \n"
    # with open("C:/Users/mcres/Documents/documentos.csv", "r", encoding="utf-8") as f:
    #     reader = csv.DictReader(f)
    #     text = ""

    #     for row in reader:
    #         text += f"{row.get('id', '')} {row.get('titulo', '')} {row.get('classe', '')} {row.get('conteúdo', '')} \n"
       
    # Dividir texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # # Salvar
    vectorstore.save_local("faiss_documento_cvs2")
        
    return vectorstore

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return {
            'name': uploaded_file.name,
            'content': extract_text_from_pdf(uploaded_file),
            'type': 'pdf'
        }
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return {
            'name': uploaded_file.name,
            'content': extract_text_from_docx(uploaded_file),
            'type': 'docx'
        }
    # elif uploaded_file.type == "text/csv":
    #     return {
    #         'name': uploaded_file.name,
    #         'content': extract_text_from_csv(uploaded_file),
    #         'type': 'csv'
    #     }
    else:
        raise ValueError("Formato de arquivo não suportado")

llm =  GoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21",
            google_api_key=google_api_key,
            temperature=0.7
        )

# Inicializar estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'documentos_contexto' not in st.session_state:
    st.session_state.documentos_contexto = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Inicializar estado da sessão
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False

if 'use_juris_search' not in st.session_state:
    st.session_state.use_juris_search = False

if "chat_session" not in st.session_state: #importante
    model = get_model()   
    st.session_state.chat_session = model.start_chat(history=[])    

if 'prompt' not in st.session_state:        
    st.session_state.prompt=None    

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

vector_store = carregar_vector_store()

def perform_web_search(query, num_results=5):
    try:
        # Configurar a busca
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": num_results,
            "hl": "pt-br",
            "gl": "br"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extrair e formatar resultados
        if "organic_results" in results:
            formatted_results = []
            for result in results["organic_results"][:num_results]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                formatted_results.append(
                    f"Título: {title}\n"
                    f"Resumo: {snippet}\n"
                    f"Link: {link}"
                )
            
            return "\n\n".join(formatted_results)
        else:
            return "Nenhum resultado encontrado."
            
    except Exception as e:
        print(f"Erro na busca: {str(e)}")  # Para debug
        return "Não foi possível realizar a busca no momento."

def gerar_resposta(query, contexto_docs, historico, uploaded_files):
    
    try:

        contexto_completo=""            

        historico = "\n".join([
            f"Humano: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
            for msg in historico[-3:]
        ])

        # Realizar busca na web apenas se o toggle estiver ativado
        if st.session_state.use_web_search:
            web_results = ""
            if st.session_state.get('web_search_toggle', True):
                with st.spinner("Buscando informações na web..."):
                    web_search_results = perform_web_search(query)
                    if web_search_results and not web_search_results.startswith("Erro"):
                        web_results = f"\n\nInformações da Web:\n{web_search_results}"

            contexto_completo = f"{web_results}"
                        
        elif st.session_state.use_juris_search :
                                        
            resultados = re.findall(r'\((.*?)\)', prompt)
                #q=cancelamento+de+voo
            url = f"https://www.jusbrasil.com.br/jurisprudencia/busca?q={resultados}"
            
            for _ in range(5):
                    try:
                        scraper = cloudscraper.create_scraper()
                        response = scraper.get(url)
                        time.sleep(2)
                        if "Just a moment" not in response.text:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            page_content = soup.get_text(separator='\n', strip=True)
                    except:
                        page_content =""
                        pass
            contexto_completo = f"{page_content}"   
            
        elif st.session_state.documentos_contexto:    
            # Formatando contexto dos documentos do vector store
            contexto_texto = ""
            if contexto_docs:
                contexto_docs = "\n\n".join([
                    f"Documento do Vector Store {i+1}:\n<assunto> {doc.metadata.get('assunto', 'N/A')}</assunto>\n<texto>: {doc.page_content}</texto>"
                    for i, doc in enumerate(contexto_docs)
                ])
            
            contexto_completo = f"{contexto_docs}"                
                        
        elif st.session_state.uploaded_files:
            # Adicionando conteúdo dos arquivos carregados
            arquivos_texto = ""
            if uploaded_files:
                arquivos_texto = "\n\n".join([
                    f"Arquivo Carregado {i+1} - {arquivo['name']}:\n{arquivo['content'][:3000]}"
                    for i, arquivo in enumerate(uploaded_files)
                ])
            contexto_completo = f"{arquivos_texto}"                
        else:#se não houver contexto
            vector_store = carregar_vector_store()
            retriever = vector_store.as_retriever()
            mq_retriever= get_mq_retriever(vector_store, llm) 
            #retrieved_docs = retriever.get_relevant_documents(query)  
            retrieved_docs = mq_retriever.get_relevant_documents(query)               

            arquivos_texto = "\n\n".join([doc.page_content for doc in retrieved_docs])
                                
            contexto_completo = arquivos_texto  
            

            #st.write(contexto_completo)

        query += f" responda sempre em português. <contexto>{contexto_completo}</contexto>"
        
        # # Carregar e executar a chain de QA
        
        return query            
    
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"


def busca_combinada(vector_store, query, campo, valor_campo, texto_livre, num_results):
    try:
        if texto_livre:
            todos_docs = vector_store.similarity_search("", k=1500)
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
                retriever = temp_store.as_retriever()
                resultados =   retriever.get_relevant_documents(query, k=min(num_results, len(resultados)))
                        
            
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

def encontrar_urls(texto):
    padrao = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(padrao, texto)
    return urls

def stream_data(resposta):
    for word in resposta.split(" "):
        yield word + " "
        time.sleep(0.02)

vector_store = carregar_vector_store()

if vector_store:
    # Sidebar
    colSideBAr, colChatbot = st.columns([1, 2])
    with colSideBAr:
    #with st.sidebar:
        
        with st.expander("⚙️ Configurações", expanded=True):
            
            use_web = st.toggle(
                "Habilitar busca na internet",
                value=st.session_state.use_web_search,
                help="Ative para permitir que o assistente busque informações atualizadas na internet",
                key='web_search_toggle'  # Adicionando uma key única
            )
            # Atualizar o estado com o valor do toggle
            st.session_state.use_web_search = use_web
            use_juris= False            
            
            
            use_juris = st.toggle(
                "Habilitar busca de jurisprudência",
                value=st.session_state.use_juris_search,
                help="Ative para permitir que o assistente busque informações de jurisprudência no site jusbrasil",
                key='juris_search_toggle'  # Adicionando uma key única
            )
            # Atualizar o estado com o valor do toggle
            st.session_state.use_juris_search = use_juris
            use_web= False   

            
        with st.expander(label="🔗 Arraste seus arquivos aqui", expanded=True):
            uploaded_files = st.file_uploader(
                label="",
                type=["pdf", "docx", "txt","csv"],
                accept_multiple_files=True,
                help="Suporta múltiplos arquivos PDF, DOCX,TXT,csv",
            )                
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                        try:
                            processed_file = process_uploaded_file(uploaded_file)
                            st.session_state.uploaded_files.append(processed_file)                                
                        except Exception as e:
                            st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
        
        expander_pesq = st.expander("🔍 Filtros de Busca", expanded=True)
        with expander_pesq:            
            query = texto_livre = st.text_input(
                "Texto Livre:", placeholder="Ex: termo1, termo2"
            )
            campo = st.selectbox(
                "Campo:", ["classe", "assunto"], help="Filtrar por campo"
            )
            valores = []
            if campo:
                classes, assuntos = extrair_campos_unicos(vector_store)
                valores = classes if campo == "classe" else assuntos
            valor_campo = st.selectbox(
                "Valor do Campo:", [""] + valores, help="Selecione o valor do campo"
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
            with st.expander(f"📚  Total de Documentos Encontrados : {len(st.session_state.documentos_contexto)}", expanded=False):
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
   

    with colChatbot:
        # # Área principal do chat
        # st.markdown("""<center><h2> 💬 Chat Assistente Jurídico</h2><center>""", unsafe_allow_html= True)

        containerChatbot= st.container(height=800, border=True)
        with containerChatbot:

            # Exibir mensagens do chat
            for message in st.session_state.messages:
                with st.chat_message("user" if message["role"] == "user" else "assistant", 
                                    avatar="👨" if message["role"] == "user" else "⚖️"):                     
                
                    st.markdown(f"""
                        <div style="text-align: justify; font-family: Verdana; font-size: 14px;">
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
                        # Chat input
        if prompt := st.chat_input("O que você gostaria de perguntar?"):
            st.session_state.messages.append({"role": "user", "content": prompt})                
            
            with containerChatbot:
                with st.chat_message("user", avatar="👨"):
                    st.markdown(prompt)  
                
                with st.spinner("🤔 Pensando..."):                        
                    
                    prompt_formatado = gerar_resposta(
                                                prompt, 
                                                st.session_state.documentos_contexto,
                                                st.session_state.chat_history, 
                                                st.session_state.uploaded_files
                                                )         
                    resposta = st.session_state.chat_session.send_message(prompt_formatado)    

                with st.chat_message("assistant", avatar="⚖️"):
                    
                    st.text(f"Resposta \n\n:{resposta.text}") 
                    #st.write_stream(stream_data(resposta.text))                     
                    st.session_state.messages.append({"role": "assistant", "content": f" {resposta.text}"})
                    st.session_state.chat_history.extend([
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': resposta.text}
                    ])    
       
            
            
else:
    st.error("Não foi possível carregar o vector store")


#if __name__ == "__main__":
 #   main()
