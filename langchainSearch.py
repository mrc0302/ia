import cloudscraper
import uuid
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
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Consulta de Documentos com faiss", layout="wide")


def main():
        
        # Configuração da página


        # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    #-------------------------------------------------------------------------------
    genai.configure(api_key=os.getenv("google_api_key"))

    # llm =  GoogleGenerativeAI(
    #         model="gemini-2.0-flash-thinking-exp-01-21",
    #         google_api_key=google_api_key,
    #         temperature=0.3
    #     )
    llm =  genai.GenerativeModel( 
        
        model_name="gemini-2.0-flash-exp",
        generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }       
    )
        #-------------------------------------------------------------------------------
    def get_mq_retriever(vector_store, llm):
        
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever= vector_store.as_retriever(),
            llm=llm
            )
        return mq_retriever

    # def get_embeddings():

    #     embeddings = GoogleGenerativeAIEmbeddings(
    #             model="models/embedding-001",
    #             google_api_key=google_api_key
    #         )
        
    #     return embeddings
     
    def clear_history():
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.documentos_contexto = []
        st.session_state.uploaded_files = []
        st.session_state.use_web_search = False
        st.session_state.chat_session=[] # importante
        st.session_state.documentos_contexto=[]
        st.session_state.uploader_key =None
        reset_uploader()
        st.rerun()

    def extract_text_from_csv(uploaded_file, local_path = "faiss_documento_cvs2"):

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
        vectorstore.save_local(local_path)
            
        return vectorstore

    def extract_text_from_pdf_vectorstore(pdf_file):
        # pdf_file = open(pdf_file, "rb")   
        # Use BytesIO para trabalhar com o conteúdo do arquivo carregado
        from io import BytesIO
        pdf_bytes = BytesIO(pdf_file.read())
        pdf_reader = PdfReader(pdf_bytes)
        
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

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
        
        # # CARREGO NA MEMÓRIA
        st.session_state.uploaded_files.append(vectorstore)
                                   
        # # ou Salvar  
       # vectorstore.save_local(pdf_file)
            
        return text    
         
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
        elif uploaded_file.type == "text/csv":
            return {
                'name': uploaded_file.name,
                'content': extract_text_from_csv(uploaded_file),
                'type': 'csv'
            }
        else:
            raise ValueError("Formato de arquivo não suportado")

   
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())
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
        model = llm   
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

            
            llm= GoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=google_api_key,
                temperature=0.7
            )
               

            historico = "\n".join([
                f"Humano: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
                for msg in historico[-3:]
            ])

            # Realizar busca na web apenas se o toggle estiver ativado
            if st.session_state.use_web_search:
                st.session_state.chat_history = []
                    
                with st.spinner("Buscando informações na web..."):
                    web_search_results = perform_web_search(query)  # Isso retorna uma string
                    
                    # Converter resultados da web em objetos Document
                    retrieved_docs = []
                    if web_search_results and not web_search_results.startswith("Erro"):
                        web_results = web_search_results.split("\n\n")
                        
                        # Transformar cada resultado em um objeto Document
                        for result in web_results:
                            doc = Document(page_content=result, metadata={"source": "web_search"})
                            retrieved_docs.append(doc)
                            
            elif st.session_state.use_juris_search :
                st.session_state.chat_history = []                           
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
                retrieved_docs = [
                    Document(
                        page_content=doc.page_content,
                        metadata={"assunto": doc.metadata.get('assunto', 'N/A')}
                    )
                    for doc in st.session_state.documentos_contexto
                ]
            elif st.session_state.uploaded_files:                          
                
               # Adicionando conteúdo dos arquivos carregados
                documentos = []
                if uploaded_files:
                    documentos = [
                        Document(
                            page_content=arquivo['content'],
                            metadata={"name": arquivo['name'], "type": arquivo['type']}
                        )
                        for arquivo in uploaded_files
                    ]
                retrieved_docs = documentos

            else:#se não houver contexto
                
                vector_store = carregar_vector_store()
                mq_retriever = MultiQueryRetriever.from_llm(retriever = vector_store.as_retriever(), llm = llm)                       
                retrieved_docs = mq_retriever.get_relevant_documents(query=query)
                
                # vector_store = carregar_vector_store()
                # retriever = vector_store.as_retriever()
                # mq_retriever= get_mq_retriever(vector_store, llm) 
                # #retrieved_docs = retriever.get_relevant_documents(query)  
                # retrieved_docs = mq_retriever.get_relevant_documents(query)               

                # arquivos_texto = "\n\n".join([doc.page_content for doc in retrieved_docs])
                                    
                # contexto_completo = arquivos_texto  
                          
            
            query +=f" responda sempre em português. "              

            chain = load_qa_chain(llm, chain_type="stuff")
            resposta = chain.run(input_documents=retrieved_docs, question=query)
           # print(resposta) 

            return resposta
            
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

    def reset_uploader():
        # Gerar uma nova chave UUID aleatória forçará o Streamlit a criar um novo componente
        st.session_state.uploader_key = str(uuid.uuid4())
       
    
    # Sidebar
    colSideBAr, colChatbot = st.columns([1, 2])
    with colSideBAr:
    #with st.sidebar:
        
        with st.expander("⚙️ Configurações", expanded=True):
            
            use_web = st.toggle(
                "Habilitar busca na internet",
                value=st.session_state.use_web_search,
                help="Ative para permitir que o assistente busque informações atualizadas na internet",
                key="web_search_toggle",  # Adicionando uma key única              
                
            )
            # Atualizar o estado com o valor do toggle
            st.session_state.use_web_search = use_web
            use_juris= False      
                            
            use_juris = st.toggle(
                "Habilitar busca de jurisprudência",
                value=st.session_state.use_juris_search,
                help="Ative para permitir que o assistente busque informações de jurisprudência no site jusbrasil",
                key="juris_search_toggle",  # Adicionando uma key única
                
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
                key=st.session_state.uploader_key  # Usar a chave dinâmica aqui
            ) 
                         
            if uploaded_files:                    
                for uploaded_file in uploaded_files:
                    if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                        try:
                            processed_file = process_uploaded_file(uploaded_file)
                            st.session_state.uploaded_files.append(processed_file)                                
                        except Exception as e:
                            st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
            st.button('remove File', on_click=reset_uploader)  
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
                buscar = st.button("Buscar", on_click=clear_history, key="buscar")
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
                st.button("🗑️ Limpar Tudo", key="limpar_tudo", on_click=clear_history)                  

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

        containerChatbot= st.container(height=600, border=True)
        
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
                    resposta =prompt_formatado #st.session_state.chat_session.send_message(prompt_formatado)    

                with st.chat_message("assistant", avatar="⚖️"):
                    
                    #st.text(f"Resposta \n\n:{resposta.text}") 
                    st.write_stream(stream_data(resposta))                     
                    st.session_state.messages.append({"role": "assistant", "content": f" {resposta}"})
                    st.session_state.chat_history.extend([
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': resposta}
                    ])    
            
    

if __name__ == "__main__":
    main()
