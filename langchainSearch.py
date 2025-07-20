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
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
import json
import pandas as pd
from io import StringIO
from langchain.docstore.document import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
           

    # Configuração da página
    st.set_page_config(
        page_title="Sistema de Modelos Judiciais",
        page_icon="🧊", 
        layout="wide",  
        initial_sidebar_state="expanded"
    )
    
    def load_css(file_name):
        with open(file_name, encoding='utf-8') as f:  # Adicione encoding='utf-8'
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    load_css("static/styles.css")  

        # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    #-------------------------------------------------------------------------------
    genai.configure(api_key=os.getenv("google_api_key"))  

    # Modelos disponíveis
    models = [
        "gemini-pro",                    # Modelo padrão
        "gemini-pro-vision",             # Com suporte a imagens
        "gemini-1.5-pro",                # Versão mais recente
        "gemini-1.5-flash",              # Mais rápido
        "learnlm-2.0-flash-experimental", # Seu modelo atual
        "gemini-1.5-pro-002",            # Versões específicas
        "gemini-1.5-flash-002"
    ]
    # Tipos MIME suportados
    mime_types = [
        "text/plain",     # Texto simples (padrão)
        "text/html",      # HTML
        "application/json", # JSON
        "text/markdown"   # Markdown
    ]
 
    
    def get_model():  # retorna o modelo para pergunta, diferente do embedding   
        generation_config = {
           "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
       
        model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config=generation_config
       )
        return model
       
        #-------------------------------------------------------------------------------
    llm =  ChatGoogleGenerativeAI(
            model="learnlm-2.0-flash-experimental",
            temperature=0.3,
            max_tokens=8192,
            top_p=0.8,
            top_k=40,
            mime_types = "text/markdown", 
            convert_system_message_to_human=True  # Importante para Gemini
        )
  
    def get_mq_retriever(vector_store, llm):
                
        # Retriever básico
        basic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # MultiQueryRetriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=basic_retriever,
            llm=llm
        )

        return multi_query_retriever
    

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
        init_session()

        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.documentos_contexto = []
        st.session_state.uploaded_files = []
        st.session_state.use_web_search = False
        st.session_state.documentos_contexto = []
        
        # CORREÇÃO: Em vez de definir como lista vazia, reinicializar o chat_session corretamente
        model = get_model()   
        st.session_state.chat_session = model.start_chat(history=[])

    def limpar_apenas_arquivos():
        """Limpa apenas os arquivos carregados, mantendo o chat"""
        st.session_state.uploaded_files = []

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

    def split_documents(documents):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # tamanho do chunk
            chunk_overlap=200,  # sobreposição
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    def create_faiss_vectorstore(chunks):
        # Usar embeddings HuggingFace (gratuito)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        return vectorstore

    def init_session():
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
     
    init_session()

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
    
    def create_vector_store(documents, vector_db_path):
        """Create and save vector store"""
        try:
            os.makedirs(vector_db_path, exist_ok=True)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
            )
            
            st.write("Dividindo documentos...")
            chunks = text_splitter.split_documents(documents)
            st.write(f"Total de chunks: {len(chunks)}")
            
            st.write("Criando vector store...")
            vector_store = FAISS.from_documents(chunks, get_embeddings)
                      
                
            return vector_store
            
        except Exception as e:
            st.error(f"Erro ao criar vector store: {str(e)}")
            
            return None


#     def gerar_resposta(query, llm, vector_store, contexto_docs, historico, uploaded_files, campo, valor_campo):
        
#         try:
           
#             # #-----------------------------------------
#             # prompt = ChatPromptTemplate.from_template(
#             # """ responda sempre em português de forma objetiva, com base nos
#             #     <contexto> {context}</contexto>.
#             #      Forença a resposta com formatação em markdown, FONTE 15 E TEXTO JUSTIFICADO 
                 
#             # Pergunta: {input}
#             # """)

#  #-----------------------------------------
#             prompt_template = """ responda sempre em português de forma objetiva, com base nos
#                 <contexto> {context}</contexto>.
#                  Forença a resposta com formatação em markdown, FONTE 15 E TEXTO JUSTIFICADO 
#                  Pergunta: {input}
#             """
#             prompt  = ChatPromptTemplate.from_template(
#                 template=prompt_template,
                
#             )

#             # Criar chains
#             #documents_chain = create_stuff_documents_chain(llm, prompt)
#              # Chain completa
#             ##retrieval_chain = create_retrieval_chain(retriever, documents_chain)
            
#             #-----------------------------------------
        
#             # # "stuff" - combina todos os docs em um prompt
#             #qa_chain = load_qa_chain(llm, chain_type="stuff")
           

#             # # "map_reduce" - processa docs separadamente, depois combina
#             # qa_chain = load_qa_chain(llm, chain_type="map_reduce")

#             # # "refine" - refina a resposta iterativamente
#             # qa_chain = load_qa_chain(llm, chain_type="refine")

#             # # "map_rerank" - rankeia respostas de cada doc
#             # qa_chain = load_qa_chain(llm, chain_type="map_rerank")
            
#             contexto_completo=" "            
            
#             retrieved_docs = []

#             arquivos_texto = [] # CRIA UMA LISTA PARA ARMAZENAR O TEXTO FORMATADO            

#             if contexto_docs:
               
#                 retrieved_docs = contexto_docs # pega os documentos encontrados na busca

                          
#                 for doc in retrieved_docs:
#                     # Formata o texto com metadados
#                     metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
#                     arquivos_texto.append(f"{metadata_str}\n{doc.page_content}")
                
                
#             elif uploaded_files:  # Se houver uma consulta de texto livre
               
#                for doc in uploaded_files:
#                     # Formata o texto com metadados
#                     metadata_str = f"[Nome: {doc['name']}, Tipo: {doc['type']}]"
                   
#                     arquivos_texto.append(f"{metadata_str}\n{doc['content']}") # adiciona o conteúdo do arquivo à lista
                                     
#             elif campo == "classe" and valor_campo:  # Se a busca na web estiver habilitada
                
#                 search_kwargs={
#                     "k": 10,           # Retorna 4 documentos
#                     "fetch_k": 100,    # Busca 10 candidatos iniciais
#                     "lambda_mult": 0.5 , # > 0.5 estou mais focado na similaridade 0.5 < esto mais focado na diversidade 
#                     "score_threshold": 0.4,  # Threshold similaridade - filtra documentos com pontuação abaixo de 0.4
#                     "filter": {campo: valor_campo}  # Filtra por classe  
#                 }                
#                 retriever = vector_store.as_retriever(search_type="mmr", search_kwargs= search_kwargs) # search_type= "similarity"
#                 retrieved_docs = retriever.invoke(query) # para gerar subconsultas
#                 for doc in retrieved_docs:
#                     # Formata o texto com metadados
#                     metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
#                     arquivos_texto.append(f"{metadata_str}\n{doc.page_content}")

#             else:
               
#                 retriever = vector_store.as_retriever()                
#                 retrieved_docs = retriever.invoke(query) # para gerar subconsultas

#                 for doc in retrieved_docs:
#                     # Formata o texto com metadados
#                     metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
#                     arquivos_texto.append(f"{metadata_str}\n{doc.page_content}")

#                 # for doc in retrieved_docs:
#                 #     st.write(f"Conteúdo: {doc.page_content}")# só conteúdo do campo texto
#                 #     st.write(f"Metadados: {doc.metadata}") # todos os metadados do documento        
             
#             contexto_completo = "\n\n".join(arquivos_texto)

            
            
#             # # Configurar retriever
#             # retriever = vector_store.as_retriever(
#             #     search_type="similarity",
#             #     search_kwargs={"k": 20}
#             # )
            
            
#             qa_chain = load_qa_chain(
#                llm=llm,
#                chain_type="stuff",
#                prompt=prompt
#             )
            
#             response = qa_chain.run(input_documents=retrieved_docs, question=query)
            
#             # # Executar
#             # response = retrieval_chain.invoke({
#             #             "input": query,                        
#             #             "context": contexto_completo,
#             #             # "histórico": historico,
#             #             # "uploaded_files": uploaded_files,   
                           
#             #         })

#             #st.write(f"Resposta gerada: {response['answer']}")
#         #--------------------------------

#          #st.write(contexto_completo)
#          #   query += f" responda sempre em português de forma objetiva, com base exclusivamente no contexto. <contexto>{contexto_completo}</contexto> . forneça a resposta com formatação em markdown, FONTE 18 E TEXTO JUSTIFICADO "
                       
#             #return query            
#             return response['answer']  # Retorna a resposta gerada pelo modelo
        
#         except Exception as e:
#             return f"Erro ao gerar resposta: {str(e)}"
    def gerar_resposta(query, llm, vector_store, contexto_docs, historico, uploaded_files, campo, valor_campo):
    
        try:
            # a variável context deve sempre estar presente. senão dá erro
            prompt_template = """Você é um assistente especialista em direito. 
                                 Responda as perguntas do usuário sempre em português de forma clara.
                                 as pergutas do usuário poderão se  exclusivamente com base no contexto fornecido ou não.
            
            <contexto>{context}{contexto}{metadados}{historico}</contexto>
            
            Pergunta: {question}
            """
            
            # OPÇÃO 1: Forma simples (recomendada)
            prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question","contexto","metadados"]
            )
                       
            contexto_completo = " "            
            retrieved_docs = []
            arquivos_texto = []
            metadados = []  # Lista para armazenar metadados

            if contexto_docs: # são os documentos da consultas
                
                retrieved_docs = contexto_docs
                
                for doc in retrieved_docs:
                    metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
                    arquivos_texto.append(f"{metadata_str}\n{doc.page_content}") # além dos metadados inclui o conteúdo dos documentos
                    metadados.append(doc.metadata)  # Adiciona metadados à lista de metadados com as informações dos documentos
                     
            elif uploaded_files:    
                     
                from langchain.schema import Document

                for doc in uploaded_files:

                    doc_obj = Document(
                        page_content=doc['content'], 
                        metadata={'name': doc['name'], 'type': doc['type']}
                    )
                    retrieved_docs.append(doc_obj) # o retrieved_docs será utilizado na resposta como parâmetro

                    for doc in retrieved_docs:
                        metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
                        arquivos_texto.append(f"{metadata_str}\n{doc.page_content}") # será utilizado no contexto não se é o melhor
                        metadados.append(doc.metadata)  # Adiciona metadados à lista                                                                           
           
            elif campo == "classe" and valor_campo:
                search_kwargs = {
                    "k": 10,
                    "fetch_k": 100,
                    "lambda_mult": 0.5,
                    "score_threshold": 0.4,
                    "filter": {campo: valor_campo}
                }                
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
                    arquivos_texto.append(f"{metadata_str}\n{doc.page_content}")
                    metadados.append(doc.metadata)  # Adiciona metadados à lista    

            else:
                retriever = vector_store.as_retriever()                
                retrieved_docs = retriever.invoke(query)
                for doc in retrieved_docs:
                    metadata_str = f"[Classe: {doc.metadata.get('classe', 'N/A')}, Assunto: {doc.metadata.get('assunto', 'N/A')}]"
                    arquivos_texto.append(f"{metadata_str}\n{doc.page_content}")
                    metadados.append(doc.metadata)  # Adiciona metadados à lista    
            
            contexto_completo = "\n\n".join(arquivos_texto)
           
            qa_chain = load_qa_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            
            # CORREÇÃO: usar 'question' em vez de 'query' para consistency com o prompt
            response = qa_chain.run(input_documents= retrieved_docs, question=query, contexto = contexto_completo,
                                     historico= historico,  metadados=metadados)  # Passar contexto e metadados
            
            return response
            
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"
        
   
    def busca_combinada(vector_store, campo, valor_campo, texto_livre, num_results):
        
        try:

            resultados = []

            total_vectors = vector_store.index.ntotal #  # Total de registros no vector store
             # Primeiro busca pelo texto livre se não houver filtro de metadados selecionado
            todos_docs = vector_store.similarity_search("", k=total_vectors)
            
            if not (campo =="classe" or campo =="assunto") and texto_livre:               
                              
                termo_lower = texto_livre.lower() # termo de busca em minúsculas
                
                for doc in todos_docs:      # doc é um objeto Document com page_content e metadata
                    
                    valor_campo = doc.metadata.get("texto", "").lower() # Se "texto" existe: retorna o valor
                                                                        # Se "texto" NÃO existe: retorna "" (string vazia)
                    
                    # Aqui está a "busca parcial" - equivalente ao LIKE '%termo%'
                    if termo_lower in valor_campo:
                        resultados.append(doc) #adiciono a lista de resultados com metadados
                         
                        if len(resultados) >= num_results:
                            break                   
            
            elif campo =="classe" and  valor_campo and texto_livre:  # Se houver filtro de metadados
                
                termo_lower = texto_livre.lower() # termo de busca em minúsculas
                
                for doc in todos_docs:      # doc é um objeto Document com page_content e metadata
                    
                    classe_content = doc.metadata.get("classe", "") 

                    if classe_content == valor_campo and termo_lower in doc.page_content.lower():# se a classe do documento é igual ao valor do campo 
                                                                                            # e o termo de busca está no conteúdo   

                        resultados.append(doc) #adiciono a lista de resultados com metadados
                        
                        if len(resultados) >= num_results:
                            break    
         
            elif campo =="assunto" and  valor_campo and texto_livre:  # Se houver filtro de metadados
                
                termo_lower = texto_livre.lower() # termo de busca em minúsculas
                
                for doc in todos_docs:      # doc é um objeto Document com page_content e metadata
                    
                    assunto_content = doc.metadata.get("assunto", "") 

                    if assunto_content == valor_campo and termo_lower in doc.page_content.lower():# se a assunto do documento é igual ao valor do campo 
                                                                                            # e o termo de busca está no conteúdo   

                        resultados.append(doc) #adiciono a lista de resultados com metadados
                        
                        if len(resultados) >= num_results:
                            break         

            elif campo  and  valor_campo and not texto_livre:  # Se houver filtro de metadados
                
                for doc in todos_docs:
                    classe_content = doc.metadata.get("classe", "") 

                    if classe_content == valor_campo:
                        resultados.append(doc) #adiciono a lista de resultados com metadados            

            st.write(f"Total de documentos encontrados: {len(resultados)}")
            return resultados[:num_results]  # retorna apenas os primeiros num_results documentos
            
        except Exception as e:
            st.error(f"Erro na busca: {str(e)}")
            print(f"Erro detalhado: {str(e)}")  # Para debug
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

   
    if vector_store:
        # Sidebar
        colSideBAr, colChatbot = st.columns([1, 20])
        #with colSideBAr:
        with st.sidebar:            
            tab1, tab2, tab3 = st.tabs(["Pesquisa", "Arquivos","Configurações"])
            with tab1:                 
                expander_pesq = st.expander("🔍 Filtros de Busca", expanded=True)                
                with expander_pesq:            
                    query = texto_livre = st.text_input(
                        "Texto Livre:", placeholder="Ex: termo1, termo2"
                    )
                
                    campo = st.selectbox(
                        "Campo:", [" ", "classe", "assunto"], help="Filtrar por campo"
                    )
                    valores = [] 
                    classes, assuntos = extrair_campos_unicos(vector_store)               
                
                    if campo:                    
                        valores = classes if campo == "classe" else assuntos
                    
                    valor_campo = st.selectbox(
                        "Valor do Campo:", [" "] + valores, help="Selecione o valor do campo"                                              
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
                                        vector_store, campo, valor_campo, texto_livre, num_results
                                    )
                                    st.session_state.documentos_contexto = docs # armazena os documentos encontrados DA BUSCA no estado da sessão
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
            with tab2:
                  with st.expander(label="🔗 Arraste seus arquivos aqui", expanded=True):
                    uploaded_files = st.file_uploader(
                        label="",
                        type=["pdf", "docx", "txt","csv"],
                        accept_multiple_files=True,
                        help="Suporta múltiplos arquivos PDF, DOCX,TXT,csv",
                    )
                    
                    # Sincronizar com o estado da sessão
                    if uploaded_files:
                        # Obter nomes dos arquivos atualmente no uploader
                        current_file_names = [f.name for f in uploaded_files]
                        
                        # Remover do estado arquivos que não estão mais no uploader
                        st.session_state.uploaded_files = [
                            f for f in st.session_state.uploaded_files 
                            if f['name'] in current_file_names
                        ]
                        
                        # Adicionar novos arquivos
                        for uploaded_file in uploaded_files:
                            if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                                try:
                                    processed_file = process_uploaded_file(uploaded_file)
                                    st.session_state.uploaded_files.append(processed_file)                                
                                except Exception as e:
                                    st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
                    else:
                        # Se não há arquivos no uploader, limpar o estado também
                        st.session_state.uploaded_files = []
               
            with tab3:
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
                
           
        with colChatbot:
            # # Área principal do chat
            # st.markdown("""<center><h2> 💬 Chat Assistente Jurídico</h2><center>""", unsafe_allow_html= True)

            containerChatbot= st.container(height=800, border=True)
            
            with containerChatbot:

                # Exibir mensagens do chat
                for message in st.session_state.messages:
                    with st.chat_message("user" if message["role"] == "user" else "assistant", 
                                        avatar="👨" if message["role"] == "user" else "⚖️"):                     
                    
                        st.markdown(f"""{message["content"]}""", unsafe_allow_html=True)
                
                            # Chat input
            if prompt := st.chat_input("O que você gostaria de perguntar?"):
                st.session_state.messages.append({"role": "user", "content": prompt})                
                
                with containerChatbot:
                    with st.chat_message("user", avatar="👨"):
                        st.markdown(prompt)  
                    
                    with st.spinner("🤔 Pensando..."):                        
                        if campo:
                            filterName = campo
                            filterValue = valor_campo
                        else:
                            filterName = None
                            filterValue = None  
                    
                        prompt_formatado =  gerar_resposta(
                                                    prompt, 
                                                    llm,
                                                    vector_store,
                                                    st.session_state.documentos_contexto,
                                                    st.session_state.chat_history, 
                                                    st.session_state.uploaded_files 
                                                    , filterName,
                                                    filterValue                                                 
                                                    )         
                        #resposta = st.session_state.chat_session.send_message(prompt_formatado) #, history=st.session_state.chat_history  
                        resposta = prompt_formatado # Resposta formatada
                    with st.chat_message("assistant", avatar="⚖️"):
                        
                        #st.text(f"Resposta \n\n:{resposta.text}") 
                        #st.write_stream(stream_data(resposta.text))
                        #st.markdown(f"Resposta \n\n:{resposta.text}", unsafe_allow_html=True)
                        st.markdown(f"""{resposta}""", unsafe_allow_html=True)

                        st.session_state.messages.append({"role": "assistant", "content": f" {resposta}"})
                        st.session_state.chat_history.extend([
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': resposta}
                        ])    
            
                
                
    else:
        st.error("Não foi possível carregar o vector store")


if __name__ == "__main__":
    main()
