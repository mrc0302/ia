import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import time
from googlesearch import search
from serpapi import GoogleSearch

from duckduckgo_search import DDGS
#from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from datetime import datetime

def main():
    # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    genai.configure(api_key=google_api_key)

    # Inicializar DuckDuckGo Search
    #search = DuckDuckGoSearchAPIWrapper()

    # Configuração da página
    st.set_page_config(page_title="Sistema Legal", layout="wide")

    # Inicializar estado da sessão
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'documentos_contexto' not in st.session_state:
        st.session_state.documentos_contexto = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'use_web_search' not in st.session_state:
        st.session_state.use_web_search = False

    def limpar_tudo():
        # Limpar todas as variáveis de estado
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinicializar os estados necessários
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.documentos_contexto = []
        st.session_state.uploaded_files = []
        st.session_state.use_web_search = True
        
        # Forçar rerun da aplicação
        st.rerun()

    def format_text(text):
        text = text.replace('\x0f', '')
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return f"""
        <div style="font-family: 'verdana', serif; 
                    font-size: 16px; 
                    line-height: 2.0;
                    padding: 18px;                    
                    word-wrap: break-word;
                    max-height: 1200px; 
                    overflow-y: auto;">
            {text}
        </div>
        """

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
        elif uploaded_file.type == "text/plain":
            return {
                'name': uploaded_file.name,
                'content': str(uploaded_file.read(), "utf-8"),
                'type': 'txt'
            }
        else:
            raise ValueError("Formato de arquivo não suportado")

    def process_query_intent(query):
        search_keywords = ['buscar', 'procurar', 'pesquisar', 'encontrar', 'localizar']
        query_lower = query.lower()
        is_direct_search = any(query_lower.startswith(keyword) for keyword in search_keywords)
        return is_direct_search, query

    def extrair_campos_unicos(vector_store):
        try:
            resultados = vector_store.similarity_search("", k=1000)
            classes = set()
            assuntos = set()
            
            for doc in resultados:
                metadata = doc.metadata
                if 'classe' in metadata:
                    classes.add(metadata['classe'])
                if 'assunto' in metadata:
                    assuntos.add(metadata['assunto'])
                    
            return sorted(list(classes)), sorted(list(assuntos))
        except Exception as e:
            st.error(f"Erro ao extrair campos: {str(e)}")
            return [], []

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
                    model="models/text-embedding-004",
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


    def direct_vector_store_query(vector_store, query, num_results=5):
        try:
            #results = vector_store.similarity_search(query, k=num_results)
            retriever = vector_store.as_retriever()
            resultados =   retriever.get_relevant_documents(query, k=min(num_results, len(resultados)))
            return results
        except Exception as e:
            st.error(f"Erro na busca direta: {str(e)}")
            return []

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

    def gerar_resposta(pergunta, contexto_docs, historico, uploaded_files):
        try:

            # llm = get_llm()
            
            # # Configurar o MultiQueryRetriever
           
            # mq_retriever = MultiQueryRetriever.from_llm(
            #     retriever=vector_store.as_retriever(),
            #     llm=llm
            # )
            
            # Adicionar contexto do histórico à pergunta
            # historico_texto = "\n".join([
            #     f"Humano: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
            #     for msg in historico[-3:]
            # ])


            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Formatando contexto dos documentos do vector store
            contexto_texto = ""
            if contexto_docs:
                contexto_texto = "\n\n".join([
                    f"Documento do Vector Store {i+1}:\nAssunto: {doc.metadata.get('assunto', 'N/A')}\nConteúdo: {doc.page_content}"
                    for i, doc in enumerate(contexto_docs)
                ])
            
            # Adicionando conteúdo dos arquivos carregados
            arquivos_texto = ""
            if uploaded_files:
                arquivos_texto = "\n\n".join([
                    f"Arquivo Carregado {i+1} - {arquivo['name']}:\n{arquivo['content'][:3000]}"
                    for i, arquivo in enumerate(uploaded_files)
                ])
            
            # Realizar busca na web apenas se o toggle estiver ativado
            web_results = ""
            if st.session_state.get('web_search_toggle', False):
                with st.spinner("Buscando informações na web..."):
                    web_search_results = perform_web_search(pergunta)
                    if web_search_results and not web_search_results.startswith("Erro"):
                        web_results = f"\n\nInformações da Web:\n{web_search_results}"
            
            # Combinando os contextos
            if contexto_texto and arquivos_texto:
                contexto_completo = f"{contexto_texto}\n\n{arquivos_texto}{web_results}"
            elif contexto_texto:
                contexto_completo = f"{contexto_texto}{web_results}"
            elif arquivos_texto:
                contexto_completo = f"{arquivos_texto}{web_results}"
            else:
                contexto_completo = web_results if web_results else ""
            
            # Formatando o histórico
            historico_texto = ""
            if historico:
                historico_texto = "\n".join([
                    f"Usuário: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
                    for msg in historico[-3:]  # Últimas 3 mensagens
                ])

            # Construindo o prompt
            prompt_parts = [
                "Você é um assistente especializado. Use as informações fornecidas para responder à pergunta de forma clara e objetiva.",
                "",
                "Documentos de Referência:" if contexto_completo else "Não foram fornecidos documentos que contenham informações relevantes.",
                contexto_completo if contexto_completo else "",
                "",
                "Histórico da Conversa:" if historico_texto else "Não há histórico de conversas anteriores.",
                historico_texto if historico_texto else "",
                "",
                f"Pergunta Atual: {pergunta}",
                "",
                "Instruções:",
                "1. Use linguagem clara e apropriada ao contexto",
                "2. Se não houver informações suficientes, indique isso claramente",
                "3. Cite trechos relevantes dos documentos quando apropriado",
                "4. Se usar informações da web, cite as fontes com os links correspondentes",
                "5. Mantenha a resposta objetiva e focada na pergunta",
                "6. Se houver informações conflitantes, aponte as diferenças",
                "7. Organize a resposta de forma clara e estruturada",
                "8. Use as informações mais recentes disponíveis",
                "",
                "Resposta:"
            ]
            
            prompt = "\n".join(prompt_parts)
            
            # Gerando a resposta
            resposta = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            # Formatando a resposta
            formatted_response = resposta.text
            
            # Adicionar metadados sobre as fontes usadas
            fontes = []
            if web_results and st.session_state.get('web_search_toggle', False):
                fontes.append("Resultados da busca na web")
            if contexto_docs:
                fontes.append(f"{len(contexto_docs)} documentos do banco de dados")
            if uploaded_files:
                fontes.append(f"{len(uploaded_files)} arquivos carregados")
                
            if fontes:
                formatted_response += "\n\nFontes consultadas:\n" + "\n".join(f"* {fonte}" for fonte in fontes)
            
            return formatted_response

        except Exception as e:
            error_msg = f"Erro ao gerar resposta: {str(e)}\nPor favor, tente reformular sua pergunta ou verificar se há documentos carregados."
            st.error(error_msg)
            return error_msg


    @st.cache_resource
    def carregar_vector_store():
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            vector_store = FAISS.load_local(
                folder_path="faiss_legal_store_gemini",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            st.error(f"Erro ao carregar vector store: {str(e)}")
            return None

    def stream_data(resposta):
        for word in resposta.split(" "):
            yield word + " "
            time.sleep(0.02)

    vector_store = carregar_vector_store()
    if vector_store:
        # Sidebar
        with st.sidebar:            
            with st.expander("⚙️ Configurações", expanded=True):
                use_web = st.toggle(
                    "Habilitar busca na internet",
                    value=st.session_state.use_web_search,
                    help="Ative para permitir que o assistente busque informações atualizadas na internet",
                    key='web_search_toggle'  # Adicionando uma key única
                )
                # Atualizar o estado com o valor do toggle
                st.session_state.use_web_search = use_web
            
            with st.expander(label="🔗 Arraste seus arquivos aqui", expanded=True):
                uploaded_files = st.file_uploader(
                    label="",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                    help="Suporta múltiplos arquivos PDF, DOCX e TXT",
                )
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                            try:
                                processed_file = process_uploaded_file(uploaded_file)
                                st.session_state.uploaded_files.append(processed_file)                                
                            except Exception as e:
                                st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
                
            # Seção de busca no sidebar
            expander_pesq = st.expander("🔍 Filtros de Busca", expanded=True)
            with expander_pesq:
                # Campo de busca semântica
                query = st.text_input(
                    "Busca Semântica:",
                    placeholder="Digite sua consulta semântica",
                    key="semantic_search"
                )
                
                # Campos de filtro
                campo = st.selectbox(
                    "Campo:",
                    options=["", "classe", "assunto"],
                    help="Selecione o campo para filtrar"
                )
                
                # Obter valores para o campo
                classes, assuntos = extrair_campos_unicos(vector_store)
                valores = classes if campo == "classe" else assuntos if campo == "assunto" else []
                
                # Campo valor sempre visível
                valor_campo = st.selectbox(
                    "Valor:",
                    options=[""] + valores,
                    help="Selecione o valor do campo"
                )
                
                # Texto livre
                texto_livre = st.text_input(
                    "Termos de Busca:",
                    placeholder="termo1, termo2, termo3",
                    help="Digite termos separados por vírgula"
                )
                
                # Número de resultados
                num_results = st.slider(
                    "Número de Resultados:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Selecione quantos resultados deseja ver"
                )
                
                # Botões de ação
                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button("🔍 Buscar", use_container_width=True):
                        if not any([query, campo and valor_campo, texto_livre]):
                            st.warning("Especifique pelo menos um critério de busca.")
                        else:
                            with st.spinner("Buscando..."):
                                docs = busca_combinada(
                                    vector_store,
                                    query,
                                    campo,
                                    valor_campo,
                                    texto_livre,
                                    num_results
                                )
                                if docs:
                                    st.session_state.documentos_contexto = docs
                                    st.success(f"Encontrados {len(docs)} documentos!")
                                else:
                                    st.warning("Nenhum documento encontrado.")
                                    st.session_state.documentos_contexto = []
                
                with col2:
                    if st.button("🗑️ Limpar", use_container_width=True):
                        st.session_state.documentos_contexto = []
                        st.rerun()

            # Resultados em um expander separado no sidebar
            if st.session_state.documentos_contexto:
                docs = st.session_state.documentos_contexto 
                with st.expander(f"📚 Total de Documentos Encontrados {len(docs)}", expanded=True):
                                        
                    # Criar lista de títulos para o selectbox
                    doc_titles = [f"Documento {i+1} - {doc.metadata.get('assunto', 'Sem assunto')}" 
                                for i, doc in enumerate(st.session_state.documentos_contexto)]
                    
                    # Selectbox para escolher o documento
                    selected_doc = st.selectbox(
                        "Selecione um documento para visualizar:",
                        doc_titles,
                        key="doc_selector"
                    )
                    
                    # Exibir conteúdo do documento selecionado
                    if selected_doc:
                        doc_index = doc_titles.index(selected_doc)
                        doc = st.session_state.documentos_contexto[doc_index]
                        st.text_area(
                            "Conteúdo:",
                            doc.page_content,
                            height=300,
                            key=f"text_{doc_index}"
                        )

        # Área principal - Chat
        st.markdown("#### 💬 Chat Assistente Jurídico")
        
        # Exibir mensagens do chat
        for message in st.session_state.messages:
            with st.chat_message("user" if message["role"] == "user" else "assistant", 
                               avatar="👨" if message["role"] == "user" else "⚖️"):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Chat input e processamento
        if prompt := st.chat_input("O que você gostaria de perguntar?"):
            st.session_state.messages.append({"role": "user", "content": prompt})            
            
            with st.chat_message("user", avatar="👨‍⚖️"):
                st.markdown(f""" <div style="text-align: justify;"> <strong>{prompt}</strong> </div>""", 
                          unsafe_allow_html=True)                      
            
            is_direct_search, query = process_query_intent(prompt)
            
            with st.spinner("🤔 Analisando documentos..."):
                if is_direct_search:
                    results = direct_vector_store_query(vector_store, query)
                    st.session_state.documentos_contexto = results
                    
                    if results:
                        response_text = "Encontrei os seguintes documentos:\n\n"
                        for i, doc in enumerate(results, 1):
                            response_text += f"{i}. Assunto: {doc.metadata.get('assunto', 'N/A')}\n"
                            response_text += f"Conteúdo: {doc.page_content[:200]}...\n\n"
                    else:
                        response_text = "Não encontrei documentos relacionados à sua busca."
                else:
                    response_text = gerar_resposta(
                        prompt,
                        st.session_state.documentos_contexto,
                        st.session_state.chat_history,
                        st.session_state.uploaded_files
                    )

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response_text}
                ])

                with st.chat_message("assistant", avatar="⚖️"):
                    st.write_stream(stream_data(response_text))

if __name__ == "__main__":
    main()
