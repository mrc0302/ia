import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from openai import OpenAI
import os
from dotenv import load_dotenv

def main():
    # Carregar variáveis de ambiente
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Configurar as APIs
    genai.configure(api_key=GOOGLE_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Configuração da página
    st.set_page_config(page_title="Sistema Legal", layout="wide")

    # Inicializar o mecanismo de busca web
    search = DuckDuckGoSearchRun()

    # Inicializar estados da sessão
    if "messages_base_dados" not in st.session_state:
        st.session_state.messages_base_dados = []
    
    if "messages_internet" not in st.session_state:
        st.session_state.messages_internet = []
    
    if "messages_vector" not in st.session_state:
        st.session_state.messages_vector = []

    if "chat_history_base_dados" not in st.session_state:
        st.session_state.chat_history_base_dados = []
    
    if "chat_history_internet" not in st.session_state:
        st.session_state.chat_history_internet = []
    
    if "chat_history_vector" not in st.session_state:
        st.session_state.chat_history_vector = []

    if 'documentos_contexto' not in st.session_state:
        st.session_state.documentos_contexto = []

    if 'modo_operacao' not in st.session_state:
        st.session_state.modo_operacao = 'base_dados'
    
    if 'ultimo_modo' not in st.session_state:
        st.session_state.ultimo_modo = 'base_dados'

    if 'modelo_ai' not in st.session_state:
        st.session_state.modelo_ai = 'gemini'

    def get_current_messages():
        if st.session_state.modo_operacao == 'base_dados':
            return st.session_state.messages_base_dados
        elif st.session_state.modo_operacao == 'internet':
            return st.session_state.messages_internet
        else:
            return st.session_state.messages_vector

    def get_current_history():
        if st.session_state.modo_operacao == 'base_dados':
            return st.session_state.chat_history_base_dados
        elif st.session_state.modo_operacao == 'internet':
            return st.session_state.chat_history_internet
        else:
            return st.session_state.chat_history_vector

    def limpar_tudo():
        st.session_state.messages_base_dados = []
        st.session_state.messages_internet = []
        st.session_state.messages_vector = []
        st.session_state.chat_history_base_dados = []
        st.session_state.chat_history_internet = []
        st.session_state.chat_history_vector = []
        st.session_state.documentos_contexto = []
        st.rerun()

    @st.cache_resource
    def carregar_vector_store():
        try:
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local(
                "faiss_legal_store_gemini", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Erro ao carregar vector store: {str(e)}")
            return None

    def realizar_busca_web(query, historico=None):
        try:
            # Criar um contexto mais rico incluindo o histórico
            contexto_query = query
            if historico:
                # Pegar as últimas 3 interações para contexto
                ultimas_interacoes = historico[-6:]  # 3 pares de pergunta/resposta
                contexto_historico = "\n".join([
                    f"Pergunta anterior: {msg['content']}" 
                    for msg in ultimas_interacoes if msg['role'] == 'user'
                ])
                if contexto_historico:
                    contexto_query = f"{contexto_historico}\nPergunta atual: {query}"

            # Realizar a busca com o contexto enriquecido
            search = DuckDuckGoSearchRun()
            resultados = search.run(contexto_query)
            return resultados
        except Exception as e:
            return f"Erro na busca web: {str(e)}"

    def consultar_vector_direto(query, vector_store):
        try:
            resultados = vector_store.similarity_search_with_score(query, k=3)
            resposta = "Resultados encontrados:\n\n"
            for doc, score in resultados:
                resposta += f"📄 Documento (Relevância: {(1-score)*100:.1f}%)\n"
                resposta += f"Assunto: {doc.metadata.get('assunto', 'Não especificado')}\n"
                resposta += f"Classe: {doc.metadata.get('classe', 'Não especificada')}\n"
                resposta += f"Conteúdo: {doc.page_content}\n\n"
            return resposta
        except Exception as e:
            return f"Erro na consulta direta: {str(e)}"

    def busca_combinada(vector_store, query, campo, valor_campo, texto_livre, num_results):
        try:
            if not any([query, campo and valor_campo, texto_livre]):
                return []
                
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
                embeddings = OpenAIEmbeddings()
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

    def gerar_resposta_openai(prompt):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "system", "content": "Você é um assistente jurídico especializado."},
                         {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro ao gerar resposta com OpenAI: {str(e)}"


    def gerar_resposta_gemini(prompt):
        try:
            # Configuração específica para o modelo experimental
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            # Criar o modelo com a configuração específica
            model = genai.GenerativeModel(
                model_name="learnlm-1.5-pro-experimental",
                generation_config=generation_config,
            )

            # Iniciar sessão de chat
            chat_session = model.start_chat(history=[])
            
            # Enviar mensagem e obter resposta
            response = chat_session.send_message(prompt)
            
            return response.text
        except Exception as e:
            return f"Erro ao gerar resposta com Gemini: {str(e)}"
    
    def gerar_resposta(pergunta, contexto_docs, historico, modo_operacao):
        try:
            # Preparar o contexto baseado no modo de operação
            if modo_operacao == 'base_dados':
                if not contexto_docs:
                    return "Por favor, realize uma busca primeiro para fornecer contexto à sua pergunta."
                
                contexto = ""
                for i, doc in enumerate(contexto_docs):
                    contexto += f"\nDocumento {i+1}:\n"
                    contexto += f"Assunto: {doc.metadata.get('assunto', 'Não especificado')}\n"
                    contexto += f"Classe: {doc.metadata.get('classe', 'Não especificada')}\n"
                    contexto += f"Conteúdo: {doc.page_content}\n"
            
            elif modo_operacao == 'internet':
                contexto = realizar_busca_web(pergunta)
            
            elif modo_operacao == 'vector_direto':
                return consultar_vector_direto(pergunta, vector_store)

            # Criar o prompt completo
            prompt = f"""Você é um assistente jurídico especializado.
            {'Informações da web:' if modo_operacao == 'internet' else 'Documentos disponíveis:'}
            {contexto}
            Pergunta: {pergunta}
            """

            if historico:
                relevant_history = historico[-6:]
                prompt += "\nHistórico da conversa:\n"
                for msg in relevant_history:
                    prompt += f"\n{msg['role']}: {msg['content']}"

            # Gerar resposta com o modelo selecionado
            if st.session_state.modelo_ai == 'openai':
                return gerar_resposta_openai(prompt)
            else:
                return gerar_resposta_gemini(prompt)
            
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {str(e)}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."

    vector_store = carregar_vector_store()
    if vector_store:
        # Sidebar
        with st.sidebar:
            # Seletor de modelo AI
            st.radio(
                "Modelo de IA:",
                ['gemini', 'openai'],
                key='modelo_ai',
                format_func=lambda x: {
                    'gemini': '🤖 Google Gemini',
                    'openai': '🔮 OpenAI GPT-4'
                }[x]
            )
            
            # Seletor de modo de operação
            modo_anterior = st.session_state.modo_operacao
            st.radio(
                "Modo de Operação:",
                ['base_dados', 'internet', 'vector_direto'],
                key='modo_operacao',
                format_func=lambda x: {
                    'base_dados': '📚 Base de Dados',
                    'internet': '🌐 Internet',
                    'vector_direto': '🔍 Consulta Direta'
                }[x]
            )
            
            # Verificar mudança de modo
            if modo_anterior != st.session_state.modo_operacao:
                st.session_state.documentos_contexto = []
                st.rerun()

            # Mostrar filtros apenas no modo base_dados
            if st.session_state.modo_operacao == 'base_dados':
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

            # Documentos no sidebar (apenas no modo base_dados)
            if st.session_state.modo_operacao == 'base_dados' and st.session_state.documentos_contexto:
                with st.expander("📚 Documentos Encontrados", expanded=False):
                    st.markdown(f"Total: {len(st.session_state.documentos_contexto)}")
                    # Documentos no sidebar (apenas no modo base_dados)
            if st.session_state.modo_operacao == 'base_dados' and st.session_state.documentos_contexto:
                with st.expander("📚 Documentos Encontrados", expanded=False):
                    st.markdown(f"Total: {len(st.session_state.documentos_contexto)}")
                    doc_titles = [f"Assunto: {doc.metadata.get('assunto', 'Sem assunto')}" 
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
        
        # Mostrar modo e modelo atual
        modo_texto = {
            'base_dados': '📚 Modo: Base de Dados',
            'internet': '🌐 Modo: Busca na Internet',
            'vector_direto': '🔍 Modo: Consulta Direta ao Vector'
        }[st.session_state.modo_operacao]
        
        modelo_texto = {
            'gemini': '🤖 Modelo: Google Gemini',
            'openai': '🔮 Modelo: OpenAI GPT-4'
        }[st.session_state.modelo_ai]
        
        st.info(f"{modo_texto} | {modelo_texto}")

        # Exibir mensagens do chat atual
        current_messages = get_current_messages()
        for message in current_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("O que você gostaria de perguntar?"):
            current_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("🤔 Pensando..."):
                current_history = get_current_history()
                resposta = gerar_resposta(
                    prompt, 
                    st.session_state.documentos_contexto, 
                    current_history,
                    st.session_state.modo_operacao
                )
                
                current_messages.append({"role": "assistant", "content": resposta})
                current_history.extend([
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': resposta}
                ])
                
            with st.chat_message("assistant"):
                st.markdown(resposta)

            # Atualizar o histórico apropriado
            if st.session_state.modo_operacao == 'base_dados':
                st.session_state.messages_base_dados = current_messages
                st.session_state.chat_history_base_dados = current_history
            elif st.session_state.modo_operacao == 'internet':
                st.session_state.messages_internet = current_messages
                st.session_state.chat_history_internet = current_history
            else:
                st.session_state.messages_vector = current_messages
                st.session_state.chat_history_vector = current_history

    else:
        st.error("Não foi possível carregar o vector store")

if __name__ == "__main__":
    main()
