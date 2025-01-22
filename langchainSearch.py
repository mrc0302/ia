import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
import os
from dotenv import load_dotenv

def main():
    # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Configuração da página
    st.set_page_config(page_title="Sistema Legal", layout="wide")

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
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.7
        )

    def gerar_resposta(pergunta, contexto_docs, historico):
        try:
            llm = get_llm()
            
            # Configurar o MultiQueryRetriever
            vector_store = carregar_vector_store()
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=vector_store.as_retriever(),
                llm=llm
            )
            
            # Adicionar contexto do histórico à pergunta
            historico_texto = "\n".join([
                f"Humano: {msg['content'] if msg['role'] == 'user' else ''}\nAssistente: {msg['content'] if msg['role'] == 'assistant' else ''}"
                for msg in historico[-3:]
            ])
            
            question = f"""
            Histórico da conversa:
            {historico_texto}
            
            Pergunta atual: {pergunta}
            """
            
            # Recuperar documentos relevantes
            retrieved_docs = mq_retriever.get_relevant_documents(query=question)
            
            # Adicionar instrução para resposta em português
            question += " responda sempre em português."
            
            # Carregar e executar a chain de QA
            chain = load_qa_chain(llm, chain_type="stuff")
            resposta = chain.run(input_documents=retrieved_docs, question=question)
            
            return str(resposta)
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def busca_combinada(vector_store, query, campo, valor_campo, texto_livre, num_results):
        try:
            # Usar MultiQueryRetriever para busca semântica
            llm = get_llm()
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=vector_store.as_retriever(),
                llm=llm
            )
            
            # Aplicar filtros básicos primeiro
            todos_docs = vector_store.similarity_search("", k=1000)
            resultados_filtrados = []
            
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
                    resultados_filtrados.append(doc)
            
            # Aplicar busca semântica nos documentos filtrados
            if query and resultados_filtrados:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=google_api_key
                )
                temp_store = FAISS.from_documents(resultados_filtrados, embeddings)
                query += " responda sempre em português."
                resultados = mq_retriever.get_relevant_documents(query=query)
                return resultados[:num_results]
            
            return resultados_filtrados[:num_results]
        except Exception as e:
            st.error(f"Erro na busca: {str(e)}")
            return []

    # O resto do código permanece igual...
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

        # Exibir mensagens do chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(f"""
                    <div style="text-align: justify; font-family: Verdana; font-size: 14px;">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("O que você gostaria de perguntar?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"""
                    <div style="text-align: justify; font-family: Verdana; font-size: 14px;">
                        {prompt}
                    
                    """, unsafe_allow_html=True)
            
            with st.spinner("🤔 Pensando..."):
                resposta = gerar_resposta(prompt, st.session_state.documentos_contexto, st.session_state.chat_history)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': resposta}
                ])
                
            with st.chat_message("assistant"):
                st.markdown(f"""
                    <div style="text-align: justify; font-family: Verdana; font-size: 14px;">
                        {resposta}
                    
                    """, unsafe_allow_html=True)

    else:
        st.error("Não foi possível carregar o vector store")

if __name__ == "__main__":
    main()
