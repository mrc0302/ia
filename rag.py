import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import time
from langchain.chains.question_answering import load_qa_chain # melhor implementação do que similaridade
from langchain.retrievers.multi_query import MultiQueryRetriever




def main(): 
 

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    load_dotenv()
    google_api_key = os.getenv("google_api_key")

    embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

    vector_store= FAISS.load_local(
                "faiss_legal_store_gemini",                
                embeddings,
                allow_dangerous_deserialization=True
            )
    llm= GoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.7
                )
    # Função para limpar o chat, histórico e resultados
    def limpar_tudo():
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.documentos_contexto = []
        st.rerun()

    # Inicializa o estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'documentos_contexto' not in st.session_state:
        st.session_state.documentos_contexto = []

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def stream_data(resposta):
        for word in resposta.split(" "):
            yield word + " "
            time.sleep(0.02)

    def gerar_resposta(question):
        try:

            
            #retriever = vector_store.as_retriever() # melhor implementação

            # retrieved_docs = retriever.get_relevant_documents(question)

            # no caso, ele gera automaticamente subconsultas
            mq_retriever = MultiQueryRetriever.from_llm(retriever = vector_store.as_retriever(), llm = llm)                       
                
            retrieved_docs = mq_retriever.get_relevant_documents(query=question)
            
            question +=f" responda sempre em português . "              

            chain = load_qa_chain(llm, chain_type="stuff")
            resposta = chain.run(input_documents=retrieved_docs, question=question)

            return str(resposta)
            
        except Exception as e:
            
            return f"Erro ao gerar resposta: {str(e)}"

    # Interface principal do chat
    st.markdown("#### 💬 Chat Assistente Jurídico")


    # Input do chat
    #if prompt := st.chat_input("O que você gostaria de perguntar?"):

    if prompt := st.text_input("O que você gostaria de perguntar?", value="", ):    
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        #with st.chat_message("user"):
        
        
    with st.container(height=700, border=True):
        
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                # Gera e exibe a resposta
        
        if prompt:
                        
            with st.spinner("🤔 Pensando..."):
                resposta = gerar_resposta(prompt)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': resposta}
                ])

            st.write_stream(stream_data(resposta))
            placeholder = st.empty()        


if __name__ == "__main__":
    main()
