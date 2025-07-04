import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.schema import Document
from typing import List
import io
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
#from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

import os
import time
import shutil
import zipfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def extract_text_from_csv(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()
    return text 

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Implementar processamento em lotes para evitar erros de rate limit
    batch_size = 20
    all_chunks = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        try:
            if i > 0:
                # Esperar um pouco entre lotes para respeitar os limites de taxa
                time.sleep(2)
            all_chunks.extend(batch)
        except Exception as e:
            st.error(f"Erro ao processar lote {i//batch_size + 1}: {str(e)}")
            time.sleep(10)  # Espera mais tempo se atingiu o limite
    
    # Criar embeddings para todos os chunks processados
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
    
    # Garantir que o diretório exista
    os.makedirs("faiss_index", exist_ok=True)
    
    # Salvar o vetor localmente
    vector_store.save_local("faiss_index")
    
    # Verificar se o arquivo foi criado corretamente
    if os.path.exists(os.path.join("faiss_index", "index.faiss")):
        # Também salvar como arquivo zip para download
        create_vector_store_zip()
        return vector_store
    else:
        st.error("Erro ao salvar o vetor FAISS. O arquivo index.faiss não foi criado.")
        return None

def create_vector_store_zip():
    """Cria um arquivo zip do diretório faiss_index para download"""
    try:
        # Verificar se o diretório existe
        if not os.path.exists("faiss_index"):
            st.error("Diretório faiss_index não encontrado!")
            return False
            
        # Verificar se os arquivos necessários existem
        if not (os.path.exists(os.path.join("faiss_index", "index.faiss")) and 
                os.path.exists(os.path.join("faiss_index", "index.pkl"))):
            st.error("Arquivos de índice FAISS não encontrados!")
            return False
        
        # Criar o zip
        shutil.make_archive("faiss_index_zip", 'zip', "faiss_index")
        return True
    except Exception as e:
        st.error(f"Erro ao criar arquivo zip: {str(e)}")
        return False

def load_uploaded_vector_store(uploaded_file):
    """Carrega um vetor salvo anteriormente do arquivo zip enviado pelo usuário"""
    try:
        # Limpar diretório existente se houver
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        
        # Salvar o arquivo enviado temporariamente
        with open("temp_vector.zip", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Criar diretório para extração
        os.makedirs("faiss_index", exist_ok=True)
        
        # Extrair o zip e examinar seu conteúdo
        with zipfile.ZipFile("temp_vector.zip", 'r') as zip_ref:
            # Listar todos os arquivos no ZIP para debug
            all_files = zip_ref.namelist()
            st.write(f"Arquivos no ZIP: {all_files}")
            
            # Procurar os arquivos necessários em qualquer localização no ZIP
            faiss_file = None
            pkl_file = None
            
            for file in all_files:
                if file.endswith(".faiss"):
                    faiss_file = file
                elif file.endswith(".pkl"):
                    pkl_file = file
            
            if faiss_file and pkl_file:
                # Extrair todos os arquivos
                zip_ref.extractall("temp_extract")
                
                # Copiar os arquivos específicos para o diretório faiss_index
                shutil.copy(os.path.join("temp_extract", faiss_file), 
                           os.path.join("faiss_index", "index.faiss"))
                shutil.copy(os.path.join("temp_extract", pkl_file), 
                           os.path.join("faiss_index", "index.pkl"))
                
                # Limpar diretório temporário
                if os.path.exists("temp_extract"):
                    shutil.rmtree("temp_extract")
            else:
                # Tentar extrair diretamente se os arquivos tiverem os nomes esperados
                zip_ref.extractall("faiss_index")
        
        # Limpar arquivo temporário
        if os.path.exists("temp_vector.zip"):
            os.remove("temp_vector.zip")
        
        # Verificar se os arquivos foram extraídos corretamente
        if not (os.path.exists(os.path.join("faiss_index", "index.faiss")) and 
                os.path.exists(os.path.join("faiss_index", "index.pkl"))):
            
            # Verificar se existem arquivos .faiss e .pkl em qualquer lugar no diretório
            faiss_files = []
            pkl_files = []
            
            for root, dirs, files in os.walk("faiss_index"):
                for file in files:
                    if file.endswith(".faiss"):
                        faiss_files.append(os.path.join(root, file))
                    elif file.endswith(".pkl"):
                        pkl_files.append(os.path.join(root, file))
            
            if faiss_files and pkl_files:
                # Mover arquivos para o local correto
                shutil.copy(faiss_files[0], os.path.join("faiss_index", "index.faiss"))
                shutil.copy(pkl_files[0], os.path.join("faiss_index", "index.pkl"))
            else:
                st.error("Os arquivos necessários não foram encontrados no ZIP carregado!")
                st.write(f"Arquivos .faiss encontrados: {faiss_files}")
                st.write(f"Arquivos .pkl encontrados: {pkl_files}")
                return False
            
        st.success("Vetor carregado com sucesso!")
        return True
    
    except Exception as e:
        st.error(f"Erro ao carregar vetor: {str(e)}")
        st.write("Detalhes do erro para debug:", e)
        return False

def save_uploaded_index_file(uploaded_file, filename):
    """Salva um arquivo de índice enviado pelo usuário"""
    try:
        # Garantir que o diretório exista
        os.makedirs("faiss_index", exist_ok=True)
        
        # Salvar o arquivo
        with open(os.path.join("faiss_index", filename), "wb") as f:
            f.write(uploaded_file.getvalue())
        
        return True
    except Exception as e:
        st.error(f"Erro ao salvar arquivo {filename}: {str(e)}")
        return False

def get_conversational_chain(model):
    prompt_template = """
    Answer the question in portuguese Brazil \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        # Verificar se os arquivos necessários existem
        if not (os.path.exists(os.path.join("faiss_index", "index.faiss")) and 
                os.path.exists(os.path.join("faiss_index", "index.pkl"))):
            st.error("Arquivos de índice FAISS não encontrados! Por favor, processe arquivos ou carregue um vetor válido primeiro.")
            return
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")   
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                                temperature=0.3)
        
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        mq_retriever = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=model) 
        
        #docs = mq_retriever.get_relevant_documents(query=user_question)

        compressor = LLMChainExtractor.from_llm(model)
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=mq_retriever
        )
        
        docs = compression_retriever.invoke( user_question)
        st.write(docs)
        chain = get_conversational_chain(model)
        
        response = chain(
            {"input_documents": docs, "question": user_question}
             , return_only_outputs=True)

        # print(response)
        st.markdown(f"""
            <div style="height: 600px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 10px; border-radius: 5px;">
                <div>**Reply:** {response['output_text']}</div>
            </div>
            """, 
            unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Erro ao processar sua pergunta: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Verificar status do vetor
    vector_status = "⚠️ Nenhum vetor carregado" 
    if os.path.exists(os.path.join("faiss_index", "index.faiss")) and os.path.exists(os.path.join("faiss_index", "index.pkl")):
        vector_status = "✅ Vetor pronto para uso"
        
    st.info(vector_status)
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # Criar abas para escolher entre carregar novo arquivo ou usar vetor salvo
        tab1, tab2, tab3 = st.tabs(["Processar Arquivos", "Carregar Vetor Salvo", "Carregar Arquivos de Índice"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload your PDF Files and Click on the Submit & Process Button", 
                type=["pdf", "csv"], 
                accept_multiple_files=True,
                key="content_files"
            )
                    
            if st.button("Submit & Process", key="process_btn"):
                with st.spinner("Processing..."):
                    if uploaded_files:  # Verificar se arquivos foram carregados
                        # Acumular texto de todos os arquivos
                        combined_text = ""
                        
                        for uploaded_file in uploaded_files:
                            try:
                                if uploaded_file.type == "application/pdf":
                                    file_text = get_pdf_text(uploaded_file)
                                elif uploaded_file.type == "text/csv":
                                    file_text = extract_text_from_csv(uploaded_file)
                                else:
                                    st.warning(f"Tipo de arquivo não suportado: {uploaded_file.name}")
                                    continue
                                    
                                combined_text += file_text + "\n\n"
                                
                            except Exception as e:
                                st.error(f"Erro ao processar arquivo {uploaded_file.name}: {str(e)}")
                        
                        if combined_text:
                            # Processar todo o texto acumulado
                            text_chunks = get_text_chunks(combined_text)
                            vector_store = get_vector_store(text_chunks)
                            
                            # Oferecer download do vetor
                            if os.path.exists("faiss_index_zip.zip"):
                                with open("faiss_index_zip.zip", "rb") as f:
                                    st.download_button(
                                        label="Download Vector Store",
                                        data=f,
                                        file_name="faiss_index.zip",
                                        mime="application/zip"
                                    )
                            
                            st.success("Processamento concluído com sucesso!")
                            st.rerun()  # Recarregar a página para atualizar o status do vetor
                        else:
                            st.warning("Nenhum texto foi extraído dos arquivos.")
                    else:
                        st.warning("Por favor, faça upload de pelo menos um arquivo.")
        
        with tab2:
            vector_file = st.file_uploader(
                "Carregar vetor salvo anteriormente", 
                type=["zip"],
                key="vector_file"
            )
            
            if st.button("Carregar Vetor", key="load_vector_btn"):
                if vector_file is not None:
                    with st.spinner("Carregando vetor..."):
                        success = load_uploaded_vector_store(vector_file)
                        if success:
                            st.success("Vetor carregado e pronto para uso!")
                            st.rerun()  # Recarregar a página para atualizar o status do vetor
                else:
                    st.warning("Por favor, selecione um arquivo de vetor (ZIP).")
        
        with tab3:
            st.write("Carregar arquivos individuais do índice FAISS:")
            
            faiss_file = st.file_uploader(
                "Arquivo .faiss", 
                type=["faiss"],
                key="faiss_file"
            )
            
            pkl_file = st.file_uploader(
                "Arquivo .pkl", 
                type=["pkl"],
                key="pkl_file"
            )
            
            if st.button("Carregar Arquivos", key="load_files_btn"):
                if faiss_file is not None and pkl_file is not None:
                    with st.spinner("Carregando arquivos..."):
                        # Limpar diretório se existir
                        if os.path.exists("faiss_index"):
                            shutil.rmtree("faiss_index")
                        os.makedirs("faiss_index", exist_ok=True)
                        
                        # Salvar os arquivos
                        with open(os.path.join("faiss_index", "index.faiss"), "wb") as f:
                            f.write(faiss_file.getvalue())
                        
                        with open(os.path.join("faiss_index", "index.pkl"), "wb") as f:
                            f.write(pkl_file.getvalue())
                        
                        if os.path.exists(os.path.join("faiss_index", "index.faiss")) and \
                           os.path.exists(os.path.join("faiss_index", "index.pkl")):
                            st.success("Arquivos de índice carregados com sucesso!")
                            st.rerun()
                        else:
                            st.error("Erro ao salvar os arquivos carregados.")
                else:
                    st.warning("Por favor, carregue ambos os arquivos .faiss e .pkl.")

if __name__ == "__main__":
    main()