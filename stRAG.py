import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from tkinter import filedialog
import tkinter as tk

def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory()
    root.destroy()
    return directory

# Rest of the imports remain the same...

def main():
    # Previous code remains the same until the directory input section...

    if mode == "Criar nova base de conhecimento":
        uploaded_file = st.sidebar.file_uploader(
            "Faça upload do arquivo CSV",
            type=['csv']
        )
        
        if st.sidebar.button("Selecionar diretório para salvar"):
            vector_db_path = select_directory()
            if vector_db_path:
                st.session_state['vector_db_path'] = vector_db_path
                st.sidebar.success(f"Diretório selecionado: {vector_db_path}")
        
        vector_db_path = st.session_state.get('vector_db_path', '')
        
        if uploaded_file and vector_db_path:
            if st.sidebar.button("Criar Base de Conhecimento"):
                # Rest of the code remains the same...
                
    else:
        if st.sidebar.button("Selecionar diretório da base"):
            vector_db_path = select_directory()
            if vector_db_path:
                st.session_state['vector_db_path'] = vector_db_path
                st.sidebar.success(f"Diretório selecionado: {vector_db_path}")
        
        vector_db_path = st.session_state.get('vector_db_path', '')
        
        if vector_db_path:
            if st.sidebar.button("Carregar Base de Conhecimento"):
                # Rest of the code remains the same...
