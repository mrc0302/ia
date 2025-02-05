import streamlit as st
import sqlite3
import os
from striprtf.striprtf import rtf_to_text
import pathlib
import pandas as pd

def get_db_path():
    """Retorna o caminho absoluto do banco de dados"""
    return os.path.abspath('documentos.db')

def create_database():
    """Cria o banco de dados e a tabela se não existirem"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            classe TEXT NOT NULL,
            conteudo TEXT NOT NULL,
            caminho_completo TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    return db_path

def insert_document(titulo, classe, conteudo, caminho_completo):
    """Insere um documento no banco de dados"""
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Verifica se o documento já existe
    cursor.execute('SELECT id FROM documentos WHERE caminho_completo = ?', (caminho_completo,))
    if cursor.fetchone() is None:
        cursor.execute('''
            INSERT INTO documentos (titulo, classe, conteudo, caminho_completo)
            VALUES (?, ?, ?, ?)
        ''', (titulo, classe, conteudo, caminho_completo))
        conn.commit()
        return True
    conn.close()
    return False

def read_rtf_file(file_path):
    """Lê e converte o conteúdo de um arquivo RTF para texto"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return rtf_to_text(content)
    except Exception as e:
        st.warning(f"Erro ao ler arquivo {file_path}: {str(e)}")
        return ""

def process_directory(root_path):
    """Processa recursivamente todos os arquivos RTF em um diretório e suas subpastas"""
    root_dir = pathlib.Path(root_path)
    
    # Encontra todos os arquivos RTF recursivamente
    rtf_files = list(root_dir.rglob("*.rtf"))
    total_files = len(rtf_files)
    
    if total_files == 0:
        return "Nenhum arquivo RTF encontrado no diretório."
    
    # Contadores para o relatório
    arquivos_processados = 0
    arquivos_novos = 0
    
    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file_path in enumerate(rtf_files):
        # Atualiza a barra de progresso
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processando arquivo {idx + 1} de {total_files}")
        
        # Obtém informações do arquivo
        titulo = file_path.stem  # Nome do arquivo sem extensão
        classe = file_path.parent.name  # Nome da pasta pai
        caminho_completo = str(file_path.absolute())
        
        # Lê o conteúdo do arquivo
        conteudo = read_rtf_file(caminho_completo)
        
        # Tenta inserir no banco de dados
        if conteudo:
            arquivos_processados += 1
            if insert_document(titulo, classe, conteudo, caminho_completo):
                arquivos_novos += 1
    
    return {
        "total": total_files,
        "processados": arquivos_processados,
        "novos": arquivos_novos
    }

def main():
    st.title("Importador de Documentos RTF para SQLite")
    
    # Cria o banco de dados e obtém o caminho
    db_path = create_database()
    
    # Mostra o caminho do banco de dados
    st.info(f"📁 Localização do banco de dados: {db_path}")
    
    # Campo para inserir o caminho do diretório
    root_path = st.text_input("Digite o caminho do diretório raiz:")
    
    if st.button("Processar Diretório") and root_path:
        if os.path.isdir(root_path):
            with st.spinner("Processando arquivos..."):
                result = process_directory(root_path)
                
                if isinstance(result, str):
                    st.warning(result)
                else:
                    st.success(f"""
                        Processamento concluído!
                        - Total de arquivos encontrados: {result['total']}
                        - Arquivos processados com sucesso: {result['processados']}
                        - Novos arquivos adicionados: {result['novos']}
                        - Arquivos já existentes: {result['processados'] - result['novos']}
                    """)
        else:
            st.error("Diretório inválido. Por favor, verifique o caminho.")
    
    # Adiciona um botão para visualizar os dados
    if st.button("Visualizar Dados"):
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query("""
            SELECT 
                id, 
                titulo, 
                classe, 
                substr(caminho_completo, -50) as caminho 
            FROM documentos
            """, conn)
        conn.close()
        
        if not df.empty:
            st.write(f"Total de documentos no banco: {len(df)}")
            st.dataframe(df)
        else:
            st.write("Nenhum documento encontrado no banco de dados.")

if __name__ == "__main__":
    main()