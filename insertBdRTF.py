import streamlit as st
import sqlite3
import os
import re
from striprtf.striprtf import rtf_to_text

def criar_banco():
    """Cria o banco de dados e a tabela se não existirem"""
    conn = sqlite3.connect('bdProjetoForense.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tbAssunto (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        assunto TEXT,
        classe TEXT,
        texto TEXT,
        img TEXT,
        tipo INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()

def limpar_nome_arquivo(nome):
    """Limpa o nome do arquivo removendo números e 'C' isolado"""
    # Remove números e traços no início do nome
    nome = re.sub(r'^[\d\-\.]+\s*', '', nome)
    
    # Remove a letra 'C' quando está sozinha (com espaços antes e depois)
    nome = re.sub(r'\s+C\s+', ' ', nome)
    
    # Remove qualquer número remanescente
    nome = re.sub(r'\d+\.*\d*', '', nome)
    
    # Remove múltiplos espaços
    nome = re.sub(r'\s+', ' ', nome)
    
    # Remove espaços no início e fim
    nome = nome.strip()
    
    return nome

def ler_arquivo_rtf(caminho):
    """Lê um arquivo RTF e retorna seu conteúdo"""
    try:
        with open(caminho, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
            return conteudo
    except Exception as e:
        st.error(f"Erro ao ler arquivo {caminho}: {str(e)}")
        return None

def verificar_duplicidade(cursor, texto):
    """Verifica se já existe um documento com o mesmo texto"""
    cursor.execute('SELECT COUNT(*) FROM tbAssunto WHERE texto = ?', (texto,))
    count = cursor.fetchone()[0]
    return count > 0

def processar_pasta(pasta_raiz):
    """Processa todos os arquivos RTF na pasta e subpastas"""
    conn = sqlite3.connect('bdProjetoForense.db')
    cursor = conn.cursor()
    arquivos_processados = 0
    arquivos_duplicados = 0
    
    for pasta_atual, _, arquivos in os.walk(pasta_raiz):
        nome_pasta = os.path.basename(pasta_atual)
        
        for arquivo in arquivos:
            if arquivo.lower().endswith('.rtf'):
                caminho_completo = os.path.join(pasta_atual, arquivo)
                
                # Lê o conteúdo RTF
                conteudo_rtf = ler_arquivo_rtf(caminho_completo)
                if conteudo_rtf is None:
                    continue
                
                # Converte RTF para texto
                try:
                    texto_plain = rtf_to_text(conteudo_rtf)
                except Exception as e:
                    st.error(f"Erro ao converter RTF para texto: {arquivo} - {str(e)}")
                    continue
                
                # Verifica se já existe documento com mesmo texto
                if verificar_duplicidade(cursor, texto_plain):
                    st.warning(f"Documento duplicado encontrado: {arquivo}")
                    arquivos_duplicados += 1
                    continue
                
                # Prepara os dados para inserção
                nome_arquivo = os.path.splitext(arquivo)[0]
                assunto = limpar_nome_arquivo(nome_arquivo)
                classe = f"SENTENÇA - {nome_pasta}"
                
                # Insere no banco de dados
                try:
                    cursor.execute('''
                    INSERT INTO tbAssunto (assunto, classe, texto, img, tipo)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (assunto, classe, texto_plain, conteudo_rtf, 1))
                    
                    arquivos_processados += 1
                    st.text(f"Arquivo processado:\nOriginal: {nome_arquivo}\nLimpo: {assunto}")
                except Exception as e:
                    st.error(f"Erro ao inserir no banco de dados: {arquivo} - {str(e)}")
                    continue
    
    conn.commit()
    conn.close()
    return arquivos_processados, arquivos_duplicados

# Interface Streamlit
st.title('Importador de Documentos RTF')

# Criar banco de dados se não existir
criar_banco()

# Campo para selecionar a pasta
pasta_raiz = st.text_input('Digite o caminho da pasta raiz contendo os arquivos RTF:')

if st.button('Processar Documentos'):
    if not pasta_raiz:
        st.error('Por favor, informe o caminho da pasta raiz.')
    elif not os.path.exists(pasta_raiz):
        st.error('Pasta não encontrada. Verifique o caminho informado.')
    else:
        with st.spinner('Processando documentos...'):
            quantidade, duplicados = processar_pasta(pasta_raiz)
            st.success(f'Processamento concluído! {quantidade} arquivos foram importados.')
            if duplicados > 0:
                st.warning(f'{duplicados} arquivos duplicados foram encontrados e ignorados.')
            
        # Exibe estatísticas
        conn = sqlite3.connect('bdProjetoForense.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tbAssunto')
        total_registros = cursor.fetchone()[0]
        conn.close()
        
        st.info(f'Total de registros no banco de dados: {total_registros}')