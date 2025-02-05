import streamlit as st
import sqlite3
import pypandoc
import tempfile
import os
import re
from typing import List, Tuple






# Adicione esta verificação de Pandoc no início
try:
    pypandoc.get_pandoc_version()
except OSError:
    st.warning("Instalando Pandoc... Aguarde.")
    pypandoc.download_pandoc()
    st.success("Pandoc instalado com sucesso!")

  
    
def connect_to_db(db_path: str) -> sqlite3.Connection:
   """Estabelece conexão com o banco de dados"""
   try:
       conn = sqlite3.connect(db_path)
       return conn
   except Exception as e:
       raise Exception(f"Erro ao conectar ao banco de dados: {str(e)}")

def get_records_to_convert(conn: sqlite3.Connection) -> List[Tuple]:
   """Recupera registros para conversão"""
   try:
       cursor = conn.cursor()
       cursor.execute("""
           SELECT id, img 
           FROM tbAssunto 
           WHERE img IS NOT NULL 
           AND img <> ''
       """)
       return cursor.fetchall()
   except Exception as e:
       raise Exception(f"Erro ao recuperar registros: {str(e)}")

def update_html_column(conn: sqlite3.Connection, id: int, html_content: str) -> None:
   """Atualiza a coluna html"""
   try:
       cursor = conn.cursor()
       cursor.execute(
           "UPDATE tbAssunto SET html = ? WHERE id = ?",
           (html_content, id)
       )
       conn.commit()
   except Exception as e:
       raise Exception(f"Erro ao atualizar registro {id}: {str(e)}")

def extract_color_table(rtf_content: str) -> dict:
   """Extrai a tabela de cores do RTF"""
   color_dict = {}
   color_match = re.search(r'\\colortbl;(.*?);', rtf_content)
   if color_match:
       colors = color_match.group(1).split(';')
       for i, color in enumerate(colors):
           rgb_match = re.search(r'\\red(\d+)\\green(\d+)\\blue(\d+)', color)
           if rgb_match:
               r, g, b = map(int, rgb_match.groups())
               color_dict[f'\\cf{i}'] = f'rgb({r}, {g}, {b})'
   return color_dict

def process_rtf_colors(html_content: str, color_table: dict) -> str:
   """Processa as cores no HTML gerado"""
   for color_cmd, rgb_value in color_table.items():
       html_content = html_content.replace(
           f'class="{color_cmd}"',
           f'style="color: {rgb_value};"'
       )
   return html_content

def convert_rtf_to_html(rtf_content: str) -> str:
   """Converte RTF para HTML usando pypandoc e processa o resultado"""
   try:
       color_table = extract_color_table(rtf_content)
       
       with tempfile.NamedTemporaryFile(mode='w', suffix='.rtf', delete=False, encoding='utf-8') as tmp_file:
           tmp_file.write(rtf_content)
           tmp_path = tmp_file.name
       
       try:
           html_content = pypandoc.convert_file(
               tmp_path,
               'html',
               format='rtf',
               extra_args=['--wrap=none']
           )
           
           html_content = process_rtf_colors(html_content, color_table)
           
           html_content = re.sub(
               r'<(p|h[1-6])[^>]*>',
               r'<\1 class="ql-align-justify">',
               html_content
           )
           
           html_content = re.sub(r'</?body[^>]*>', '', html_content)
           html_content = re.sub(r'</?html[^>]*>', '', html_content)
           html_content = re.sub(r'<head>.*?</head>', '', html_content, flags=re.DOTALL)
           
           return html_content.strip()
           
       finally:
           if os.path.exists(tmp_path):
               os.unlink(tmp_path)
               
   except Exception as e:
       raise Exception(f"Erro na conversão: {str(e)}")


def main():
    #st.set_page_config(page_title="Conversor RTF para HTML", page_icon="🔄")

    st.title("Conversor RTF para HTML")

    # Campo para digitar caminho do banco
    db_path = st.text_input('Digite o caminho do banco de dados:', value='bdProjetoForense.db')

    # Upload do arquivo
    uploaded_file = st.file_uploader("Ou selecione o arquivo do banco de dados", type=['db'])

    if uploaded_file is not None:
        with open(db_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Arquivo carregado: {db_path}")

    if st.button("Verificar Conexão"):
        try:
            conn = connect_to_db(db_path)
            st.success(f"Conexão estabelecida com sucesso: {db_path}")
            conn.close()
        except Exception as e:
            st.error(f"Erro na conexão: {str(e)}")

    if st.button("Visualizar Registros"):
        try:
            conn = connect_to_db(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, img FROM tbAssunto WHERE img IS NOT NULL")
            records = cursor.fetchall()
            conn.close()
            
            if records:
                st.write(f"Total de registros encontrados: {len(records)}")
                for id, img in records:
                    st.write(f"ID: {id}")
                    st.text(img[:200] + "..." if len(img) > 200 else img)
                    st.write("---")
            else:
                st.warning("Nenhum registro encontrado.")
            
        except Exception as e:
            st.error(f"Erro ao visualizar registros: {str(e)}")

    if st.button("Iniciar Conversão", type="primary"):
        try:
            with st.spinner('Conectando ao banco de dados...'):
                conn = connect_to_db(db_path)
            
            records = get_records_to_convert(conn)
            total_records = len(records)
            
            if total_records == 0:
                st.warning("Nenhum registro para converter!")
                conn.close()
            
            
            st.info(f"Total de registros para converter: {total_records}")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            converted_count = 0
            error_count = 0
            error_records = []
            
            for i, (id, rtf_content) in enumerate(records, 1):
                try:
                    status_text.text(f"Convertendo registro {i} de {total_records}")
                    
                    if rtf_content:
                        html_content = convert_rtf_to_html(rtf_content)
                        update_html_column(conn, id, html_content)
                        converted_count += 1
                    
                    progress_bar.progress(i / total_records)
                
                except Exception as e:
                    error_count += 1
                    error_records.append((id, str(e)))
                    continue
            
            conn.close()
            
            st.success(f"""✅ Processo concluído!
            - Registros convertidos: {converted_count}
            - Erros: {error_count}
            """)
            
            if error_records:
                st.error("Detalhes dos erros:")
                for id, error in error_records:
                    st.write(f"- Registro {id}: {error}")
            
        except Exception as e:
            st.error(f"Erro durante o processo: {str(e)}")
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    main()

