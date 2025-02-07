import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
from google.api_core import retry

def main():
    # Configuração da página Streamlit
    st.set_page_config(page_title="Pesquisa em CSV", layout="wide")
    st.title("Visualização e Pesquisa em CSV")

    # Carregar variáveis de ambiente
    load_dotenv()
    google_api_key = os.getenv("google_api_key")

    # Configurar Gemini com tratamento de erro
    try:
        if google_api_key:
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('gemini-pro')
            gemini_available = True
        else:
            gemini_available = False
    except Exception:
        gemini_available = False

    def search_in_dataframe(df, query, columns_to_search=['assunto', 'texto']):
        """
        Realiza uma busca case-insensitive no DataFrame
        """
        combined_mask = pd.Series([False] * len(df))
        
        for column in columns_to_search:
            if column in df.columns:
                mask = df[column].astype(str).str.contains(query, case=False, na=False)
                combined_mask |= mask
        
        return df[combined_mask]

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def generate_response(query, context):
        """
        Gera resposta com retry em caso de erro
        """
        try:
            prompt = f"""
            Contexto: {context}
            
            Pergunta: {query}
            
            Por favor, responda à pergunta acima usando apenas as informações fornecidas no contexto.
            Se a informação não estiver disponível no contexto, diga que não pode responder.
            
            Resposta:
            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Erro ao gerar resposta com Gemini: {str(e)}")
            return None

    # Interface Streamlit
    st.sidebar.header("Configurações")

    # Indicador de status do Gemini
    if gemini_available:
        st.sidebar.success("Gemini API disponível")
    else:
        st.sidebar.warning("Gemini API não disponível - Modo somente visualização")

    # Upload do arquivo CSV
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo CSV",
        type=['csv']
    )

    # Layout em duas colunas
    col1, col2 = st.columns([2, 1])

    if uploaded_file:
        try:
            # Carregar o DataFrame
            df = pd.read_csv(uploaded_file)
            
            # Mostrar informações sobre o DataFrame
            with st.sidebar:
                st.write("Informações do arquivo:")
                st.write(f"Total de registros: {len(df)}")
                st.write(f"Colunas disponíveis: {', '.join(df.columns)}")
            
            # Seleção de colunas para pesquisa
            columns_to_search = st.sidebar.multiselect(
                "Selecione as colunas para pesquisa:",
                df.columns,
                default=['assunto', 'texto'] if 'assunto' in df.columns and 'texto' in df.columns else []
            )
            
            # Número de registros por página
            rows_per_page = st.sidebar.slider('Registros por página:', 5, 50, 10)
            
            # Exibir todos os registros na primeira coluna
            with col1:
                st.subheader("Registros")
                
                # Adicionar campos de filtro para cada coluna
                filters = {}
                col_filters = st.columns(3)  # Organiza filtros em 3 colunas
                for i, column in enumerate(df.columns):
                    with col_filters[i % 3]:
                        if df[column].dtype == 'object':  # Para colunas de texto
                            unique_values = ['Todos'] + sorted(df[column].unique().tolist())
                            selected = st.selectbox(f'Filtrar {column}:', unique_values)
                            if selected != 'Todos':
                                filters[column] = selected

                # Aplicar filtros
                filtered_df = df.copy()
                for column, value in filters.items():
                    filtered_df = filtered_df[filtered_df[column] == value]

                # Paginação manual
                total_pages = len(filtered_df) // rows_per_page + (1 if len(filtered_df) % rows_per_page else 0)
                current_page = st.number_input('Página', min_value=1, max_value=total_pages, value=1) - 1
                start_idx = current_page * rows_per_page
                end_idx = start_idx + rows_per_page
                
                # Exibir DataFrame paginado
                st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
                st.write(f"Página {current_page + 1} de {total_pages}")
            
            # Interface de pesquisa na segunda coluna
            with col2:
                st.subheader("Pesquisa")
                query = st.text_input("Digite sua pesquisa:")
                
                if query:
                    with st.spinner("Pesquisando..."):
                        # Realizar a busca
                        results = search_in_dataframe(filtered_df, query, columns_to_search)
                        
                        st.write(f"Encontrados {len(results)} resultados:")
                        
                        if not results.empty:
                            # Exibir resultados filtrados
                            st.subheader("Resultados da pesquisa:")
                            st.dataframe(results, use_container_width=True)
                            
                            # Gerar resposta com Gemini se disponível
                            if gemini_available:
                                context = "\n\n".join([
                                    f"ID: {row['id']}\nAssunto: {row['assunto']}\nClasse: {row['classe']}\nTexto: {row['texto']}"
                                    for _, row in results.iterrows()
                                ])
                                
                                response = generate_response(query, context)
                                if response:
                                    st.subheader("Análise do Gemini:")
                                    st.write(response)
                        else:
                            st.warning("Nenhum resultado encontrado para sua pesquisa.")
                        
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")
    else:
        st.info("Por favor, faça upload de um arquivo CSV para começar.")
if __name__ == "__main__":
    main()
