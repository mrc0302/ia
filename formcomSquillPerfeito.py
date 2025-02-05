
import streamlit as st
import time
from streamlit_quill import st_quill
import re


# Configuração da página deve ser o primeiro comando Streamlit
st.set_page_config(page_title="Legal Document Application", layout="wide")

from datetime import datetime
from database import Database

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Modifique a função transform_results_to_items:
def transform_results_to_items(results):
    # Agrupar por classe
    classes = {}
    for assunto, classe, preview in results:
        if classe not in classes:
            classes[classe] = []
        classes[classe].append((assunto, preview))
    
    # Criar itens para o accordion
    items = []
    for classe in sorted(classes.keys()):
        children = []
        for assunto, preview in sorted(classes[classe]):
            children.append({
                "label": assunto,
                "key": assunto,
                "description": preview[:100] + "..." if preview else ""
            })
        
        items.append({
            "label": classe,
            "children": children,
            "key": f"class_{classe}"
        })
    
    return items

# Função para inicializar o estado da sessão
def init_session_state():
    if "form_assunto" not in st.session_state:
        st.session_state.form_assunto = ""
    if "form_classe" not in st.session_state:
        st.session_state.form_classe = ""
    if "form_content" not in st.session_state:
        st.session_state.form_content = ""
    if "search_text" not in st.session_state:
        st.session_state.search_text = ""
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Assunto"
    if "show_dialog" not in st.session_state:
        st.session_state.show_dialog = False
    if "action_type" not in st.session_state:
        st.session_state.action_type = None
    if "temp_data" not in st.session_state:
        st.session_state.temp_data = {}

# Função de confirmação
@st.dialog("Confirmação")
def show_confirmation_dialog():
    st.write(f"### Deseja {st.session_state.action_type}?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Sim"):
            if st.session_state.action_type == "salvar":
                execute_save()
            elif st.session_state.action_type == "excluir":
                execute_delete()
            elif st.session_state.action_type == "limpar":
                execute_clear()
            st.session_state.show_dialog = False
            st.rerun()
    
    with col2:
        if st.button("Não"):
            st.session_state.show_dialog = False
            st.session_state.action_type = None
            st.rerun()

# Funções de execução
def execute_save():
    app = Database()
    assunto = st.session_state.temp_data.get('assunto')
    classe = st.session_state.temp_data.get('classe')
    content = st.session_state.temp_data.get('content')
    
    # O st_quill retorna o conteúdo HTML diretamente como string
    content_to_save = {
        'text': content,  # texto plano
        'html': content   # conteúdo HTML
    }
    
    if not assunto:
        st.error("O campo Assunto é obrigatório")
    elif not classe:
        st.error("O campo Classe é obrigatório")
    else:
        success, message = app.save_or_update_document(assunto, classe, content_to_save)
        if success:
            set_form_data(assunto, classe, content)
            st.success("Documento salvo com sucesso!")
            time.sleep(2)
        else:
            st.error(message)

def execute_delete():
    app = Database()
    assunto = st.session_state.form_assunto

    if assunto:
        success, message = app.delete_document(assunto)
        if success:
            set_form_data()
            st.success("Documento excluído com sucesso!")
        else:
            st.error(message)
    else:
        st.warning("Nenhum documento selecionado")

def execute_clear():
    set_form_data()
    st.success("Formulário limpo com sucesso!")

# Função para atualizar os dados do formulário
def set_form_data(assunto="", classe="", content=""):
    st.session_state.form_assunto = assunto
    st.session_state.form_classe = classe
    st.session_state.form_content = content

def highlight_search_terms(html_content, search_text):
    """Destaca todos os termos encontrados no conteúdo HTML."""
    if not search_text or not html_content:
        return html_content
    
    # Divide os termos por vírgula e limpa
    terms = [term.strip() for term in search_text.split(',')]
    terms = [term for term in terms if term]
    
    highlighted_content = html_content
    
    for term in terms:
        try:
            # Escapa caracteres especiais do regex
            escaped_term = re.escape(term)
            
            # Cria padrão case-insensitive que preserva caso original
            pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
            
            # Aplica o highlight mantendo o caso original do texto
            highlighted_content = pattern.sub(
                lambda m: f'<span style="background-color: yellow;">{m.group(1)}</span>',
                highlighted_content
            )
        except Exception as e:
            print(f"Erro ao destacar termo '{term}': {str(e)}")
            continue
    
    return highlighted_content

    # #"""Destaca múltiplos termos no conteúdo HTML."""
    # if not search_text or not html_content:
    #     return html_content
    
    # # Divide por vírgulas e limpa cada termo
    # search_terms = [term.strip() for term in search_text.split(',')]
    # # Remove termos vazios
    # search_terms = [term for term in search_terms if term]
    
    # # Aplica highlight para cada termo
    # highlighted_content = html_content
    # for term in search_terms:
    #     if term:
    #         # Escapa caracteres especiais do regex
    #         escaped_term = re.escape(term)
    #         # Cria padrão case-insensitive
    #         pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
    #         # Aplica o highlight
    #         highlighted_content = pattern.sub(
    #             r'<span style="background-color: yellow;">\1</span>',
    #             highlighted_content
    #         )
    
    # return highlighted_content


init_session_state()
app = Database()

with st.sidebar:
    st.markdown("<h1 style='text-align: left; margin-top: -2.5rem'> Pesquisa de Modelos</h1>", unsafe_allow_html=True)
    search_text = st.text_input("Parâmetro:", key="search_text")
    search_mode = st.radio("Modalidade:", 
                        ["Assunto", "Classe", "Conteúdo", "Todos", "Multi-termo"],
                        key="search_mode")

    if search_text:
        try:
            # st.write("Iniciando busca...")
            # st.write(f"Texto de busca: {search_text}")
            # st.write(f"Modo: {search_mode}")
            
            results, error_message = app.search_documents(search_text, search_mode)
            
            if error_message:
                st.error(f"Erro na busca: {error_message}")
            elif not results:
                st.warning("Nenhum documento encontrado!")
            else:
                st.success(f"Encontrados {len(results)} resultados")
                
                classes = {}
                for assunto, classe, preview in results:
                    if classe not in classes:
                        classes[classe] = []
                    classes[classe].append((assunto, preview))

                for classe in sorted(classes.keys()):
                    with st.expander(f"► {classe}"):
                        for assunto, preview in sorted(classes[classe]):
                            if st.button(f"📄 {assunto}", key=f"btn_{classe}_{assunto}"):
                                doc, doc_message = app.get_document(assunto)
                                if doc:
                                    # Aplica o highlight antes de definir o conteúdo
                                    highlighted_content = highlight_search_terms(doc[3], search_text)
                                    set_form_data(doc[1], doc[2], highlighted_content)
                                else:
                                    st.error(doc_message)
                    
        except Exception as e:
            st.error(f"Erro inesperado: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h4 style='text-align: center; margin-top: -4rem'> Biblioteca de Decisões Judiciais e Despachos </h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; margin-top: -1rem'>Modelos</h5>", unsafe_allow_html=True)

    with st.form("document_form"):
        assunto = st.text_input("Assunto:", value=st.session_state.form_assunto)
        classe = st.text_input("Classe:", value=st.session_state.form_classe)
        
        toolbar = [
            [{'header': '1'}, {'header': '2'}, {'font': []}],
            [{'list': 'ordered'}, {'list': 'bullet'}, {'indent': '-1'}, {'indent': '+1'}],
            [{'bold': True}, {'italic': True}, {'underline': True}, {'strike': True}],
            [{'color': []}, {'background': []}],
            [{'link': True}],
            [{'align': []}]
        ]

        content = st_quill(
            value=st.session_state.form_content,
            html=True,
            toolbar=toolbar,
            history=True,
            preserve_whitespace=True,
            readonly=False,
            key=None
        )
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.form_submit_button("Salvar"):
                if not assunto:
                    st.error("O campo Assunto é obrigatório")
                elif not classe:
                    st.error("O campo Classe é obrigatório")
                else:
                    content_to_save = {
                        'text': content,
                        'html': content
                    }
                    success, message = app.save_or_update_document(assunto, classe, content_to_save)
                    if success:
                        st.success("Documento salvo com sucesso!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)

        with col2:
            if st.form_submit_button("Novo"):
                set_form_data()
                st.rerun()

        with col3:
            if st.form_submit_button("Excluir"):
                if st.session_state.form_assunto:
                    success, message = app.delete_document(st.session_state.form_assunto)
                    if success:
                        st.success("Documento excluído com sucesso!")
                        set_form_data()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Nenhum documento selecionado para excluir")

        with col4:
            if st.form_submit_button("Limpar"):
                set_form_data()
                st.rerun()
