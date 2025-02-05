import streamlit as st
from streamlit_option_menu import option_menu
from formSquill as fs
from langchainSearch as ls
from conversorRTF_HTML as crtf
from rag as rg




#st.set_page_config(page_title="Gerenciador de Modelos Judiciais", layout="wide")
st.set_page_config(
    page_title="Meu App",
    page_icon="🎈",    
    layout="wide",
       
)


st.markdown("""
    <style>
        .main {
            overflow: hidden;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            height: 100vh;
        }
        footer {
            display: none;
        }
        #MainMenu {
            visibility: hidden;
        }
        header {
            visibility: hidden;
        }
        
            
        
    </style>
""", unsafe_allow_html=True)


#page = st_navbar(["Home", "Documentation", "Examples", "Community", "About"])
#st.write(page)


# with st.sidebar:
#     selected = option_menu(
#         menu_title="Main Menu",  # required
#         options=["Home", "Projects", "Contact"],  # required
#     )

# if selected == "Home":
#     st.title(f"You have selected {selected}")
# if selected == "Projects":
#     st.title(f"You have selected {selected}")




st.markdown(f"""<header tabindex="-1" class="stAppHeader st-emotion-cache-12fmjuu e10jh26i0"><center><h2>💬 Chat Assistente Jurídico</h2></center></header> 
    """,
    unsafe_allow_html=True
)


selected = option_menu(
    menu_title=None,
    options=["Formulário", "Chatbot Jurídico", "Pesquisa Avançada","Conversor de banco de dados"], # Você pode alterar estas opções
    icons=["house", "file-text", "eye"],         # E estes ícones
    default_index=0,                             # Página inicial (0 = primeiro item)
    orientation="horizontal",                     # Navegação horizontal
    styles={
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#0083B8"},
    }
)

# Lógica das páginas
if selected == "Formulário":
    
    fs.main()

elif selected == "Chatbot Jurídico":
    
    rg.main() 
    
elif selected == "Conversor de banco de dados":
    
    crtf.main()

elif selected == "Pesquisa Avançada":
   
    ls.main()
    




# st.header("Bem-vindo ao Langchain Search")
# pg = st.navigation([st.Page("langchainSearch.py", title= "Pesquisa Avançada por IA", icon="🎯"), 
#                     st.Page("rag.py",title= "Chatbot RAG", icon="🤖"), 
#                     st.Page("formSquill.py",title= "Tela de Formulário", icon="📝") , 
#                     st.Page("conversorRTF_HTML.py",title= "Conversor de banco de dados RTF para HTML", icon="📄")])
# pg.run()
        
    
   

#st.caption("Desenvolvido por: [Langchain](https://www.langchain.com.br/)")

#stAppHeader st-emotion-cache-12fmjuu e10jh26i0
