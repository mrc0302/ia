import streamlit as st
from streamlit_option_menu import option_menu
import formSquill as fs
import langchainSearch as ls
import conversorRTF_HTML as crtf
import ragTeste as rg



def main(): 
    
    # Configuração da página deve ser o primeiro comando Streamlit
    st.set_page_config(
        page_title="Ex-stream-ly Cool App",
        page_icon="🧊", 
        layout="wide",  
       # initial_sidebar_state="expanded"
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


if __name__ == "__main__":
    main()
