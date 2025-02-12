import streamlit as st
from streamlit_option_menu import option_menu
import formSquill as fs
import langchainSearch as ls
import conversorRTF_HTML as crtf
import ragTeste as rg

st.set_page_config(page_title="Gerenciador de Modelos Judiciais", layout="wide")

# st.markdown("""
#     <style>
#         .main {
#             overflow: hidden;
#         }
#         .block-container {
#             padding-top: 1rem;
#             padding-bottom: 0rem;
#             height: 100vh;
#         }
#         footer {
#             display: none;
#         }
#         #MainMenu {
#             visibility: hidden;
#         }
#         header {
#             visibility: hidden;
#         }                  
        
#     </style>
# """, unsafe_allow_html=True)


# st.markdown(f"""<header tabindex="-1" class="stAppHeader st-emotion-cache-12fmjuu e10jh26i0"><center><h2>💬 Chat Assistente Jurídico</h2></center></header> 
#     """,
#     unsafe_allow_html=True
# )


with st.sidebar:

    
  pg = st.navigation([st.Page("langchainSearch.py", title= "Pesquisa Avançada por IA", icon="🎯"), 
                              st.Page("ragTeste.py",title= "Chatbot RAG", icon="🤖"), 
                              st.Page("formSquill.py",title= "Tela de Formulário", icon="📝") , 
                              st.Page("conversorRTF_HTML.py",title= "Conversor de banco de dados RTF para HTML", icon="📄")])
pg.run()   
   



    
   

#st.caption("Desenvolvido por: [Langchain](https://www.langchain.com.br/)")

#stAppHeader st-emotion-cache-12fmjuu e10jh26i0
