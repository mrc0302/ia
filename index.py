# streamlit_app.py

import hmac
import streamlit as st

import importlib

PAGES = {

    "Home": "index",
    "Pesquisa": "langchainSearch",
    
    #,
    #"Consulta por IA": "Page2",
}


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """AUTENTICAÇÃO"""
        with st.form("Credentials"):
            st.text_input("Login", key="username")
            st.text_input("Senha", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Main Streamlit app starts here
#page = st.sidebar.radio("Ir para", list(PAGES.keys())[1])
# Importa e executa a página selecionada
#module = importlib.import_module(PAGES[page])
#module = importlib.import_module((PAGES.keys())[1])
module = importlib.import_module(PAGES["Pesquisa"])
module.main()