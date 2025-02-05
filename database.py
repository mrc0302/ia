# database.py

import sqlite3
import unicodedata
from contextlib import contextmanager
import threading

class Database:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_name="bdProjetoForense.db"):
        if not hasattr(self, 'initialized'):
            self.db_name = db_name
            self.initialized = True
            self._local = threading.local()
            self.init_db()

    @contextmanager
    def get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_name)
            self.create_normalize_function(self._local.connection)
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e

    @contextmanager
    def get_cursor(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                cursor.close()

    def create_normalize_function(self, connection):
        def normalize_text(text):
            return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                         if unicodedata.category(c) != 'Mn')
        connection.create_function("normalize", 1, normalize_text)

    def init_db(self):
        with self.get_cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tbAssunto (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assunto TEXT,
                classe TEXT,
                html TEXT
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assunto ON tbAssunto(assunto)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classe ON tbAssunto(classe)")

    def save_or_update_document(self, assunto, classe, content):
        try:
            # Verifica se o content é um dicionário com chave 'html'
            if isinstance(content, dict) and 'html' in content:
                content = content['html']

            with self.get_cursor() as cursor:
                cursor.execute("SELECT id FROM tbAssunto WHERE assunto = ?", (assunto,))
                existing_record = cursor.fetchone()

                if existing_record:
                    query = """
                    UPDATE tbAssunto
                    SET classe = ?, html = ?
                    WHERE assunto = ?
                    """
                    cursor.execute(query, (classe, content, assunto))
                else:
                    query = """
                    INSERT INTO tbAssunto (assunto, classe, html)
                    VALUES (?, ?, ?)
                    """
                    cursor.execute(query, (assunto, classe, content))

            return True, "Documento salvo com sucesso!"

        except sqlite3.Error as e:
            return False, f"Erro ao salvar documento: {str(e)}"

    def delete_document(self, assunto):
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT id FROM tbAssunto WHERE assunto = ?", (assunto,))
                if not cursor.fetchone():
                    return False, "Documento não encontrado para exclusão"
                
                cursor.execute("DELETE FROM tbAssunto WHERE assunto = ?", (assunto,))
                if cursor.rowcount == 0:
                    return False, "Nenhum documento foi excluído"

            return True, "Documento excluído com sucesso!"
        except sqlite3.Error as e:
            return False, f"Erro ao excluir documento: {str(e)}"

    def get_document(self, assunto):
        try:
            with self.get_cursor() as cursor:
                cursor.execute('''
                    SELECT id, assunto, classe, html FROM tbAssunto 
                    WHERE assunto = ?
                ''', (assunto,))
                result = cursor.fetchone()
                return (result, None) if result else (None, "Documento não encontrado")
        except sqlite3.Error as e:
            return None, f"Erro ao buscar: {str(e)}"

    def search_documents(self, search_text, search_mode):
        try:
            if search_mode == "Multi-termo":
                return self.search_documents_multi_terms(search_text)
                
            search_param = f"%{search_text}%"
            
            with self.get_cursor() as cursor:
                if search_mode == "Assunto":
                    query = """
                    SELECT assunto, classe, substr(html, 1, 200) as preview
                    FROM tbAssunto
                    WHERE assunto LIKE ?
                    ORDER BY classe, assunto
                    LIMIT 50
                    """
                elif search_mode == "Classe":
                    query = """
                    SELECT assunto, classe, substr(html, 1, 200) as preview
                    FROM tbAssunto
                    WHERE classe LIKE ?
                    ORDER BY classe, assunto
                    LIMIT 50
                    """
                else:  # Conteúdo
                    query = """
                    SELECT assunto, classe, substr(html, 1, 200) as preview
                    FROM tbAssunto
                    WHERE html LIKE ?
                    ORDER BY classe, assunto
                    LIMIT 50
                    """
                
                cursor.execute(query, (search_param,))
                results = cursor.fetchall()
                return results if results else [], None
            
        except sqlite3.Error as e:
            return [], f"Erro na pesquisa: {str(e)}"

    def search_documents_multi_terms(self, search_text):
        try:
            terms = [term.strip() for term in search_text.split(',')]
            terms = [term for term in terms if term]

            with self.get_cursor() as cursor:
                query = """
                    SELECT assunto, classe, substr(html, 1, 200) as preview
                    FROM tbAssunto
                    WHERE 1=1
                """
                for _ in terms:
                    query += " AND html LIKE ?"
                query += " ORDER BY classe, assunto LIMIT 50"

                params = [f"%{term}%" for term in terms]
                cursor.execute(query, params)
                results = cursor.fetchall()
                return results, None

        except sqlite3.Error as e:
            return [], f"Erro na pesquisa: {str(e)}" 