# database.py - Implementação para PostgreSQL
import psycopg2
import unicodedata
import logging
import threading
from bs4 import BeautifulSoup

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Database')

class Database:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, connection_string="Host=dbpostgres.c69sswyckdq3.us-east-1.rds.amazonaws.com;Port=5432;Database=postgres;Username=postgres;Password=lwm4r*s4*"):
        if not hasattr(self, 'initialized'):
            print("Inicializando classe Database PostgreSQL")
            # Parsear a string de conexão no formato ADO.NET
            params = {}
            for param in connection_string.split(';'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key.lower()] = value
            
            # Configurar parâmetros para o psycopg2
            self.connection_params = {
                'host': params.get('host'),
                'port': params.get('port'),
                'dbname': params.get('database'),  # Note: é 'dbname' no psycopg2, não 'database'
                'user': params.get('username'),
                'password': params.get('password')
            }
            
            # Adicionar sslmode apenas se especificado
            if 'sslmode' in params:
                ssl_value = params.get('sslmode', '').lower()
                self.connection_params['sslmode'] = 'require' if ssl_value == 'require' else 'prefer'
            
            self.initialized = True
            self.connection = None
            self.init_db()
    
    def connect(self):
        """Estabelece uma conexão com o banco de dados"""
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(**self.connection_params)
                logger.info("Conexão estabelecida com sucesso")
            return self.connection
        except Exception as e:
            logger.error(f"Erro ao conectar: {str(e)}")
            raise

    def init_db(self):
        """Inicializa a estrutura do banco de dados"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Verificar se a tabela existe
            cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'tbassunto'
            )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                # Criar a tabela
                cursor.execute("""
                CREATE TABLE tbAssunto (
                    id SERIAL PRIMARY KEY,
                    assunto TEXT,
                    classe TEXT,
                    texto TEXT,
                    html TEXT,
                    img TEXT
                )
                """)
                conn.commit()
                logger.info("Tabela tbAssunto criada com sucesso")
            
            # Verificar se a coluna 'texto' existe
            if table_exists:
                cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'tbassunto' AND column_name = 'texto'
                )
                """)
                column_exists = cursor.fetchone()[0]
                
                # Se a coluna 'texto' não existir, adicioná-la
                if not column_exists:
                    cursor.execute("ALTER TABLE tbAssunto ADD COLUMN texto TEXT")
                    conn.commit()
                    logger.info("Coluna texto adicionada com sucesso")
            
            # Verificar os índices
            cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'tbassunto' AND indexname = 'idx_assunto'
            )
            """)
            idx_assunto_exists = cursor.fetchone()[0]
            
            if not idx_assunto_exists:
                cursor.execute("CREATE INDEX idx_assunto ON tbAssunto(assunto)")
                conn.commit()
                logger.info("Índice idx_assunto criado com sucesso")
            
            cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'tbassunto' AND indexname = 'idx_classe'
            )
            """)
            idx_classe_exists = cursor.fetchone()[0]
            
            if not idx_classe_exists:
                cursor.execute("CREATE INDEX idx_classe ON tbAssunto(classe)")
                conn.commit()
                logger.info("Índice idx_classe criado com sucesso")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {str(e)}")
            if 'conn' in locals() and conn:
                conn.rollback()
            raise

    def extract_text_from_html(self, html_content):
        """Extrai texto puro do HTML"""
        if not html_content:
            return ""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.warning(f"Erro ao extrair texto do HTML: {str(e)}")
            return html_content  # Retorna o HTML original como fallback

    def save_or_update_document(self, assunto, classe, content):
        """Salva ou atualiza um documento no banco de dados"""
        try:
            # Extrair conteúdo HTML e RTF (se disponível)
            html_content = None
            rtf_content = None
            
            if isinstance(content, dict):
                if 'html' in content:
                    html_content = content['html']
                if 'rtf' in content:
                    rtf_content = content['rtf']
                elif 'text' in content:
                    rtf_content = content['text']
            else:
                # Se não for um dicionário, assume que é o conteúdo HTML
                html_content = content
            
            # Extrair texto puro do HTML
            text_content = self.extract_text_from_html(html_content)
            
            # Obter conexão
            conn = self.connect()
            cursor = conn.cursor()
            
            # Verificar se o registro já existe
            cursor.execute("SELECT id FROM tbAssunto WHERE assunto = %s", (assunto,))
            existing_record = cursor.fetchone()

            if existing_record:
                # Atualizar registro existente
                if rtf_content:
                    query = """
                    UPDATE tbAssunto
                    SET classe = %s, texto = %s, html = %s, img = %s
                    WHERE assunto = %s
                    """
                    cursor.execute(query, (classe, text_content, html_content, rtf_content, assunto))
                else:
                    query = """
                    UPDATE tbAssunto
                    SET classe = %s, texto = %s, html = %s
                    WHERE assunto = %s
                    """
                    cursor.execute(query, (classe, text_content, html_content, assunto))
            else:
                # Inserir novo registro
                if rtf_content:
                    query = """
                    INSERT INTO tbAssunto (assunto, classe, texto, html, img)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(query, (assunto.upper(), classe.upper(), text_content, html_content, rtf_content))
                else:
                    query = """
                    INSERT INTO tbAssunto (assunto, classe, texto, html)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(query, (assunto, classe, text_content, html_content))
            
            # Commit e fechar cursor
            conn.commit()
            cursor.close()
            
            return True, "Documento salvo com sucesso!"
            
        except Exception as e:
            logger.error(f"Erro ao salvar documento: {str(e)}")
            if 'conn' in locals() and conn:
                conn.rollback()
            if 'cursor' in locals() and cursor:
                cursor.close()
            return False, f"Erro ao salvar documento: {str(e)}"

    def delete_document(self, assunto):
        """Exclui um documento do banco de dados"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Verificar se o documento existe
            cursor.execute("SELECT id FROM tbAssunto WHERE assunto = %s", (assunto,))
            if not cursor.fetchone():
                cursor.close()
                return False, "Documento não encontrado para exclusão"
            
            # Excluir o documento
            cursor.execute("DELETE FROM tbAssunto WHERE assunto = %s", (assunto,))
            conn.commit()
            cursor.close()
            
            return True, "Documento excluído com sucesso!"
            
        except Exception as e:
            logger.error(f"Erro ao excluir documento: {str(e)}")
            if 'conn' in locals() and conn:
                conn.rollback()
            if 'cursor' in locals() and cursor:
                cursor.close()
            return False, f"Erro ao excluir documento: {str(e)}"

    def get_document(self, assunto):
        """Recupera um documento do banco de dados"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, assunto, classe, html, img FROM tbAssunto 
                WHERE assunto = %s
            ''', (assunto,))
            result = cursor.fetchone()
            cursor.close()
            
            return (result, None) if result else (None, "Documento não encontrado")
            
        except Exception as e:
            logger.error(f"Erro ao buscar documento: {str(e)}")
            if 'cursor' in locals() and cursor:
                cursor.close()
            return None, f"Erro ao buscar documento: {str(e)}"

    def search_documents(self, search_text, search_mode):
        """Pesquisa documentos no banco de dados"""
        try:
            if search_mode == "Multi-termo":
                return self.search_documents_multi_terms(search_text)
                
            # Usando lower() para busca case-insensitive
            search_param = f"%{search_text.lower()}%"
            
            conn = self.connect()
            cursor = conn.cursor()
            
            if search_mode == "Assunto":
                query = """
                SELECT assunto, classe, substring(html, 1, 200) as preview
                FROM tbAssunto
                WHERE lower(assunto) LIKE %s
                ORDER BY classe, assunto
                LIMIT 50
                """
            elif search_mode == "Classe":
                query = """
                SELECT assunto, classe, substring(html, 1, 200) as preview
                FROM tbAssunto
                WHERE lower(classe) LIKE %s
                ORDER BY classe, assunto
                LIMIT 50
                """
            else:  # Conteúdo
                query = """
                SELECT assunto, classe, substring(html, 1, 200) as preview
                FROM tbAssunto
                WHERE lower(html) LIKE %s
                ORDER BY classe, assunto
                LIMIT 50
                """
            
            cursor.execute(query, (search_param,))
            results = cursor.fetchall()
            cursor.close()
            
            return results if results else [], None
            
        except Exception as e:
            logger.error(f"Erro na pesquisa: {str(e)}")
            if 'cursor' in locals() and cursor:
                cursor.close()
            return [], f"Erro na pesquisa: {str(e)}"

    def search_documents_multi_terms(self, search_text):
        """Pesquisa documentos usando múltiplos termos"""
        try:
            terms = [term.strip() for term in search_text.split(',')]
            terms = [term for term in terms if term]
            
            conn = self.connect()
            cursor = conn.cursor()
            
            query = """
                SELECT assunto, classe, substring(html, 1, 200) as preview
                FROM tbAssunto
                WHERE 1=1
            """
            for _ in terms:
                query += " AND lower(html) LIKE %s"
            query += " ORDER BY classe, assunto LIMIT 50"

            params = [f"%{term.lower()}%" for term in terms]
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            return results, None
            
        except Exception as e:
            logger.error(f"Erro na pesquisa multi-termos: {str(e)}")
            if 'cursor' in locals() and cursor:
                cursor.close()
            return [], f"Erro na pesquisa multi-termos: {str(e)}"

# Função para testar a conexão
def test_database_connection():
    try:
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        cursor.close()
        print(f"Conexão PostgreSQL bem-sucedida. Versão: {version[0]}")
        return True
    except Exception as e:
        print(f"Erro ao conectar ao PostgreSQL: {str(e)}")
        return False

# Executar teste quando executado diretamente
if __name__ == "__main__":
    test_database_connection()
