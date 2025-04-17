import psycopg2
import psycopg2.extras
import sqlite3
import datetime
import json
import numpy as np
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path

# Tải biến môi trường
load_dotenv()

# Đường dẫn đến file SQLite
SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_analytics.db")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Biến global để theo dõi loại database đang sử dụng
DB_TYPE = "sqlite"  # hoặc "postgres"

# Database setup
def init_db():
    """
    Khởi tạo kết nối đến database và tạo bảng nếu chưa tồn tại
    """
    global DB_TYPE
    
    try:
        # Thử kết nối PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "sentiment_analysis"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "your_password")
        )
        
        print("Kết nối PostgreSQL thành công!")
        DB_TYPE = "postgres"
        
        with conn.cursor() as cur:
            print("Đang tạo bảng...")
            # Sử dụng cú pháp PostgreSQL (SERIAL thay vì AUTOINCREMENT)
            cur.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                language TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                entities JSONB,
                keywords JSONB,
                source TEXT,
                fetch_date TIMESTAMP
            )
            ''')
            
            cur.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES articles(id),
                model_used TEXT,
                analysis_type TEXT,
                analysis_result JSONB,
                analysis_date TIMESTAMP
            )
            ''')
            
            cur.execute('''
            CREATE TABLE IF NOT EXISTS model_evaluations (
                id SERIAL PRIMARY KEY,
                model_name TEXT,
                model_type TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                user_rating INTEGER,
                user_feedback TEXT,
                evaluation_date TIMESTAMP
            )
            ''')
            
            # Tạo các index
            cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment_label)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)')
            
            conn.commit()
            print("Đã tạo các bảng PostgreSQL thành công!")
            
        return conn
    except Exception as e:
        print(f"Không thể kết nối PostgreSQL: {e}")
        print("Fallback sang SQLite...")
        DB_TYPE = "sqlite"
        
        # Tạo kết nối SQLite
        conn = sqlite3.connect(SQLITE_PATH)
        # Cho phép truy cập cột bằng tên
        conn.row_factory = sqlite3.Row
        
        c = conn.cursor()
        
        # Tạo bảng articles với cú pháp SQLite
        c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            content TEXT,
            language TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            entities TEXT,
            keywords TEXT,
            source TEXT,
            fetch_date TIMESTAMP
        )
        ''')
        
        # Tạo bảng analysis_history với cú pháp SQLite
        c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            model_used TEXT,
            analysis_type TEXT,
            analysis_result TEXT,
            analysis_date TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES articles (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_type TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            user_rating INTEGER,
            user_feedback TEXT,
            evaluation_date TIMESTAMP
        )
        ''')
        
        # Tạo các index
        c.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment_label)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_history_article_id ON analysis_history(article_id)')
        
        conn.commit()
        print("Đã tạo các bảng SQLite thành công!")
        
        return conn

# Save analysis results to database
def save_to_db(conn, article_data, analysis_results):
    """
    Lưu kết quả phân tích vào cơ sở dữ liệu
    """
    if DB_TYPE == "postgres":
        return save_to_db_postgres(conn, article_data, analysis_results)
    else:
        return save_to_db_sqlite(conn, article_data, analysis_results)

def save_to_db_postgres(conn, article_data, analysis_results):
    """Lưu kết quả phân tích vào PostgreSQL"""
    try:
        with conn.cursor() as cur:
            # Kiểm tra URL đã tồn tại chưa
            cur.execute("SELECT id FROM articles WHERE url = %s", (article_data['url'],))
            existing = cur.fetchone()
            
            if existing:
                article_id = existing[0]
                # Cập nhật bản ghi hiện có
                cur.execute("""
                    UPDATE articles 
                    SET sentiment_score = %s, sentiment_label = %s, entities = %s, keywords = %s
                    WHERE id = %s
                """, (
                    float(analysis_results['overall_sentiment'].get('score', 0)), 
                    analysis_results['overall_sentiment'].get('label', 'NEUTRAL'),
                    json.dumps(analysis_results.get('entities', []), cls=NumpyEncoder),
                    json.dumps(analysis_results.get('keywords', []), cls=NumpyEncoder),
                    article_id
                ))
            else:
                # Thêm bản ghi mới
                cur.execute("""
                    INSERT INTO articles (url, title, content, language, sentiment_score, sentiment_label, entities, keywords, source, fetch_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    article_data['url'],
                    article_data['title'],
                    article_data['content'],
                    article_data['language'],
                    float(analysis_results['overall_sentiment'].get('score', 0)),
                    analysis_results['overall_sentiment'].get('label', 'NEUTRAL'),
                    json.dumps(analysis_results.get('entities', []), cls=NumpyEncoder),
                    json.dumps(analysis_results.get('keywords', []), cls=NumpyEncoder),
                    article_data['source'],
                    datetime.datetime.now()
                ))
                article_id = cur.fetchone()[0]
            
            # Lưu lịch sử phân tích
            cur.execute("""
                INSERT INTO analysis_history (article_id, model_used, analysis_type, analysis_result, analysis_date)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                article_id,
                analysis_results['model_used'],
                'sentiment',
                json.dumps(analysis_results['sentence_sentiments'], cls=NumpyEncoder),
                datetime.datetime.now()
            ))
            
            conn.commit()
            return article_id
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
        return None

def save_to_db_sqlite(conn, article_data, analysis_results):
    """Lưu kết quả phân tích vào SQLite"""
    try:
        c = conn.cursor()
        # Kiểm tra URL đã tồn tại chưa
        c.execute("SELECT id FROM articles WHERE url = ?", (article_data['url'],))
        existing = c.fetchone()
        
        if existing:
            article_id = existing[0]
            # Cập nhật bản ghi hiện có
            c.execute("""
                UPDATE articles 
                SET sentiment_score = ?, sentiment_label = ?, entities = ?, keywords = ?
                WHERE id = ?
            """, (
                float(analysis_results['overall_sentiment'].get('score', 0)), 
                analysis_results['overall_sentiment'].get('label', 'NEUTRAL'),
                json.dumps(analysis_results.get('entities', []), cls=NumpyEncoder),
                json.dumps(analysis_results.get('keywords', []), cls=NumpyEncoder),
                article_id
            ))
        else:
            # Thêm bản ghi mới
            c.execute("""
                INSERT INTO articles (url, title, content, language, sentiment_score, sentiment_label, entities, keywords, source, fetch_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_data['url'],
                article_data['title'],
                article_data['content'],
                article_data['language'],
                float(analysis_results['overall_sentiment'].get('score', 0)),
                analysis_results['overall_sentiment'].get('label', 'NEUTRAL'),
                json.dumps(analysis_results.get('entities', []), cls=NumpyEncoder),
                json.dumps(analysis_results.get('keywords', []), cls=NumpyEncoder),
                article_data['source'],
                datetime.datetime.now().isoformat()
            ))
            article_id = c.lastrowid
        
        # Lưu lịch sử phân tích
        c.execute("""
            INSERT INTO analysis_history (article_id, model_used, analysis_type, analysis_result, analysis_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            article_id,
            analysis_results['model_used'],
            'sentiment',
            json.dumps(analysis_results['sentence_sentiments'], cls=NumpyEncoder),
            datetime.datetime.now().isoformat()
        ))
        
        conn.commit()
        return article_id
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
        return None

# Get analysis history from database
def get_analysis_history(conn, limit=10):
    """Lấy lịch sử phân tích từ cơ sở dữ liệu"""
    if DB_TYPE == "postgres":
        return get_analysis_history_postgres(conn, limit)
    else:
        return get_analysis_history_sqlite(conn, limit)

def get_analysis_history_postgres(conn, limit=10):
    """Lấy lịch sử phân tích từ PostgreSQL"""
    try:
        import pandas as pd
        query = """
            SELECT a.id, a.url, a.title, a.language, a.sentiment_score, a.sentiment_label, a.source, a.fetch_date
            FROM articles a
            ORDER BY a.fetch_date DESC
            LIMIT %s
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    except Exception as e:
        print(f"Error fetching history from Postgres: {e}")
        return None

def get_analysis_history_sqlite(conn, limit=10):
    """Lấy lịch sử phân tích từ SQLite"""
    try:
        import pandas as pd
        query = """
            SELECT a.id, a.url, a.title, a.language, a.sentiment_score, a.sentiment_label, a.source, a.fetch_date
            FROM articles a
            ORDER BY a.fetch_date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    except Exception as e:
        print(f"Error fetching history from SQLite: {e}")
        return None

# Hàm mới cho phần Settings
def check_database_exists():
    """Kiểm tra database file có tồn tại không"""
    return os.path.exists(SQLITE_PATH)

def get_database_info(conn):
    """Lấy thông tin về database"""
    db_info = {
        "Type": DB_TYPE,
        "Path": SQLITE_PATH if DB_TYPE == "sqlite" else "PostgreSQL connection"
    }
    
    try:
        # Đếm số bài viết và phân tích
        if DB_TYPE == "postgres":
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM articles")
                article_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM analysis_history")
                analysis_count = cur.fetchone()[0]
        else:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM articles")
            article_count = c.fetchone()[0]
            
            c.execute("SELECT COUNT(*) FROM analysis_history")
            analysis_count = c.fetchone()[0]
            
        db_info["Articles"] = article_count
        db_info["Analysis Records"] = analysis_count
        
    except Exception as e:
        print(f"Error getting database info: {e}")
        db_info["Error"] = str(e)
    
    return db_info

# Backup database
def backup_database():
    """Tạo bản sao lưu của database SQLite"""
    if DB_TYPE != "sqlite":
        return False
    
    try:
        backup_dir = os.path.join(os.path.dirname(SQLITE_PATH), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"news_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        
        if os.path.exists(SQLITE_PATH):
            shutil.copy2(SQLITE_PATH, backup_file)
            return True
        return False
    except Exception as e:
        print(f"Backup error: {e}")
        return False

# Clear database
def clear_database(conn, confirm=False):
    """Xóa tất cả dữ liệu trong database"""
    if not confirm:
        return False
    
    try:
        if DB_TYPE == "postgres":
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analysis_history")
                cur.execute("DELETE FROM articles")
                conn.commit()
        else:
            c = conn.cursor()
            c.execute("DELETE FROM analysis_history")
            c.execute("DELETE FROM articles")
            conn.commit()
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        conn.rollback()
        return False

def get_database_path():
    """
    Trả về đường dẫn đến file database
    
    Returns:
        Path: Đường dẫn đến file database
    """
    return SQLITE_PATH 

# Thêm đánh giá mô hình
def save_model_evaluation(conn, evaluation_data):
    """
    Lưu đánh giá mô hình vào database
    
    Args:
        conn: Kết nối database
        evaluation_data (dict): Dữ liệu đánh giá mô hình
        
    Returns:
        int: ID của đánh giá
    """
    if DB_TYPE == "postgres":
        return save_model_evaluation_postgres(conn, evaluation_data)
    else:
        return save_model_evaluation_sqlite(conn, evaluation_data)

def save_model_evaluation_postgres(conn, evaluation_data):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_evaluations (
                    model_name, model_type, accuracy, precision, recall, 
                    f1_score, user_rating, user_feedback, evaluation_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                evaluation_data.get('model_name', ''),
                evaluation_data.get('model_type', ''),
                float(evaluation_data.get('accuracy', 0)),
                float(evaluation_data.get('precision', 0)),
                float(evaluation_data.get('recall', 0)),
                float(evaluation_data.get('f1_score', 0)),
                int(evaluation_data.get('user_rating', 0)),
                evaluation_data.get('user_feedback', ''),
                datetime.datetime.now()
            ))
            
            evaluation_id = cur.fetchone()[0]
            conn.commit()
            return evaluation_id
    except Exception as e:
        print(f"Error saving model evaluation: {e}")
        conn.rollback()
        return None

def save_model_evaluation_sqlite(conn, evaluation_data):
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO model_evaluations (
                model_name, model_type, accuracy, precision, recall, 
                f1_score, user_rating, user_feedback, evaluation_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_data.get('model_name', ''),
            evaluation_data.get('model_type', ''),
            float(evaluation_data.get('accuracy', 0)),
            float(evaluation_data.get('precision', 0)),
            float(evaluation_data.get('recall', 0)),
            float(evaluation_data.get('f1_score', 0)),
            int(evaluation_data.get('user_rating', 0)),
            evaluation_data.get('user_feedback', ''),
            datetime.datetime.now().isoformat()
        ))
        
        evaluation_id = c.lastrowid
        conn.commit()
        return evaluation_id
    except Exception as e:
        print(f"Error saving model evaluation: {e}")
        conn.rollback()
        return None

# Lấy đánh giá mô hình
def get_model_evaluations(conn, model_type=None):
    """
    Lấy dữ liệu đánh giá mô hình từ database
    
    Args:
        conn: Kết nối database
        model_type (str, optional): Loại mô hình cần lọc
        
    Returns:
        DataFrame: Dữ liệu đánh giá mô hình
    """
    try:
        import pandas as pd
        
        if DB_TYPE == "postgres":
            if model_type:
                query = """
                    SELECT * FROM model_evaluations
                    WHERE model_type = %s
                    ORDER BY evaluation_date DESC
                """
                df = pd.read_sql_query(query, conn, params=(model_type,))
            else:
                query = """
                    SELECT * FROM model_evaluations
                    ORDER BY evaluation_date DESC
                """
                df = pd.read_sql_query(query, conn)
        else:
            if model_type:
                query = """
                    SELECT * FROM model_evaluations
                    WHERE model_type = ?
                    ORDER BY evaluation_date DESC
                """
                df = pd.read_sql_query(query, conn, params=(model_type,))
            else:
                query = """
                    SELECT * FROM model_evaluations
                    ORDER BY evaluation_date DESC
                """
                df = pd.read_sql_query(query, conn)
                
        return df
    except Exception as e:
        print(f"Error getting model evaluations: {e}")
        return None

# Lấy thống kê đánh giá mô hình
def get_model_evaluation_stats(conn):
    """
    Lấy thống kê đánh giá mô hình
    
    Args:
        conn: Kết nối database
        
    Returns:
        dict: Thống kê đánh giá mô hình
    """
    try:
        import pandas as pd
        
        if DB_TYPE == "postgres":
            query = """
                SELECT 
                    model_type,
                    AVG(accuracy) as avg_accuracy,
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1_score,
                    AVG(user_rating) as avg_user_rating,
                    COUNT(*) as evaluation_count
                FROM model_evaluations
                GROUP BY model_type
            """
        else:
            query = """
                SELECT 
                    model_type,
                    AVG(accuracy) as avg_accuracy,
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1_score,
                    AVG(user_rating) as avg_user_rating,
                    COUNT(*) as evaluation_count
                FROM model_evaluations
                GROUP BY model_type
            """
            
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error getting model evaluation stats: {e}")
        return None

def get_article_by_id(conn, article_id):
    """
    Lấy thông tin chi tiết của một bài viết theo ID
    
    Args:
        conn: Kết nối database
        article_id (int): ID của bài viết
        
    Returns:
        dict: Thông tin bài viết hoặc None nếu không tìm thấy
    """
    try:
        if DB_TYPE == "postgres":
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
                article = cur.fetchone()
                if article:
                    return dict(article)
        else:
            c = conn.cursor()
            c.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
            article = c.fetchone()
            if article:
                return dict(zip([column[0] for column in c.description], article))
        return None
    except Exception as e:
        print(f"Error retrieving article: {e}")
        return None

def get_analysis_result(conn, article_id, analysis_type='sentiment'):
    """
    Lấy kết quả phân tích gần nhất của một bài viết
    
    Args:
        conn: Kết nối database
        article_id (int): ID của bài viết
        analysis_type (str): Loại phân tích (sentiment, entity, etc.)
        
    Returns:
        dict or list: Kết quả phân tích hoặc None nếu không tìm thấy
    """
    try:
        if DB_TYPE == "postgres":
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT analysis_result FROM analysis_history 
                    WHERE article_id = %s AND analysis_type = %s 
                    ORDER BY analysis_date DESC LIMIT 1
                """, (article_id, analysis_type))
                result = cur.fetchone()
                if result:
                    # Kiểm tra xem result[0] đã là object hay vẫn là chuỗi JSON
                    if isinstance(result[0], (dict, list)):
                        return result[0]  # Đã là object Python
                    else:
                        return json.loads(result[0])  # Cần chuyển từ chuỗi JSON
        else:
            c = conn.cursor()
            c.execute("""
                SELECT analysis_result FROM analysis_history 
                WHERE article_id = ? AND analysis_type = ? 
                ORDER BY analysis_date DESC LIMIT 1
            """, (article_id, analysis_type))
            result = c.fetchone()
            if result:
                try:
                    return json.loads(result[0])  # Thử chuyển từ chuỗi JSON
                except (TypeError, json.JSONDecodeError):
                    # Nếu không phải chuỗi JSON hoặc không thể decode
                    return result[0]
        return None
    except Exception as e:
        print(f"Error retrieving analysis result: {e}")
        return None 

def serialize_json(data):
    """Chuyển đổi dữ liệu Python thành chuỗi JSON để lưu vào database"""
    if data is None:
        return None
    return json.dumps(data, cls=NumpyEncoder)

def deserialize_json(json_data):
    """Chuyển đổi chuỗi JSON từ database thành đối tượng Python"""
    if not json_data:
        return None
    
    if isinstance(json_data, (dict, list)):
        return json_data
    
    try:
        return json.loads(json_data)
    except (TypeError, json.JSONDecodeError) as e:
        print(f"JSON deserialization error: {e}")
        return json_data 