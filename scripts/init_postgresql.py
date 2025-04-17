import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Kết nối thông tin database từ biến môi trường
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sentiment_analysis")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")

def create_database():
    """Tạo database nếu chưa tồn tại"""
    # Kết nối đến PostgreSQL mặc định
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database="postgres",
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # Kiểm tra database đã tồn tại chưa
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()
        
        if not exists:
            print(f"Creating database '{DB_NAME}'...")
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
    
    conn.close()

def create_tables():
    """Tạo các bảng cần thiết"""
    # Kết nối đến database mới tạo
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    with conn.cursor() as cur:
        # Tạo bảng articles
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
        
        # Tạo bảng analysis_history
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
        
        # Tạo các index
        cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment_label)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_history_article_id ON analysis_history(article_id)')
        
        # Commit các thay đổi
        conn.commit()
        print("Tables created successfully.")
    
    conn.close()

def main():
    """Hàm chính để khởi tạo database"""
    try:
        create_database()
        create_tables()
        print("PostgreSQL database setup completed successfully.")
    except Exception as e:
        print(f"Error setting up PostgreSQL database: {e}")

if __name__ == "__main__":
    main()
