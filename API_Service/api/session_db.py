"""Session storage database"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class SessionDB:
    def __init__(self):
        db_path = Path(__file__).parent.parent.parent / 'logs' / 'sessions.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_query TEXT,
            bot_response TEXT,
            intent TEXT,
            metadata TEXT
        )
        """
        self.conn.execute(sql)
        self.conn.commit()
    
    def log_interaction(self, query: str, response: str, intent: str = None, metadata: dict = None):
        """Log a chat interaction"""
        self.conn.execute(
            "INSERT INTO sessions (timestamp, user_query, bot_response, intent, metadata) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), query, response, intent, json.dumps(metadata or {}))
        )
        self.conn.commit()
    
    def get_recent_sessions(self, limit: int = 10):
        """Get recent chat sessions"""
        cursor = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()

# Global instance
session_db = SessionDB()
