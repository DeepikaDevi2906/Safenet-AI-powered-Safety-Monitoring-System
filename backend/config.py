# config.py

class Config:
    SECRET_KEY = "safenet_secret_key"  # Used for sessions, CSRF protection (keep private)
    SQLALCHEMY_DATABASE_URI = "sqlite:///safenet.db"  # SQLite database file
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disable event system to save memory
