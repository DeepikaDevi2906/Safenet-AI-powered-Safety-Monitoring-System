from flask import Flask
from flask_cors import CORS
from backend.config import Config
from backend.extensions import db, socketio
from backend.routes import routes_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    app.register_blueprint(routes_bp)
    return app

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
