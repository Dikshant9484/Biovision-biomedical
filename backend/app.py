"""
BioVision AI - Flask Backend Entry Point
"""
import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from routes.breast import breast_bp
from routes.skin import skin_bp
from routes.blood import blood_bp
from routes.ecg import ecg_bp
from routes.lung import lung_bp
from routes.xray import xray_bp
from routes.universal import universal_bp
from routes.chat import chat_bp
from routes.health import health_bp


def create_app():
    app = Flask(__name__)

    # Read allowed origins from environment
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")

    # Enable CORS properly for all API routes
    CORS(
        app,
        resources={r"/api/*": {"origins": allowed_origins}},
        supports_credentials=False,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Type"],
    )

    # Extra manual CORS headers for file upload / preflight reliability
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = allowed_origins
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    # Flask config
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "biovision-dev-key")

    # Root route (so Render / browser root check doesn't show 404)
    @app.route("/")
    def home():
        return {
            "message": "BioVision AI Backend is running",
            "status": "ok"
        }

    # Register Blueprints
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(breast_bp, url_prefix="/api/predict")
    app.register_blueprint(skin_bp, url_prefix="/api/predict")
    app.register_blueprint(blood_bp, url_prefix="/api/predict")
    app.register_blueprint(ecg_bp, url_prefix="/api/predict")
    app.register_blueprint(lung_bp, url_prefix="/api/predict")
    app.register_blueprint(xray_bp, url_prefix="/api/predict")
    app.register_blueprint(universal_bp, url_prefix="/api/predict")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)