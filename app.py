# from flask import Flask
# from flask_pymongo import PyMongo
from flask import Flask
from flask_pymongo

from flask_login import LoginManager

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "your_secure_key_here_change_me"  # Update this!
    app.config["MONGO_URI"] = "mongodb://localhost:27017/pothole_db"

    # Initialize after app config
    mongo = PyMongo(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"  # Fixed blueprint prefix

    # Now safe to import blueprints
    from auth import auth_bp
    from detection_routes import main_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
