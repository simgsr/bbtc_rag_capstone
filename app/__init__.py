import os
from flask import Flask
from flask_session import Session

def create_app():
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
    app.config['SESSION_TYPE'] = 'filesystem'
    
    # Initialize Session
    Session(app)
    
    # Create necessary directories
    os.makedirs(os.getenv('STORAGE_PATH', 'vectorstore'), exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    with app.app_context():
        from . import routes
        app.register_blueprint(routes.bp)
        
    return app
