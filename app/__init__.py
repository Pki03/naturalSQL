from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Import routes here, after app initialization to avoid circular imports
    from .routes import home  
    app.register_blueprint(home)

    return app
