import os
from flask import Flask
from server.routes import register_routes
from flask_session import Session

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'server', 'templates'))
app.secret_key = os.urandom(24)

# --- SERVER SESSION CONFIGURATION ---
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session/"
Session(app)
# -------------------------------------------

register_routes(app)

if __name__ == '__main__':
    if not os.path.exists("./flask_session/"):
        os.makedirs("./flask_session/")

    app.run(debug=True, host='127.0.0.1', port=5000)