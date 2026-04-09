import os
from flask import Flask
from server.routes import register_routes

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'server', 'templates'))
app.secret_key = os.urandom(24)
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
