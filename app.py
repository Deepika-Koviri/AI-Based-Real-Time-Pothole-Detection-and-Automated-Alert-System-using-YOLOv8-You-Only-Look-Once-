from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
from twilio.rest import Client
from flask_socketio import SocketIO
from geopy.distance import geodesic
import threading
import time
import os
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()  # Load .env file

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "your_secure_key_here_change_me"
    app.config["MONGO_URI"] = "mongodb://localhost:27017/pothole_db"

    # Initialize existing extensions
    mongo = PyMongo(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # NEW: Initialize SocketIO and Twilio
    socketio = SocketIO(app, cors_allowed_origins="*")
    app.config['SOCKETIO'] = socketio
    client = Client(os.getenv('TWILIO_SID'), os.getenv('TWILIO_TOKEN'))


    # NEW: Global variables (accessible via app.config)
    app.config['potholes'] = []  # Load from MongoDB later
    app.config['active_users'] = {}
    app.config['twilio_client'] = client

    # NEW: SocketIO event handlers (inside create_app)
    @socketio.on('user_location')
    def handle_location(data):
        sid = request.sid
        app.config['active_users'][sid] = {'lat': data['lat'], 'lon': data['lon']}

    @socketio.on('disconnect')
    def handle_disconnect():
        sid = request.sid
        if sid in app.config['active_users']:
            del app.config['active_users'][sid]

    # NEW: Alert checking function
    def check_alerts():
        while True:
            for sid, pos in list(app.config['active_users'].items()):
                for pothole in app.config['potholes']:
                    dist = geodesic((pos['lat'], pos['lon']), (pothole['lat'], pothole['lon'])).meters
                    if dist <= 100:
                        socketio.emit('pothole_alert', {
                            'msg': f'Pothole {dist:.0f}m ahead!', 
                            'pothole': pothole
                        }, room=sid)
            time.sleep(10)

    # Start alert thread
    threading.Thread(target=check_alerts, daemon=True).start()

    # NEW: SMS function
    def send_sms_alert(to, body):
        client = app.config['twilio_client']
        message = client.messages.create(
            body=body, 
            from_=os.getenv('TWILIO_PHONE'), 
            to=to
        )
        print(f'SMS sent: {message.sid}')
        return message.sid

    # NEW: SMS API route
    @app.route('/api/send_sms', methods=['POST'])
    def api_send_sms():
        from flask import request, jsonify
        data = request.json
        sid = send_sms_alert(data['to'], data['body'])
        return jsonify({'sid': sid})

    # Import and register blueprints (KEEP EXISTING)
    from auth import auth_bp
    from detection_routes import main_bp
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)

    return app, socketio  # Return both!

# Create app and socketio
app, socketio = create_app()

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)  # CHANGED!
