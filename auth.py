# from flask import Blueprint, render_template, request, redirect, url_for, flash
# from flask_login import UserMixin, login_user, logout_user, login_required, current_user
# from werkzeug.security import generate_password_hash, check_password_hash
# from bson.objectid import ObjectId
# from datetime import datetime
# from app import mongo, login_manager

# auth_bp = Blueprint("auth", __name__)

# class User(UserMixin):
#     def __init__(self, user_doc):
#         self.id = str(user_doc["_id"])
#         self.name = user_doc["name"]
#         self.email = user_doc["email"]
#         self.role = user_doc.get("role", "user")

# @login_manager.user_loader
# def load_user(user_id):
#     doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
#     return User(doc) if doc else None

# @auth_bp.route("/register", methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#         name = request.form["name"]
#         email = request.form["email"]
#         password = request.form["password"]

#         if mongo.db.users.find_one({"email": email}):
#             flash("Email already registered")
#             return redirect(url_for("auth.register"))

#         user_doc = {
#             "name": name,
#             "email": email,
#             "password_hash": generate_password_hash(password),
#             "role": "user",
#             "created_at": datetime.utcnow(),
#         }
#         result = mongo.db.users.insert_one(user_doc)
#         user_doc["_id"] = result.inserted_id
#         login_user(User(user_doc))
#         return redirect(url_for("main.dashboard"))
#     return render_template("register.html")

# @auth_bp.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form["email"]
#         password = request.form["password"]
#         doc = mongo.db.users.find_one({"email": email})
#         if doc and check_password_hash(doc["password_hash"], password):
#             login_user(User(doc))
#             return redirect(url_for("main.dashboard"))
#         flash("Invalid email or password")
#     return render_template("login.html")

# @auth_bp.route("/logout")
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for("auth.login"))


from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime

auth_bp = Blueprint("auth", __name__)

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])
        self.name = user_doc["name"]
        self.email = user_doc["email"]
        self.role = user_doc.get("role", "user")

# This will be set by app.py after init
def load_user(user_id):
    from app import mongo  # Delayed import inside function
    doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None

@auth_bp.record_once
def on_load(state):
    """Called once when blueprint loads - sets login_manager.user_loader"""
    from app import login_manager
    login_manager.user_loader(load_user)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    from app import mongo  # Lazy import
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        if mongo.db.users.find_one({"email": email}):
            flash("Email already registered")
            return redirect(url_for("auth.register"))
        user_doc = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": "user",
            "created_at": datetime.utcnow(),
        }
        result = mongo.db.users.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        login_user(User(user_doc))
        return redirect(url_for("main.dashboard"))
    return render_template("register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    from app import mongo  # Lazy import
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        doc = mongo.db.users.find_one({"email": email})
        if doc and check_password_hash(doc["password_hash"], password):
            login_user(User(doc))
            return redirect(url_for("main.dashboard"))
        flash("Invalid email or password")
    return render_template("login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
