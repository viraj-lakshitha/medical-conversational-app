from flask import jsonify
import hashlib
import secrets
from utils.mongo import get_user_by_username, insert_user
from utils.jwt_auth import generate_token

def hash_password(password):
    """Hash password with salt using SHA256"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password, stored_password):
    """Verify password against stored hash"""
    try:
        salt, password_hash = stored_password.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False

def signup(request):
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if get_user_by_username(username):
        return jsonify({"error": "Username already exists"}), 409

    hashed_password = hash_password(password)
    insert_user(username, hashed_password)

    return jsonify({"message": "User created successfully"}), 201


def login(request):
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_token(user["_id"])
    return jsonify({"access_token": token}), 200
