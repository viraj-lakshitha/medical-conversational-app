from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "app")

# Initialize MongoDB connection
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

# Collections
users_collection = db["users"]
conversations = db["conversations"]

# ---------------------------
# User Functions
# ---------------------------

def get_user_by_username(username=None, user_id=None, by_id=False):
    if by_id and user_id:
        return users_collection.find_one({"_id": ObjectId(user_id)})
    if username:
        return users_collection.find_one({"username": username})
    return None


def insert_user(username, hashed_password):
    user_data = {
        "username": username,
        "password": hashed_password
    }
    users_collection.insert_one(user_data)


# ---------------------------
# Conversation-based Chat History
# ---------------------------

def save_chat_history(user_id, query, response, conversation_id=None):
    timestamp = datetime.utcnow()

    user_msg = {
        "sender": "user",
        "text": query,
        "timestamp": timestamp
    }
    bot_msg = {
        "sender": "bot",
        "text": response,
        "timestamp": timestamp
    }

    if conversation_id:
        # Append to existing conversation
        conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"messages": {"$each": [user_msg, bot_msg]}}}
        )
    else:
        # Create new conversation
        new_convo = {
            "user_id": ObjectId(user_id),
            "created_at": timestamp,
            "messages": [user_msg, bot_msg]
        }
        conversation_id = conversations.insert_one(new_convo).inserted_id

    return str(conversation_id)


def get_user_conversations(user_id):
    convos = conversations.find({"user_id": ObjectId(user_id)}).sort("created_at", -1)
    return [
        {
            "_id": str(c["_id"]),
            "latest": c["messages"][-1]["text"][:30] if c.get("messages") else "Untitled",
            "created_at": c["created_at"]
        }
        for c in convos
    ]


def get_conversation_messages(convo_id):
    convo = conversations.find_one({"_id": ObjectId(convo_id)})
    if convo:
        return convo["messages"]
    return []
