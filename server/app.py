# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_PORT = int(os.getenv("API_PORT", 8080))
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")
from modules import auth
from modules.medical_pipeline import MedicalConversationalPipeline
from utils.jwt_auth import token_required
from utils.mongo import (
    save_chat_history,
    get_user_conversations,
    get_conversation_messages
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app)

# Initialize Medical Pipeline
logger.info("Initializing Medical Conversational Pipeline...")
try:
    medical_pipeline = MedicalConversationalPipeline()
    logger.info("Medical Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Medical Pipeline: {str(e)}")
    medical_pipeline = None

# ----------------------------
# Routes
# ----------------------------

@app.route("/signup", methods=["POST"])
def signup():
    return auth.signup(request)


@app.route("/login", methods=["POST"])
def login():
    return auth.login(request)


@app.route("/api/v1/chat", methods=["POST"])
@token_required
def chat(current_user):
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Get conversation context for better responses
    conversation_id = data.get("conversation_id")
    conversation_context = []
    if conversation_id:
        try:
            messages = get_conversation_messages(conversation_id)
            conversation_context = [
                {"role": msg["sender"], "content": msg["text"]}
                for msg in messages[-5:]  # Last 5 messages for context
            ]
        except Exception as e:
            logger.warning(f"Failed to load conversation context: {str(e)}")

    # Process medical query through pipeline
    if medical_pipeline:
        try:
            logger.info(f"Processing medical query from user {current_user.get('email', 'unknown')}")
            result = medical_pipeline.process_medical_query(query, conversation_context)
            
            answer = result.response
            
            # Add metadata for debugging (optional)
            metadata = {
                "entities_found": len(result.entities),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "warnings": result.warnings
            }
            
            if result.warnings:
                logger.warning(f"Medical query warnings: {result.warnings}")
            
        except Exception as e:
            logger.error(f"Medical pipeline error: {str(e)}")
            answer = "I apologize, but I'm experiencing technical difficulties. Please try again in a moment or consult with a healthcare professional for urgent medical matters."
            metadata = {"error": str(e)}
    else:
        # Fallback if medical pipeline failed to initialize
        answer = "Medical AI services are currently initializing. Please ensure all required services (Neo4j, ChromaDB) are running and try again."
        metadata = {"pipeline_status": "unavailable"}

    # Save chat and return updated conversation ID
    conversation_id = save_chat_history(
        user_id=current_user["_id"],
        query=query,
        response=answer,
        conversation_id=conversation_id
    )

    response_data = {
        "response": answer,
        "conversation_id": conversation_id
    }
    
    # Include metadata in development mode
    if app.debug:
        response_data["metadata"] = metadata

    return jsonify(response_data)


@app.route("/conversations", methods=["GET"])
@token_required
def conversations(current_user):
    conversations = get_user_conversations(current_user["_id"])
    return jsonify({"conversations": conversations})


@app.route("/conversation/<convo_id>", methods=["GET"])
@token_required
def conversation_messages(current_user, convo_id):
    messages = get_conversation_messages(convo_id)

    print('Calling this function', current_user)
    print('Messages', messages)

    chat_history = [
        {
            "sender": msg["sender"],
            "text": msg["text"],
            "timestamp": msg["timestamp"]
        }
        for msg in messages
    ]
    return jsonify({"messages": chat_history})


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """System health check endpoint"""
    if medical_pipeline:
        health_status = medical_pipeline.get_system_health()
        health_status["api_status"] = "healthy"
    else:
        health_status = {
            "api_status": "degraded",
            "medical_pipeline": "unavailable",
            "message": "Medical pipeline failed to initialize"
        }
    
    return jsonify(health_status)


@app.route("/api/v1/entities", methods=["POST"])
@token_required
def extract_entities(current_user):
    """Extract medical entities from text"""
    data = request.get_json()
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Empty text"}), 400
    
    if medical_pipeline and medical_pipeline.components_loaded.get("ned", False):
        try:
            entities = medical_pipeline._extract_entities(text)
            return jsonify({
                "entities": entities,
                "count": len(entities)
            })
        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")
            return jsonify({"error": "Entity extraction failed"}), 500
    else:
        return jsonify({"error": "Entity extraction service unavailable"}), 503


@app.route("/api/v1/search", methods=["POST"])
@token_required
def search_documents(current_user):
    """Search medical documents"""
    data = request.get_json()
    query = data.get("query", "").strip()
    max_results = data.get("max_results", 5)
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    if medical_pipeline and medical_pipeline.components_loaded.get("rag", False):
        try:
            results = medical_pipeline._retrieve_context(query, [])
            return jsonify({
                "documents": results[:max_results],
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return jsonify({"error": "Document search failed"}), 500
    else:
        return jsonify({"error": "Document search service unavailable"}), 503


@app.route("/api/v1/predict", methods=["POST"])
@token_required
def predict_diseases(current_user):
    """Predict diseases based on symptoms"""
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    
    if medical_pipeline and medical_pipeline.components_loaded.get("neuro_symbolic", False):
        try:
            # Create mock entities for prediction
            mock_entities = [{"concept_name": symptom, "concept_type": "SYMPTOM"} for symptom in symptoms]
            predictions = medical_pipeline._predict_diseases(mock_entities, {})
            return jsonify({
                "predictions": predictions,
                "symptoms_analyzed": symptoms
            })
        except Exception as e:
            logger.error(f"Disease prediction error: {str(e)}")
            return jsonify({"error": "Disease prediction failed"}), 500
    else:
        return jsonify({"error": "Disease prediction service unavailable"}), 503


@app.route("/api/v1/metrics", methods=["GET"])
@token_required
def get_metrics(current_user):
    """Get system performance metrics"""
    if medical_pipeline and hasattr(medical_pipeline, 'metrics_collector'):
        try:
            hours = int(request.args.get('hours', 1))
            
            metrics = {
                "performance": medical_pipeline.metrics_collector.get_performance_summary(hours=hours),
                "medical_accuracy": medical_pipeline.metrics_collector.get_medical_accuracy_summary(hours=hours),
                "errors": medical_pipeline.metrics_collector.get_error_summary(hours=hours),
                "cache_stats": medical_pipeline.cache_manager.get_cache_stats() if hasattr(medical_pipeline, 'cache_manager') else {}
            }
            
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Metrics retrieval error: {str(e)}")
            return jsonify({"error": "Failed to retrieve metrics"}), 500
    else:
        return jsonify({"error": "Metrics service unavailable"}), 503


@app.route("/api/v1/analytics/dashboard", methods=["GET"])
@token_required
def get_dashboard_data(current_user):
    """Get analytics dashboard data"""
    if medical_pipeline and hasattr(medical_pipeline, 'metrics_collector'):
        try:
            from modules.monitoring_system import MedicalAnalyticsDashboard
            
            dashboard = MedicalAnalyticsDashboard(medical_pipeline.metrics_collector)
            dashboard_data = dashboard.generate_dashboard_data()
            
            return jsonify(dashboard_data)
        except Exception as e:
            logger.error(f"Dashboard data error: {str(e)}")
            return jsonify({"error": "Failed to generate dashboard data"}), 500
    else:
        return jsonify({"error": "Analytics service unavailable"}), 503


@app.route("/api/v1/cache/invalidate", methods=["POST"])
@token_required
def invalidate_cache(current_user):
    """Invalidate cache entries"""
    if medical_pipeline and hasattr(medical_pipeline, 'cache_manager'):
        try:
            data = request.get_json()
            component = data.get('component')
            pattern = data.get('pattern')
            
            deleted_count = medical_pipeline.cache_manager.invalidate_cache(component, pattern)
            
            return jsonify({
                "message": f"Invalidated {deleted_count} cache entries",
                "deleted_count": deleted_count
            })
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return jsonify({"error": "Cache invalidation failed"}), 500
    else:
        return jsonify({"error": "Cache service unavailable"}), 503


@app.route("/api/v1/llm/info", methods=["GET"])
@token_required
def get_llm_info(current_user):
    """Get LLM model information"""
    if medical_pipeline and hasattr(medical_pipeline, 'llm_interface'):
        try:
            llm_info = medical_pipeline.llm_interface.get_model_info()
            return jsonify(llm_info)
        except Exception as e:
            logger.error(f"LLM info error: {str(e)}")
            return jsonify({"error": "Failed to get LLM info"}), 500
    else:
        return jsonify({"error": "LLM service unavailable"}), 503


@app.route("/api/v1/conversation/context", methods=["GET"])
@token_required
def get_conversation_context(current_user):
    """Get conversation context information"""
    if medical_pipeline and hasattr(medical_pipeline, 'conversation_manager'):
        try:
            context_summary = medical_pipeline.conversation_manager.get_context_summary()
            context_length = len(medical_pipeline.conversation_manager.context_history)
            
            return jsonify({
                "context_summary": context_summary,
                "conversation_length": context_length,
                "entity_timeline": medical_pipeline.conversation_manager.entity_timeline,
                "topic_tracking": medical_pipeline.conversation_manager.topic_tracking
            })
        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return jsonify({"error": "Failed to get conversation context"}), 500
    else:
        return jsonify({"error": "Conversation service unavailable"}), 503


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=API_PORT)
