#!/usr/bin/env python3
import os
import json
import time
import uuid
import logging
import subprocess
from pathlib import Path
from threading import Thread, Lock

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

from sgpt.config import cfg
from sgpt.handlers.chat_handler import ChatHandler
from sgpt.handlers.default_handler import DefaultHandler
from sgpt.handlers.repl_handler import ReplHandler
from sgpt.role import DefaultRoles, SystemRole
from sgpt.function import get_openai_schemas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / ".config" / "shell_gpt" / "server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store active REPL sessions
# The key is the repl_id and the value is a dict with session data
repl_sessions = {}
repl_lock = Lock()  # Lock for thread-safe access to repl_sessions

# API key for authentication (should be set via environment variable)
API_KEY = os.environ.get("SGPT_API_KEY", "default-key-change-me")

def authenticate():
    """Simple authentication check."""
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return False
    return True

@app.before_request
def before_request():
    """Check authentication before processing request."""
    # Skip authentication for status endpoint
    if request.path == "/api/v1/status":
        return
    
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401

@app.route("/api/v1/status", methods=["GET"])
def status():
    """Get server status."""
    with repl_lock:
        active_sessions = len(repl_sessions)
    
    return jsonify({
        "status": "running",
        "active_repl_sessions": active_sessions
    })

def create_streaming_handler(handler_class, role_class, md):
    """Create a custom handler that yields streaming tokens."""
    class StreamingHandler(handler_class):
        def handle_streaming(self, **kwargs):
            """Handle request and yield streaming tokens."""
            disable_stream = cfg.get("DISABLE_STREAMING") == "true"
            messages = self.make_messages(kwargs.pop("prompt").strip())
            
            generator = self.get_completion(
                messages=messages,
                **kwargs
            )
            
            full_response = ""
            for token in generator:
                if token:  # Only yield non-empty tokens
                    full_response += token
                    yield token
            
            # Yield the final complete response as a special message
            yield f"\n__SGPT_COMPLETE__{full_response}__SGPT_COMPLETE__"
    
    return StreamingHandler(role_class, md)

@app.route("/api/v1/completion", methods=["POST"])
def completion():
    """Generate a completion for a prompt."""
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", cfg.get("DEFAULT_MODEL"))
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    shell = data.get("shell", False)
    describe_shell = data.get("describe_shell", False)
    code = data.get("code", False)
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    role_name = data.get("role")
    role_class = (
        DefaultRoles.check_get(shell, describe_shell, code)
        if not role_name
        else SystemRole.get(role_name)
    )
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    try:
        if stream:
            # Return streaming response
            def generate():
                handler = create_streaming_handler(DefaultHandler, role_class, md)
                for token in handler.handle_streaming(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    caching=cache,
                    functions=function_schemas,
                ):
                    # Send as Server-Sent Events format
                    if token.startswith("\n__SGPT_COMPLETE__"):
                        # Extract final response and send completion event
                        final_response = token.replace("\n__SGPT_COMPLETE__", "").replace("__SGPT_COMPLETE__", "")
                        yield f"event: complete\ndata: {json.dumps({'completion': final_response, 'model': model, 'role': role_name})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                yield f"event: end\ndata: {{}}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Return regular non-streaming response
            handler = DefaultHandler(role_class, md)
            full_completion = handler.handle(
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                caching=cache,
                functions=function_schemas,
            )
            
            return jsonify({
                "completion": full_completion,
                "model": model,
                "role": role_name,
            })
    except Exception as e:
        logger.exception("Error generating completion")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """Generate a chat completion."""
    data = request.json
    prompt = data.get("prompt", "")
    chat_id = data.get("chat_id")
    model = data.get("model", cfg.get("DEFAULT_MODEL"))
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    role_name = data.get("role")
    role_class = DefaultRoles.DEFAULT.get_role() if not role_name else SystemRole.get(role_name)
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    try:
        if stream:
            # Return streaming response
            def generate():
                handler = create_streaming_handler(ChatHandler, role_class, md)
                handler.chat_id = chat_id  # Set chat_id for ChatHandler
                
                for token in handler.handle_streaming(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    caching=cache,
                    functions=function_schemas,
                    chat_id=chat_id,
                ):
                    # Send as Server-Sent Events format
                    if token.startswith("\n__SGPT_COMPLETE__"):
                        # Extract final response and send completion event
                        final_response = token.replace("\n__SGPT_COMPLETE__", "").replace("__SGPT_COMPLETE__", "")
                        yield f"event: complete\ndata: {json.dumps({'completion': final_response, 'chat_id': chat_id, 'model': model, 'role': role_name})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                yield f"event: end\ndata: {{}}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Return regular non-streaming response
            handler = ChatHandler(chat_id, role_class, md)
            full_completion = handler.handle(
                prompt=prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                caching=cache,
                functions=function_schemas,
            )
            
            return jsonify({
                "completion": full_completion,
                "chat_id": chat_id,
                "model": model,
                "role": role_name,
            })
    except Exception as e:
        logger.exception(f"Error generating chat completion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/repl/start", methods=["POST"])
def repl_start():
    """Start a new REPL session."""
    data = request.json
    repl_id = data.get("repl_id")
    if not repl_id:
        repl_id = str(uuid.uuid4())
    
    init_prompt = data.get("prompt", "")
    model = data.get("model", cfg.get("DEFAULT_MODEL"))
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    role_name = data.get("role")
    role_class = DefaultRoles.DEFAULT.get_role() if not role_name else SystemRole.get(role_name)
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    # Create a new REPL handler
    repl_handler = ReplHandler(repl_id, role_class, md)
    
    # Store in sessions
    with repl_lock:
        repl_sessions[repl_id] = {
            "handler": repl_handler,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "cache": cache,
            "functions": function_schemas,
            "last_activity": time.time(),
        }
    
    # Process initial prompt if provided
    if init_prompt:
        if stream:
            def generate():
                try:
                    streaming_handler = create_streaming_handler(ReplHandler, role_class, md)
                    streaming_handler.chat_id = repl_id  # Set repl_id as chat_id
                    
                    for token in streaming_handler.handle_streaming(
                        prompt=init_prompt,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        caching=cache,
                        functions=function_schemas,
                        chat_id=repl_id,
                    ):
                        if token.startswith("\n__SGPT_COMPLETE__"):
                            final_response = token.replace("\n__SGPT_COMPLETE__", "").replace("__SGPT_COMPLETE__", "")
                            yield f"event: complete\ndata: {json.dumps({'repl_id': repl_id, 'response': final_response, 'model': model})}\n\n"
                        else:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    
                    yield f"event: end\ndata: {{}}\n\n"
                except Exception as e:
                    logger.exception(f"Error processing initial REPL prompt: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            try:
                response = repl_handler.process_single_input(
                    init_prompt,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    caching=cache,
                    functions=function_schemas,
                )
                return jsonify({
                    "repl_id": repl_id,
                    "response": response,
                    "model": model,
                })
            except Exception as e:
                logger.exception(f"Error processing initial REPL prompt: {e}")
                return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "repl_id": repl_id,
        "response": None,
        "model": model,
    })

@app.route("/api/v1/repl/<repl_id>", methods=["POST"])
def repl_process(repl_id):
    """Process input for an existing REPL session."""
    with repl_lock:
        if repl_id not in repl_sessions:
            return jsonify({"error": "REPL session not found"}), 404
        
        session = repl_sessions[repl_id]
        session["last_activity"] = time.time()
    
    data = request.json
    user_input = data.get("input", "")
    stream = data.get("stream", False)
    
    try:
        if stream:
            def generate():
                try:
                    repl_handler = session["handler"]
                    streaming_handler = create_streaming_handler(ReplHandler, repl_handler.role, repl_handler.markdown)
                    streaming_handler.chat_id = repl_id
                    
                    for token in streaming_handler.handle_streaming(
                        prompt=user_input,
                        model=session["model"],
                        temperature=session["temperature"],
                        top_p=session["top_p"],
                        caching=session["cache"],
                        functions=session["functions"],
                        chat_id=repl_id,
                    ):
                        if token.startswith("\n__SGPT_COMPLETE__"):
                            final_response = token.replace("\n__SGPT_COMPLETE__", "").replace("__SGPT_COMPLETE__", "")
                            yield f"event: complete\ndata: {json.dumps({'response': final_response, 'repl_id': repl_id})}\n\n"
                        else:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    
                    yield f"event: end\ndata: {{}}\n\n"
                except Exception as e:
                    logger.exception(f"Error processing REPL input: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Process the input
            repl_handler = session["handler"]
            response = repl_handler.process_single_input(
                user_input,
                model=session["model"],
                temperature=session["temperature"],
                top_p=session["top_p"],
                caching=session["cache"],
                functions=session["functions"],
            )
            
            return jsonify({
                "response": response,
                "repl_id": repl_id,
            })
    except Exception as e:
        logger.exception(f"Error processing REPL input: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/repl/<repl_id>", methods=["DELETE"])
def repl_end(repl_id):
    """End a REPL session."""
    with repl_lock:
        if repl_id in repl_sessions:
            del repl_sessions[repl_id]
    
    return jsonify({"status": "session ended"})

@app.route("/api/v1/chats", methods=["GET"])
def list_chats():
    """List all chat sessions."""
    try:
        chat_ids = ChatHandler.list_ids(standalone=True)
        return jsonify({"chats": chat_ids})
    except Exception as e:
        logger.exception(f"Error listing chats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/chats/<chat_id>", methods=["GET"])
def show_chat(chat_id):
    """Show messages from a specific chat."""
    try:
        messages = ChatHandler.get_messages(chat_id)
        return jsonify({"chat_id": chat_id, "messages": messages})
    except Exception as e:
        logger.exception(f"Error showing chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/roles", methods=["GET"])
def list_roles():
    """List all available roles."""
    try:
        roles = SystemRole.list(standalone=True)
        return jsonify({"roles": roles})
    except Exception as e:
        logger.exception(f"Error listing roles: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/roles/<role_name>", methods=["GET"])
def show_role(role_name):
    """Show a specific role."""
    try:
        role = SystemRole.show(role_name, standalone=True)
        return jsonify({"role": role})
    except Exception as e:
        logger.exception(f"Error showing role: {e}")
        return jsonify({"error": str(e)}), 500

def get_providers_config_path() -> Path:
    """Get the path to the providers configuration file."""
    return Path.home() / ".config" / "shell_gpt" / "providers.json"

def load_providers_config() -> dict:
    """Load providers configuration from JSON file."""
    providers_config_path = get_providers_config_path()
    if not providers_config_path.exists():
        # Create default config if it doesn't exist
        providers_config_path.parent.mkdir(parents=True, exist_ok=True)
        default_config = {
            "local": {
                "type": "ollama",
                "url": "http://localhost:11434"
            }
        }
        with open(providers_config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(providers_config_path, 'r') as f:
        return json.load(f)

@app.route("/api/v1/models", methods=["GET"])
def list_models():
    """List available models from all providers."""
    all_models = request.args.get("all", "false").lower() == "true"
    models = []
    
    try:
        providers = load_providers_config()
        
        # List Ollama models
        for name, details in providers.items():
            if details["type"] == "ollama":
                try:
                    # Set the OLLAMA_HOST environment variable for each provider
                    env = os.environ.copy()
                    env["OLLAMA_HOST"] = details["url"].split("//")[1]
                    
                    output = subprocess.check_output("ollama list", shell=True, env=env).decode()
                    lines = output.split('\n')[1:]
                    ollama_models = sorted([f"{name}/{line.split()[0]}" for line in lines if line])
                    
                    for model in ollama_models:
                        models.append({
                            "provider": name,
                            "name": model.split('/', 1)[1],
                            "full_name": model,
                            "type": "Ollama"
                        })
                except Exception as e:
                    logger.warning(f"Failed to list models from {details['url']}: {e}")
        
        # List OpenRouter models if requested
        if all_models:
            try:
                response = requests.get('https://openrouter.ai/api/v1/models', timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for item in data['data']:
                    models.append({
                        "provider": "openrouter",
                        "name": item['id'],
                        "full_name": item['id'],
                        "type": "OpenRouter"
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch OpenRouter models: {e}")
        
        return jsonify({"models": models})
    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from file."""
    try:
        key_path = Path.home() / ".openrouter" / "key"
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error("OpenRouter API key file not found at ~/.openrouter/key")
        return ""
    except Exception as e:
        logger.error(f"Error reading OpenRouter API key: {e}")
        return ""

def load_model_config(model: str, provider: str, url: str):
    """Load model configuration into ShellGPT config."""
    # Base ShellGPT configuration
    sgptrc = {
        "CHAT_CACHE_PATH": "/tmp/chat_cache",
        "CACHE_PATH": "/tmp/cache",
        "CHAT_CACHE_LENGTH": "100",
        "CACHE_LENGTH": "100",
        "REQUEST_TIMEOUT": "60",
        "DEFAULT_COLOR": "magenta",
        "ROLE_STORAGE_PATH": str(Path.home() / ".config" / "shell_gpt" / "roles"),
        "SYSTEM_ROLES": "false",
        "DEFAULT_EXECUTE_SHELL_CMD": "false",
        "DISABLE_STREAMING": "false",
        "CODE_THEME": "dracula",
        "OPENAI_FUNCTIONS_PATH": str(Path.home() / ".config" / "shell_gpt" / "functions"),
        "OPENAI_USE_FUNCTIONS": "false",
        "SHOW_FUNCTIONS_OUTPUT": "false",
        "PRETTIFY_MARKDOWN": "true",
        "SHELL_INTERACTION": "true",
        "OS_NAME": "auto",
        "SHELL_NAME": "auto"
    }

    # Provider-specific configuration
    if provider == 'ollama':
        config = {**sgptrc, **{
            "API_BASE_URL": url,
            "USE_LITELLM": "true",
            "OPENAI_API_KEY": ""
        }}
    elif provider == 'openrouter':
        config = {**sgptrc, **{
            "API_BASE_URL": url,
            "USE_LITELLM": "false",
            "OPENAI_API_KEY": get_openrouter_api_key()
        }}
    else:
        logger.error(f"Unknown provider: {provider}")
        return False

    # Set the model
    config["DEFAULT_MODEL"] = model

    # Write configuration to file
    config_path = Path.home() / ".config" / "shell_gpt" / ".sgptrc"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        for key, value in config.items():
            if value == '':
                value = '""'
            f.write(f'{key}={value}\n')

    return True

@app.route("/api/v1/models/load", methods=["POST"])
def load_model():
    """Load a specific model for use with ShellGPT."""
    data = request.json
    model_name = data.get("model")
    
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    
    try:
        providers = load_providers_config()
        matched_models = {}
        
        # Try to find the model in Ollama providers
        for provider, details in providers.items():
            if details["type"] == "ollama":
                try:
                    # Set the OLLAMA_HOST environment variable for each provider
                    env = os.environ.copy()
                    env["OLLAMA_HOST"] = details["url"].split("//")[1]
                    
                    output = subprocess.check_output("ollama list", shell=True, env=env).decode()
                    lines = output.split('\n')[1:]
                    ollama_models = sorted([f"{provider}/{line.split()[0]}" for line in lines if line])
                    
                    for model in ollama_models:
                        model_short = model.split('/', 1)[1]
                        if (model_name in model) or (model_name in model_short):
                            matched_models[model] = (provider, model_short, "ollama", details["url"])
                except Exception as e:
                    logger.warning(f"Failed to list models from {provider}: {e}")
        
        # If exactly one Ollama model matches, load it
        if len(matched_models) == 1:
            model_info = next(iter(matched_models.items()))
            full_model_name = model_info[0]  # This is "provider/model:tag"
            provider, model_short, model_type, url = model_info[1]
            
            if load_model_config(full_model_name, model_type, url):
                return jsonify({"status": "success", "model": full_model_name})
            else:
                return jsonify({"error": "Failed to load model configuration"}), 500
        
        # If no Ollama matches or multiple matches, try OpenRouter
        if len(matched_models) == 0:
            try:
                response = requests.get('https://openrouter.ai/api/v1/models', timeout=10)
                response.raise_for_status()
                data = response.json()
                
                openrouter_matches = []
                for item in data['data']:
                    if model_name in item['id']:
                        openrouter_matches.append(item['id'])
                
                # Check for exact match
                if model_name in openrouter_matches:
                    openrouter_matches = [model_name]
                
                if len(openrouter_matches) == 1:
                    if load_model_config(openrouter_matches[0], "openrouter", "https://openrouter.ai/api/v1"):
                        return jsonify({"status": "success", "model": openrouter_matches[0]})
                    else:
                        return jsonify({"error": "Failed to load model configuration"}), 500
                elif len(openrouter_matches) > 1:
                    return jsonify({
                        "status": "error",
                        "error": f"Multiple OpenRouter models match '{model_name}'",
                        "matches": openrouter_matches[:10]
                    }), 400
            except Exception as e:
                logger.warning(f"Failed to fetch OpenRouter models: {e}")
        
        # Handle multiple Ollama matches
        if len(matched_models) > 1:
            return jsonify({
                "status": "error",
                "error": f"Multiple models match '{model_name}'",
                "matches": list(matched_models.keys())[:10]
            }), 400
        
        # No matches found
        return jsonify({
            "status": "error",
            "error": f"No models match '{model_name}'"
        }), 404
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/model/current", methods=["GET"])
def get_current_model():
    """Get the currently configured model."""
    config_path = Path.home() / ".config" / "shell_gpt" / ".sgptrc"
    
    if not config_path.exists():
        return jsonify({"error": "No configuration found"}), 404
    
    try:
        with open(config_path, 'r') as file:
            for line in file:
                if line.startswith('DEFAULT_MODEL='):
                    model = line.split('=', 1)[1].strip().strip('"')
                    return jsonify({"model": model})
        
        return jsonify({"error": "DEFAULT_MODEL not found in configuration"}), 404
    except Exception as e:
        logger.exception(f"Error getting current model: {e}")
        return jsonify({"error": str(e)}), 500

def cleanup_inactive_sessions():
    """Clean up inactive REPL sessions."""
    with repl_lock:
        current_time = time.time()
        inactive_sessions = []
        
        for repl_id, session in repl_sessions.items():
            # If inactive for more than 1 hour, mark for removal
            if current_time - session["last_activity"] > 3600:
                inactive_sessions.append(repl_id)
        
        for repl_id in inactive_sessions:
            logger.info(f"Cleaning up inactive REPL session: {repl_id}")
            del repl_sessions[repl_id]

# Add REPL handler method for single input processing
def process_single_input(self, prompt, model, temperature, top_p, caching, functions):
    """Process a single REPL input and return the response."""
    self.init_history(init_prompt="")  # Make sure history is initialized
    
    # Add user prompt to history
    self.history.append({"role": "user", "content": prompt})
    
    # Get completion
    messages = []
    
    # Add system role if specified
    if self.role_class:
        messages.append({"role": "system", "content": self.role_class})
    
    # Add conversation history
    messages.extend(self.history)
    
    from sgpt.client import Client
    client = Client(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        caching=caching,
        functions=functions,
    )
    
    # Get completion
    completion = client.get_completion()
    
    # Add assistant response to history
    self.history.append({"role": "assistant", "content": completion})
    
    # Save history
    self.save_history()
    
    return completion

# Add method to ReplHandler class
ReplHandler.process_single_input = process_single_input

def cleanup_thread_func():
    """Background thread to clean up inactive sessions."""
    while True:
        time.sleep(1800)  # Run every 30 minutes
        try:
            cleanup_inactive_sessions()
        except Exception as e:
            logger.exception(f"Error in cleanup thread: {e}")

# Start cleanup thread
cleanup_thread = Thread(target=cleanup_thread_func, daemon=True)
cleanup_thread.start()

def main():
    app.run(host="127.0.0.1", port=5000)
