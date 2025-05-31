#!/usr/bin/env python3
import os
import json
import time
import uuid
import logging
import argparse
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
from sgpt.model import BaseModelConfigManager

# Global verbose flag
VERBOSE_MODE = False

# Initialize logger (will be properly configured in main())
logger = logging.getLogger(__name__)

class Server:
    def __init__(self):
        self.providers_config_path = Path.home() / ".config" / "shell_gpt" / "providers.json"
        self.providers = self._load_providers_config()
        self._models_lock = Lock()
        self._models = []
        self._update_models()
        # Start background thread to update models periodically
        self._models_updater_thread = Thread(target=self._models_updater, daemon=True)
        self._models_updater_thread.start()

    def _load_providers_config(self) -> dict:
        """Load providers configuration from JSON file."""
        if not self.providers_config_path.exists():
            # Create default config if it doesn't exist
            self.providers_config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = {
                "local": {
                    "type": "ollama",
                    "url": "http://localhost:11434"
                }
            }
            with open(self.providers_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            if VERBOSE_MODE:
                logger.debug(f"Created default providers config at {self.providers_config_path}")
            return default_config

        with open(self.providers_config_path, 'r') as f:
            config = json.load(f)
            if VERBOSE_MODE:
                logger.debug(f"Loaded providers config: {list(config.keys())}")
            return config

    def _list_endpoint_models(self, provider_name: str, url: str, key: str) -> list:
        """Fetch models from a provider endpoint and extract using the given key."""
        models = []
        try:
            if VERBOSE_MODE:
                logger.debug(f"Fetching models from provider: {provider_name} at {url} (key: {key})")

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            model_list = data.get(key, [])
            if VERBOSE_MODE:
                logger.debug(f"API response: {len(model_list)} models found for key '{key}'")

            for model_info in model_list:
                model_name = model_info.get('id', '') or model_info.get('name')
                if model_name:
                    full_name = f"{provider_name}/{model_name}"
                    models.append({
                        "provider": provider_name,
                        "name": model_name,
                        "type": model_info.get('type', provider_name),
                        "size": model_info.get('size', 0),
                        "modified_at": model_info.get('modified_at', ''),
                        "digest": model_info.get('digest', ''),
                        "details": model_info.get('details', {}),
                    })
                    if VERBOSE_MODE:
                        logger.debug(f"Added model: {full_name}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch models from provider {provider_name} at {url}: {e}")
        except Exception as e:
            logger.warning(f"Error processing models from {provider_name}: {e}")

        return models

    def _list_ollama_models(self, provider_name: str, base_url: str) -> list:
        """Get models from an Ollama provider using the REST API."""
        return self._list_endpoint_models(provider_name, f"{base_url}/api/tags", "models")

    def _list_openai_models(self, provider_name: str, base_url: str) -> list:
        """Get models from an OpenAI-compatible provider using the REST API."""
        return self._list_endpoint_models(provider_name, f"{base_url}/models", "data")

    def _update_models(self):
        """Fetch and update the cached list of models atomically."""
        models = []
        for provider_name, provider_details in self.providers.items():
            if provider_details["type"] == "ollama":
                ollama_models = self._list_ollama_models(provider_name, provider_details["url"])
                models.extend(ollama_models)
        # Optionally include all models (OpenRouter etc.)
        for provider_name, provider_details in self.providers.items():
            if provider_details["type"] in ("ollama", "openrouter"):
                openai_models = self._list_openai_models(provider_name, provider_details["url"])
                models.extend(openai_models)
        with self._models_lock:
            self._models = models
        if VERBOSE_MODE:
            logger.debug(f"Model cache updated: {len(models)} models")

    def _models_updater(self):
        """Background thread to periodically update the model cache."""
        while True:
            try:
                self.providers = self._load_providers_config()
                self._update_models()
            except Exception as e:
                logger.warning(f"Error updating model cache: {e}")
            time.sleep(1800)  # Update every 30 minutes

    def list_models(self, all_models=True):
        """Return the cached list of models."""
        with self._models_lock:
            if all_models:
                return list(self._models)
            else:
                return [m for m in self._models if m.get("type") == "ollama"]

    def get_provider_details(self, provider):
        """
        Return (url, type) for the given provider name.
        Raises ValueError if provider not found.
        """
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found in configuration")
        provider_info = self.providers[provider]
        return provider_info.get("url"), provider_info.get("type")

class ServerModelConfigManager(BaseModelConfigManager):
    """
    Model config manager that uses a fixed config_path and server for list_models.
    """

    def __init__(self, server):
        config_path = Path.home() / ".config" / "shell_gpt" / "sgpt.toml"
        super().__init__(config_path)
        self.server = server

    def list_models(self, all_models=True):
        """
        List available models using the client.
        Args:
            all_models (bool): If False, fetch local models only.
        Returns:
            dict: Dictionary containing available models.
        Do not cache the model list, always fetch from client
        """
        return self.server.list_models()

server = Server()
model_config_manager = ServerModelConfigManager(server)

def setup_logging(verbose=False):
    """Configure logging based on verbose mode."""
    global VERBOSE_MODE
    VERBOSE_MODE = verbose
    
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if verbose:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(Path.home() / ".config" / "shell_gpt" / "server.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set more verbose logging for specific modules in verbose mode
    if verbose:
        logging.getLogger('flask').setLevel(logging.DEBUG)
        logging.getLogger('werkzeug').setLevel(logging.INFO)  # Keep werkzeug less verbose
    
    return logging.getLogger(__name__)

def log_request_details(endpoint, data=None):
    """Log detailed request information in verbose mode."""
    if not VERBOSE_MODE:
        return
    
    logger = logging.getLogger(__name__)
    logger.debug(f"=== {endpoint} REQUEST ===")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Method: {request.method}")
    logger.debug(f"URL: {request.url}")
    
    if data:
        # Sanitize sensitive data
        sanitized_data = data.copy() if isinstance(data, dict) else data
        if isinstance(sanitized_data, dict) and 'api_key' in sanitized_data:
            sanitized_data['api_key'] = '[REDACTED]'
        logger.debug(f"Request Data: {json.dumps(sanitized_data, indent=2, default=str)}")

def log_completion_details(prompt, completion, model, **kwargs):
    """Log completion details in verbose mode."""
    if not VERBOSE_MODE:
        return
    
    logger = logging.getLogger(__name__)
    logger.debug(f"=== COMPLETION DETAILS ===")
    logger.debug(f"Model: {model}")
    logger.debug(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    logger.debug(f"Completion: {completion[:200]}{'...' if len(completion) > 200 else ''}")
    logger.debug(f"Parameters: {kwargs}")

def log_streaming_token(token, token_count=None):
    """Log streaming token details in verbose mode."""
    if not VERBOSE_MODE:
        return
    
    logger = logging.getLogger(__name__)
    if token_count is not None:
        logger.debug(f"Streaming token #{token_count}: '{token[:50]}{'...' if len(token) > 50 else ''}'")
    else:
        logger.debug(f"Streaming token: '{token[:50]}{'...' if len(token) > 50 else ''}'")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store active REPL sessions
# The key is the repl_id and the value is a dict with session data
repl_sessions = {}
repl_lock = Lock()  # Lock for thread-safe access to repl_sessions

def authenticate():
    """Simple authentication check."""
    api_key = request.headers.get("X-API-Key")
    if VERBOSE_MODE:
        logger.debug(f"Authentication check - API key present: {'Yes' if api_key else 'No'}")
    return True

def create_role_from_content(role_content):
    """Create a role object from role content string."""
    if not role_content:
        return DefaultRoles.DEFAULT.get_role()
    
    # Create a simple role object that can be used by handlers
    class CustomRole:
        def __init__(self, content):
            self.content = content
            self.role = content  # Add role attribute for compatibility with handlers
            self.name = "CustomRole"  # Add name attribute
        
        def __str__(self):
            return self.content
        
        def get_role_name(self, initial_message: str):
            """Get role name from initial message."""
            if not initial_message:
                return None
            message_lines = initial_message.splitlines()
            if "You are" in message_lines[0]:
                return message_lines[0].split("You are ")[1].strip()
            return None
        
        def same_role(self, initial_message: str) -> bool:
            """Check if this role matches the initial message."""
            if not initial_message:
                return False
            return True if f"You are {self.name}" in initial_message else False
    
    return CustomRole(role_content)

def resolve_model_configuration(model_param):
    """Resolve model configuration and configure ShellGPT appropriately."""
    if model_param is None:
        # Use default model from config
        model_param = model_config_manager.get_default_model()
        if VERBOSE_MODE:
            logger.debug(f"Using default model: {model_param}")
        return model_param
    
    if not model_config_manager.validate_model(model_param):
        raise ValueError("Invalid model parameter format")
    
    provider = model_param['provider']
    model_name = model_param['name']
    model_type = model_param['type']
    
    if VERBOSE_MODE:
        logger.debug(f"Resolving model - Provider: {provider}, Name: {model_name}, Type: {model_type}")
    
    if not model_config_manager.model_exists(model_param):
        raise ValueError(f"Model '{model_name}' not found for provider '{provider}'")

    # Configure ShellGPT for the selected model
    try:
        provider_url, provider_type = server.get_provider_details(provider)
        success = load_model_config(model_name, provider, provider_url, provider_type)
        if not success:
            raise ValueError(f"Failed to configure model {model_name} for provider {provider}")
        
        if VERBOSE_MODE:
            logger.debug(f"Successfully configured model: {model_name}")
        
        return model_name
    except Exception as e:
        if VERBOSE_MODE:
            logger.error(f"Error configuring model: {e}")
        raise ValueError(f"Error configuring model: {e}")

@app.before_request
def before_request():
    """Check authentication before processing request."""
    if VERBOSE_MODE:
        logger.debug(f"Incoming request: {request.method} {request.path}")
    
    # Skip authentication for status endpoint
    if request.path == "/api/v1/status":
        return
    
    if not authenticate():
        return jsonify({"error": "Unauthorized"}), 401

@app.route("/api/v1/status", methods=["GET"])
def status():
    """Get server status."""
    log_request_details("STATUS")
    
    with repl_lock:
        active_sessions = len(repl_sessions)
    
    response = {
        "status": "running",
        "active_repl_sessions": active_sessions,
        "verbose_mode": VERBOSE_MODE
    }
    
    if VERBOSE_MODE:
        logger.debug(f"Status response: {response}")
    
    return jsonify(response)

def create_streaming_handler(handler_class, role, md, chat_id=None):
    """Create a custom handler that yields streaming tokens."""
    class StreamingHandler(handler_class):
        def handle_streaming(self, **kwargs):
            """Handle request and yield streaming tokens."""
            if VERBOSE_MODE:
                logger.debug(f"Creating streaming handler with kwargs: {kwargs}")
            
            disable_stream = cfg.get("DISABLE_STREAMING") == "true"
            prompt = kwargs.pop("prompt").strip()
            
            if VERBOSE_MODE:
                logger.debug(f"Streaming handler - prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            messages = self.make_messages(prompt)
            
            if VERBOSE_MODE:
                logger.debug(f"Generated messages for completion: {len(messages)} messages")
                for i, msg in enumerate(messages):
                    logger.debug(f"Message {i}: {msg['role']} - {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            
            generator = self.get_completion(
                messages=messages,
                **kwargs
            )
            
            full_response = ""
            token_count = 0
            for token in generator:
                if token:  # Only yield non-empty tokens
                    full_response += token
                    token_count += 1
                    log_streaming_token(token, token_count)
                    yield token
            
            if VERBOSE_MODE:
                logger.debug(f"Streaming complete - total tokens: {token_count}, full response length: {len(full_response)}")
            
            # Yield the final complete response as a special message
            yield f"\n__SGPT_COMPLETE__{full_response}__SGPT_COMPLETE__"
    
    # Handle different constructor signatures
    from sgpt.handlers.chat_handler import ChatHandler
    from sgpt.handlers.repl_handler import ReplHandler
    from sgpt.handlers.default_handler import DefaultHandler
    
    if issubclass(handler_class, ChatHandler):
        # ChatHandler and ReplHandler need chat_id
        if chat_id is None:
            chat_id = "temp"  # Use temp as default
        return StreamingHandler(chat_id, role, md)
    else:
        # DefaultHandler only needs role and markdown
        return StreamingHandler(role, md)

@app.route("/api/v1/completion", methods=["POST"])
def completion():
    """Generate a completion for a prompt."""
    data = request.json
    log_request_details("COMPLETION", data)
    
    prompt = data.get("prompt", "")
    model_param = data.get("model")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    shell = data.get("shell", False)
    describe_shell = data.get("describe_shell", False)
    code = data.get("code", False)
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    # Validate and resolve model configuration
    try:
        if model_param is not None and not model_config_manager.validate_model(model_param):
            return jsonify({"error": "Invalid model parameter format. Expected dictionary with 'provider', 'name', and 'type' fields."}), 400
        
        # Verify model availability if specified
        if model_param is not None and not model_config_manager.model_exists(model_param):
            return jsonify({"error": f"Model '{model_param.get('name', 'unknown')}' not available from provider '{model_param.get('provider', 'unknown')}'"}), 400
        
        # Resolve the actual model name to use
        model = resolve_model_configuration(model_param)
        
    except ValueError as e:
        if VERBOSE_MODE:
            logger.error(f"Model resolution error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        if VERBOSE_MODE:
            logger.error(f"Unexpected error during model resolution: {e}")
        return jsonify({"error": f"Error resolving model configuration: {e}"}), 500
    
    # Handle role - can be content string or None
    role_content = data.get("role")
    if role_content:
        role_class = create_role_from_content(role_content)
    else:
        role_class = DefaultRoles.check_get(shell, describe_shell, code)
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    if VERBOSE_MODE:
        logger.debug(f"Completion parameters - Model: {model}, Model param: {model_param}, Temperature: {temperature}, "
                    f"Top-p: {top_p}, Stream: {stream}, Role content: {role_content[:100] if role_content else 'Default'}")
        logger.debug(f"Function schemas enabled: {functions}, Cache: {cache}")
    
    try:
        if stream:
            if VERBOSE_MODE:
                logger.debug("Starting streaming completion")
            
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
                        if VERBOSE_MODE:
                            logger.debug("Streaming completion finished")
                        log_completion_details(prompt, final_response, model, 
                                             temperature=temperature, top_p=top_p, cache=cache)
                        yield f"event: complete\ndata: {json.dumps({'completion': final_response, 'model': model_param})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                yield f"event: end\ndata: {{}}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            if VERBOSE_MODE:
                logger.debug("Starting non-streaming completion")
            
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
            
            log_completion_details(prompt, full_completion, model, 
                                 temperature=temperature, top_p=top_p, cache=cache)
            
            response = {
                "completion": full_completion,
                "model": model_param,
            }
            
            if VERBOSE_MODE:
                logger.debug(f"Non-streaming completion finished - response length: {len(full_completion)}")
            
            return jsonify(response)
    except Exception as e:
        logger.exception("Error generating completion")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/chat", methods=["POST"])
def chat():
    """Generate a chat completion."""
    data = request.json
    log_request_details("CHAT", data)
    
    prompt = data.get("prompt", "")
    chat_id = data.get("chat_id")
    model_param = data.get("model")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    # Validate and resolve model configuration
    try:
        if model_param is not None and not model_config_manager.validate_model(model_param):
            return jsonify({"error": "Invalid model parameter format. Expected dictionary with 'provider', 'name', and 'type' fields."}), 400
        
        # Verify model availability if specified
        if model_param is not None and not model_config_manager.model_exists(model_param):
            return jsonify({"error": f"Model '{model_param.get('name', 'unknown')}' not available from provider '{model_param.get('provider', 'unknown')}'"}), 400
        
        # Resolve the actual model name to use
        model = resolve_model_configuration(model_param)
        
    except ValueError as e:
        if VERBOSE_MODE:
            logger.error(f"Model resolution error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        if VERBOSE_MODE:
            logger.error(f"Unexpected error during model resolution: {e}")
        return jsonify({"error": f"Error resolving model configuration: {e}"}), 500
    
    # Handle role - can be content string or None
    role_content = data.get("role")
    if role_content:
        role_class = create_role_from_content(role_content)
    else:
        role_class = DefaultRoles.DEFAULT.get_role()
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    if VERBOSE_MODE:
        logger.debug(f"Chat parameters - Chat ID: {chat_id}, Model: {model}, Model param: {model_param}, Stream: {stream}")
        logger.debug(f"Role content: {role_content[:100] if role_content else 'Default'}")
    
    try:
        if stream:
            if VERBOSE_MODE:
                logger.debug("Starting streaming chat completion")
            
            # Return streaming response
            def generate():
                handler = create_streaming_handler(ChatHandler, role_class, md, chat_id)
                
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
                        log_completion_details(prompt, final_response, model, chat_id=chat_id)
                        yield f"event: complete\ndata: {json.dumps({'completion': final_response, 'chat_id': chat_id, 'model': model_param})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                yield f"event: end\ndata: {{}}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            if VERBOSE_MODE:
                logger.debug("Starting non-streaming chat completion")
            
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
            
            log_completion_details(prompt, full_completion, model, chat_id=chat_id)
            
            return jsonify({
                "completion": full_completion,
                "chat_id": chat_id,
                "model": model_param,
            })
    except Exception as e:
        logger.exception(f"Error generating chat completion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/repl/start", methods=["POST"])
def repl_start():
    """Start a new REPL session."""
    data = request.json
    log_request_details("REPL_START", data)
    
    repl_id = data.get("repl_id")
    if not repl_id:
        repl_id = str(uuid.uuid4())
    
    if VERBOSE_MODE:
        logger.debug(f"Starting REPL session with ID: {repl_id}")
    
    init_prompt = data.get("prompt", "")
    model_param = data.get("model")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    md = data.get("md", cfg.get("PRETTIFY_MARKDOWN") == "true")
    functions = data.get("functions", cfg.get("OPENAI_USE_FUNCTIONS") == "true")
    cache = data.get("cache", True)
    stream = data.get("stream", False)
    
    # Validate and resolve model configuration
    try:
        if model_param is not None and not model_config_manager.validate_model(model_param):
            return jsonify({"error": "Invalid model parameter format. Expected dictionary with 'provider', 'name', and 'type' fields."}), 400
        
        # Verify model availability if specified
        if model_param is not None and not model_config_manager.model_exists(model_param):
            return jsonify({"error": f"Model '{model_param.get('name', 'unknown')}' not available from provider '{model_param.get('provider', 'unknown')}'"}), 400
        
        # Resolve the actual model name to use
        model = resolve_model_configuration(model_param)
        
    except ValueError as e:
        if VERBOSE_MODE:
            logger.error(f"Model resolution error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        if VERBOSE_MODE:
            logger.error(f"Unexpected error during model resolution: {e}")
        return jsonify({"error": f"Error resolving model configuration: {e}"}), 500
    
    # Handle role - can be content string or None
    role_content = data.get("role")
    if role_content:
        role_class = create_role_from_content(role_content)
    else:
        role_class = DefaultRoles.DEFAULT.get_role()
    
    function_schemas = (get_openai_schemas() or None) if functions else None
    
    # Create a new REPL handler
    repl_handler = ReplHandler(repl_id, role_class, md)
    
    # Store in sessions
    with repl_lock:
        repl_sessions[repl_id] = {
            "handler": repl_handler,
            "model": model,
            "model_param": model_param,
            "temperature": temperature,
            "top_p": top_p,
            "cache": cache,
            "functions": function_schemas,
            "last_activity": time.time(),
        }
        
        if VERBOSE_MODE:
            logger.debug(f"REPL session stored - Active sessions: {len(repl_sessions)}")
    
    # Process initial prompt if provided
    if init_prompt:
        if VERBOSE_MODE:
            logger.debug(f"Processing initial REPL prompt: {init_prompt[:100]}{'...' if len(init_prompt) > 100 else ''}")
        
        if stream:
            def generate():
                try:
                    streaming_handler = create_streaming_handler(ReplHandler, role_class, md, repl_id)
                    
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
                            log_completion_details(init_prompt, final_response, model, repl_id=repl_id)
                            yield f"event: complete\ndata: {json.dumps({'repl_id': repl_id, 'response': final_response, 'model': model_param})}\n\n"
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
                log_completion_details(init_prompt, response, model, repl_id=repl_id)
                return jsonify({
                    "repl_id": repl_id,
                    "response": response,
                    "model": model_param,
                })
            except Exception as e:
                logger.exception(f"Error processing initial REPL prompt: {e}")
                return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "repl_id": repl_id,
        "response": None,
        "model": model_param,
    })

@app.route("/api/v1/repl/<repl_id>", methods=["POST"])
def repl_process(repl_id):
    """Process input for an existing REPL session."""
    data = request.json
    log_request_details(f"REPL_PROCESS/{repl_id}", data)
    
    with repl_lock:
        if repl_id not in repl_sessions:
            if VERBOSE_MODE:
                logger.debug(f"REPL session not found: {repl_id}")
            return jsonify({"error": "REPL session not found"}), 404
        
        session = repl_sessions[repl_id]
        session["last_activity"] = time.time()
        
        if VERBOSE_MODE:
            logger.debug(f"Processing input for REPL session: {repl_id}")
    
    user_input = data.get("input", "")
    stream = data.get("stream", False)
    
    if VERBOSE_MODE:
        logger.debug(f"REPL input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    
    try:
        if stream:
            def generate():
                try:
                    repl_handler = session["handler"]
                    streaming_handler = create_streaming_handler(ReplHandler, repl_handler.role, repl_handler.markdown, repl_id)
                    
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
                            log_completion_details(user_input, final_response, session["model"], repl_id=repl_id)
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
            
            log_completion_details(user_input, response, session["model"], repl_id=repl_id)
            
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
    log_request_details(f"REPL_END/{repl_id}")
    
    with repl_lock:
        if repl_id in repl_sessions:
            if VERBOSE_MODE:
                logger.debug(f"Ending REPL session: {repl_id}")
            del repl_sessions[repl_id]
        else:
            if VERBOSE_MODE:
                logger.debug(f"REPL session not found for deletion: {repl_id}")
    
    return jsonify({"status": "session ended"})

@app.route("/api/v1/chats", methods=["GET"])
def list_chats():
    """List all chat sessions."""
    log_request_details("LIST_CHATS")
    
    try:
        chat_ids = ChatHandler.list_ids(standalone=True)
        if VERBOSE_MODE:
            logger.debug(f"Found {len(chat_ids)} chat sessions")
        return jsonify({"chats": chat_ids})
    except Exception as e:
        logger.exception(f"Error listing chats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/chats/<chat_id>", methods=["GET"])
def show_chat(chat_id):
    """Show messages from a specific chat."""
    log_request_details(f"SHOW_CHAT/{chat_id}")
    
    try:
        messages = ChatHandler.get_messages(chat_id)
        if VERBOSE_MODE:
            logger.debug(f"Retrieved {len(messages)} messages for chat {chat_id}")
        return jsonify({"chat_id": chat_id, "messages": messages})
    except Exception as e:
        logger.exception(f"Error showing chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/models", methods=["GET"])
def list_models():
    """List available models from all providers."""
    log_request_details("LIST_MODELS")

    try:    
        all_models = request.args.get("all", "false").lower() == "true"
    
        if VERBOSE_MODE:
            logger.debug(f"Listing models - include all: {all_models}")

        models = server.list_models(all_models=all_models)

        if VERBOSE_MODE:
            logger.debug(f"Total models found: {len(models)}")
       
        return jsonify({"models": models})
    except Exception as e:
        logger.exception(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from file."""
    try:
        key_path = Path.home() / ".openrouter" / "key"
        with open(key_path, 'r') as f:
            key = f.read().strip()
            if VERBOSE_MODE:
                logger.debug("OpenRouter API key loaded successfully")
            return key
    except FileNotFoundError:
        logger.error("OpenRouter API key file not found at ~/.openrouter/key")
        return ""
    except Exception as e:
        logger.error(f"Error reading OpenRouter API key: {e}")
        return ""

def load_model_config(model: str, provider: str, url: str, provider_type: str):
    """Load model configuration into ShellGPT config."""
    if VERBOSE_MODE:
        logger.debug(f"Loading model config - Model: {model}, Provider: {provider}, URL: {url}, Type: {provider_type}")
    
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
    if provider_type == 'ollama':
        config = {**sgptrc, **{
            "API_BASE_URL": url,
            "USE_LITELLM": "true",
            "OPENAI_API_KEY": ""
        }}
        if VERBOSE_MODE:
            logger.debug("Configured for Ollama provider")
    elif provider_type in ['openai', 'openrouter']:
        config = {**sgptrc, **{
            "API_BASE_URL": url,
            "USE_LITELLM": "false",
            "OPENAI_API_KEY": get_openrouter_api_key()
        }}
        if VERBOSE_MODE:
            logger.debug("Configured for OpenRouter/OpenAI provider")
    else:
        logger.error(f"Unknown provider type: {provider_type}")
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

    if VERBOSE_MODE:
        logger.debug(f"Model configuration written to {config_path}")

    return True

def cleanup_inactive_sessions():
    """Clean up inactive REPL sessions."""
    if VERBOSE_MODE:
        logger.debug("Running REPL session cleanup")
    
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
        
        if VERBOSE_MODE and inactive_sessions:
            logger.debug(f"Cleaned up {len(inactive_sessions)} inactive sessions")

# Add REPL handler method for single input processing
def process_single_input(self, prompt, model, temperature, top_p, caching, functions):
    """Process a single REPL input and return the response."""
    if VERBOSE_MODE:
        logger.debug(f"Processing single REPL input - Model: {model}, Prompt length: {len(prompt)}")
    
    self.init_history(init_prompt="")  # Make sure history is initialized
    
    # Add user prompt to history
    self.history.append({"role": "user", "content": prompt})
    
    # Get completion
    messages = []
    
    # Add system role if specified
    if self.role_class:
        messages.append({"role": "system", "content": str(self.role_class)})
    
    # Add conversation history
    messages.extend(self.history)
    
    if VERBOSE_MODE:
        logger.debug(f"REPL messages prepared - {len(messages)} messages")
    
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
    
    if VERBOSE_MODE:
        logger.debug(f"REPL completion received - length: {len(completion)}")
    
    # Add assistant response to history
    self.history.append({"role": "assistant", "content": completion})
    
    # Save history
    self.save_history()
    
    return completion

# Add method to ReplHandler class
ReplHandler.process_single_input = process_single_input

def cleanup_thread_func():
    """Background thread to clean up inactive sessions."""
    if VERBOSE_MODE:
        logger.debug("Starting cleanup thread")
    
    while True:
        time.sleep(1800)  # Run every 30 minutes
        try:
            cleanup_inactive_sessions()
        except Exception as e:
            logger.exception(f"Error in cleanup thread: {e}")

# Start cleanup thread
cleanup_thread = Thread(target=cleanup_thread_func, daemon=True)
cleanup_thread.start()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ShellGPT Server - HTTP API for ShellGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start server with default settings
  %(prog)s --verbose          # Start server with verbose logging
  %(prog)s --host 0.0.0.0     # Start server accessible from all interfaces
  %(prog)s --port 8080        # Start server on port 8080
  %(prog)s --verbose --port 8080 --host 0.0.0.0  # All options combined
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging for debugging (traces all server events, prompts, completions, etc.)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to bind the server to (default: 5000)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the server."""
    args = parse_args()
    
    # Setup logging based on verbose flag
    global logger
    logger = setup_logging(verbose=args.verbose)
    
    if args.verbose:
        logger.info("=" * 60)
        logger.info("SHELLGPT SERVER STARTING IN VERBOSE MODE")
        logger.info("=" * 60)
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Log Level: DEBUG")
        logger.info("=" * 60)
    else:
        logger.info(f"Starting ShellGPT server on {args.host}:{args.port}")
    
    try:
        app.run(host=args.host, port=args.port, debug=args.verbose)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")

import sys

if __name__ == "__main__":
    # For debugging: override sys.argv here if needed
    sys.argv = ["sgpt_server", "--verbose"]
    main()