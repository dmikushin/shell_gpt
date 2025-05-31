import json
import requests
from typing import Dict, Any, Optional, List, Union


class ShellGPTError(Exception):
    """Base exception for ShellGPT client errors."""
    pass


class ShellGPTAuthenticationError(ShellGPTError):
    """Authentication error (401)."""
    pass


class ShellGPTNotFoundError(ShellGPTError):
    """Resource not found error (404)."""
    pass


class ShellGPTBadRequestError(ShellGPTError):
    """Bad request error (400)."""
    pass


class ShellGPTServerError(ShellGPTError):
    """Internal server error (500)."""
    pass


class ShellGPTConnectionError(ShellGPTError):
    """Connection error."""
    pass


class ShellGPTClient:
    """Client for ShellGPT REST API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        try:
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise ShellGPTBadRequestError(error_data.get("error", "Bad request"))
            elif response.status_code == 401:
                error_data = response.json() if response.content else {}
                raise ShellGPTAuthenticationError(error_data.get("error", "Missing or invalid API key"))
            elif response.status_code == 404:
                error_data = response.json() if response.content else {}
                raise ShellGPTNotFoundError(error_data.get("error", "Resource not found"))
            elif response.status_code == 500:
                error_data = response.json() if response.content else {}
                raise ShellGPTServerError(error_data.get("error", "Internal server error"))
            else:
                response.raise_for_status()
                return response.json()
        except requests.exceptions.JSONDecodeError:
            raise ShellGPTServerError(f"Invalid JSON response: {response.text}")
    
    def _validate_model_parameter(self, model: Union[Dict[str, str], None]) -> bool:
        """Prepare model parameter for API request."""
        if model is None:
            return True
        
        if isinstance(model, dict):
            # Validate dictionary format
            required_keys = {"provider", "name", "type"}
            if not all(key in model for key in required_keys):
                raise ShellGPTBadRequestError(
                    f"Model dictionary must contain keys: {required_keys}. "
                    f"Got: {set(model.keys())}"
                )
            return True
        
        raise ShellGPTBadRequestError(
            f"Model parameter must be a dictionary or None. Got: {type(model)}"
        )
    
    def check_server_status(self) -> bool:
        """Check if the ShellGPT server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get detailed server status."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/status", timeout=10)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def _handle_streaming_response(self, response: requests.Response, silent: bool = False) -> Dict[str, Any]:
        """Handle streaming response from the API."""
        if response.status_code != 200:
            return self._handle_response(response)
        
        full_completion = ""
        result_data = {}
        
        if not silent:
            print("🤖 AI: ", end="", flush=True)
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "error" in data:
                            if not silent:
                                print()  # New line before error
                            raise ShellGPTServerError(data["error"])
                        
                        if "token" in data:
                            # Stream token
                            token = data["token"]
                            if not silent:
                                print(token, end="", flush=True)
                            full_completion += token
                        
                        elif "completion" in data or "response" in data:
                            # Complete event
                            result_data = data
                            if "completion" in data:
                                result_data["completion"] = full_completion
                            else:
                                result_data["response"] = full_completion
                    
                    except json.JSONDecodeError:
                        continue
                
                elif line.startswith("event: error"):
                    if not silent:
                        print()  # New line before error
                    # Next line should contain error data
                    continue
        
        except requests.exceptions.RequestException as e:
            if not silent:
                print()  # New line before error
            raise ShellGPTConnectionError(f"Streaming connection error: {e}")
        
        if not silent:
            print()  # New line after streaming
        return result_data
    
    def get_streaming_response(self, url: str, data: Dict[str, Any]) -> requests.Response:
        """Get raw streaming response for custom handling."""
        try:
            response = self.session.post(url, json=data, stream=True)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def completion(self, prompt: str, model: Union[Dict[str, str], None] = None, 
                  temperature: float = 0.0, top_p: float = 1.0, md: bool = True, 
                  shell: bool = False, describe_shell: bool = False, code: bool = False, 
                  functions: bool = False, cache: bool = True, stream: bool = False, 
                  role: Optional[str] = None, silent_stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Generate a completion for a given prompt."""
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "md": md,
            "shell": shell,
            "describe_shell": describe_shell,
            "code": code,
            "functions": functions,
            "cache": cache,
            "stream": stream,
            **kwargs
        }
        
        if self._validate_model_parameter(model):
            data["model"] = model
        if role:
            data["role"] = role
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/completion", json=data)
            
            if stream:
                return self._handle_streaming_response(response, silent=silent_stream)
            else:
                return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def completion_stream(self, prompt: str, model: Union[Dict[str, str], None] = None, 
                         temperature: float = 0.0, top_p: float = 1.0, md: bool = True, 
                         shell: bool = False, describe_shell: bool = False, code: bool = False, 
                         functions: bool = False, cache: bool = True, role: Optional[str] = None,
                         **kwargs) -> requests.Response:
        """Get streaming completion response for custom handling."""
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "md": md,
            "shell": shell,
            "describe_shell": describe_shell,
            "code": code,
            "functions": functions,
            "cache": cache,
            "stream": True,
            **kwargs
        }
        
        if self._validate_model_parameter(model):
            data["model"] = model
        if role:
            data["role"] = role
        
        return self.get_streaming_response(f"{self.base_url}/api/v1/completion", data)
    
    def chat_completion(self, prompt: str, chat_id: Optional[str] = None, 
                       model: Union[Dict[str, str], None] = None, temperature: float = 0.0,
                       top_p: float = 1.0, md: bool = True, functions: bool = False,
                       cache: bool = True, stream: bool = True, role: Optional[str] = None,
                       silent_stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Send a chat completion request."""
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "md": md,
            "functions": functions,
            "cache": cache,
            "stream": stream,
            **kwargs
        }
        
        if chat_id:
            data["chat_id"] = chat_id
        
        if self._validate_model_parameter(model):
            data["model"] = model
        if role:
            data["role"] = role
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/chat", json=data)
            
            if stream:
                return self._handle_streaming_response(response, silent=silent_stream)
            else:
                return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def chat_completion_stream(self, prompt: str, chat_id: Optional[str] = None, 
                              model: Union[Dict[str, str], None] = None, temperature: float = 0.0,
                              top_p: float = 1.0, md: bool = True, functions: bool = False,
                              cache: bool = True, role: Optional[str] = None,
                              **kwargs) -> requests.Response:
        """Get streaming chat completion response for custom handling."""
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "md": md,
            "functions": functions,
            "cache": cache,
            "stream": True,
            **kwargs
        }
        
        if chat_id:
            data["chat_id"] = chat_id
        
        if self._validate_model_parameter(model):
            data["model"] = model
        if role:
            data["role"] = role
        
        return self.get_streaming_response(f"{self.base_url}/api/v1/chat", data)
    
    def start_repl_session(self, repl_id: Optional[str] = None, prompt: Optional[str] = None,
                          model: Union[Dict[str, str], None] = None, temperature: float = 0.0,
                          top_p: float = 1.0, md: bool = True, functions: bool = False,
                          cache: bool = True, stream: bool = False, role: Optional[str] = None,
                          silent_stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Start a new REPL session."""
        data = {
            "temperature": temperature,
            "top_p": top_p,
            "md": md,
            "functions": functions,
            "cache": cache,
            "stream": stream,
            **kwargs
        }
        
        if repl_id:
            data["repl_id"] = repl_id
        if prompt:
            data["prompt"] = prompt
        
        if self._validate_model_parameter(model):
            data["model"] = model
        if role:
            data["role"] = role
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/repl/start", json=data)
            if stream:
                return self._handle_streaming_response(response, silent=silent_stream)
            else:
                return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def process_repl_input(self, repl_id: str, input_text: str, stream: bool = False, silent_stream: bool = False) -> Dict[str, Any]:
        """Process input for an existing REPL session."""
        data = {
            "input": input_text,
            "stream": stream
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/repl/{repl_id}", json=data)
            
            if stream:
                return self._handle_streaming_response(response, silent=silent_stream)
            else:
                return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def repl_input_stream(self, repl_id: str, input_text: str) -> requests.Response:
        """Get streaming REPL input response for custom handling."""
        data = {
            "input": input_text,
            "stream": True
        }
        
        return self.get_streaming_response(f"{self.base_url}/api/v1/repl/{repl_id}", data)
    
    def end_repl_session(self, repl_id: str) -> Dict[str, Any]:
        """End a REPL session and clean up resources."""
        try:
            response = self.session.delete(f"{self.base_url}/api/v1/repl/{repl_id}")
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def list_chats(self) -> Dict[str, Any]:
        """List all available chat sessions."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/chats")
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def get_chat_history(self, chat_id: str) -> Dict[str, Any]:
        """Get chat history for a specific chat session."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/chats/{chat_id}")
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")
    
    def list_models(self, all_models: bool = False) -> Dict[str, Any]:
        """List all available models from configured providers."""
        params = {"all": all_models} if all_models else {}
        
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models", params=params)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            raise ShellGPTConnectionError(f"Failed to connect to server: {e}")

