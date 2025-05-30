# ShellGPT Server API

## Base Configuration

- **Base URL**: `http://127.0.0.1:5000` (default)
- **Content-Type**: `application/json`
- **Authentication**: Optional `X-API-Key` header (currently accepts any value)
- **CORS**: Enabled for all origins

## Endpoints

### 1. Server Status

**GET** `/api/v1/status`

Get server status and active sessions.

**Response:**
```json
{
  "status": "running",
  "active_repl_sessions": 0,
  "verbose_mode": false
}
```

**Error Responses:**
- No specific error codes (endpoint always returns 200 OK)

### 2. Text Completion

**POST** `/api/v1/completion`

Generate a completion for a given prompt.

**Request Body:**
```json
{
  "prompt": "string (required)",
  "model": "string (optional, uses DEFAULT_MODEL if not specified)",
  "temperature": 0.0,
  "top_p": 1.0,
  "md": true,
  "shell": false,
  "describe_shell": false,
  "code": false,
  "functions": false,
  "cache": true,
  "stream": false,
  "role": "string (optional, custom system role content)"
}
```

**Response (Non-streaming):**
```json
{
  "completion": "string",
  "model": "string"
}
```

**Response (Streaming):**
- Content-Type: `text/event-stream`
- Data events: `{"token": "string"}`
- Complete event: `{"completion": "string", "model": "string"}`
- End event: `{}`

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error generating completion (model issues, configuration problems, etc.)

### 3. Chat Completion

**POST** `/api/v1/chat`

Generate a chat completion with conversation history.

**Request Body:**
```json
{
  "prompt": "string (required)",
  "chat_id": "string (optional, for conversation continuity)",
  "model": "string (optional)",
  "temperature": 0.0,
  "top_p": 1.0,
  "md": true,
  "functions": false,
  "cache": true,
  "stream": false,
  "role": "string (optional, custom system role content)"
}
```

**Response (Non-streaming):**
```json
{
  "completion": "string",
  "chat_id": "string",
  "model": "string"
}
```

**Response (Streaming):**
- Content-Type: `text/event-stream`
- Data events: `{"token": "string"}`
- Complete event: `{"completion": "string", "chat_id": "string", "model": "string"}`
- End event: `{}`

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error generating chat completion (model issues, configuration problems, etc.)

### 4. REPL Session Management

#### Start REPL Session

**POST** `/api/v1/repl/start`

Start a new REPL (Read-Eval-Print Loop) session.

**Request Body:**
```json
{
  "repl_id": "string (optional, auto-generated if not provided)",
  "prompt": "string (optional, initial prompt)",
  "model": "string (optional)",
  "temperature": 0.0,
  "top_p": 1.0,
  "md": true,
  "functions": false,
  "cache": true,
  "stream": false,
  "role": "string (optional, custom system role content)"
}
```

**Response:**
```json
{
  "repl_id": "string",
  "response": "string (null if no initial prompt)",
  "model": "string"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error processing initial REPL prompt (model issues, configuration problems, etc.)

#### Process REPL Input

**POST** `/api/v1/repl/{repl_id}`

Process input for an existing REPL session.

**Request Body:**
```json
{
  "input": "string (required)",
  "stream": false
}
```

**Response (Non-streaming):**
```json
{
  "response": "string",
  "repl_id": "string"
}
```

**Response (Streaming):**
- Content-Type: `text/event-stream`
- Data events: `{"token": "string"}`
- Complete event: `{"response": "string", "repl_id": "string"}`
- End event: `{}`

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: REPL session not found (invalid repl_id or session expired)
- `500 Internal Server Error`: Error processing REPL input (model issues, configuration problems, etc.)

#### End REPL Session

**DELETE** `/api/v1/repl/{repl_id}`

End a REPL session and clean up resources.

**Response:**
```json
{
  "status": "session ended"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- No error returned if session doesn't exist (always returns 200 OK)

### 5. Chat History Management

#### List All Chats

**GET** `/api/v1/chats`

List all available chat sessions.

**Response:**
```json
{
  "chats": ["chat_id_1", "chat_id_2", "..."]
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error accessing chat storage or listing chat sessions

#### Show Chat Messages

**GET** `/api/v1/chats/{chat_id}`

Retrieve all messages from a specific chat session.

**Response:**
```json
{
  "chat_id": "string",
  "messages": [
    {
      "role": "user|assistant|system",
      "content": "string"
    }
  ]
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error retrieving chat messages (invalid chat_id, storage issues, etc.)

### 6. Model Management

#### List Available Models

**GET** `/api/v1/models`

List all available models from configured providers.

**Query Parameters:**
- `all`: boolean (default: false) - Include OpenRouter models

**Response:**
```json
{
  "models": [
    {
      "provider": "string",
      "name": "string",
      "full_name": "string",
      "type": "Ollama|OpenRouter"
    }
  ]
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `500 Internal Server Error`: Error listing models (provider connection issues, configuration problems, etc.)

#### Load Model

**POST** `/api/v1/models/load`

Load a specific model for use with ShellGPT.

**Request Body:**
```json
{
  "model": "string (required, model name or partial match)"
}
```

**Response (Success):**
```json
{
  "status": "success",
  "model": "string"
}
```

**Response (Multiple Matches):**
```json
{
  "status": "error",
  "error": "Multiple models match 'model_name'",
  "matches": ["model1", "model2", "..."]
}
```

**Response (No Matches):**
```json
{
  "status": "error",
  "error": "No models match 'model_name'"
}
```

**Error Responses:**
- `400 Bad Request`: Model name is required (missing "model" parameter)
- `400 Bad Request`: Multiple models match the provided name (ambiguous model selection)
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: No models match the provided name
- `500 Internal Server Error`: Failed to load model configuration (file system issues, provider problems, etc.)

#### Get Current Model

**GET** `/api/v1/model/current`

Get the currently configured model.

**Response:**
```json
{
  "model": "string"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: No configuration found (missing .sgptrc file)
- `404 Not Found`: DEFAULT_MODEL not found in configuration file
- `500 Internal Server Error`: Error reading configuration file (file system issues, permission problems, etc.)

## General Error Response Format

All endpoints may return error responses in the following format:

```json
{
  "error": "string (error description)"
}
```

## Streaming Responses

Streaming endpoints use Server-Sent Events (SSE) format:

- **Content-Type**: `text/event-stream`
- **Data Event**: `data: {"token": "string"}\n\n`
- **Complete Event**: `event: complete\ndata: {"completion": "string", ...}\n\n`
- **End Event**: `event: end\ndata: {}\n\n`
- **Error Event**: `event: error\ndata: {"error": "string"}\n\n`

## Configuration

The server loads configuration from:
- `~/.config/shell_gpt/.sgptrc` - ShellGPT configuration
- `~/.config/shell_gpt/providers.json` - Provider configuration
- `~/.openrouter/key` - OpenRouter API key

## Session Management

- REPL sessions are automatically cleaned up after 1 hour of inactivity
- Chat sessions persist until manually deleted
- All sessions are stored in memory and lost on server restart
