import os
import sys
import json
import argparse
from pathlib import Path
from io import StringIO

from rich.console import Console
from rich.live import Live
from rich.text import Text

# Import the client and exceptions
from sgpt.client import (
    ShellGPTClient,
    ShellGPTError,
    ShellGPTAuthenticationError,
    ShellGPTNotFoundError,
    ShellGPTBadRequestError,
    ShellGPTServerError,
    ShellGPTConnectionError
)

from sgpt.model import BaseModelConfigManager

# Monkey patch Console.print to handle markdown parameter
original_print = Console.print

def patched_print(self, *args, **kwargs):
    """Patched version of Console.print that handles markdown parameter."""
    markdown_enabled = kwargs.pop('markdown', False) 
    if markdown_enabled and args and args[0]:
        from rich.markdown import Markdown
        return original_print(self, Markdown(args[0]), **kwargs)
    return original_print(self, *args, **kwargs)

Console.print = patched_print

from sgpt.utils import (
    get_edited_prompt,
    get_sgpt_version,
    install_shell_integration,
    run_command,
)

console = Console()

# Default server configuration
SERVER_URL = os.environ.get("SGPT_SERVER_URL", "http://localhost:5000")

# Initialize the client globally
client = ShellGPTClient(base_url=SERVER_URL)

def get_roles_dir():
    """Get the roles directory path."""
    roles_dir = Path.home() / ".config" / "shell_gpt" / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    return roles_dir

def create_role(role_name, description):
    """Create a new role on the client side."""
    roles_dir = get_roles_dir()
    role_file = roles_dir / f"{role_name}.json"
    
    if role_file.exists():
        console.print(f"[red]Role '{role_name}' already exists.[/red]")
        return False
    
    role_data = {
        "name": role_name,
        "description": description,
        "created_at": str(Path.ctime(Path.now()) if hasattr(Path, 'now') else "unknown")
    }
    
    try:
        with open(role_file, 'w') as f:
            json.dump(role_data, f, indent=2)
        console.print(f"[green]Role '{role_name}' created successfully.[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error creating role: {e}[/red]")
        return False

def list_roles():
    """List all available roles on the client side."""
    roles_dir = get_roles_dir()
    
    # Built-in roles
    builtin_roles = ["default", "shell", "code", "describe_shell"]
    
    # Custom roles from files
    custom_roles = []
    for role_file in roles_dir.glob("*.json"):
        custom_roles.append(role_file.stem)
    
    console.print("[bold]Built-in roles:[/bold]")
    for role in builtin_roles:
        console.print(f"  {role}")
    
    if custom_roles:
        console.print("\n[bold]Custom roles:[/bold]")
        for role in sorted(custom_roles):
            console.print(f"  {role}")
    else:
        console.print("\n[dim]No custom roles found.[/dim]")

def show_role(role_name):
    """Show details of a specific role."""
    # Check built-in roles first
    builtin_descriptions = {
        "default": "Default role for general conversation",
        "shell": "Role for generating shell commands",
        "code": "Role for generating code",
        "describe_shell": "Role for describing shell commands"
    }
    
    if role_name in builtin_descriptions:
        console.print(f"[bold]Role: {role_name}[/bold]")
        console.print(f"Type: Built-in")
        console.print(f"Description: {builtin_descriptions[role_name]}")
        return
    
    # Check custom roles
    roles_dir = get_roles_dir()
    role_file = roles_dir / f"{role_name}.json"
    
    if not role_file.exists():
        console.print(f"[red]Role '{role_name}' not found.[/red]")
        return
    
    try:
        with open(role_file, 'r') as f:
            role_data = json.load(f)
        
        console.print(f"[bold]Role: {role_name}[/bold]")
        console.print(f"Type: Custom")
        console.print(f"Description: {role_data.get('description', 'No description')}")
        if 'created_at' in role_data:
            console.print(f"Created: {role_data['created_at']}")
    except Exception as e:
        console.print(f"[red]Error reading role: {e}[/red]")

def delete_role(role_name):
    """Delete a custom role."""
    # Prevent deletion of built-in roles
    builtin_roles = ["default", "shell", "code", "describe_shell"]
    if role_name in builtin_roles:
        console.print(f"[red]Cannot delete built-in role '{role_name}'.[/red]")
        return False
    
    roles_dir = get_roles_dir()
    role_file = roles_dir / f"{role_name}.json"
    
    if not role_file.exists():
        console.print(f"[red]Role '{role_name}' not found.[/red]")
        return False
    
    try:
        role_file.unlink()
        console.print(f"[green]Role '{role_name}' deleted successfully.[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error deleting role: {e}[/red]")
        return False

def get_role_content(role_name):
    """Get the content/description of a role to send to server."""
    if not role_name:
        return None
    
    # Built-in roles
    builtin_roles = {
        "default": "You are ShellGPT, a helpful AI assistant.",
        "shell": "You are ShellGPT, an AI assistant that generates shell commands. Provide only the command without explanation unless asked.",
        "code": "You are ShellGPT, an AI assistant that generates code. Provide only clean, executable code without explanations unless asked.",
        "describe_shell": "You are ShellGPT, an AI assistant that explains shell commands in detail."
    }
    
    if role_name in builtin_roles:
        return builtin_roles[role_name]
    
    # Custom roles
    roles_dir = get_roles_dir()
    role_file = roles_dir / f"{role_name}.json"
    
    if role_file.exists():
        try:
            with open(role_file, 'r') as f:
                role_data = json.load(f)
            return role_data.get('description', '')
        except Exception:
            return None
    
    return None
        
def stream_response(response, md=False):
    """Handle streaming response with Rich Live display."""
    accumulated_text = ""
    lines_iter = response.iter_lines(decode_unicode=True)
    
    try:
        with Live(console=console, refresh_per_second=10) as live:
            for line in lines_iter:
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        token = data.get('token', '')
                        if token:
                            accumulated_text += token
                            # Update the live display
                            if md:
                                from rich.markdown import Markdown
                                live.update(Markdown(accumulated_text))
                            else:
                                live.update(Text(accumulated_text, style="white"))
                    except json.JSONDecodeError:
                        continue
                elif line.startswith('event: complete'):
                    # Get the next line which should contain the complete response
                    try:
                        complete_line = next(lines_iter, "")
                        if complete_line and complete_line.startswith('data: '):
                            complete_data = json.loads(complete_line[6:])
                            final_response = complete_data.get('completion') or complete_data.get('response', '')
                            # Final update with complete response
                            if md:
                                from rich.markdown import Markdown
                                live.update(Markdown(final_response))
                            else:
                                live.update(Text(final_response, style="white"))
                            return final_response
                    except (StopIteration, json.JSONDecodeError):
                        # If we can't get the complete response, return what we have
                        break
                elif line.startswith('event: end'):
                    break
                elif line.startswith('event: error'):
                    try:
                        error_line = next(lines_iter, "")
                        if error_line and error_line.startswith('data: '):
                            error_data = json.loads(error_line[6:])
                            error_msg = error_data.get('error', 'Unknown error')
                            live.update(Text(f"Error: {error_msg}", style="red"))
                            return None
                    except (StopIteration, json.JSONDecodeError):
                        live.update(Text("Error: Unknown error occurred", style="red"))
                        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Streaming interrupted by user[/yellow]")
        return accumulated_text
    
    return accumulated_text

def handle_client_streaming_response(response_data, md=False):
    """Handle streaming response from the client with Rich Live display."""
    if not isinstance(response_data, dict):
        return str(response_data)
    
    # For client streaming, the response is already handled by the client
    # We just need to return the completion or response
    return response_data.get('completion') or response_data.get('response', '')


class ClientModelConfigManager(BaseModelConfigManager):
    """
    Model config manager that uses a fixed config_path and client for list_models.
    """

    def __init__(self, client):
        config_path = Path.home() / ".config" / "shell_gpt" / "sgpt.toml"
        super().__init__(config_path)
        self.client = client

    def list_models(self, all_models=True):
        """
        List available models using the client.
        Args:
            all_models (bool): If False, fetch local models only.
        Returns:
            dict: Dictionary containing available models.
        Do not cache the model list, always fetch from client
        """
        return self.client.list_models(all_models=all_models).get('models', [])

# Instantiate for use in the rest of the code
model_config_manager = ClientModelConfigManager(client)
get_default_model = model_config_manager.get_default_model
set_default_model = model_config_manager.set_default_model
find_matching_models = model_config_manager.find_matching_models
                
def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="ShellGPT - AI-powered command line assistant",
        prog="sgpt"
    )
    
    # Main command arguments
    parser.add_argument('prompt', nargs='?', default='', help='The prompt to generate completions for')
    
    # Model options
    parser.add_argument('--model', help='Large language model to use')
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help='Randomness of generated output (0.0-2.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Limits highest probable tokens (0.0-1.0)')
    
    # Output options
    parser.add_argument('--md', action='store_true', help='Prettify markdown output')
    parser.add_argument('--no-md', action='store_true', help='Disable markdown output')
    
    # Assistance options
    assistance_group = parser.add_argument_group('assistance options')
    assistance_group.add_argument('-s', '--shell', action='store_true',
                                help='Generate and execute shell commands')
    assistance_group.add_argument('--interaction', action='store_true', default=True,
                                help='Interactive mode for --shell option')
    assistance_group.add_argument('--no-interaction', action='store_true',
                                help='Disable interactive mode for --shell option')
    assistance_group.add_argument('-d', '--describe-shell', action='store_true',
                                help='Describe a shell command')
    assistance_group.add_argument('-c', '--code', action='store_true',
                                help='Generate only code')
    assistance_group.add_argument('--functions', action='store_true', default=False,
                                help='Allow function calls')
    assistance_group.add_argument('--no-functions', action='store_true',
                                help='Disable function calls')
    
    # Editor and cache options
    parser.add_argument('--editor', action='store_true', help='Open $EDITOR to provide a prompt')
    parser.add_argument('--cache', action='store_true', default=True,
                       help='Cache completion results')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--stream', action='store_true', default=True,
                       help='Enable streaming output')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming output')
    
    # Version
    parser.add_argument('--version', action='store_true', help='Show version')
    
    # Chat options
    chat_group = parser.add_argument_group('chat options')
    chat_group.add_argument('--chat', help='Follow conversation with id, use "temp" for quick session')
    chat_group.add_argument('--repl', help='Start a REPL session')
    chat_group.add_argument('--show-chat', help='Show all messages from provided chat id')
    chat_group.add_argument('-lc', '--list-chats', action='store_true',
                          help='List all existing chat ids')
    
    # Role options
    role_group = parser.add_argument_group('role options')
    role_group.add_argument('--role', help='System role for GPT model')
    role_group.add_argument('--create-role', help='Create role')
    role_group.add_argument('--delete-role', help='Delete role')
    role_group.add_argument('--show-role', help='Show role')
    role_group.add_argument('-lr', '--list-roles', action='store_true', help='List roles')
    
    # Hidden/special options
    parser.add_argument('--install-integration', action='store_true', 
                       help=argparse.SUPPRESS)  # Hidden option
    
    return parser

def create_model_parser():
    """Create a separate parser for model commands."""
    parser = argparse.ArgumentParser(
        description="Model management commands",
        prog="sgpt model"
    )
    
    subparsers = parser.add_subparsers(dest='model_command', help='Model commands')
    
    # Model avail command
    avail_parser = subparsers.add_parser('avail', help='List available models')
    avail_parser.add_argument('--all', action='store_true', help='Include OpenRouter models')
    
    # Model load command
    load_parser = subparsers.add_parser('load', help='Load a specific model')
    load_parser.add_argument('model_name', help='Model name to load')
    
    # Model status command
    subparsers.add_parser('status', help='Show currently loaded model')
    
    return parser

def handle_model_commands(args):
    """Handle model management commands using the client."""
    try:        
        if args.model_command == 'avail':
            all_models = getattr(args, 'all', False)
            data = client.list_models(all_models=all_models)
            
            from rich.table import Table
            table = Table(title="Available Models")
            table.add_column("Provider", style="cyan")
            table.add_column("Model", style="green")
            table.add_column("Type", style="yellow")
            
            for model in data.get("models", []):
                table.add_row(
                    model.get("provider", ""),
                    model.get("name", ""),
                    model.get("type", "")
                )
            
            if table.row_count == 0:
                console.print("[yellow]No models found. Check your provider configuration.[/yellow]")
            else:
                console.print(table)
        
        elif args.model_command == 'load':
            try:
                model_name = args.model_name
                try:
                    exact_match, name_matches, ollama_matches = find_matching_models(model_name)
                except ShellGPTConnectionError:
                    raise Exception("Cannot get the list of available models, is the server running?")

                if exact_match:
                    model = exact_match
                elif len(ollama_matches) == 1:
                    model = ollama_matches[0]
                elif len(name_matches) == 1:
                    model = name_matches[0]
                elif not name_matches:
                    raise Exception(f"No models match '{model_name}'")
                else:
                    raise Exception(f"Multiple models match '{model_name}'")

                model_name = model[0]
                set_default_model(model)
                console.print(f"[green]✓ Loaded model: {model_name}[/green]")
            except Exception as e:
                console.print(f"[red]✗ {str(e)}[/red]")
        
        elif args.model_command == 'status':
            model = get_default_model()
            console.print(f"[green]Currently using: {model.get("name", "")}[/green]")
 
    except ShellGPTConnectionError as e:
        console.print(f"[red]Connection error: {str(e)}[/red]")
    except ShellGPTAuthenticationError as e:
        console.print(f"[red]Authentication error: {str(e)}[/red]")
    except ShellGPTError as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def validate_args(args):
    """Validate argument combinations."""
    # Check for conflicting options
    if sum([args.shell, args.describe_shell, args.code]) > 1:
        console.print("[red]Error: Only one of --shell, --describe-shell, and --code options can be used at a time.[/red]")
        sys.exit(2)
    
    if args.chat and args.repl:
        console.print("[red]Error: --chat and --repl options cannot be used together.[/red]")
        sys.exit(2)
    
    stdin_passed = not sys.stdin.isatty()
    if args.editor and stdin_passed:
        console.print("[red]Error: --editor option cannot be used with stdin input.[/red]")
        sys.exit(2)
    
    # Handle conflicting boolean options
    if args.no_md:
        args.md = False
    if args.no_interaction:
        args.interaction = False
    if args.no_functions:
        args.functions = False
    if args.no_cache:
        args.cache = False
    if args.no_stream:
        args.stream = False

def main():
    """Main function."""
    # Check if the first argument is 'model' - handle model commands separately
    if len(sys.argv) > 1 and sys.argv[1] == 'model':
        # Handle model commands
        model_parser = create_model_parser()
        args = model_parser.parse_args(sys.argv[2:])  # Skip 'sgpt' and 'model'
        
        # Add the model_command info to args for compatibility
        args.command = 'model'
        
        # Handle special options first
        if hasattr(args, 'version') and args.version:
            get_sgpt_version()
            return
        
        # Handle model commands
        handle_model_commands(args)
        return
    
    # For all other commands, use the main parser
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special options first
    if args.version:
        get_sgpt_version()
        return
    
    if args.install_integration:
        install_shell_integration()
        return
    
    # Handle client-side role operations (no server required)
    if args.create_role:
        role_description = input("Enter role description: ")
        create_role(args.create_role, role_description)
        return
    
    if args.delete_role:
        delete_role(args.delete_role)
        return
    
    if args.show_role:
        show_role(args.show_role)
        return
    
    if args.list_roles:
        list_roles()
        return
    
    # Validate arguments
    validate_args(args)
        
    # Handle stdin input
    stdin_passed = not sys.stdin.isatty()
    prompt = args.prompt
    
    if stdin_passed:
        stdin = ""
        # Process stdin input
        for line in sys.stdin:
            if "__sgpt__eof__" in line:
                break
            stdin += line
        prompt = f"{stdin}\n\n{prompt}" if prompt else stdin
        try:
            # Switch to stdin for interactive input.
            if os.name == "posix":
                sys.stdin = open("/dev/tty", "r")
            elif os.name == "nt":
                sys.stdin = open("CON", "r")
        except OSError:
            # Non-interactive shell.
            pass
    
    # If no prompt was provided and no special modes are active,
    # show help and exit
    if (not prompt and not stdin_passed and 
        not args.editor and not args.repl and not args.chat and 
        not args.show_chat and not args.list_chats):
        # Show help when called without arguments
        parser = create_parser()
        parser.print_help()
        return
    
    # Handle special display commands
    if args.show_chat:
        try:
            data = client.get_chat_history(args.show_chat)
            
            # Format and display messages
            for message in data.get("messages", []):
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    console.print("[bold blue]You:[/bold blue]")
                else:
                    console.print("[bold green]Assistant:[/bold green]")
                
                console.print(content, style="white" if not args.md else None, markdown=args.md)
                console.print("")
            
            return
        except ShellGPTNotFoundError:
            console.print(f"[red]Chat '{args.show_chat}' not found.[/red]")
            return
        except ShellGPTError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    if args.list_chats:
        try:
            data = client.list_chats()
            
            console.print("[bold]Available chat sessions:[/bold]")
            for chat_id in data.get("chats", []):
                console.print(f"  {chat_id}")
            
            return
        except ShellGPTError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    # Handle editor input
    if args.editor:
        prompt = get_edited_prompt()
    
    # Get role content to send to server
    role_content = get_role_content(args.role)

    model = args.model or get_default_model()
    
    # Process REPL command
    if args.repl:
        try:
            # Start REPL session
            if args.stream:
                # Handle streaming start
                response = client.get_streaming_response(
                    f"{client.base_url}/api/v1/repl/start",
                    {
                        "repl_id": args.repl,
                        "prompt": prompt,
                        "model": model,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "md": args.md,
                        "cache": args.cache,
                        "functions": args.functions,
                        "role": role_content,
                        "stream": args.stream
                    }
                )
                initial_response = stream_response(response, args.md)
                repl_id = args.repl
            else:
                response = client.start_repl_session(
                    repl_id=args.repl,
                    prompt=prompt,
                    model=model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    md=args.md,
                    cache=args.cache,
                    functions=args.functions,
                    role=role_content,
                    stream=args.stream
                )
                
                repl_id = response.get("repl_id", args.repl)
                initial_response = response.get("response")
                
                if initial_response:
                    console.print("[bold green]Assistant:[/bold green]")
                    console.print(initial_response, style="white" if not args.md else None, markdown=args.md)
            
            # Start REPL loop
            try:
                while True:
                    # Get user input
                    user_input = input("\n>>> ")
                    
                    if user_input.lower() in ("exit", "quit", "q"):
                        # End REPL session
                        client.end_repl_session(repl_id)
                        console.print("[yellow]REPL session ended.[/yellow]")
                        break
                    
                    # Process user input
                    console.print("[bold green]Assistant:[/bold green]")
                    if args.stream:
                        response = client.repl_input_stream(repl_id, user_input)
                        stream_response(response, args.md)
                    else:
                        response = client.process_repl_input(repl_id, user_input, stream=args.stream)
                        response_text = handle_client_streaming_response(response, args.md)
                        if response_text:
                            console.print(response_text, style="white" if not args.md else None, markdown=args.md)
                        
            except KeyboardInterrupt:
                # End REPL session on Ctrl+C
                try:
                    client.end_repl_session(repl_id)
                except:
                    pass
                console.print("\n[yellow]REPL session ended.[/yellow]")
            
            return
            
        except ShellGPTError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    # Process chat command
    if args.chat:
        try:
            if args.stream:
                response = client.chat_completion_stream(
                    prompt=prompt,
                    chat_id=args.chat,
                    model=model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    md=args.md,
                    cache=args.cache,
                    functions=args.functions,
                    role=role_content
                )
                full_completion = stream_response(response, args.md)
            else:
                response = client.chat_completion(
                    prompt=prompt,
                    chat_id=args.chat,
                    model=model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    md=args.md,
                    cache=args.cache,
                    functions=args.functions,
                    role=role_content,
                    stream=args.stream
                )
                full_completion = handle_client_streaming_response(response, args.md)
                if full_completion:
                    console.print(full_completion, style="white" if not args.md else None, markdown=args.md)
            
            return
            
        except ShellGPTError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    # Default command - regular completion
    try:
        if args.stream:
            response = client.completion_stream(
                prompt=prompt,
                model=model,
                temperature=args.temperature,
                top_p=args.top_p,
                md=args.md,
                shell=args.shell,
                describe_shell=args.describe_shell,
                code=args.code,
                functions=args.functions,
                cache=args.cache,
                role=role_content
            )
            full_completion = stream_response(response, args.md)
        else:
            response = client.completion(
                prompt=prompt,
                model=model,
                temperature=args.temperature,
                top_p=args.top_p,
                md=args.md,
                shell=args.shell,
                describe_shell=args.describe_shell,
                code=args.code,
                functions=args.functions,
                cache=args.cache,
                role=role_content,
                stream=args.stream
            )
            full_completion = handle_client_streaming_response(response, args.md)
            if full_completion:
                console.print(full_completion, style="white" if not args.md else None, markdown=args.md)
        
        # Handle shell command execution
        if args.shell and args.interaction and full_completion:
            while True:
                choice = input("[E]xecute, [D]escribe, [A]bort (e/d/a): ").lower()
                if choice in ('e', 'execute', 'y', 'yes'):
                    # Run the command
                    run_command(full_completion)
                    break
                elif choice in ('d', 'describe'):
                    # Describe the shell command
                    try:
                        if args.stream:
                            describe_response = client.completion_stream(
                                prompt=full_completion,
                                model=model,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                md=args.md,
                                describe_shell=True,
                                cache=args.cache,
                                role=get_role_content("describe_shell")
                            )
                            stream_response(describe_response, args.md)
                        else:
                            describe_response = client.completion(
                                prompt=full_completion,
                                model=model,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                md=args.md,
                                describe_shell=True,
                                cache=args.cache,
                                role=get_role_content("describe_shell"),
                                stream=args.stream
                            )
                            describe_text = handle_client_streaming_response(describe_response, args.md)
                            if describe_text:
                                console.print(describe_text, style="white" if not args.md else None, markdown=args.md)
                        break
                    except ShellGPTError as e:
                        console.print(f"[red]Error describing command: {str(e)}[/red]")
                        break
                elif choice in ('a', 'abort'):
                    break
                else:
                    console.print("Please enter 'e', 'd', or 'a'")
                    
    except ShellGPTError as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def entry_point():
    """Entry point for the application."""
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    # For debugging: override sys.argv here if needed
    # Example:
    # sys.argv = ["sgpt", "--version"]
    # sys.argv = ["sgpt", "--shell", "list files in current directory"]
    # sys.argv = ["sgpt", "model", "avail"]
    sys.argv = ["sgpt", "Please create a simple hello world program in C++"]
    entry_point()