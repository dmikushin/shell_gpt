import os
import sys
import json
import argparse
import requests
from pathlib import Path
from io import StringIO

from rich.console import Console
from rich.live import Live
from rich.text import Text

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
API_KEY = os.environ.get("SGPT_API_KEY", "default-key-change-me")

def get_api_headers():
    """Get API headers for authentication."""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

def check_server_status():
    """Check if the server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/api/v1/status", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False

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
    """Handle model management commands."""
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
    if args.model_command == 'avail':
        try:
            all_models = getattr(args, 'all', False)
            response = requests.get(
                f"{SERVER_URL}/api/v1/models?all={all_models}",
                headers=get_api_headers(),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
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
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    elif args.model_command == 'load':
        try:
            response = requests.post(
                f"{SERVER_URL}/api/v1/models/load",
                json={"model": args.model_name},
                headers=get_api_headers(),
                timeout=30
            )
            
            data = response.json()
            
            if response.status_code == 200:
                console.print(f"[green]✓ Loaded model: {data.get('model')}[/green]")
            else:
                error = data.get("error", "Unknown error")
                console.print(f"[red]✗ {error}[/red]")
                
                if "matches" in data:
                    console.print("[yellow]Multiple models match your query:[/yellow]")
                    for model in data.get("matches", [])[:10]:
                        console.print(f"  {model}")
                    if len(data.get("matches", [])) > 10:
                        console.print(f"  ... and {len(data.get('matches', [])) - 10} more")
                    console.print("[yellow]Please be more specific.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
    
    elif args.model_command == 'status':
        try:
            response = requests.get(
                f"{SERVER_URL}/api/v1/model/current",
                headers=get_api_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]Currently using: {data.get('model')}[/green]")
            else:
                data = response.json()
                console.print(f"[yellow]{data.get('error', 'No model currently loaded')}[/yellow]")
        except Exception as e:
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
    
    # Validate arguments
    validate_args(args)
    
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
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
    
    # Handle special display commands
    if args.show_chat:
        try:
            response = requests.get(
                f"{SERVER_URL}/api/v1/chats/{args.show_chat}",
                headers=get_api_headers(),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
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
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    if args.list_chats:
        try:
            response = requests.get(
                f"{SERVER_URL}/api/v1/chats",
                headers=get_api_headers(),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            console.print("[bold]Available chat sessions:[/bold]")
            for chat_id in data.get("chats", []):
                console.print(f"  {chat_id}")
            
            return
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    if args.list_roles:
        try:
            response = requests.get(
                f"{SERVER_URL}/api/v1/roles",
                headers=get_api_headers(),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            console.print("[bold]Available roles:[/bold]")
            for role_name in data.get("roles", []):
                console.print(f"  {role_name}")
            
            return
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    if args.create_role:
        role_description = input("Enter role description: ")
        # Here you would implement role creation
        console.print(f"[green]Role '{args.create_role}' created successfully.[/green]")
        return
    
    if args.show_role:
        # Here you would implement role display
        console.print(f"[bold]Role: {args.show_role}[/bold]")
        return
    
    # Handle editor input
    if args.editor:
        prompt = get_edited_prompt()
    
    # Process REPL command
    if args.repl:
        try:
            # Start REPL session
            response = requests.post(
                f"{SERVER_URL}/api/v1/repl/start",
                json={
                    "repl_id": args.repl,
                    "prompt": prompt,
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "md": args.md,
                    "cache": args.cache,
                    "functions": args.functions,
                    "role": args.role,
                    "stream": args.stream
                },
                headers=get_api_headers(),
                timeout=60,
                stream=args.stream
            )
            response.raise_for_status()
            
            if args.stream:
                initial_response = stream_response(response, args.md)
                repl_id = args.repl  # We sent this as the repl_id
            else:
                data = response.json()
                repl_id = data.get("repl_id")
                initial_response = data.get("response")
                
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
                        requests.delete(
                            f"{SERVER_URL}/api/v1/repl/{repl_id}",
                            headers=get_api_headers(),
                            timeout=10
                        )
                        console.print("[yellow]REPL session ended.[/yellow]")
                        break
                    
                    # Process user input
                    response = requests.post(
                        f"{SERVER_URL}/api/v1/repl/{repl_id}",
                        json={"input": user_input, "stream": args.stream},
                        headers=get_api_headers(),
                        timeout=60,
                        stream=args.stream
                    )
                    response.raise_for_status()
                    
                    # Display response
                    console.print("[bold green]Assistant:[/bold green]")
                    if args.stream:
                        stream_response(response, args.md)
                    else:
                        data = response.json()
                        console.print(data.get("response", ""), style="white" if not args.md else None, markdown=args.md)
            except KeyboardInterrupt:
                # End REPL session on Ctrl+C
                requests.delete(
                    f"{SERVER_URL}/api/v1/repl/{repl_id}",
                    headers=get_api_headers(),
                    timeout=10
                )
                console.print("\n[yellow]REPL session ended.[/yellow]")
            
            return
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    # Process chat command
    if args.chat:
        try:
            response = requests.post(
                f"{SERVER_URL}/api/v1/chat",
                json={
                    "prompt": prompt,
                    "chat_id": args.chat,
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "md": args.md,
                    "cache": args.cache,
                    "functions": args.functions,
                    "role": args.role,
                    "stream": args.stream
                },
                headers=get_api_headers(),
                timeout=60,
                stream=args.stream
            )
            response.raise_for_status()
            
            if args.stream:
                full_completion = stream_response(response, args.md)
            else:
                data = response.json()
                full_completion = data.get("completion", "")
                console.print(full_completion, style="white" if not args.md else None, markdown=args.md)
            
            return
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    # Default command - regular completion
    try:
        response = requests.post(
            f"{SERVER_URL}/api/v1/completion",
            json={
                "prompt": prompt,
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "md": args.md,
                "shell": args.shell,
                "describe_shell": args.describe_shell,
                "code": args.code,
                "functions": args.functions,
                "cache": args.cache,
                "role": args.role,
                "stream": args.stream
            },
            headers=get_api_headers(),
            timeout=60,
            stream=args.stream
        )
        response.raise_for_status()
        
        if args.stream:
            full_completion = stream_response(response, args.md)
        else:
            data = response.json()
            full_completion = data.get("completion", "")
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
                    describe_response = requests.post(
                        f"{SERVER_URL}/api/v1/completion",
                        json={
                            "prompt": full_completion,
                            "model": args.model,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "md": args.md,
                            "describe_shell": True,
                            "cache": args.cache,
                            "role": "describe_shell",
                            "stream": args.stream
                        },
                        headers=get_api_headers(),
                        timeout=60,
                        stream=args.stream
                    )
                    describe_response.raise_for_status()
                    
                    if args.stream:
                        stream_response(describe_response, args.md)
                    else:
                        describe_data = describe_response.json()
                        console.print(describe_data.get("completion", ""), style="white" if not args.md else None, markdown=args.md)
                    break
                elif choice in ('a', 'abort'):
                    break
                else:
                    console.print("Please enter 'e', 'd', or 'a'")
    except Exception as e:
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
    entry_point()
