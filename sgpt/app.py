import os
import sys
import json
import requests
from pathlib import Path
from io import StringIO

import typer
from click import BadArgumentUsage
from click.types import Choice
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

# Create the main app and subcommands
app = typer.Typer(help="ShellGPT - AI-powered command line assistant")
model_app = typer.Typer(help="Model management commands")
app.add_typer(model_app, name="model")

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

@model_app.command("avail")
def list_available_models(
    all_models: bool = typer.Option(False, "--all", help="Include OpenRouter models")
):
    """List available models from configured providers."""
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
    try:
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

@model_app.command("load")
def load_model(model_name: str = typer.Argument(..., help="Model name to load")):
    """Load a specific model for use with ShellGPT."""
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
    try:
        response = requests.post(
            f"{SERVER_URL}/api/v1/models/load",
            json={"model": model_name},
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

@model_app.command("status")
def show_current_model():
    """Show the currently loaded model."""
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
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

@app.command()
def main(
    prompt: str = typer.Argument(
        "",
        show_default=False,
        help="The prompt to generate completions for.",
    ),
    model: str = typer.Option(
        None,
        help="Large language model to use.",
    ),
    temperature: float = typer.Option(
        0.0,
        min=0.0,
        max=2.0,
        help="Randomness of generated output.",
    ),
    top_p: float = typer.Option(
        1.0,
        min=0.0,
        max=1.0,
        help="Limits highest probable tokens (words).",
    ),
    md: bool = typer.Option(
        True,
        help="Prettify markdown output.",
    ),
    shell: bool = typer.Option(
        False,
        "--shell",
        "-s",
        help="Generate and execute shell commands.",
        rich_help_panel="Assistance Options",
    ),
    interaction: bool = typer.Option(
        True,
        help="Interactive mode for --shell option.",
        rich_help_panel="Assistance Options",
    ),
    describe_shell: bool = typer.Option(
        False,
        "--describe-shell",
        "-d",
        help="Describe a shell command.",
        rich_help_panel="Assistance Options",
    ),
    code: bool = typer.Option(
        False,
        "--code",
        "-c",
        help="Generate only code.",
        rich_help_panel="Assistance Options",
    ),
    functions: bool = typer.Option(
        False,
        help="Allow function calls.",
        rich_help_panel="Assistance Options",
    ),
    editor: bool = typer.Option(
        False,
        help="Open $EDITOR to provide a prompt.",
    ),
    cache: bool = typer.Option(
        True,
        help="Cache completion results.",
    ),
    stream: bool = typer.Option(
        True,
        help="Enable streaming output (default: True).",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version.",
        callback=get_sgpt_version,
    ),
    chat: str = typer.Option(
        None,
        help="Follow conversation with id, " 'use "temp" for quick session.',
        rich_help_panel="Chat Options",
    ),
    repl: str = typer.Option(
        None,
        help="Start a REPL (Read–eval–print loop) session.",
        rich_help_panel="Chat Options",
    ),
    show_chat: str = typer.Option(
        None,
        help="Show all messages from provided chat id.",
        rich_help_panel="Chat Options",
    ),
    list_chats: bool = typer.Option(
        False,
        "--list-chats",
        "-lc",
        help="List all existing chat ids.",
        rich_help_panel="Chat Options",
    ),
    role: str = typer.Option(
        None,
        help="System role for GPT model.",
        rich_help_panel="Role Options",
    ),
    list_roles: bool = typer.Option(
        False,
        "--list-roles",
        "-lr",
        help="List roles.",
        rich_help_panel="Role Options",
    ),
    install_integration: bool = typer.Option(
        False,
        help="Install shell integration (ZSH and Bash only)",
        callback=install_shell_integration,
        hidden=True,  # Hiding since should be used only once.
    ),
) -> None:
    """ShellGPT client - sends requests to the ShellGPT server."""
    if not check_server_status():
        console.print("[red]Server is not running. Please start the server first.[/red]")
        return
    
    stdin_passed = not sys.stdin.isatty()
    
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
    
    if show_chat:
        try:
            response = requests.get(
                f"{SERVER_URL}/api/v1/chats/{show_chat}",
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
                
                console.print(content, style="white" if not md else None, markdown=md)
                console.print("")
            
            return
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return
    
    if list_chats:
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
    
    if sum((shell, describe_shell, code)) > 1:
        raise BadArgumentUsage(
            "Only one of --shell, --describe-shell, and --code options can be used at a time."
        )
    
    if chat and repl:
        raise BadArgumentUsage("--chat and --repl options cannot be used together.")
    
    if editor and stdin_passed:
        raise BadArgumentUsage("--editor option cannot be used with stdin input.")
    
    if editor:
        prompt = get_edited_prompt()
    
    if list_roles:
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
    
    # Process REPL command
    if repl:
        try:
            # Start REPL session
            response = requests.post(
                f"{SERVER_URL}/api/v1/repl/start",
                json={
                    "repl_id": repl,
                    "prompt": prompt,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "md": md,
                    "cache": cache,
                    "functions": functions,
                    "role": role,
                    "stream": stream
                },
                headers=get_api_headers(),
                timeout=60,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                initial_response = stream_response(response, md)
                repl_id = repl  # We sent this as the repl_id
            else:
                data = response.json()
                repl_id = data.get("repl_id")
                initial_response = data.get("response")
                
                if initial_response:
                    console.print("[bold green]Assistant:[/bold green]")
                    console.print(initial_response, style="white" if not md else None, markdown=md)
            
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
                        json={"input": user_input, "stream": stream},
                        headers=get_api_headers(),
                        timeout=60,
                        stream=stream
                    )
                    response.raise_for_status()
                    
                    # Display response
                    console.print("[bold green]Assistant:[/bold green]")
                    if stream:
                        stream_response(response, md)
                    else:
                        data = response.json()
                        console.print(data.get("response", ""), style="white" if not md else None, markdown=md)
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
    if chat:
        try:
            response = requests.post(
                f"{SERVER_URL}/api/v1/chat",
                json={
                    "prompt": prompt,
                    "chat_id": chat,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "md": md,
                    "cache": cache,
                    "functions": functions,
                    "role": role,
                    "stream": stream
                },
                headers=get_api_headers(),
                timeout=60,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                full_completion = stream_response(response, md)
            else:
                data = response.json()
                full_completion = data.get("completion", "")
                console.print(full_completion, style="white" if not md else None, markdown=md)
            
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
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "md": md,
                "shell": shell,
                "describe_shell": describe_shell,
                "code": code,
                "functions": functions,
                "cache": cache,
                "role": role,
                "stream": stream
            },
            headers=get_api_headers(),
            timeout=60,
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            full_completion = stream_response(response, md)
        else:
            data = response.json()
            full_completion = data.get("completion", "")
            console.print(full_completion, style="white" if not md else None, markdown=md)
        
        # Handle shell command execution
        if shell and interaction and full_completion:
            option = typer.prompt(
                text="[E]xecute, [D]escribe, [A]bort",
                type=Choice(("e", "d", "a", "y"), case_sensitive=False),
                default="a",
                show_choices=False,
                show_default=False,
            )
            
            if option in ("e", "y"):
                # Run the command
                run_command(full_completion)
            elif option == "d":
                # Describe the shell command
                describe_response = requests.post(
                    f"{SERVER_URL}/api/v1/completion",
                    json={
                        "prompt": full_completion,
                        "model": model,
                        "temperature": temperature,
                        "top_p": top_p,
                        "md": md,
                        "describe_shell": True,
                        "cache": cache,
                        "role": "describe_shell",
                        "stream": stream
                    },
                    headers=get_api_headers(),
                    timeout=60,
                    stream=stream
                )
                describe_response.raise_for_status()
                
                if stream:
                    stream_response(describe_response, md)
                else:
                    describe_data = describe_response.json()
                    console.print(describe_data.get("completion", ""), style="white" if not md else None, markdown=md)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def entry_point() -> None:
    # Check if args look like a prompt without a command
    # (first arg doesn't start with - and doesn't match any command)
    import sys

    if len(sys.argv) > 1 and sys.argv[1] not in ["--install-completion", "--show-completion", "--help", "model", "main"]:
        # Insert 'main' command before the args
        sys.argv.insert(1, "main")

    app()

if __name__ == "__main__":
    entry_point()
