import os
import json
import subprocess
import sys
from pathlib import Path

# To allow users to use arrow keys in the REPL.
import readline  # noqa: F401

import typer
import requests
from click import BadArgumentUsage
from click.types import Choice
from rich.console import Console
from rich.table import Table

from sgpt.config import cfg
from sgpt.function import get_openai_schemas
from sgpt.handlers.chat_handler import ChatHandler
from sgpt.handlers.default_handler import DefaultHandler
from sgpt.handlers.repl_handler import ReplHandler
from sgpt.llm_functions.init_functions import install_functions as inst_funcs
from sgpt.role import DefaultRoles, SystemRole
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


def list_ollama_models(name: str, provider_url: str) -> list[str]:
    """List Ollama models from a provider."""
    # Set the OLLAMA_HOST environment variable for each provider
    env = os.environ.copy()
    env["OLLAMA_HOST"] = provider_url.split("//")[1]

    try:
        output = subprocess.check_output("ollama list", shell=True, env=env).decode()
        lines = output.split('\n')[1:]
        ollama_models = sorted([f"{name}/{line.split()[0]}" for line in lines if line])
        return ollama_models
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to list models from {provider_url}: {e}[/red]")
        return []


def get_openrouter_models() -> list[str]:
    """Get available OpenRouter models."""
    try:
        response = requests.get('https://openrouter.ai/api/v1/models', timeout=10)
        response.raise_for_status()
        data = response.json()
        return sorted([item['id'] for item in data['data']])
    except Exception as e:
        console.print(f"[red]Failed to fetch OpenRouter models: {e}[/red]")
        return []


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from file."""
    try:
        key_path = Path.home() / ".openrouter" / "key"
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        console.print("[red]OpenRouter API key file not found at ~/.openrouter/key[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading OpenRouter API key: {e}[/red]")
        sys.exit(1)


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
        console.print(f"[red]Unknown provider: {provider}[/red]")
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


@model_app.command("avail")
def list_available_models(
    all_models: bool = typer.Option(False, "--all", help="Include OpenRouter models")
):
    """List available models from configured providers."""
    providers = load_providers_config()

    table = Table(title="Available Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Type", style="yellow")

    # List Ollama models
    for name, details in providers.items():
        if details["type"] == "ollama":
            models = list_ollama_models(name, details["url"])
            for model in models:
                table.add_row(name, model.split('/', 1)[1], "Ollama")

    # List OpenRouter models if requested
    if all_models:
        openrouter_models = get_openrouter_models()
        for model in openrouter_models:
            table.add_row("openrouter", model, "OpenRouter")

    if table.row_count == 0:
        console.print("[yellow]No models found. Check your provider configuration.[/yellow]")
    else:
        console.print(table)


@model_app.command("load")
def load_model(model_name: str = typer.Argument(..., help="Model name to load")):
    """Load a specific model for use with ShellGPT."""
    providers = load_providers_config()
    matched_models = {}

    # Try to find the model in Ollama providers
    for provider, details in providers.items():
        if details["type"] == "ollama":
            try:
                models = list_ollama_models(provider, details["url"])
                for model in models:
                    model_short = model.split('/', 1)[1]
                    if (model_name in model) or (model_name in model_short):
                        matched_models[model] = (provider, model_short, "ollama", details["url"])
            except Exception as e:
                console.print(f"[yellow]Warning: Could not list models from {provider}: {e}[/yellow]")

    # If exactly one Ollama model matches, load it
    if len(matched_models) == 1:
        model_info = next(iter(matched_models.items()))
        full_model_name = model_info[0]  # This is "provider/model:tag"
        provider, model_short, model_type, url = model_info[1]

        # Use the full model name including provider prefix
        if load_model_config(full_model_name, model_type, url):
            console.print(f"[green]✓ Loaded model: {full_model_name}[/green]")
            return

    # If no Ollama matches or multiple matches, try OpenRouter
    if len(matched_models) == 0:
        openrouter_models = get_openrouter_models()
        openrouter_matches = []

        for model in openrouter_models:
            if model_name in model:
                openrouter_matches.append(model)

        # Check for exact match
        if model_name in openrouter_matches:
            openrouter_matches = [model_name]

        if len(openrouter_matches) == 1:
            if load_model_config(openrouter_matches[0], "openrouter", "https://openrouter.ai/api/v1"):
                console.print(f"[green]✓ Loaded model: {openrouter_matches[0]} from OpenRouter[/green]")
                return
        elif len(openrouter_matches) > 1:
            console.print(f"[yellow]Multiple OpenRouter models match '{model_name}':[/yellow]")
            for model in openrouter_matches[:10]:  # Limit to first 10
                console.print(f"  {model}")
            if len(openrouter_matches) > 10:
                console.print(f"  ... and {len(openrouter_matches) - 10} more")
            console.print("[yellow]Please be more specific.[/yellow]")
            return

    # Handle multiple Ollama matches
    if len(matched_models) > 1:
        console.print(f"[yellow]Multiple models match '{model_name}':[/yellow]")
        for model_key in list(matched_models.keys())[:10]:  # Limit to first 10
            console.print(f"  {model_key}")
        if len(matched_models) > 10:
            console.print(f"  ... and {len(matched_models) - 10} more")
        console.print("[yellow]Please be more specific.[/yellow]")
        return

    # No matches found
    console.print(f"[red]✗ No models match '{model_name}'[/red]")
    console.print("[yellow]Run 'sgpt model avail' to see available models[/yellow]")
    

@model_app.command("status")
def show_current_model():
    """Show the currently loaded model."""
    config_path = Path.home() / ".config" / "shell_gpt" / ".sgptrc"

    if not config_path.exists():
        console.print("[yellow]No configuration found. No model is currently loaded.[/yellow]")
        return

    try:
        with open(config_path, 'r') as file:
            for line in file:
                if line.startswith('DEFAULT_MODEL='):
                    model = line.split('=', 1)[1].strip()
                    console.print(f"[green]Currently using: {model}[/green]")
                    return

        console.print("[yellow]DEFAULT_MODEL not found in configuration.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error reading configuration: {e}[/red]")


@app.command()
def main(
    prompt: str = typer.Argument(
        "",
        show_default=False,
        help="The prompt to generate completions for.",
    ),
    model: str = typer.Option(
        cfg.get("DEFAULT_MODEL"),
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
        cfg.get("PRETTIFY_MARKDOWN") == "true",
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
        cfg.get("SHELL_INTERACTION") == "true",
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
        cfg.get("OPENAI_USE_FUNCTIONS") == "true",
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
        callback=ChatHandler.list_ids,
        rich_help_panel="Chat Options",
    ),
    role: str = typer.Option(
        None,
        help="System role for GPT model.",
        rich_help_panel="Role Options",
    ),
    create_role: str = typer.Option(
        None,
        help="Create role.",
        callback=SystemRole.create,
        rich_help_panel="Role Options",
    ),
    show_role: str = typer.Option(
        None,
        help="Show role.",
        callback=SystemRole.show,
        rich_help_panel="Role Options",
    ),
    list_roles: bool = typer.Option(
        False,
        "--list-roles",
        "-lr",
        help="List roles.",
        callback=SystemRole.list,
        rich_help_panel="Role Options",
    ),
    install_integration: bool = typer.Option(
        False,
        help="Install shell integration (ZSH and Bash only)",
        callback=install_shell_integration,
        hidden=True,  # Hiding since should be used only once.
    ),
    install_functions: bool = typer.Option(
        False,
        help="Install default functions.",
        callback=inst_funcs,
        hidden=True,  # Hiding since should be used only once.
    ),
) -> None:
    stdin_passed = not sys.stdin.isatty()

    if stdin_passed:
        stdin = ""
        # TODO: This is very hacky.
        # In some cases, we need to pass stdin along with inputs.
        # When we want part of stdin to be used as a init prompt,
        # but rest of the stdin to be used as a inputs. For example:
        # echo "hello\n__sgpt__eof__\nThis is input" | sgpt --repl temp
        # In this case, "hello" will be used as a init prompt, and
        # "This is input" will be used as "interactive" input to the REPL.
        # This is useful to test REPL with some initial context.
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
        ChatHandler.show_messages(show_chat, md)

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

    role_class = (
        DefaultRoles.check_get(shell, describe_shell, code)
        if not role
        else SystemRole.get(role)
    )

    function_schemas = (get_openai_schemas() or None) if functions else None

    if repl:
        # Will be in infinite loop here until user exits with Ctrl+C.
        ReplHandler(repl, role_class, md).handle(
            init_prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            caching=cache,
            functions=function_schemas,
        )

    if chat:
        full_completion = ChatHandler(chat, role_class, md).handle(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            caching=cache,
            functions=function_schemas,
        )
    else:
        full_completion = DefaultHandler(role_class, md).handle(
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            caching=cache,
            functions=function_schemas,
        )

    while shell and interaction:
        option = typer.prompt(
            text="[E]xecute, [D]escribe, [A]bort",
            type=Choice(("e", "d", "a", "y"), case_sensitive=False),
            default="e" if cfg.get("DEFAULT_EXECUTE_SHELL_CMD") == "true" else "a",
            show_choices=False,
            show_default=False,
        )
        if option in ("e", "y"):
            # "y" option is for keeping compatibility with old version.
            run_command(full_completion)
        elif option == "d":
            DefaultHandler(DefaultRoles.DESCRIBE_SHELL.get_role(), md).handle(
                full_completion,
                model=model,
                temperature=temperature,
                top_p=top_p,
                caching=cache,
                functions=function_schemas,
            )
            continue
        break


def entry_point() -> None:
    app()


if __name__ == "__main__":
    entry_point()
