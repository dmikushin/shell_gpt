import os
import platform
import shlex
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from sgpt.__version__ import __version__
from sgpt.integration import bash_integration, zsh_integration


def get_edited_prompt() -> str:
    """
    Opens the user's default editor to let them
    input a prompt, and returns the edited text.

    :return: String prompt.
    """
    with NamedTemporaryFile(suffix=".txt", delete=False) as file:
        # Create file and store path.
        file_path = file.name
    editor = os.environ.get("EDITOR", "vim")
    # This will write text to file using $EDITOR.
    os.system(f"{editor} {file_path}")
    # Read file when editor is closed.
    with open(file_path, "r", encoding="utf-8") as file:
        output = file.read()
    os.remove(file_path)
    if not output:
        print("Error: Couldn't get valid PROMPT from $EDITOR", file=sys.stderr)
        sys.exit(1)
    return output


def run_command(command: str) -> None:
    """
    Runs a command in the user's shell.
    It is aware of the current user's $SHELL.
    :param command: A shell command to run.
    """
    if platform.system() == "Windows":
        is_powershell = len(os.getenv("PSModulePath", "").split(os.pathsep)) >= 3
        full_command = (
            f'powershell.exe -Command "{command}"'
            if is_powershell
            else f'cmd.exe /c "{command}"'
        )
    else:
        shell = os.environ.get("SHELL", "/bin/sh")
        full_command = f"{shell} -c {shlex.quote(command)}"

    os.system(full_command)


def install_shell_integration() -> None:
    """
    Installs shell integration. Currently only supports ZSH and Bash.
    Allows user to get shell completions in terminal by using hotkey.
    Replaces current "buffer" of the shell with the completion.
    """
    # TODO: Add support for Windows.
    # TODO: Implement updates.
    shell = os.getenv("SHELL", "")
    if "zsh" in shell:
        print("Installing ZSH integration...")
        with open(os.path.expanduser("~/.zshrc"), "a", encoding="utf-8") as file:
            file.write(zsh_integration)
    elif "bash" in shell:
        print("Installing Bash integration...")
        with open(os.path.expanduser("~/.bashrc"), "a", encoding="utf-8") as file:
            file.write(bash_integration)
    else:
        print("Error: ShellGPT integrations only available for ZSH and Bash.", file=sys.stderr)
        sys.exit(1)

    print("Done! Restart your shell to apply changes.")
    sys.exit(0)


def get_sgpt_version() -> None:
    """
    Displays the current installed version of ShellGPT
    """
    print(f"ShellGPT {__version__}")
    sys.exit(0)
