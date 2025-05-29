from typing import Any
import sys

from rich import print as rich_print
from rich.rule import Rule

from ..role import DefaultRoles, SystemRole
from ..utils import run_command
from .chat_handler import ChatHandler
from .default_handler import DefaultHandler


class ReplHandler(ChatHandler):
    def __init__(self, chat_id: str, role: SystemRole, markdown: bool) -> None:
        super().__init__(chat_id, role, markdown)

    @classmethod
    def _get_multiline_input(cls) -> str:
        multiline_input = ""
        while True:
            try:
                user_input = input("...")
                if user_input == '"""':
                    break
                multiline_input += user_input + "\n"
            except (EOFError, KeyboardInterrupt):
                break
        return multiline_input

    def handle(self, init_prompt: str, **kwargs: Any) -> None:  # type: ignore
        if self.initiated:
            rich_print(Rule(title="Chat History", style="bold magenta"))
            self.show_messages(self.chat_id, self.markdown)
            rich_print(Rule(style="bold magenta"))

        info_message = (
            "Entering REPL mode, press Ctrl+C to exit."
            if not self.role.name == DefaultRoles.SHELL.value
            else (
                "Entering shell REPL mode, type [e] to execute commands "
                "or [d] to describe the commands, press Ctrl+C to exit."
            )
        )
        print(f"\033[33m{info_message}\033[0m")  # Yellow text

        if init_prompt:
            rich_print(Rule(title="Input", style="bold purple"))
            print(init_prompt)
            rich_print(Rule(style="bold purple"))

        full_completion = ""
        while True:
            # Infinite loop until user exits with Ctrl+C.
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting REPL...")
                sys.exit(0)
                
            if prompt == '"""':
                prompt = self._get_multiline_input()
            if prompt in ("exit()", "exit", "quit", "q"):
                print("Exiting REPL...")
                sys.exit(0)
            if init_prompt:
                prompt = f"{init_prompt}\n\n\n{prompt}"
                init_prompt = ""
            if self.role.name == DefaultRoles.SHELL.value and prompt == "e":
                print()
                run_command(full_completion)
                print()
                rich_print(Rule(style="bold magenta"))
            elif self.role.name == DefaultRoles.SHELL.value and prompt == "d":
                DefaultHandler(
                    DefaultRoles.DESCRIBE_SHELL.get_role(), self.markdown
                ).handle(prompt=full_completion, **kwargs)
            else:
                full_completion = super().handle(prompt=prompt, **kwargs)
