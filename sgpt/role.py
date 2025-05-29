import json
import platform
import sys
from enum import Enum
from os import getenv, pathsep
from os.path import basename
from pathlib import Path
from typing import Dict, Optional

from distro import name as distro_name

from .config import cfg

SHELL_ROLE = """Provide only {shell} commands for {os} without any description.
If there is a lack of details, provide most logical solution.
Ensure the output is a valid shell command.
If multiple steps required try to combine them together using &&.
Provide only plain text without Markdown formatting.
Do not provide markdown formatting such as ```.
"""

DESCRIBE_SHELL_ROLE = """Provide a terse, single sentence description of the given shell command.
Describe each argument and option of the command.
Provide short responses in about 80 words.
APPLY MARKDOWN formatting when possible."""
# Note that output for all roles containing "APPLY MARKDOWN" will be formatted as Markdown.

CODE_ROLE = """Provide only code as output without any description.
Provide only code in plain text format without Markdown formatting.
Do not include symbols such as ``` or ```python.
If there is a lack of details, provide most logical solution.
You are not allowed to ask for more details.
For example if the prompt is "Hello world Python", you should return "print('Hello world')"."""

DEFAULT_ROLE = """You are programming and system administration assistant.
You are managing {os} operating system with {shell} shell.
Provide short responses in about 100 words, unless you are specifically asked for more details.
If you need to store any data, assume it will be stored in the conversation.
APPLY MARKDOWN formatting when possible."""
# Note that output for all roles containing "APPLY MARKDOWN" will be formatted as Markdown.

ROLE_TEMPLATE = "You are {name}\n{role}"


class SystemRole:
    storage: Path = Path(cfg.get("ROLE_STORAGE_PATH"))

    def __init__(
        self,
        name: str,
        role: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> None:
        self.storage.mkdir(parents=True, exist_ok=True)
        self.name = name
        if variables:
            role = role.format(**variables)
        self.role = role

    @classmethod
    def create_defaults(cls) -> None:
        cls.storage.parent.mkdir(parents=True, exist_ok=True)
        variables = {"shell": cls._shell_name(), "os": cls._os_name()}
        for default_role in (
            SystemRole("ShellGPT", DEFAULT_ROLE, variables),
            SystemRole("Shell Command Generator", SHELL_ROLE, variables),
            SystemRole("Shell Command Descriptor", DESCRIBE_SHELL_ROLE, variables),
            SystemRole("Code Generator", CODE_ROLE),
        ):
            if not default_role._exists:
                default_role._save()

    @classmethod
    def get(cls, name: str) -> "SystemRole":
        file_path = cls.storage / f"{name}.json"
        if not file_path.exists():
            print(f'Error: Role "{name}" not found.', file=sys.stderr)
            sys.exit(1)
        return cls(**json.loads(file_path.read_text()))

    @classmethod
    def create(cls, name: str, description: str) -> None:
        role = cls(name, description)
        role._save()
        print(f'Role "{name}" created successfully.')

    @classmethod
    def list(cls, standalone: bool = False) -> list:
        if not cls.storage.exists():
            if standalone:
                return []
            return
        # Get all files in the folder.
        files = cls.storage.glob("*")
        # Sort files by last modification time in ascending order.
        role_names = []
        for path in sorted(files, key=lambda f: f.stat().st_mtime):
            role_name = path.stem  # Get filename without extension
            role_names.append(role_name)
            if not standalone:
                print(role_name)
        return role_names if standalone else None

    @classmethod
    def show(cls, name: str, standalone: bool = False) -> Optional[str]:
        role = cls.get(name)
        if standalone:
            return role.role
        print(role.role)
        return None

    @classmethod
    def get_role_name(cls, initial_message: str) -> Optional[str]:
        if not initial_message:
            return None
        message_lines = initial_message.splitlines()
        if "You are" in message_lines[0]:
            return message_lines[0].split("You are ")[1].strip()
        return None

    @classmethod
    def _os_name(cls) -> str:
        if cfg.get("OS_NAME") != "auto":
            return cfg.get("OS_NAME")
        current_platform = platform.system()
        if current_platform == "Linux":
            return "Linux/" + distro_name(pretty=True)
        if current_platform == "Windows":
            return "Windows " + platform.release()
        if current_platform == "Darwin":
            return "Darwin/MacOS " + platform.mac_ver()[0]
        return current_platform

    @classmethod
    def _shell_name(cls) -> str:
        if cfg.get("SHELL_NAME") != "auto":
            return cfg.get("SHELL_NAME")
        current_platform = platform.system()
        if current_platform in ("Windows", "nt"):
            is_powershell = len(getenv("PSModulePath", "").split(pathsep)) >= 3
            return "powershell.exe" if is_powershell else "cmd.exe"
        return basename(getenv("SHELL", "/bin/sh"))

    @property
    def _exists(self) -> bool:
        return self._file_path.exists()

    @property
    def _file_path(self) -> Path:
        return self.storage / f"{self.name}.json"

    def _save(self) -> None:
        if self._exists:
            response = input(f'Role "{self.name}" already exists, overwrite it? (y/N): ')
            if response.lower() not in ('y', 'yes'):
                print("Aborted.")
                sys.exit(1)

        self.role = ROLE_TEMPLATE.format(name=self.name, role=self.role)
        self._file_path.write_text(json.dumps(self.__dict__), encoding="utf-8")

    def delete(self) -> None:
        if self._exists:
            response = input(f'Role "{self.name}" exists, delete it? (y/N): ')
            if response.lower() not in ('y', 'yes'):
                print("Aborted.")
                return
        self._file_path.unlink()

    def same_role(self, initial_message: str) -> bool:
        if not initial_message:
            return False
        return True if f"You are {self.name}" in initial_message else False


class DefaultRoles(Enum):
    DEFAULT = "ShellGPT"
    SHELL = "Shell Command Generator"
    DESCRIBE_SHELL = "Shell Command Descriptor"
    CODE = "Code Generator"

    @classmethod
    def check_get(cls, shell: bool, describe_shell: bool, code: bool) -> SystemRole:
        if shell:
            return SystemRole.get(DefaultRoles.SHELL.value)
        if describe_shell:
            return SystemRole.get(DefaultRoles.DESCRIBE_SHELL.value)
        if code:
            return SystemRole.get(DefaultRoles.CODE.value)
        return SystemRole.get(DefaultRoles.DEFAULT.value)

    def get_role(self) -> SystemRole:
        return SystemRole.get(self.value)


SystemRole.create_defaults()
