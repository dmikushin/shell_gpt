import subprocess
import sys
from datetime import datetime
from typing import List

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from sgpt.config import cfg


def mock_comp(tokens_string):
    return [
        ChatCompletionChunk(
            id="foo",
            model=cfg.get("DEFAULT_MODEL"),
            object="chat.completion.chunk",
            choices=[
                StreamChoice(
                    index=0,
                    finish_reason=None,
                    delta=ChoiceDelta(content=token, role="assistant"),
                ),
            ],
            created=int(datetime.now().timestamp()),
        )
        for token in tokens_string
    ]


def run_sgpt_command(args: List[str], input_text: str = None) -> subprocess.CompletedProcess:
    """Run sgpt command with given arguments and return the result."""
    cmd = [sys.executable, "-m", "sgpt"] + args
    
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=30
    )
    
    return result


def cmd_args(prompt="", **kwargs) -> List[str]:
    """Convert keyword arguments to command line arguments."""
    arguments = []
    
    # Add prompt if provided
    if prompt:
        arguments.append(prompt)
    
    # Convert kwargs to command line arguments
    for key, value in kwargs.items():
        arguments.append(key)
        if isinstance(value, bool):
            continue
        arguments.append(str(value))
    
    # Add default test arguments
    arguments.extend(["--no-cache", "--no-functions"])
    return arguments


def comp_args(role, prompt, **kwargs):
    """Create completion arguments for testing."""
    return {
        "messages": [
            {"role": "system", "content": role.role},
            {"role": "user", "content": prompt},
        ],
        "model": cfg.get("DEFAULT_MODEL"),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        **kwargs,
    }
