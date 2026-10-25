"""Render the production env file without logging secret values."""

import os
import re
import sys


def render_environment(base, callback_secret):
    if not base.strip():
        raise ValueError("BACKEND_ENV_FILE must not be empty")
    if not callback_secret or any(char in callback_secret for char in "\r\n\0"):
        raise ValueError("BACKEND_CALLBACK_SECRET must be a non-empty, single-line value")

    # Compose leaves single-quoted values literal, including dollar signs.
    quoted_secret = callback_secret.replace("'", "\\'")
    lines = [
        line for line in base.splitlines()
        if not re.match(r"^\s*(?:export\s+)?CALLBACK_SECRET\s*=", line)
    ]
    lines.append(f"CALLBACK_SECRET='{quoted_secret}'")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        rendered = render_environment(
            os.environ.get("BACKEND_ENV_FILE", ""),
            os.environ.get("BACKEND_CALLBACK_SECRET", ""),
        )
    except ValueError as error:
        sys.exit(str(error))
    sys.stdout.write(rendered)
