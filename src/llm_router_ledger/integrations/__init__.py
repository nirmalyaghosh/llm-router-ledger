"""
Integrations with agent frameworks that own their own call path.

Each module here is behind an optional extra, following the [anthropic]
convention: importing one without its extra installed raises ConfigError
naming the extra to install, rather than ImportError.
"""

from __future__ import annotations
