#!/usr/bin/env python3
"""
Main entry point for the MCP Nautobot Server.
"""

import asyncio
from .fast_server import main

if __name__ == "__main__":
    asyncio.run(main())
