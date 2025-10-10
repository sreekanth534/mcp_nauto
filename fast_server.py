import asyncio
import logging
import os
import json
from typing import Optional, Dict

from fastmcp.server import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_nautobot_server.nautobot_client import NautobotClient, NautobotConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
server = FastMCP()

try:
    config = NautobotConfig()
    client = NautobotClient(config)
except Exception as e:
    logger.error(f"Failed to initialize Nautobot client: {e}")
    # Exit if the client cannot be initialized
    exit(1)


@server.tool()
async def get_ip_addresses(
    address: Optional[str] = Field(None, description="Specific IP address to search for"),
    prefix: Optional[str] = Field(None, description="Network prefix to filter by (e.g., 10.0.0.0/24)"),
    status: Optional[str] = Field(None, description="Status to filter by (e.g., active, reserved, deprecated)"),
    role: Optional[str] = Field(None, description="Role to filter by (e.g., loopback, secondary, anycast)"),
    tenant: Optional[str] = Field(None, description="Tenant to filter by"),
    vrf: Optional[str] = Field(None, description="VRF to filter by"),
    limit: int = Field(100, description="Maximum number of results to return (default: 100, max: 1000)", ge=1, le=1000),
    offset: int = Field(0, description="Number of results to skip for pagination (default: 0)", ge=0),
) -> Dict:
    """Retrieve IP addresses from Nautobot with filtering options"""
    logger.info(f"Retrieving IP addresses with filters: {locals()}")
    ip_addresses = await client.get_ip_addresses(
        address=address,
        prefix=prefix,
        status=status,
        role=role,
        tenant=tenant,
        vrf=vrf,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(ip_addresses),
        "filters_applied": {k: v for k, v in locals().items() if v is not None},
        "results": [ip.model_dump() for ip in ip_addresses],
    }


@server.tool()
async def get_prefixes(
    prefix: Optional[str] = Field(None, description="Specific network prefix to search for"),
    status: Optional[str] = Field(None, description="Status to filter by"),
    site: Optional[str] = Field(None, description="Site to filter by"),
    role: Optional[str] = Field(None, description="Role to filter by"),
    tenant: Optional[str] = Field(None, description="Tenant to filter by"),
    vrf: Optional[str] = Field(None, description="VRF to filter by"),
    limit: int = Field(100, description="Maximum number of results to return (default: 100, max: 1000)", ge=1, le=1000),
    offset: int = Field(0, description="Number of results to skip for pagination (default: 0)", ge=0),
) -> Dict:
    """Retrieve network prefixes from Nautobot with filtering options"""
    logger.info(f"Retrieving prefixes with filters: {locals()}")
    prefixes = await client.get_prefixes(
        prefix=prefix,
        status=status,
        site=site,
        role=role,
        tenant=tenant,
        vrf=vrf,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(prefixes),
        "filters_applied": {k: v for k, v in locals().items() if v is not None},
        "results": [p.model_dump() for p in prefixes],
    }


@server.tool()
async def get_ip_address_by_id(
    ip_id: str = Field(..., description="The Nautobot ID of the IP address"),
) -> Dict:
    """Retrieve a specific IP address by its Nautobot ID"""
    logger.info(f"Retrieving IP address by ID: {ip_id}")
    ip_address = await client.get_ip_address_by_id(ip_id)
    if ip_address is None:
        return {"error": f"IP address with ID '{ip_id}' not found."}
    return ip_address.model_dump()


@server.tool()
async def search_ip_addresses(
    query: str = Field(..., description="Search query (can match IP address, description, etc.)"),
    limit: int = Field(50, description="Maximum number of results to return (default: 50, max: 500)", ge=1, le=500),
) -> Dict:
    """Search IP addresses using a general query string"""
    logger.info(f"Searching IP addresses with query: {query}")
    ip_addresses = await client.search_ip_addresses(query, limit)
    return {
        "query": query,
        "count": len(ip_addresses),
        "results": [ip.model_dump() for ip in ip_addresses],
    }


@server.tool()
async def test_connection() -> Dict:
    """Test the connection to the Nautobot API"""
    logger.info("Testing Nautobot API connection")
    is_connected = await client.test_connection()
    return {
        "connected": is_connected,
        "nautobot_url": client.base_url,
        "timestamp": str(asyncio.get_event_loop().time()),
    }


@server.custom_route("/tools", methods=["GET"])
async def list_tools(request: Request) -> JSONResponse:
    """An endpoint to list all available tools in JSON format."""
    tools = await server.get_tools()
    tools_json = {}
    for name, tool in tools.items():
        properties = {}
        required = []
        # The tool.model is a Pydantic model representing the tool's arguments
        for field_name, field in tool.model_fields.items():
            # Get basic type information, converting None to 'any'
            field_type = "any"
            if field.annotation is not None:
                # Get the name of the type, e.g., 'str', 'int'
                field_type = getattr(field.annotation, '__name__', str(field.annotation))

            properties[field_name] = {
                "description": field.description or "",
                "type": field_type,
            }
            if field.is_required():
                required.append(field_name)

        tools_json[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.fn.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    return JSONResponse(tools_json)


def main():
    """Start the server."""
    server.run(transport="streamable-http", port=8050)

if __name__ == "__main__":
    main()
