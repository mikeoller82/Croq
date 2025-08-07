"""
Model Context Protocol (MCP) Server Integration
Provides standardized interface for external tools and services
"""
import asyncio
import json
import logging
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import aiohttp
import websockets
from datetime import datetime

from core.hooks import hook_manager, HookType, HookContext

logger = logging.getLogger(__name__)

T = TypeVar('T')

class MCPTransport(Enum):
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"

class MCPMethod(Enum):
    INITIALIZE = "initialize"
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    LIST_RESOURCES = "list_resources"
    READ_RESOURCE = "read_resource"
    LIST_PROMPTS = "list_prompts"
    GET_PROMPT = "get_prompt"
    NOTIFICATION = "notification"

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPPrompt:
    """MCP Prompt template"""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MCPRequest:
    """MCP JSON-RPC request"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str = ""
    params: Optional[Dict[str, Any]] = None

@dataclass
class MCPResponse:
    """MCP JSON-RPC response"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class MCPServerConfig:
    """Configuration for MCP server instances"""
    def __init__(
        self,
        name: str,
        command: Optional[List[str]] = None,
        transport: MCPTransport = MCPTransport.STDIO,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        auto_restart: bool = True,
        max_restarts: int = 3
    ):
        self.name = name
        self.command = command or []
        self.transport = transport
        self.url = url
        self.env = env or {}
        self.timeout = timeout
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts

class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.connected = False
        self.restart_count = 0
        self._request_id = 0
        
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            if self.config.transport == MCPTransport.STDIO:
                return await self._connect_stdio()
            elif self.config.transport == MCPTransport.HTTP:
                return await self._connect_http()
            elif self.config.transport == MCPTransport.WEBSOCKET:
                return await self._connect_websocket()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False
    
    async def _connect_stdio(self) -> bool:
        """Connect via stdio transport"""
        try:
            self.process = subprocess.Popen(
                self.config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**self.config.env}
            )
            
            # Initialize the connection
            init_request = MCPRequest(
                id=self._next_id(),
                method=MCPMethod.INITIALIZE.value,
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "clientInfo": {
                        "name": "croq-ai-assistant",
                        "version": "1.0.0"
                    }
                }
            )
            
            await self._send_stdio_request(init_request)
            response = await self._receive_stdio_response()
            
            if response and not response.error:
                self.connected = True
                await self._discover_capabilities()
                logger.info(f"Connected to MCP server: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"STDIO connection failed: {e}")
        
        return False
    
    async def _connect_http(self) -> bool:
        """Connect via HTTP transport"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection with initialize
            response = await self._send_http_request(MCPRequest(
                id=self._next_id(),
                method=MCPMethod.INITIALIZE.value
            ))
            
            if response and not response.error:
                self.connected = True
                await self._discover_capabilities()
                return True
                
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            
        return False
    
    async def _connect_websocket(self) -> bool:
        """Connect via WebSocket transport"""
        try:
            self.websocket = await websockets.connect(self.config.url)
            
            # Initialize connection
            init_request = MCPRequest(
                id=self._next_id(),
                method=MCPMethod.INITIALIZE.value
            )
            
            await self.websocket.send(json.dumps(asdict(init_request)))
            response_data = await self.websocket.recv()
            response = MCPResponse(**json.loads(response_data))
            
            if not response.error:
                self.connected = True
                await self._discover_capabilities()
                return True
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            
        return False
    
    async def _discover_capabilities(self):
        """Discover server capabilities (tools, resources, prompts)"""
        # List tools
        tools_response = await self.send_request(MCPMethod.LIST_TOOLS)
        if tools_response and tools_response.result:
            for tool_data in tools_response.result.get("tools", []):
                tool = MCPTool(**tool_data)
                self.tools[tool.name] = tool
        
        # List resources
        resources_response = await self.send_request(MCPMethod.LIST_RESOURCES)
        if resources_response and resources_response.result:
            for resource_data in resources_response.result.get("resources", []):
                resource = MCPResource(**resource_data)
                self.resources[resource.name] = resource
        
        # List prompts
        prompts_response = await self.send_request(MCPMethod.LIST_PROMPTS)
        if prompts_response and prompts_response.result:
            for prompt_data in prompts_response.result.get("prompts", []):
                prompt = MCPPrompt(**prompt_data)
                self.prompts[prompt.name] = prompt
        
        logger.info(f"Discovered {len(self.tools)} tools, {len(self.resources)} resources, {len(self.prompts)} prompts")
    
    async def send_request(self, method: MCPMethod, params: Optional[Dict[str, Any]] = None) -> Optional[MCPResponse]:
        """Send request to MCP server"""
        if not self.connected:
            return None
            
        request = MCPRequest(
            id=self._next_id(),
            method=method.value,
            params=params
        )
        
        if self.config.transport == MCPTransport.STDIO:
            return await self._send_stdio_request(request)
        elif self.config.transport == MCPTransport.HTTP:
            return await self._send_http_request(request)
        elif self.config.transport == MCPTransport.WEBSOCKET:
            return await self._send_websocket_request(request)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Call a tool on the MCP server"""
        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found on server {self.config.name}")
            return None
        
        response = await self.send_request(
            MCPMethod.CALL_TOOL,
            {"name": tool_name, "arguments": arguments}
        )
        
        if response and response.result:
            return response.result.get("content")
        elif response and response.error:
            logger.error(f"Tool call failed: {response.error}")
        
        return None
    
    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource from the MCP server"""
        response = await self.send_request(
            MCPMethod.READ_RESOURCE,
            {"uri": uri}
        )
        
        if response and response.result:
            return response.result.get("contents")
        return None
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get a prompt template from the server"""
        response = await self.send_request(
            MCPMethod.GET_PROMPT,
            {"name": name, "arguments": arguments or {}}
        )
        
        if response and response.result:
            return response.result.get("messages")
        return None
    
    def _next_id(self) -> int:
        """Get next request ID"""
        self._request_id += 1
        return self._request_id
    
    async def _send_stdio_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send request via stdio"""
        if not self.process or not self.process.stdin:
            return None
        
        try:
            request_json = json.dumps(asdict(request))
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()
            
            return await self._receive_stdio_response()
        except Exception as e:
            logger.error(f"STDIO request failed: {e}")
            return None
    
    async def _receive_stdio_response(self) -> Optional[MCPResponse]:
        """Receive response via stdio"""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            line = self.process.stdout.readline()
            if line:
                response_data = json.loads(line.strip())
                return MCPResponse(**response_data)
        except Exception as e:
            logger.error(f"STDIO response failed: {e}")
        
        return None
    
    async def _send_http_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send request via HTTP"""
        if not self.session:
            return None
        
        try:
            async with self.session.post(
                self.config.url,
                json=asdict(request),
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return MCPResponse(**response_data)
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
        
        return None
    
    async def _send_websocket_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send request via WebSocket"""
        if not self.websocket:
            return None
        
        try:
            await self.websocket.send(json.dumps(asdict(request)))
            response_data = await self.websocket.recv()
            response = json.loads(response_data)
            return MCPResponse(**response)
        except Exception as e:
            logger.error(f"WebSocket request failed: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        self.connected = False
        
        if self.process:
            self.process.terminate()
            self.process = None
        
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info(f"Disconnected from MCP server: {self.config.name}")

class MCPManager:
    """Manages multiple MCP server connections"""
    
    def __init__(self):
        self.servers: Dict[str, MCPClient] = {}
        self.configs: Dict[str, MCPServerConfig] = {}
        
    def add_server(self, config: MCPServerConfig) -> bool:
        """Add a new MCP server configuration"""
        if config.name in self.configs:
            logger.warning(f"MCP server '{config.name}' already exists")
            return False
        
        self.configs[config.name] = config
        self.servers[config.name] = MCPClient(config)
        logger.info(f"Added MCP server configuration: {config.name}")
        return True
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured servers"""
        results = {}
        
        tasks = []
        for name, client in self.servers.items():
            tasks.append(self._connect_server(name, client))
        
        connection_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, _), result in zip(self.servers.items(), connection_results):
            results[name] = result if isinstance(result, bool) else False
        
        return results
    
    async def _connect_server(self, name: str, client: MCPClient) -> bool:
        """Connect to a single server"""
        try:
            success = await client.connect()
            if success:
                logger.info(f"Connected to MCP server: {name}")
            else:
                logger.error(f"Failed to connect to MCP server: {name}")
            return success
        except Exception as e:
            logger.error(f"Error connecting to MCP server {name}: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Call a tool on a specific server"""
        if server_name not in self.servers:
            logger.error(f"MCP server '{server_name}' not found")
            return None
        
        client = self.servers[server_name]
        return await client.call_tool(tool_name, arguments)
    
    async def find_and_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Find and call a tool across all connected servers"""
        for server_name, client in self.servers.items():
            if client.connected and tool_name in client.tools:
                logger.info(f"Calling tool '{tool_name}' on server '{server_name}'")
                return await client.call_tool(tool_name, arguments)
        
        logger.error(f"Tool '{tool_name}' not found on any connected server")
        return None
    
    def get_all_tools(self) -> Dict[str, Dict[str, MCPTool]]:
        """Get all tools from all connected servers"""
        all_tools = {}
        for server_name, client in self.servers.items():
            if client.connected:
                all_tools[server_name] = client.tools
        return all_tools
    
    def get_all_resources(self) -> Dict[str, Dict[str, MCPResource]]:
        """Get all resources from all connected servers"""
        all_resources = {}
        for server_name, client in self.servers.items():
            if client.connected:
                all_resources[server_name] = client.resources
        return all_resources
    
    async def disconnect_all(self):
        """Disconnect from all servers"""
        tasks = []
        for client in self.servers.values():
            if client.connected:
                tasks.append(client.disconnect())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Global MCP manager instance
mcp_manager = MCPManager()

# Add built-in MCP servers
def setup_builtin_servers():
    """Setup built-in MCP server configurations"""
    
    # File system server
    filesystem_config = MCPServerConfig(
        name="filesystem",
        command=["mcp-server-filesystem", "--root", "."],
        transport=MCPTransport.STDIO,
        auto_restart=True
    )
    mcp_manager.add_server(filesystem_config)
    
    # Git server
    git_config = MCPServerConfig(
        name="git",
        command=["mcp-server-git"],
        transport=MCPTransport.STDIO,
        auto_restart=True
    )
    mcp_manager.add_server(git_config)
    
    # Web search server (if available)
    search_config = MCPServerConfig(
        name="search",
        command=["mcp-server-search"],
        transport=MCPTransport.STDIO,
        auto_restart=True
    )
    mcp_manager.add_server(search_config)

# Hook integration
@hook_manager.hook(HookType.PRE_GENERATION, description="Enhance context with MCP resources")
async def enhance_with_mcp_context(context: HookContext):
    """Use MCP to enhance generation context"""
    prompt = context.get("prompt", "")
    
    # Try to find relevant resources or tools
    # This is a simple example - you could implement more sophisticated matching
    if "file:" in prompt or "read file" in prompt.lower():
        # Try to read file content via MCP
        try:
            result = await mcp_manager.find_and_call_tool("read_file", {"path": prompt})
            if result:
                context.set("additional_context", result)
        except Exception as e:
            logger.debug(f"MCP context enhancement failed: {e}")

# Initialize built-in servers
setup_builtin_servers()
