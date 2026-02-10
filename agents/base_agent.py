"""
Base agent class for LangGraph agents.
Provides common functionality for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel


class AgentState(BaseModel):
    """Base state for agents"""
    
    messages: List[BaseMessage]
    current_tool: Optional[str] = None
    tool_result: Optional[str] = None
    final_response: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class BaseTool(ABC):
    """Abstract base class for agent tools"""
    
    name: str
    description: str
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute the tool with given input"""
        pass


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.tools: Dict[str, BaseTool] = {}
        self.graph: Optional[StateGraph] = None
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the agent"""
        self.tools[tool.name] = tool
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the LangGraph graph for this agent"""
        pass
    
    def get_tools_description(self) -> str:
        """Get description of all available tools"""
        tools_desc = []
        for tool_name, tool in self.tools.items():
            tools_desc.append(f"- {tool_name}: {tool.description}")
        return "\n".join(tools_desc)
    
    @abstractmethod
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message and return response"""
        pass