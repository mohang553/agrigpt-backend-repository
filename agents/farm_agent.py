"""
Farm Assistant Agent - Main agent implementation
Routes user queries to appropriate tools based on intent.
"""

import json
import os
import asyncio
from typing import Any, Dict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel

from agents.base_agent import BaseAgent, AgentState, BaseTool
from agents.tools import PestsDiseasesTool, GovtSchemesTool
from dotenv import load_dotenv

load_dotenv()

class ToolSelection(BaseModel):
    """Model for tool selection response"""
    tool_name: str
    confidence: float
    reasoning: str


class FarmAssistantAgent(BaseAgent):
    """
    Farm Assistant Agent using LangGraph.
    Routes queries to pests/diseases or government schemes tools.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=temperature,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

        
        # Register tools
        self.register_tool(PestsDiseasesTool())
        self.register_tool(GovtSchemesTool())
        
        # Build the agent graph
        self.graph = self.build_graph()
    
    def build_graph(self):
        """
        Build the LangGraph graph.
        Current implementation: Simple tool routing without StateGraph
        (can be upgraded to full StateGraph later)
        """
        # Note: Full graph implementation would use StateGraph
        # For now, we use a simpler routing mechanism
        return None
    
    async def _select_tool(self, user_message: str) -> ToolSelection:
        """
        Use LLM to select the appropriate tool based on user message.
        
        Args:
            user_message: The user's input query
        
        Returns:
            ToolSelection with tool name and confidence
        """
        
        prompt = f"""You are a farm assistant that helps farmers with agricultural queries.
        
Available tools:
1. pests_and_diseases - For questions about crop diseases, insect pests, plant protection, and crop health
2. govt_schemes - For questions about government schemes, subsidies, loans, and financial assistance

User Query: {user_message}

Based on the user query, select the BEST matching tool. Respond in JSON format:
{{
    "tool_name": "pests_and_diseases" or "govt_schemes",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this tool was selected"
}}

Important: You MUST respond ONLY with valid JSON, no additional text."""
        
        messages = [HumanMessage(content=prompt)]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Parse the JSON response
            content = response.content.strip()
            # Clean up if there's markdown code formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            tool_data = json.loads(content.strip())
            return ToolSelection(**tool_data)
        
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            print(f"Failed to parse tool selection response: {response.content}")
            # Default to pests_and_diseases as fallback
            return ToolSelection(
                tool_name="pests_and_diseases",
                confidence=0.5,
                reasoning="Fallback selection due to parsing error"
            )
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message through the agent.
        
        Args:
            message: User's input query
        
        Returns:
            Dictionary containing tool selection and results
        """
        
        # Step 1: Select appropriate tool
        tool_selection = await self._select_tool(message)
        
        # Step 2: Get the selected tool
        selected_tool: BaseTool = self.tools.get(tool_selection.tool_name)
        
        if not selected_tool:
            return {
                "status": "error",
                "message": f"Tool '{tool_selection.tool_name}' not found",
                "tool_selected": None
            }
        
        # Step 3: Execute the tool
        tool_result = await selected_tool.execute({"query": message})
        
        # Step 4: Format response
        return {
            "status": "success",
            "user_query": message,
            "tool_selected": tool_selection.tool_name,
            "tool_confidence": tool_selection.confidence,
            "tool_reasoning": tool_selection.reasoning,
            "tool_result": tool_result
        }
    
    async def process_message_streaming(self, message: str):
        """
        Process a message with streaming output.
        
        Args:
            message: User's input query
        
        Yields:
            Tool selection updates and results
        """
        
        yield {
            "event": "thinking",
            "data": "Analyzing your query..."
        }
        
        # Select tool
        tool_selection = await self._select_tool(message)
        
        yield {
            "event": "tool_selected",
            "data": {
                "tool_name": tool_selection.tool_name,
                "confidence": tool_selection.confidence,
                "reasoning": tool_selection.reasoning
            }
        }
        
        # Execute tool
        selected_tool = self.tools.get(tool_selection.tool_name)
        if selected_tool:
            result = await selected_tool.execute({"query": message})
            
            yield {
                "event": "result",
                "data": result
            }
        else:
            yield {
                "event": "error",
                "data": f"Tool '{tool_selection.tool_name}' not found"
            }


# Standalone functions for integration with FastAPI

async def run_farm_agent(message: str) -> Dict[str, Any]:
    """
    Standalone function to run the farm agent.
    
    Args:
        message: User query
    
    Returns:
        Agent response
    """
    agent = FarmAssistantAgent()
    return await agent.process_message(message)


async def run_farm_agent_streaming(message: str):
    """
    Standalone function to run the farm agent with streaming.
    
    Args:
        message: User query
    
    Yields:
        Streaming responses
    """
    agent = FarmAssistantAgent()
    async for event in agent.process_message_streaming(message):
        yield event