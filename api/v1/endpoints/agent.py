"""
Agent API endpoint for FastAPI.
Handles /agent endpoint that accepts user messages and routes them to appropriate tools.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio
from typing import Optional

from agents.farm_agent import run_farm_agent, run_farm_agent_streaming


# Request/Response Models
class AgentRequest(BaseModel):
    """Request model for agent endpoint"""
    message: str = Field(..., description="User's query message")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the best ways to prevent rice leaf blast?",
                "stream": False
            }
        }


class ToolInfo(BaseModel):
    """Information about selected tool"""
    tool_name: str
    confidence: float
    reasoning: str


class AgentResponse(BaseModel):
    """Response model for agent endpoint"""
    status: str
    user_query: str
    tool_selected: str
    tool_confidence: float
    tool_reasoning: str
    tool_result: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "user_query": "What are the best ways to prevent rice leaf blast?",
                "tool_selected": "pests_and_diseases",
                "tool_confidence": 0.95,
                "tool_reasoning": "User is asking about crop disease prevention",
                "tool_result": "[PESTS & DISEASES TOOL]..."
            }
        }


# Create router
router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/",
    response_model=AgentResponse,
    summary="Process user message through farm agent",
    description="Analyzes user query and routes it to the appropriate tool (pests/diseases or govt schemes)"
)
async def agent_endpoint(request: AgentRequest):
    """
    Agent endpoint that processes user messages.
    
    The agent automatically selects the appropriate tool based on the query:
    - pests_and_diseases: For crop health, disease, and pest-related questions
    - govt_schemes: For government scheme, subsidy, and loan-related questions
    
    Args:
        request: AgentRequest containing the user message
    
    Returns:
        AgentResponse with selected tool and results
    
    Raises:
        HTTPException: If processing fails
    """
    
    try:
        # Process the message through the agent
        result = await run_farm_agent(request.message)
        
        if result["status"] != "success":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Agent processing failed")
            )
        
        return AgentResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )



@router.get(
    "/tools",
    summary="Get available tools",
    description="Returns list of available tools and their descriptions"
)
async def get_tools():
    """
    Returns information about available tools.
    
    Returns:
        Dictionary with tool names and descriptions
    """
    return {
        "tools": [
            {
                "name": "pests_and_diseases",
                "description": "Information about crop pests and diseases, symptoms, and treatment"
            },
            {
                "name": "govt_schemes",
                "description": "Information about government schemes, subsidies, and financial assistance"
            }
        ]
    }


@router.post(
    "/test",
    summary="Test agent with sample queries",
    description="Test the agent with predefined sample queries"
)
async def test_agent():
    """
    Test the agent with sample queries to verify functionality.
    
    Returns:
        Test results for each sample query
    """
    
    test_queries = [
        "How can I prevent powdery mildew on wheat?",
        "What are government subsidies available for organic farming?",
        "My rice plants have brown spots, what should I do?",
        "Tell me about PM-KISAN scheme eligibility",
    ]
    
    results = []
    
    for query in test_queries:
        result = await run_farm_agent(query)
        results.append(result)
    
    return {
        "status": "success",
        "test_count": len(test_queries),
        "results": results
    }