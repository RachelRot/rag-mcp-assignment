"""
RAG Agent Module.

Integrates Ollama LLM with MCP tools to create a ReAct agent.
"""

from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from tools import read_file, chunk_and_store, search_document
import logging

logger = logging.getLogger(__name__)

class AgentLogger(BaseCallbackHandler):
    """Callback handler to track agent actions and tool usage."""
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT ACTION: Using tool '{tool_name}'")
        logger.info(f"Input: {input_str}")
        logger.info(f"{'='*60}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"\n{'='*60}")
        logger.info(f"TOOL RESULT:")
        logger.info(f"Output: {output[:200]}..." if len(str(output)) > 200 else f"Output: {output}")
        logger.info(f"{'='*60}\n")
    
    def on_agent_action(self, action, **kwargs):
        logger.info(f"\nAGENT THOUGHT: {action.log}")

def build_agent():
    """
    Build and return a complete RAG agent.
    
    The agent integrates Ollama (Mistral) with three tools:
    - read_file: Read files from disk
    - chunk_and_store: Split text and store in ChromaDB
    - search_document: Semantic search over stored documents
    
    Returns:
        AgentExecutor: Ready-to-use agent executor
    """
    llm = Ollama(model="mistral", temperature=0)
    
    tools = [
        read_file,
        chunk_and_store,
        search_document,
    ]
    
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (ONLY the parameter value, NOT the function call)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- Action Input should be ONLY the parameter value, for example: data/document.txt (NOT read_file("data/document.txt"))
- Do NOT write Final Answer until you have completed ALL required steps
- Follow the instructions in the question step by step

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    ).partial(tool_names=", ".join([t.name for t in tools]))
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10,
        callbacks=[AgentLogger()],
    )
    
    return agent_executor
