import json
from typing import Any, Callable, Dict, Generator, List, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from openai import OpenAI
from pydantic import BaseModel

############################################
# Define your LLM endpoint and system prompt
############################################
# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
You are a helpful assistant.
"""


###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable


TOOL_INFOS = []


# Simple addition tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the result."""
    return a + b


# Define the tool specification
add_tool_spec = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
            },
            "required": ["a", "b"],
        },
    },
}

# Add the addition tool to our tools list
TOOL_INFOS.append(ToolInfo(name="add_numbers", spec=add_tool_spec, exec_fn=add_numbers))

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# for details
VECTOR_SEARCH_TOOLS = []

# TODO: Add vector search indexes
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )
for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(
        ToolInfo(
            name=vs_tool.tool["function"]["name"],
            spec=vs_tool.tool,
            exec_fn=vs_tool.execute,
        )
    )


class ToolCallingAgent(ChatAgent):
    """
    Class representing a tool-calling Agent
    """

    def get_tool_specs(self):
        """
        Returns tool specifications in the format OpenAI expects.
        """
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """
        Executes the specified tool with the given arguments.

        Args:
            tool_name (str): The name of the tool to execute.
            args (dict): Arguments for the tool.

        Returns:
            Any: The tool's output.
        """
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def __init__(self, llm_endpoint: str, tools: Dict[str, Dict[str, Any]]):
        """
        Initializes the ToolCallingAgent with tools.

        Args:
            tools (Dict[str, Dict[str, Any]]): A dictionary where each key is a tool name,
            and the value is a dictionary containing:
                - "spec" (dict): JSON description of the tool (matches OpenAI format)
                - "function" (Callable): Function that implements the tool logic
        """
        super().__init__()
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {
            tool.name: tool for tool in tools
        }  # Store tools for later execution

    def prepare_messages_for_llm(
        self, messages: list[ChatAgentMessage]
    ) -> list[dict[str, Any]]:
        """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        return [
            {
                k: v
                for k, v in m.model_dump_compat(exclude_none=True).items()
                if k in compatible_keys
            }
            for m in messages
        ]

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        Primary function that takes a user's request and generates a response.
        """
        # NOTE: this assumes that each chunk streamed by self.call_and_run_tools contains
        # a full message (i.e. chunk.delta is a complete message).
        # This is simple to implement, but you can also stream partial response messages from predict_stream,
        # and aggregate them in predict_stream by message ID
        response_messages = [
            chunk.delta
            for chunk in self.predict_stream(messages, context, custom_inputs)
        ]
        return ChatAgentResponse(messages=response_messages)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        if len(messages) == 0:
            raise ValueError(
                "The list of `messages` passed to predict(...) must contain at least one message"
            )
        all_messages = [
            ChatAgentMessage(role="system", content=SYSTEM_PROMPT)
        ] + messages

        try:
            for message in self.call_and_run_tools(messages=all_messages):
                yield ChatAgentChunk(delta=message)
        except openai.BadRequestError as e:
            error_data = getattr(e, "response", {}).get("json", lambda: None)()
            if error_data and "external_model_message" in error_data:
                external_error = error_data["external_model_message"].get("error", {})
                if external_error.get("code") == "content_filter":
                    yield ChatAgentChunk(
                        messages=[
                            ChatAgentMessage(
                                role="assistant",
                                content="I'm sorry, I can't respond to that request.",
                                id=str(uuid4()),
                            )
                        ]
                    )
            raise  # Re-raise if it's not a content filter error

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def chat_completion(self, messages: List[ChatAgentMessage]) -> ChatAgentResponse:
        return self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
            tools=self.get_tool_specs(),
            # stream=True,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def call_and_run_tools(
        self, messages, max_iter=10
    ) -> Generator[ChatAgentMessage, None, None]:
        current_msg_history = messages.copy()
        for i in range(max_iter):
            with mlflow.start_span(span_type="AGENT", name=f"iteration_{i + 1}"):
                # Get an assistant response from the model, add it to the running history
                # and yield it to the caller
                # NOTE: we perform a simple non-streaming chat completions here
                # Use the streaming API if you'd like to additionally do token streaming
                # of agent output.
                response = self.chat_completion(messages=current_msg_history)
                llm_message = response.choices[0].message
                assistant_message = ChatAgentMessage(
                    **llm_message.to_dict(), id=str(uuid4())
                )
                current_msg_history.append(assistant_message)
                yield assistant_message

                tool_calls = assistant_message.tool_calls
                if not tool_calls:
                    return  # Stop streaming if no tool calls are needed

                # Execute tool calls, add them to the running message history,
                # and yield their results as tool messages
                for tool_call in tool_calls:
                    function = tool_call.function
                    args = json.loads(function.arguments)
                    # Cast tool result to a string, since not all tools return as tring
                    result = str(self.execute_tool(tool_name=function.name, args=args))
                    tool_call_msg = ChatAgentMessage(
                        role="tool",
                        name=function.name,
                        tool_call_id=tool_call.id,
                        content=result,
                        id=str(uuid4()),
                    )
                    current_msg_history.append(tool_call_msg)
                    yield tool_call_msg

        yield ChatAgentMessage(
            content=f"I'm sorry, I couldn't determine the answer after trying {max_iter} times.",
            role="assistant",
            id=str(uuid4()),
        )


set_model(ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS))
