from typing import Any, Optional, Sequence, Union

import mlflow
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import (
    ChatDatabricks,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.models import set_model
from sentence_transformers import SentenceTransformer

# source: https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-tool-calling-agent.html

mlflow.langchain.autolog()


############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
N_DOCUMENTS_TO_RETRIEVE = 3
N_DOCUMENTS_TO_RETRIEVE_RERANKED = 4
VECTOR_SEARCH_ENDPOINT_NAME = "generative_ai_toolkit_vs_endpoint"
CATALOG = "generative_ai_toolkit"
SCHEMA = "use_cases"
INDEX = "delta_lake_definitive_guide_index"


vector_search_client = VectorSearchClient()
embeddings_client = SentenceTransformer(
    EMBEDDING_MODEL,
    trust_remote_code=True,
    model_kwargs={"default_task": "retrieval.query"},
)
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = (
    "You are a helpful assistant that can answer questions and help with tasks"
    "You have two tools, a calculator and a vector search tool for databricks documentation"
)

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
# for details

vector_search_index = vector_search_client.get_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=f"{CATALOG}.{SCHEMA}.{INDEX}",
)


def embed_query(query: str) -> list[float]:
    """Embed a query using the embeddings client"""
    return embeddings_client.encode(query).tolist()


def parse_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the context into a list of dictionaries"""
    data_array = context["result"]["data_array"]
    columns = context["manifest"]["columns"]

    parsed_results = []
    for item in data_array:
        parsed_result = {}
        for i, column in enumerate(columns):
            column_name = column["name"]
            parsed_result[column_name] = item[i]
        parsed_results.append(parsed_result)

    return parsed_results


@tool
def retrieve(query: str) -> list[dict[str, Any]]:
    """Retrieve documents from a vector store that contains a book about data engineering and databricks"""

    # https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/
    writer = get_stream_writer()

    query_vector = embed_query(query)
    context = vector_search_index.similarity_search(
        query_vector=query_vector,
        columns=["id", "text"],
        query_text=query,
        num_results=N_DOCUMENTS_TO_RETRIEVE,
        query_type="hybrid",
    )

    parsed_context = parse_context(context)

    writer({"retrieved_documents": parsed_context})
    return parsed_context


# Fix: Add individual tools to the list instead of a nested list
tools.append(retrieve)

#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(mlflow.pyfunc.PythonModel):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(self, context, model_input):
        model_input_dict = model_input.to_dict(orient="records")[0]

        return self.agent.invoke(model_input_dict)

    def predict_stream(self, context, model_input):
        return self.agent.stream(model_input, stream_mode=["custom", "messages"])


agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
set_model(AGENT)
