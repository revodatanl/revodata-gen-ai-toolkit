# Adapted from: https://docs.databricks.com/aws/en/notebooks/source/generative-ai/openai-pyfunc-simple-agent.html
# Notes on tracing: https://mlflow.org/docs/latest/tracing/tracing-schema

import os
from typing import Any, Generator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from mlflow.entities import SpanType
from mlflow.models import set_model
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
    ChatUsage,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

mlflow.openai.autolog()

LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
N_DOCUMENTS_TO_RETRIEVE = 20
N_DOCUMENTS_TO_RETRIEVE_RERANKED = 4
VECTOR_SEARCH_ENDPOINT_NAME = "generative_ai_toolkit_vs_endpoint"
CATALOG = "generative_ai_toolkit"
SCHEMA = "use_cases"
INDEX = "delta_lake_definitive_guide_index"


class ConversationalRAG(ChatAgent):
    """
    Conversational RAG chain.
    """

    # "databricks-agents>=0.19.0",
    def __init__(self):
        """
        Initialize the ConversationalRAG chain with necessary clients and models.

        Sets up connections to:
        - Databricks workspace client for API access
        - LLM serving endpoint for text generation
        - Vector search client and index for retrieval
        - Sentence transformer model for query embedding
        - CrossEncoder model for reranking
        """

        self.workspace_client = WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"),
            client_id=os.getenv("DATABRICKS_CLIENT_ID"),
            client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
        )
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.vector_search_client = VectorSearchClient(
            workspace_url=os.getenv("DATABRICKS_HOST"),
            service_principal_client_id=os.getenv("DATABRICKS_CLIENT_ID"),
            service_principal_client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
        )
        self.vector_search_index = self.vector_search_client.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=f"{CATALOG}.{SCHEMA}.{INDEX}",
        )
        self.embeddings_client = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            model_kwargs={"default_task": "retrieval.query"},
        )
        self.reranker_client = CrossEncoder(
            RERANKER_MODEL,
            model_kwargs={"torch_dtype": "auto"},
            trust_remote_code=True,
        )

    @mlflow.trace(span_type=SpanType.RERANKER)
    def rerank(self, query: str, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rerank the context using the reranker client"""
        documents = [result["text"] for result in context]
        reranked_context = self.reranker_client.rank(
            query,
            documents,
            return_documents=True,
            convert_to_tensor=True,
            top_k=N_DOCUMENTS_TO_RETRIEVE_RERANKED,
        )

        # Convert tensor scores to float
        for item in reranked_context:
            if "score" in item and hasattr(item["score"], "item"):
                item["score"] = float(item["score"].item())

        return reranked_context

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve the most relevant documents from the vector search index"""
        query_vector = self.embed_query(query)
        context = self.vector_search_index.similarity_search(
            query_vector=query_vector,
            columns=["id", "text"],
            query_text=query,
            num_results=N_DOCUMENTS_TO_RETRIEVE,
            query_type="hybrid",
        )
        return self.parse_context(context)

    @mlflow.trace(span_type=SpanType.EMBEDDING)
    def embed_query(self, query: str) -> list[float]:
        """Embed a query using the embeddings client"""
        return self.embeddings_client.encode(query).tolist()

    @mlflow.trace(span_type=SpanType.PARSER)
    def parse_context(self, context: dict[str, Any]) -> list[dict[str, Any]]:
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

    @mlflow.trace(span_type=SpanType.PARSER)
    def augment_latest_message(
        self, latest_message: str, reranked_context: list[dict[str, Any]]
    ) -> str:
        """Augment the latest message with the context"""
        context_str = ""
        for i, doc in enumerate(reranked_context):
            context_str += f"<document_id: {doc['corpus_id']}>\n{doc['text']}\n</document_id: {doc['corpus_id']}>\n\n"

        latest_message_augmented = (
            f"Question: {latest_message}\n\nContext:\n{context_str}"
        )

        return latest_message_augmented

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

    @mlflow.trace(span_type=SpanType.LLM)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        parsed_messages = self.prepare_messages_for_llm(messages)

        # Retrieve relevant context
        latest_message = parsed_messages[-1]["content"]
        context = self.retrieve(latest_message)

        # Rerank the context
        reranked_context = self.rerank(latest_message, context)

        # Augment the latest message with the context
        latest_message_augmented = self.augment_latest_message(
            latest_message, reranked_context
        )
        parsed_messages[-1]["content"] = latest_message_augmented

        # Generate the response
        resp = self.client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=parsed_messages,
        )

        return ChatAgentResponse(
            finish_reason=resp.choices[0].finish_reason,
            messages=[
                ChatAgentMessage(
                    **resp.choices[0].message.to_dict(),
                    id=resp.id,
                )
            ],
            usage=ChatUsage(
                **resp.usage.to_dict(),
            ),
        )

    @mlflow.trace(span_type=SpanType.LLM)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        parsed_messages = self.prepare_messages_for_llm(messages)

        # Retrieve relevant context
        latest_message = parsed_messages[-1]["content"]
        context = self.retrieve(latest_message)

        # Rerank the context
        reranked_context = self.rerank(latest_message, context)

        # Augment the latest message with the context
        latest_message_augmented = self.augment_latest_message(
            latest_message, reranked_context
        )
        parsed_messages[-1]["content"] = latest_message_augmented

        # Generate the response
        for chunk in self.client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=parsed_messages,
            stream=True,
        ):
            if not chunk.choices:
                continue

            # Extract role from the first chunk that contains it
            if chunk.choices[0].delta.role:
                role = chunk.choices[0].delta.role

            # Skip chunks without content
            if not chunk.choices[0].delta.content:
                continue

            yield ChatAgentChunk(
                finish_reason=chunk.choices[0].finish_reason,
                delta=ChatAgentMessage(
                    **chunk.choices[0].delta.to_dict(),
                    id=chunk.id,
                    role=role,
                ),
                usage=ChatUsage(
                    **chunk.usage.to_dict(),
                ),
            )


set_model(ConversationalRAG())
