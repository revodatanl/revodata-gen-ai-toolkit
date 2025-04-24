import json
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from mlflow.entities import SpanType
from mlflow.models import set_model
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


class BusinessAutomation(mlflow.pyfunc.PythonModel):
    """Business automation."""

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

        self.workspace_client = WorkspaceClient()
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.vector_search_client = VectorSearchClient()
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

    def predict(self, context, model_input) -> dict[str, Any]:
        latest_message = model_input.iloc[0, 0]

        context = self.retrieve(latest_message)

        # Rerank the context
        reranked_context = self.rerank(latest_message, context)

        # Augment the latest message with the context
        latest_message_augmented = self.augment_latest_message(
            latest_message, reranked_context
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Business_automation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                },
                "strict": True,
            },
        }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant, tasked with answering questions about the provided context.",
            },
            {"role": "user", "content": latest_message_augmented},
        ]

        response = self.client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=messages,
            response_format=response_format,
        )
        response_json = json.loads(response.choices[0].message.content)

        # Add the documents to the response
        response_json["context"] = reranked_context

        return response_json

    def predict_stream(self, context, model_input):
        # Since this is a Business automation, we don't need to stream the response.
        # Streaming is usually prefered for user facing applications, as the time to first token is reduced.
        raise NotImplementedError(
            "Streaming is not implemented for business automation processes"
        )


set_model(BusinessAutomation())
