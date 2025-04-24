# Adapted from: https://docs.databricks.com/aws/en/notebooks/source/generative-ai/openai-pyfunc-simple-agent.html
# Notes on tracing: https://mlflow.org/docs/latest/tracing/tracing-schema

########################################################
# This code contains a simple example of a conversational
# chat agent, it does not do RAG,
# but simulates a chat like ChatGPT
########################################################

from typing import Any, Generator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
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

mlflow.openai.autolog()
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"


class ConversationalChat(ChatAgent):
    def __init__(self):
        """
        Initialize the conversational chat agent with necessary clients and models.
        """
        self.workspace_client = WorkspaceClient()
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()

    @mlflow.trace(span_type=SpanType.PARSER)
    def _custom_convert_messages_to_dict(
        self, messages: list[ChatAgentMessage]
    ) -> list[dict[str, Any]]:
        # Depending the input/output requirements of the LLM provider, you might need to convert the messages to a different format.
        # For example, anthropic expects a different format than openai.
        pass

    @mlflow.trace(span_type=SpanType.LLM)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        messages = self._convert_messages_to_dict(messages)

        # Generate the response
        resp = self.client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=messages,
        )

        print(resp)

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
        messages = self._convert_messages_to_dict(messages)

        role = None
        for chunk in self.client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=messages,
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


set_model(ConversationalChat())
