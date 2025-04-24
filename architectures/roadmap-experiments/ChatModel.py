import os
from typing import Generator

from mlflow.models import set_model
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)
from openai import OpenAI


class ChatModelPrototype(ChatModel):
    def __init__(self):
        self.model_name = None
        self.max_tokens = None
        self.client = None

    def load_context(self, context):
        self.model_name = "databricks-meta-llama-3-3-70b-instruct"
        self.max_tokens = 800
        self.client = OpenAI(
            api_key=os.getenv("DATABRICKS_TOKEN"),
            base_url=os.getenv("DATABRICKS_SERVING_ENDPOINT"),
        )

    def _prepare_messages(self, messages):
        return [msg.to_dict() for msg in messages]

    def predict(self, context, messages, params=None) -> ChatCompletionResponse:
        messages = self._prepare_messages(messages)

        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=False,
        )

        chat_completion_dict = chat_completion.model_dump()

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        content=chat_completion_dict["choices"][0]["message"][
                            "content"
                        ],
                        role=chat_completion_dict["choices"][0]["message"]["role"],
                    ),
                )
            ],
            model=chat_completion_dict["model"],
        )

    def predict_stream(
        self, context, messages, params=None
    ) -> Generator[ChatCompletionChunk, None, None]:
        messages = self._prepare_messages(messages)

        chat_completion_chunks = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in chat_completion_chunks:
            yield ChatCompletionChunk(
                choices=[
                    ChatChunkChoice(
                        delta=ChatChoiceDelta(
                            content=choice.delta.content, role=choice.delta.role
                        ),
                    )
                    for choice in chunk.choices
                ],
                model=chunk.model,
            )


set_model(ChatModelPrototype())
