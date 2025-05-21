# Conversational RAG Chatbot with toolcall feedback

In this module, we will expand on the [Conversational RAG Chatbot](../1-conversational-rag). We are going to skip over many of the details discussed there already, so we recommend starting going through that code first if you have not read it yet. The main difference is that with this version we will not only provide an answer, but also stream the tool calls and results, providing the user more context of what the AI is doing. At the end of this module, we will be able to use this updated model in the Databricks Playground.

https://github.com/user-attachments/assets/14e01cca-60a7-4bc1-a808-47c9cf83aee3

## The Architecture of a Conversational RAG Chatbot with Custom Responses on Databricks

In this section, we will discuss the required changes to make the endpoint return not only the final answer but also the steps it is taking to reach that answer. Since it only requires a few small changes, we will start with the detailed overview.

### Detailed Architecture

In the [Conversational RAG Chatbot](../1-conversational-rag), the response from the API was a single message with the answer. In this module, we will return three messages:

1. An assistant message, which contains tool call details; this is the first element you see in the UI in the video above.
2. A tool message; this is the second message you see in the UI and contains the documents after the reranking step.
3. Another assistant message that contains the answer to the user's question.

This will look something like this:

```JSON
{
  "messages": [
    {
      "role": "assistant",
      "name": "retrieve_docs",
      "id": "2cdae5c3-55be-4a80-9e1b-2996f4d7d495",
      "tool_calls": [
        {
          "id": "eb7d3e2d-585d-4882-b42c-005006c4ec57",
          "type": "function",
          "function": {
            "name": "retrieve",
            "arguments": "How did ETL work in the first generation platforms?"
          }
        }
      ],
      "tool_call_id": "eb7d3e2d-585d-4882-b42c-005006c4ec57"
    },
    {
      "role": "tool",
      "content": "Document: Sharing data safely and reliably between internal and external stakeholders is one of the hardest...",
      "name": "retrieve_docs",
      "id": "1c05dd93-ec49-4a6a-b042-b1dedd77c6e6",
      "tool_call_id": "eb7d3e2d-585d-4882-b42c-005006c4ec57"
    },
    {
      "role": "assistant",
      "content":"The provided documents do not explicitly describe how ETL (Extract, Transform, Load) worked in the first generation platforms...",
      "id": "chatcmpl_50b15779-f628-449e-8c9f-ebd5a43d3d00"
    }
  ],
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 1273,
    "completion_tokens": 488,
    "total_tokens": 1761
  }
}
```

In order to make these custom responses work, we need to implement a few workarounds. We do this by making the following two changes:

- **Retrieve context:** Wrap the input to the Vector Index in a message so we can return it as the first chat message.
- **Rerank:** Wrap the output of the reranking step in a tool message so we can return it as the second chat message.

<p align="center">
  <img src="../../../assets/level-3-conversational-rag.png" alt="Interactive question-answering application">
</p>

In this implementation, we 'mimic' tool calls. In full agentic implementations, you would typically let an LLM decide which tool it should use; however, since we have this hardcoded, we construct the tool call messages ourselves. If you want to explore tool use further, take a look at [this example](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/openai-pyfunc-responses-tool-calling-agent.html). OpenAI also provides [good documentation](https://platform.openai.com/docs/guides/function-calling?api-mode=chat) on function calling or tools use.

There are other ways to customize your responses beyond what we've shown here. For example, we're still using the [`mlflow.pyfunc.ChatAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html?highlight=chatagent#mlflow.pyfunc.ChatAgent), which restricts responses to the [`ChatAgentResponse`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatAgentResponse) object. If you need even more customization, you could use [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html). With this approach, you have complete freedom in defining your response formats.

Additionally, you could implement a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, which supports multiple [streaming modes](https://langchain-ai.github.io/langgraph/concepts/streaming/) and even allows you to define custom streaming modes. For a simple use case like ours, we recommend avoiding these frameworks. However, as your application logic becomes more complex and you transition toward [agentic](https://www.anthropic.com/engineering/building-effective-agents) flows, it might be worth exploring such frameworks as they abstract away some of the complexity. Keep in mind that depending on which streaming mode you select, the output from LangGraph will differ, which would require updates on the frontend and might not be compatible with the Databricks Playground.

## Dive into the Code

Now that you are up to speed on what we are going to build, you can find the full implementation [here](ConversationalRAGCustom.py). If you want to register the model in MLflow, consult this [notebook](ConversationalRAGCustom.ipynb). It goes through the steps of registering the model in Unity Catalog and deploying it to a Model Serving Endpoint so that you can use it in the Databricks Playground.
