# Revodata Generative AI Toolkit
 
 > *Building Generative AI Applications on Databricks*

The RevoData Generative AI Toolkit is designed to accelerate the development of generative AI applications within the Databricks ecosystem by providing example architectures and modular components.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Prerequisites](#2-prerequisites)
- [3. Architectures](#3-architectures)
  - [3.1 Retrieval Augmented Generation (RAG)](#31-retrieval-augmented-generation-rag)
  - [3.2 Agents (upcoming)](#32-agents-upcoming)
- [4. Components](#4-components)
  - [4.1 Embeddings](#41-embeddings)
  - [4.2 Rerankers](#42-rerankers)
  - [4.3 Large Language Models](#43-large-language-models)
- [5. Modularizing Generative AI Applications](#5-modularizing-generative-ai-applications)
- [6. Roadmap](#6-roadmap)
- [7. Contact Information](#7-contact-information)


## 1. Introduction

Building Generative AI applications requires careful consideration of architectures and technical approaches. As an engineer, you'll need to make several key design decisions:

- **Chain vs. Agent Architecture**: Determine whether a sequential processing chain or an autonomous agent better suits your use case.
- **Single Agent vs. Multi-Agent System**: Evaluate if your application needs a single agent or would benefit from multiple specialized agents collaborating.
- **Single-Turn vs. Multi-Turn Interactions**: Assess whether your application requires contextual memory to maintain conversation history for more coherent and personalized follow-up responses.
- **User Interface**: Decide between direct user-facing interactions requiring thoughtful frontend design, or backend automation processes that operate without human involvement.
- **Retrieval-Augmented Generation**: Assess whether your application requires external knowledge retrieval to enhance the quality and accuracy of generated content.
- **Tool Integrations**: Assess which external tools (databases, APIs, code interpreters, search engines) your application needs to access.
- **Model Customization Needs**: Determine if fine-tuning foundation models would significantly improve performance for your specific domain or tasks.
- **Evaluation Framework**: Establish robust metrics and testing strategies to measure efficacy, reliability, and alignment with business objectives.

This repository helps you prototype and build faster through two main sections: **Architectures**, which are pre-built frameworks ready for experimentation, and **Components**, which are modular building blocks you can combine to create custom applications.

## 2. Prerequisites

In order to work with the code in this repository you need:

1. A Databricks workspace with Vector Search and Serving Endpoints enabled
2. Familiarity with MLflow and Databricks Serving Endpoints
3. The credentials as outlined in the `.env.example` file

When developing user-facing architectures, we will utilize the Databricks Playground to interact with the application. Note that multiple options exist for [accessing your application](/frontend-guide.md) both within and outside the Databricks ecosystem.

## 3. Architectures

The pre-built architectures make it possible to quickly test your use case without having to write all the application logic. At this moment we have the following implementations:

### [3.1 Retrieval Augmented Generation (RAG)](/architectures/retrieval-augmented-generation/)

- **Conversational RAG Chatbot:** Supports multi-turn interactions, allowing for dialogue with contextual understanding across multiple exchanges.
- **Conversational RAG Chatbot with toolcall feedback:** Supports multi-turn interactions with an integrated UI feedback mechanisms, enabling users to see the actions the chatbot is taking and which documents it has retrieved.
- **Business Process Automation:** Designed for single-turn interactions that can be implemented in business process where standalone and structured responses are required (i.e., answering support tickets).

### [3.2 Agents (upcoming)](/architectures/roadmap-experiments/)

While not completed yet, we made some of our experiments available to you.

## 4. Components

Within our architectures, all processing steps are implemented in single Python files. As a result, there is some duplication. Components solve this issue by extracting common transformations into standalone serving endpoints. We have implemented several foundational components:

### [4.1 Embeddings](/components/embeddings/)

- **Sentence Transformers Integration:** Implements an embeddings endpoint using the Sentence Transformers package.
- **Transformers Integration:** Implements an embeddings endpoint with the Transformers package.

### [4.2 Rerankers](/components/rerankers/)

- **Sentence Transformers Reranking:** Implements a reranking endpoint using the Sentence Transformers package.

### [4.3 Large Language Models](/components/large-language-models/)

- **Transformers Text Generation Model Deployment:** Enables deployment of open-source LLMs from Hugging Face on Model Serving Endpoints.

> All these components require hosting on GPU-equipped model serving endpoints. The standard CPU instances only have 4GB of memory which is often not sufficient for these models.

## 5. Modularizing Generative AI Applications

The components outlined above provide a foundation for developing modular Generative AI applications, offering several significant advantages:

- **Cost Efficiency & Scalability:** Deploy compute-intensive components (embeddings, rerankers, LLMs) on specialized GPU instances while running application logic on cost-effective CPU instances. This enables multiple applications to leverage shared AI resources, optimizing costs while maintaining performance.
- **Code Maintainability (DRY Principle):** Consolidate core functionality into reusable components, eliminating redundancy across applications. This approach ensures that updates, optimizations, and bug fixes need to be implemented only once, reducing maintenance overhead while improving system reliability and consistency.

We strongly recommend modularizing your Generative AI applications by building a Python package with reusable application logic, and implementing separate endpoints for compute-intensive components.

## 6. Roadmap

We are actively implementing new architectures and components, the following are planned in the near future:

- [ ]  Agentic architectures
- [ ]  Framework specific examples (LangChain, LangGraph, DSPy, PydanticAI, CrewAI)
- [ ]  Advanced RAG architectures (late chunking, adaptive rag, reflective rag, etc)
- [ ]  Unstructured data parsing pipelines, keep your vector store up to date automatically
- [ ]  Evaluation strategies and data fly-wheel examples

## 7. Contact Information

Do you have questions, feedback, or would you like to see a specific implementation? We'd love to hear from you! Please open an issue on our GitHub repository or reach out to our team at [blog@revodata.nl](mailto:blog@revodata.nl)