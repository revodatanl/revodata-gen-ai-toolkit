# Revodata Generative AI Toolkit

<div align="center">

![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Flake8](https://img.shields.io/badge/code%20style-flake8-blue.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![uv](https://img.shields.io/badge/uv-0.1.11-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-brightgreen)

</div>

The RevoData Generative AI Toolkit is designed to accelerate the development of generative AI applications within the Databricks ecosystem by providing example architectures and modular components.

## Table of Contents

- [Revodata Generative AI Toolkit](#revodata-generative-ai-toolkit)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Architectures](#architectures)
    - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    - [Agents (upcoming)](#agents-upcoming)
  - [Components](#components)
    - [Embeddings](#embeddings)
    - [Rerankers](#rerankers)
    - [Large Language Models](#large-language-models)
  - [Modularizing Generative AI Applications](#modularizing-generative-ai-applications)
  - [Roadmap](#roadmap)
  - [Contact Information](#contact-information)

## Introduction

Building Generative AI applications involves key architectural and technical decisions. This repository accelerates development with two main sections:

- [Architectures](#architectures): Pre-built frameworks that consist of multiple components for quick experimentation
- [Components](#components): Modular building blocks that together form an architecture

These two sections are designed to work together, allowing you to start with pre-built architectures and then customize them using the provided components.

The architectures and components are designed to help you make the following decisions:

- Chain vs. Agent: Choose between sequential processing or autonomous agents
- Single vs. Multi-Agent: Decide if your app needs one agent or multiple collaborating agents
- Single vs. Multi-Turn: Determine if your app requires contextual memory for conversations
- User Interface: Choose between user-facing interactions or backend automation
- Retrieval-Augmented Generation: Assess if external knowledge retrieval is needed
- Tool Integrations: Identify necessary external tools (databases, APIs, etc.)
- Model Customization: Consider fine-tuning models for your domain
- Evaluation: Establish metrics and testing strategies for efficacy and reliability

## Prerequisites

In order to work with the code in this repository you need:

1. A Databricks workspace with Vector Search and Serving Endpoints enabled
2. Familiarity with MLflow and Databricks Serving Endpoints
3. The credentials as outlined in the `.env.example` file

When developing user-facing architectures, we will utilize the Databricks Playground to interact with the application. Because it is the easiest way to get started and test your application without having to write any frontend code.
Note that multiple options exist for [accessing your application](/frontend-guide.md) both within and outside the Databricks ecosystem.

## Architectures

The pre-built architectures make it possible to quickly test your use case without having to write all the application logic. At this moment we have the following implementations:

The pre-configured architectures make it possible to quickly test your use case without having to write all the application logic and just focus on YOUR business logic.

### [Retrieval Augmented Generation (RAG)](/architectures/retrieval-augmented-generation/)

- Conversational RAG Chatbot: Supports multi-turn interactions, allowing for dialogue with contextual understanding across multiple exchanges.
- Conversational RAG Chatbot with toolcall feedback: Supports multi-turn interactions with an integrated UI feedback mechanisms, enabling users to see the actions the chatbot is taking and which documents it has retrieved.
- Business Process Automation: Designed for single-turn interactions that can be implemented in business process where standalone and structured responses are required (i.e., answering support tickets).

### [Agents (upcoming)](/architectures/roadmap-experiments/)

While not completed yet, we made some of our experiments available to you. usage at your own risk.

## Components

Within our architectures, all processing steps are implemented in single Python files. As a result, there is some duplication. Components solve this issue by extracting common transformations into standalone serving endpoints. We have implemented several foundational components:

### [Embeddings](/components/embeddings/)

- Sentence Transformers Integration: Implements an embeddings endpoint using the Sentence Transformers package.
- Transformers Integration: Implements an embeddings endpoint with the Transformers package.

### [Rerankers](/components/rerankers/)

- Sentence Transformers Reranking: Implements a reranking endpoint using the Sentence Transformers package.

### [Large Language Models](/components/large-language-models/)

- Transformers Text Generation Model Deployment: Enables deployment of open-source LLMs from Hugging Face on Model Serving Endpoints.

> All these components require hosting on GPU-equipped model serving endpoints. The standard CPU instances only have 4GB of memory which is often not sufficient for these models.

## Modularizing Generative AI Applications

The components outlined above provide a foundation for developing modular Generative AI applications, offering several significant advantages:

- Cost Efficiency & Scalability: Deploy compute-intensive components (embeddings, rerankers, LLMs) on specialized GPU instances while running application logic on cost-effective CPU instances. This enables multiple applications to leverage shared AI resources, optimizing costs while maintaining performance.
- Code Maintainability (DRY Principle): Consolidate core functionality into reusable components, eliminating redundancy across applications. This approach ensures that updates, optimizations, and bug fixes need to be implemented only once, reducing maintenance overhead while improving system reliability and consistency.

We strongly recommend modularizing your Generative AI applications by building a Python package with reusable application logic, and implementing separate endpoints for compute-intensive components.

## Roadmap

We are actively implementing new architectures and components, the following are planned in the near future:

- [ ]  Agentic architectures
- [ ]  Framework specific examples (LangChain, LangGraph, DSPy, PydanticAI, CrewAI)
- [ ]  Advanced RAG architectures (late chunking, adaptive rag, reflective rag, etc)
- [ ]  Unstructured data parsing pipelines, keep your vector store up to date automatically
- [ ]  Evaluation strategies and data fly-wheel examples

## Contact Information

Do you have questions, feedback, or would you like to see a specific implementation? We'd love to hear from you! Please open an issue on our GitHub repository or reach out to our team at [blog@revodata.nl](mailto:blog@revodata.nl)
