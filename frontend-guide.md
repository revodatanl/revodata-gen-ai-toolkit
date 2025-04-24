# How to Access Databricks Serving Endpoints and Data from a Frontend

> *Integrating Databricks capabilities into web applications and services*

This guide outlines the various methods for accessing Databricks serving endpoints and data from frontend applications. Each approach offers different capabilities, performance characteristics, and implementation complexities to match your specific requirements.

## Databricks REST APIs

Databricks provides comprehensive REST APIs that enable access to virtually all platform capabilities. To use these APIs, you'll need:

- A Databricks workspace URL
- A valid authentication token
- API endpoint mappings in your preferred programming language

### Data Operations with Statement Execution

The [Statement Execution API](https://docs.databricks.com/api/azure/workspace/statementexecution) provides a flexible interface for working with your data:

- **Data Retrieval Methods**:
  - **INLINE**: Retrieve data directly as JSON with pagination if needed
    - Ideal for smaller datasets (limited to 25MiB)
    - Simplifies implementation with direct response handling
  
  - **EXTERNAL_LINKS**: Retrieve links to data in JSON, Apache Arrow, or CSV formats
    - Supports larger datasets (up to 100GiB)
    - Requires additional code to fetch and process linked data
    - Better for production environments with significant data volumes

- **Data Modification**:
  - Databricks doesn't provide dedicated endpoints for INSERT, UPDATE, or DELETE operations
  - Instead, execute SQL queries directly via the Statement Execution API
  - Examples: `INSERT INTO`, `UPDATE`, `MERGE INTO`, `DELETE FROM`

### Vector Search Operations

For applications requiring similarity search or semantic retrieval:

- The [Query an Index API](https://docs.databricks.com/api/azure/workspace/vectorsearchindexes/queryindex) enables vector searches on existing indexes
- Essential for implementing Retrieval Augmented Generation (RAG) patterns
- Supports both exact and approximate nearest neighbors search algorithms

### ML and LLM Model Access

To integrate machine learning capabilities:

- Use the [Query a Serving Endpoint API](https://docs.databricks.com/api/azure/workspace/servingendpoints/query) to interact with deployed models
- Works with any MLflow model flavor deployed to a serving endpoint
- Supports various inference types from classification to text generation

## Databricks SQL Driver for Node.js

For JavaScript/TypeScript applications focused on data operations:

- The [SQL Driver for Node.js](https://github.com/databricks/databricks-sql-nodejs) provides a dedicated client library
- Requires a workspace URL, authentication token, and SQL warehouse ID
- Supports both synchronous and asynchronous query execution
- Includes connection pooling and parameterized queries
- Limited to SQL operations (doesn't support ML model serving)

## Node.js Delta Sharing Connector

For open table formats and cross-organization data sharing:

- [Node.js Delta Sharing Client](https://github.com/goodwillpunning/nodejs-sharing-client) implements the Delta Sharing protocol

## Databricks Foundation Model APIs

For applications requiring simple generative AI capabilities:

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) provide simplified access to large language models
- Follow the [OpenAI API standard](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#chat) for compatibility with existing libraries
- Ideal for implementing chatbots, content generation, and other LLM-based features
