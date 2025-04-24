from typing import List, Literal

from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StreamWriter
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

########################################################
# Configs
########################################################

EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
VECTOR_SEARCH_ENDPOINT_NAME = "generative_ai_toolkit_vs_endpoint"
CATALOG = "generative_ai_toolkit"
SCHEMA = "use_cases"
INDEX = "delta_lake_definitive_guide_index"

########################################################
# Embeddings & Vector Store
########################################################

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True},
)

vector_store = DatabricksVectorSearch(
    endpoint=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=f"{CATALOG}.{SCHEMA}.{INDEX}",
    embedding=embeddings,
    text_column="text",
)

retriever = vector_store.as_retriever()

########################################################
# LLM Setup
########################################################

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

########################################################
# Router
########################################################


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "not_in_relevant"] = Field(
        ...,
        description="Given a user question choose to route it to a non relevant answer response or a vectorstore.",
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or response to a non relevant question.
The vectorstore contains documents related to the datalake architecture in databricks. All questions related to
the databricks/datalake architecture should be routed to the vectorstore. Otherwise, use a non relevant answer response."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

########################################################
# Retrieval grader
########################################################


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

########################################################
# Generate
########################################################

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

########################################################
# Generate - answer to non relevant question
########################################################

system = """You have received a question that is not related to databricks/datalake architecture. Please apologize and say that you are not able to answer that question. Keep it brief and short."""
denied_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

denied_chain = denied_prompt | llm | StrOutputParser()

########################################################
# Generate - can not answer
########################################################

system = """You have received a question from the user. You are not able to answer that question. Please apologize and say that you are not able to answer that question, suggest rephrasing or asking a different question. Keep it brief and short."""
can_not_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

can_not_answer_chain = can_not_answer_prompt | llm | StrOutputParser()

########################################################
# Hallucination grader
########################################################


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

########################################################
# Answer grader
########################################################


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_answer_grader

########################################################
# Question rewriter
########################################################

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Only return the  rewritten question, no other text."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

########################################################
# Construct the graph
########################################################


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    generation_attempts: int


########################################################
# Graph flow
########################################################
def retrieve(state, writer: StreamWriter):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    writer({"documents": documents})
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    # Initialize or increment the generation counter
    if "generation_attempts" not in state:
        state["generation_attempts"] = 1
    else:
        state["generation_attempts"] += 1

    # Check if we've exceeded the maximum attempts
    if state["generation_attempts"] > 3:
        return Command(name="can_not_answer", kwargs={})

    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_attempts": state["generation_attempts"],
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def generate_denied(state):
    """
    Generate an answer to a non relevant question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]

    generation = denied_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


########################################################
# Graph Edges
########################################################
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "not_in_relevant":
        print("---ROUTE QUESTION GENERATE AN ANSWER FOR NON RELATED QUESTION---")
        return "not_in_relevant"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generation_attempts = state.get("generation_attempts", 0)

    # Check if we've exceeded the maximum attempts
    if generation_attempts >= 3:
        print("---DECISION: TOO MANY GENERATION ATTEMPTS, CANNOT ANSWER---")
        return "can_not_answer"

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def can_not_answer(state):
    """
    Node to handle cases where we've tried generation too many times

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with a message indicating we cannot answer
    """
    print("---CAN NOT ANSWER---")

    question = state["question"]
    generation = can_not_answer_chain.invoke({"question": question})

    return {
        "generation": generation,
    }


########################################################
# Graph
########################################################
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("not_in_relevant", generate_denied)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("can_not_answer", can_not_answer)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "not_in_relevant": "not_in_relevant",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("not_in_relevant", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("can_not_answer", END)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "can_not_answer": "can_not_answer",
    },
)

# Compile the graph
app = workflow.compile()
