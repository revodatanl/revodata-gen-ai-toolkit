import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0.1, max_tokens=1000
)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

mlflow.models.set_model(chain)
