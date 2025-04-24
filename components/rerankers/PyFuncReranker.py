import mlflow
from mlflow.models import set_model


class Reranker(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def load_context(self, context):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            "jinaai/jina-reranker-v2-base-multilingual",
            model_kwargs={"torch_dtype": "auto"},
            trust_remote_code=True,
        )

    def predict(self, context, model_input, params=None):
        model_input_dict = model_input.to_dict(orient="records")[0]

        query = model_input_dict.get("query")
        documents = model_input_dict.get("documents")

        rankings = self.model.rank(
            query, documents, return_documents=True, convert_to_tensor=True, top_k=10
        )

        # Convert tensor scores to float and sort by score in descending order
        for item in rankings:
            if hasattr(item["score"], "item"):
                item["score"] = item["score"].item()
            else:
                item["score"] = float(item["score"])

        # Sort rankings by score in descending order
        rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)

        return rankings


set_model(Reranker())
