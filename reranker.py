import os
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, trust_remote_code=True)

    def rerank(self, question: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [(question, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata = {
                **doc.metadata,
                "rerank_score": f"{float(score):.4f}",
            }
            scored_docs.append(doc)

        return sorted(
            scored_docs,
            key=lambda item: float(item.metadata.get("rerank_score", "0")),
            reverse=True,
        )
