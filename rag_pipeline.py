from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sentence_transformers import CrossEncoder
from langfuse import Langfuse

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class MetricsCollector:
    retrieval_times: List[float] = field(default_factory=list)
    rerank_times: List[float] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)
    requests: int = 0
    estimated_cost_per_token: float = field(default_factory=lambda: float(os.environ.get("ESTIMATED_COST_PER_TOKEN", "0.000001")))

    def record_request(self) -> None:
        self.requests += 1

    def record_retrieval(self, value: float) -> None:
        self.retrieval_times.append(value)

    def record_rerank(self, value: float) -> None:
        self.rerank_times.append(value)

    def record_generation(self, value: float) -> None:
        self.generation_times.append(value)

    def record_cost(self, prompt_tokens: int, answer_tokens: int) -> float:
        return round((prompt_tokens + answer_tokens) * self.estimated_cost_per_token, 6)

    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]

    def summary(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "retrieval_p50_ms": round(self._percentile(self.retrieval_times, 0.5) * 1000.0, 2),
            "retrieval_p95_ms": round(self._percentile(self.retrieval_times, 0.95) * 1000.0, 2),
            "rerank_p50_ms": round(self._percentile(self.rerank_times, 0.5) * 1000.0, 2),
            "rerank_p95_ms": round(self._percentile(self.rerank_times, 0.95) * 1000.0, 2),
            "generation_p50_ms": round(self._percentile(self.generation_times, 0.5) * 1000.0, 2),
            "generation_p95_ms": round(self._percentile(self.generation_times, 0.95) * 1000.0, 2),
            "estimated_cost_per_token_usd": round(self.estimated_cost_per_token, 9),
        }


class ProductionRAG:
    def __init__(self) -> None:
        self.data_path = Path(os.environ.get("DATA_PATH", "Data")).resolve()
        self.vector_store_path = Path(os.environ.get("VECTOR_STORE_PATH", "faiss_store")).resolve()
        self.bm25_k = int(os.environ.get("BM25_K", "10"))
        self.vector_k = int(os.environ.get("VECTOR_K", "10"))
        self.top_k = int(os.environ.get("TOP_K", "5"))
        self.rerank_pool_size = int(os.environ.get("RERANK_POOL_SIZE", "20"))
        self.embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self.llm_model = os.environ.get("LLM_MODEL", "phi3:mini")
        self.reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.langfuse = Langfuse()
        self.metrics = MetricsCollector()
        self._built = False
        self.document_count = 0
        self.cache_enabled = os.environ.get("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
        self.query_cache: Dict[str, List[str]] = {}
        self.retrieval_cache: Dict[str, List[Any]] = {}
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.step_logs: List[str] = []

    def _log_step(self, message: str) -> None:
        logger.info(message)
        self.step_logs.append(message)

    def build(self) -> None:
        if self._built:
            return
        self.step_logs = []
        self._log_step("Starting Production RAG build")
        self._load_documents()
        self._build_retrievers()
        self._prepare_prompt_chains()
        self._built = True
        self._log_step("Production RAG pipeline is built and ready.")

    def _load_documents(self) -> None:
        self.documents = []
        pdf_files = sorted(self.data_path.glob("*.pdf"))
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            self.documents.extend(loader.load())

        for doc in self.documents:
            doc.page_content = self.clean_text(doc.page_content)
            self._add_metadata(doc)

        self.document_count = len(self.documents)
        self._log_step(f"Loaded {self.document_count} pages from {self.data_path}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(self.documents)
        self._log_step(f"Split documents into {len(self.chunks)} chunks")

    def _build_retrievers(self) -> None:
        self._log_step("Initializing BM25 retriever")
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = self.bm25_k

        self._log_step(f"Initializing embeddings with model {self.embed_model}")
        self.embedding = OllamaEmbeddings(model=self.embed_model)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        try:
            self.vectorstore = FAISS.load_local(str(self.vector_store_path), embeddings=self.embedding)
            self._log_step(f"Loaded existing FAISS store from {self.vector_store_path}")
        except Exception:
            self._log_step("FAISS store not found, building a new one")
            self.vectorstore = FAISS.from_documents(self.chunks, self.embedding)
            self.vectorstore.save_local(str(self.vector_store_path))
            self._log_step(f"Created FAISS store at {self.vector_store_path}")

        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.vector_k})
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.3, 0.7],
        )
        self._log_step(f"Initializing reranker with model {self.reranker_model}")
        self.reranker = CrossEncoder(self.reranker_model)

    def _prepare_prompt_chains(self) -> None:
        multi_query_prompt = ChatPromptTemplate.from_template(
            """
You are a legal search expert.

Generate 5 alternative search queries for the question below.
Return ONLY the queries, one per line, with no numbering or extra text.

Question:
{question}
"""
        )
        self.query_generator = (multi_query_prompt | ChatOllama(model=self.llm_model) | StrOutputParser())

        answer_prompt = ChatPromptTemplate.from_template(
            """
You are a legal assistant.

Rules:
- Use ONLY the provided context.
- If the answer is not in the context, say exactly "Not found in documents".
- Always include citations in the form [SOURCE=<Act>; SECTION=<Section>; PAGE=<Page>].
- Do not hallucinate or introduce information that is not supported.

Context:
{context}

Question:
{question}

Answer:
"""
        )
        self._log_step("Preparing prompt chains")
        self.answer_chain = answer_prompt | ChatOllama(model=self.llm_model)

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\.{2,}", ".", text)
        return text.strip()

    def _add_metadata(self, doc: Any) -> None:
        source = doc.metadata.get("source", "Unknown")
        act_name = Path(source).stem.replace("_", " ")
        section_match = re.search(r"Section\s+([\dA-Za-z\.]+)", doc.page_content)
        section = section_match.group(1) if section_match else doc.metadata.get("section", "Unknown")
        year_match = re.search(r"(19|20)\d{2}", source)
        year = int(year_match.group()) if year_match else None
        doc.metadata.update({"section": section, "act": act_name, "year": year})

    def generate_queries(self, question: str) -> List[str]:
        self._log_step("Generating alternative search queries")
        if self.cache_enabled and question in self.query_cache:
            self._log_step("Query expansion cache hit")
            return self.query_cache[question]

        output = self.query_generator.invoke({"question": question})
        queries = [q.strip() for q in output.split("\n") if q.strip()]
        result = [question] + queries[:5]
        if self.cache_enabled:
            self.query_cache[question] = result
        self._log_step(f"Generated {len(result)} queries")
        return result

    def reciprocal_rank_fusion(self, results: Iterable[List[Any]], k: int = 60) -> List[Any]:
        fused_scores: Dict[Any, float] = defaultdict(float)
        doc_map: Dict[Any, Any] = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content[:100],
                )
                doc_map[key] = doc
                fused_scores[key] += 1.0 / (k + rank + 1)

        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in reranked]

    def rerank(self, query: str, docs: List[Any], k_top: int | None = None) -> List[Any]:
        if not docs:
            return []
        k_top = k_top or self.top_k
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k_top]]

    def retrieve_and_rerank(self, query: str) -> List[Any]:
        self.build()
        if self.cache_enabled and query in self.retrieval_cache:
            self._log_step("Retrieval cache hit")
            return self.retrieval_cache[query]

        self._log_step("Starting hybrid retrieval")
        trace = self.langfuse.trace(name="lexretriever-query", input={"question": query})

        retrieval_span = trace.span(name="hybrid-multiquery-retrieval")
        start = time.perf_counter()
        queries = self.generate_queries(query)
        retrieved = [self.hybrid_retriever.invoke(q) for q in queries]
        retrieval_latency = time.perf_counter() - start
        self.metrics.record_retrieval(retrieval_latency)
        retrieval_span.end(output={"queries": len(queries), "raw_docs": sum(len(d) for d in retrieved)})
        self._log_step(f"Retrieved {sum(len(d) for d in retrieved)} raw documents using {len(queries)} queries")

        self._log_step("Fusing retrieval results")
        fused_docs = self.reciprocal_rank_fusion(retrieved)
        rerank_span = trace.span(name="cross-encoder-rerank")
        start = time.perf_counter()
        self._log_step(f"Reranking top {min(self.rerank_pool_size, len(fused_docs))} candidate documents")
        final_docs = self.rerank(query, fused_docs[: self.rerank_pool_size], k_top=self.top_k)
        rerank_latency = time.perf_counter() - start
        self.metrics.record_rerank(rerank_latency)
        rerank_span.end(output={"top_docs": len(final_docs)})
        self._log_step(f"Reranking complete, returning {len(final_docs)} final documents")

        if self.cache_enabled:
            self.retrieval_cache[query] = final_docs

        trace.event(
            name="retrieved_documents",
            metadata={
                "documents": [
                    {
                        "source": doc.metadata.get("source"),
                        "act": doc.metadata.get("act"),
                        "section": doc.metadata.get("section"),
                        "page": doc.metadata.get("page"),
                    }
                    for doc in final_docs
                ]
            },
        )

        return final_docs

    def format_context(self, docs: List[Any]) -> str:
        context_lines = []
        for doc in docs:
            meta = doc.metadata
            citation = f"[SOURCE={meta.get('act')}; SECTION={meta.get('section')}; PAGE={meta.get('page')}]"
            context_lines.append(f"{citation}\n{doc.page_content}")
        return "\n\n".join(context_lines)

    def generate_answer(self, question: str, context: str) -> tuple[str, int, float]:
        self._log_step("Generating answer from LLM")
        start = time.perf_counter()
        answer_raw = self.answer_chain.invoke({"context": context, "question": question})
        if hasattr(answer_raw, "content"):
            answer = answer_raw.content
        elif hasattr(answer_raw, "output_text"):
            answer = answer_raw.output_text
        else:
            answer = str(answer_raw)
        answer = answer if isinstance(answer, str) else str(answer)
        latency = time.perf_counter() - start
        self.metrics.record_generation(latency)
        prompt_tokens = len((context + " " + question).split())
        self._log_step(f"Answer generation complete in {latency:.2f} seconds")
        return answer, prompt_tokens, latency

    def ask(self, question: str) -> Dict[str, Any]:
        self.step_logs = []
        self._log_step("Received question")
        self.metrics.record_request()
        request_id = str(uuid.uuid4())
        if self.cache_enabled and question in self.response_cache:
            self._log_step("Returning cached response")
            cached_result = copy.deepcopy(self.response_cache[question])
            cached_result["request_id"] = request_id
            cached_result["step_logs"] = self.step_logs.copy()
            return cached_result

        try:
            docs = self.retrieve_and_rerank(question)
            context = self.format_context(docs)
            answer, prompt_tokens, generation_latency = self.generate_answer(question, context)
            cost = self.metrics.record_cost(prompt_tokens, len(answer.split()))

            sources = [
                {
                    "act": doc.metadata.get("act"),
                    "section": doc.metadata.get("section"),
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
                for doc in docs
            ]

            result = {
                "request_id": request_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "latency_ms": round((self.metrics.retrieval_times[-1] + self.metrics.rerank_times[-1] + generation_latency) * 1000.0, 2),
                "retrieval_ms": round(self.metrics.retrieval_times[-1] * 1000.0, 2) if self.metrics.retrieval_times else 0.0,
                "rerank_ms": round(self.metrics.rerank_times[-1] * 1000.0, 2) if self.metrics.rerank_times else 0.0,
                "generation_ms": round(generation_latency * 1000.0, 2),
                "cost_usd": cost,
                "step_logs": self.step_logs.copy(),
            }

            if self.cache_enabled:
                cache_entry = copy.deepcopy(result)
                cache_entry.pop("request_id", None)
                self.response_cache[question] = cache_entry

            return result
        except Exception:
            logger.exception("Error during RAG ask() pipeline for question: %s", question)
            return {
                "request_id": request_id,
                "question": question,
                "answer": "Not found in documents.",
                "sources": [],
                "latency_ms": 0.0,
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
                "cost_usd": 0.0,
                "step_logs": self.step_logs.copy(),
            }

    def is_ready(self) -> bool:
        return self._built and self.document_count > 0


_rag_instance: ProductionRAG | None = None


def get_rag_instance() -> ProductionRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ProductionRAG()
    return _rag_instance


if __name__ == "__main__":
    rag = get_rag_instance()
    rag.build()
    print("Production RAG pipeline built.")

