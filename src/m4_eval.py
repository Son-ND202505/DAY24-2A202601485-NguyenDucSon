from __future__ import annotations

"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH  # noqa: E402


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _fallback_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question: list[EvalResult] = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs)
        q_tokens = _tokens(question)
        a_tokens = _tokens(answer)
        gt_tokens = _tokens(ground_truth)
        ctx_tokens = _tokens(context_text)

        faithfulness = len(a_tokens & ctx_tokens) / max(len(a_tokens), 1) if a_tokens else 0.0
        answer_relevancy = len(q_tokens & a_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
        context_precision = len(gt_tokens & ctx_tokens) / max(len(ctx_tokens & (gt_tokens | q_tokens)), 1)
        context_recall = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1) if gt_tokens else 0.0

        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=float(faithfulness),
                answer_relevancy=float(answer_relevancy),
                context_precision=float(min(context_precision, 1.0)),
                context_recall=float(context_recall),
            )
        )

    return {
        "faithfulness": _mean([r.faithfulness for r in per_question]),
        "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
        "context_precision": _mean([r.context_precision for r in per_question]),
        "context_recall": _mean([r.context_recall for r in per_question]),
        "per_question": per_question,
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation, falling back to deterministic local scores."""
    if not OPENAI_API_KEY or len(questions) <= 1 or os.getenv("USE_RAGAS_API", "0") != "1":
        return _fallback_eval(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
            "per_question": per_question,
        }
    except Exception as exc:
        print(f"  Warning: RAGAS evaluation failed, using fallback scores: {exc}")
        return _fallback_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, cite retrieved context"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 terms, or increase retrieval top_k"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filters, or stricter RRF cutoff"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template and query rewriting"),
    }
    failures = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        avg_score = sum(metrics.values()) / len(metrics)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": round(float(avg_score), 4),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
