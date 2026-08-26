# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Đức Sơn
**Ngày:** 2026-08-26

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~20ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency(), đo trên 10 adversarial inputs)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 6.82 | 19.94 | 19.94 | <10ms |
| NeMo Input Rail | 1.16 | 1.28 | 1.28 | <300ms |
| RAG Pipeline | (không đo trong Task 12 — pipeline chạy riêng ở `setup_answers.py`) | — | — | <2000ms |
| NeMo Output Rail | (dùng chung code path với Input Rail, latency tương đương ~1ms) | — | — | <300ms |
| **Total Guard** | 8.01 | **21.23** | 21.23 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No
**Comment:** Guard stack rất nhanh, đạt ~4% ngân sách 500ms. Presidio (regex-based) vượt nhẹ mục tiêu lý tưởng <10ms ở P95 (19.94ms) — do phải chạy nhiều PatternRecognizer (VN_CCCD, VN_PHONE, PHONE_NUMBER, EMAIL_ADDRESS) tuần tự trên mỗi input, nhưng vẫn nằm sâu trong ngân sách tổng 500ms nên không phải bottleneck thực sự. NeMo input rail cực nhanh (~1ms) vì chặn ngay bằng colang canonical-form matching (embedding similarity), không cần gọi LLM — chỉ khi không rail nào match thì mới phải gọi LLM thật (chậm hơn nhiều, ~1-3s tùy OpenRouter). Layer chậm nhất trong toàn hệ thống thực chất là RAG Pipeline (gọi LLM sinh câu trả lời cuối), không nằm trong phạm vi đo P95 của Task 12.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.678 (trung bình 3 distribution: 0.861 / 0.600 / 0.574) |
| Worst metric | answer_relevancy (27/50 câu có đây là worst_metric) |
| Dominant failure distribution | factual (nhiều lượt "worst" nhất do số câu đông; nhưng avg_score thấp nhất thực chất là adversarial 0.574) |
| Cohen's κ | 0.000 (poor agreement — xem `analysis/bias_report.md`) |
| Adversarial pass rate | 20 / 20 (100%) |
| Guard P95 latency | 21.23 ms |

---

## Nhận xét & Cải tiến

> Guardrail stack (Phase C) hoạt động rất tốt: chặn 20/20 adversarial input với latency chỉ 21ms — vượt xa yêu cầu ≥15/20 và ngân sách 500ms. Điểm sáng lớn nhất là NeMo input rail chặn được toàn bộ jailbreak/off-topic/prompt-injection chỉ bằng colang pattern matching, không cần gọi LLM, nên vừa nhanh vừa rẻ. Điểm cần cải thiện nằm ở Phase A và B: RAGAS avg_score thấp nhất ở adversarial (0.574) do context_recall kém khi corpus có nhiều phiên bản chính sách (v2023/v2024) — cần thêm metadata filter theo version và bật reranker thật (`USE_REAL_MODELS`) thay vì lexical fallback. Phase B cho thấy LLM-as-judge không đáng tin (κ=0, verbosity bias 100%) khi so sánh câu trả lời ngắn với ground truth dài — nếu deploy production thật, tôi sẽ không dùng judge kiểu pairwise so với ground truth để quyết định chất lượng tự động, mà dùng judge với rubric chấm điểm độc lập theo từng tiêu chí (accuracy tách biệt hoàn toàn khỏi độ dài câu trả lời), đồng thời tăng cỡ mẫu human-labeled để hiệu chỉnh threshold trước khi tin tưởng judge trong CI gate.
