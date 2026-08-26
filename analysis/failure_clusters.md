# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Đức Sơn
**Ngày:** 2026-08-26

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.8919 | 0.5799 | 0.6970 |
| answer_relevancy | 0.7359 | 0.4997 | 0.4231 |
| context_precision | 0.9324 | 0.7263 | 0.7125 |
| context_recall | 0.8840 | 0.5939 | 0.4626 |
| **avg_score** | **0.8611** | **0.6000** | **0.5738** |

*(Nguồn: `reports/ragas_50q.json`, chạy trên 50/50 câu qua `evaluate_ragas()` — fallback deterministic scorer của m4_eval.py vì `USE_RAGAS_API` không bật.)*

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách? | 0.2071 | answer_relevancy |
| 2 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.2667 | faithfulness |
| 3 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.3034 | faithfulness |
| 4 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.3184 | answer_relevancy |
| 5 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.3408 | answer_relevancy |
| 6 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.3611 | answer_relevancy |
| 7 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.4026 | answer_relevancy |
| 8 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.4339 | answer_relevancy |
| 9 | multi_hop | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn 15 ngày). Ai phê duyệt khoản này và phí phạt là bao nhiêu? | 0.4605 | answer_relevancy |
| 10 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.4872 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col; tổng mỗi cột = 20/20/10)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | 4 | 1 | 7 |
| answer_relevancy | 14 | 9 | 4 | 27 |
| context_precision | 2 | 2 | 0 | 4 |
| context_recall | 2 | 5 | 5 | 12 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (20 câu tổng cộng có worst_metric rơi vào 1 trong 4 nhóm, nhưng vì factual có nhiều câu nhất trong 3 nhóm nên tổng số "lượt là worst" theo cột cao nhất)
**Dominant metric:** answer_relevancy (27/50 câu, tức hơn một nửa toàn bộ test set, có answer_relevancy là điểm yếu nhất)

**Lý do phân tích:**

> `answer_relevancy` áp đảo hoàn toàn ở cả 3 distribution (14/20 factual, 9/20 multi_hop, 4/10 adversarial). Nguyên nhân gốc nằm ở cách `setup_answers.py` sinh câu trả lời: prompt hệ thống yêu cầu "Trả lời CHỈ dựa trên context", nên khi context retrieval không khớp chính xác với cách diễn đạt câu hỏi, LLM có xu hướng trả lời đúng chính sách nhưng lệch trọng tâm câu hỏi (ví dụ trả lời cả đoạn chính sách thay vì con số cụ thể được hỏi). Đây là vấn đề template + query-context alignment, không phải retrieval sai hoàn toàn (context_precision vẫn cao ở factual: 0.93). `context_recall` là điểm yếu thứ hai và rõ nhất ở `adversarial` (0.46) — đúng như thiết kế test set: các câu hỏi liên quan version conflict (v2023/v2024) hoặc câu hỏi phủ định khiến hybrid search (BM25+dense, do reranker đang chạy ở chế độ lexical fallback vì `USE_REAL_MODELS` chưa bật ở Day 18) khó chọn đúng chunk phiên bản mới nhất.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Tighten system prompt, lower temperature, buộc trích dẫn nguyên văn câu trong context khi trả lời số liệu |
| context_recall | Missing relevant chunks | Bật `USE_REAL_MODELS` để dùng cross-encoder reranker thật thay vì lexical fallback; tăng `top_k` cho câu multi-hop/adversarial |
| context_precision | Too many irrelevant chunks | Thêm metadata filter theo `version` (v2023 vs v2024) để loại chunk lỗi thời trước khi đưa vào context |
| answer_relevancy | Answer doesn't match question | Redesign prompt template: yêu cầu trả lời thẳng vào con số/điều kiện được hỏi trước, giải thích sau; thêm few-shot ví dụ cho câu hỏi multi-hop |

---

## 6. Nhận xét về Adversarial Distribution

> Avg_score: factual (0.861) > multi_hop (0.600) > adversarial (0.574) — đúng với giả thuyết ban đầu rằng adversarial là distribution khó nhất, nhưng thú vị là multi_hop và adversarial khá sát nhau, cho thấy pipeline hiện tại "sợ" việc kết hợp nhiều tài liệu/tính toán ngang với việc bị bẫy bởi version conflict. Trong bottom 10, có 3/10 câu thuộc adversarial (#3, #5, #7) — đều là các câu về thâm niên/VPN cá nhân/bảo hiểm thử việc, nơi context chứa cả quy định cũ và mới hoặc câu trả lời "an toàn nhưng sai chính sách" (ví dụ #5 trả lời "được dùng VPN cá nhân" trong khi chính sách v1.3 cấm hoàn toàn). Điều này khẳng định pipeline bị nhầm bởi version conflict như thiết kế test set dự đoán, và context_recall thấp nhất ở adversarial (0.463) là bằng chứng định lượng rõ ràng nhất.
