# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Đức Sơn
**Ngày:** 2026-08-26
**Judge model:** openai/gpt-4o-mini (qua OpenRouter)

---

## 1. Pairwise Judge Results

*(`pairwise_judge()` chạy trên 10 cặp answer_a = model_answer (Day 18) vs answer_b = ground_truth, lấy từ `human_labels_10q.json` + `test_set_50q.json`)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ bao nhiêu ngày khi kết hôn? | B | B bổ sung chi tiết "không trừ vào phép năm" mà A thiếu |
| 5 | Mua thiết bị 55 triệu cần ai duyệt? | B | B nêu đúng ngưỡng 50 triệu → CEO duyệt; A mơ hồ ("Giám đốc phòng ban") |
| 12 | Thưởng Tết tối thiểu (≥6 tháng)? | B | B nêu thêm điều kiện cho nhân viên <6 tháng mà A không có |
| 21 | Senior 9 năm: phép năm + lương? | B | B giải thích rõ cách tính thâm niên hơn A |
| 23 | Hoàn trả học phí nghỉ việc sau 8 tháng? | B | B nêu rõ điều kiện cam kết 1 năm làm căn cứ hoàn trả |
| 29 | Tạm ứng 8 triệu quá hạn: ai duyệt, phạt bao nhiêu? | B | B nêu chi tiết quy trình duyệt + cách tính phạt hơn A |
| 33 | Manager 12 năm: phụ cấp + phép năm? | B | B giải thích nguồn gốc số liệu rõ hơn A |
| 41 | Nghỉ bao nhiêu ngày phép năm? | B | B chính xác theo chính sách hiện hành (v2024); A dùng số liệu cũ (v2023) |
| 46 | Thử việc có được nghỉ phép năm? | B | Cả hai đúng nội dung nhưng B nêu thêm bước cần duyệt |
| 50 | Manager dùng VPN cá nhân khi WFH? | B | B đúng chính sách cấm VPN cá nhân; A sai (nói "được") |

---

## 2. Swap-and-Average Results

*(Pass 1 = judge(question, A, B); Pass 2 = judge(question, B, A) rồi convert ngược lại)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | ✓ |
| 5 | B | B | B | ✓ |
| 12 | B | B | B | ✓ |
| 21 | B | B | B | ✓ |
| 23 | B | B | B | ✓ |
| 29 | B | B | B | ✓ |
| 33 | B | B | B | ✓ |
| 41 | B | B | B | ✓ |
| 46 | B | B | B | ✓ |
| 50 | B | B | B | ✓ |

**Position bias rate:** 0.0% (0/10 case không nhất quán — judge cực kỳ ổn định về mặt thứ tự, nhưng đó lại là dấu hiệu của bias khác, xem mục 4)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)
**Judge labels:** kết quả `swap_and_average()` — judge_label = 1 nếu final_winner ∈ {A, tie} (nghĩa là model_answer được coi là tốt), = 0 nếu final_winner = B

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 0 | ✗ |
| 5 | 0 | 0 | ✓ |
| 12 | 1 | 0 | ✗ |
| 21 | 1 | 0 | ✗ |
| 23 | 1 | 0 | ✗ |
| 29 | 0 | 0 | ✓ |
| 33 | 1 | 0 | ✗ |
| 41 | 0 | 0 | ✓ |
| 46 | 1 | 0 | ✗ |
| 50 | 0 | 0 | ✓ |

**Cohen's κ:** 0.000
**Interpretation:** poor (κ ≤ 0 theo thang Landis-Koch) — judge không đồng thuận với con người hơn mức ngẫu nhiên.

---

## 4. Verbosity Bias

Trong 10 case có winner rõ ràng (không có tie nào):
- A thắng + A dài hơn B: 0 / 10 cases
- B thắng + B dài hơn A: 10 / 10 cases
- **Verbosity bias rate:** 100%

**Kết luận:** Đây là bias cực kỳ rõ ràng — judge chọn `ground_truth` (B) trong toàn bộ 10/10 case, và `ground_truth` luôn dài/chi tiết hơn `model_answer` (A). Judge lẫn lộn giữa "đầy đủ chi tiết hơn" với "đúng hơn". Trong 4 câu mà human_label=0 (model_answer thực sự sai), judge tình cờ đúng — nhưng trong 6 câu mà human_label=1 (model_answer thực sự đúng, chỉ ngắn gọn hơn), judge vẫn chọn B chỉ vì B dài hơn. Verbosity bias ở đây là nguyên nhân trực tiếp khiến κ = 0, không phải vì judge kém năng lực đánh giá factual correctness.

---

## 5. Nhận xét chung

> κ = 0.000 — thấp hơn nhiều so với ngưỡng "substantial agreement" (>0.6), cho thấy trong thiết lập so sánh này (model_answer ngắn vs ground_truth dài), LLM judge **không đáng tin cậy** như một proxy cho đánh giá của con người. Position bias rate = 0% nhìn qua có vẻ tốt (judge ổn định), nhưng thực chất "ổn định" ở đây là bias khác che khuất bias vị trí — judge nhất quán chọn văn bản dài hơn ở CẢ HAI lượt swap, nên position_consistent luôn True dù quyết định vẫn sai. Swap-and-average trong trường hợp này chỉ giúp phát hiện position bias (không có), chứ không giúp phát hiện hay giảm verbosity bias. Trong môi trường production, không nên dùng pairwise judge kiểu "model_answer vs ground_truth dài" để đo factual correctness; nên: (1) yêu cầu judge chấm điểm accuracy độc lập với độ dài bằng rubric rõ ràng (ví dụ: liệt kê các fact cần có, chấm theo checklist), (2) chuẩn hóa độ dài hai câu trả lời trước khi đưa vào judge, hoặc (3) dùng con người làm nhãn chính và chỉ dùng LLM judge để lọc sơ bộ số lượng lớn, không dùng làm quyết định cuối.
