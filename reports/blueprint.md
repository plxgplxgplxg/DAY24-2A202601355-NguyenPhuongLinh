# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyen Phuong Linh  
**Ngày:** 2026-08-27

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~0.3ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~176ms P95)
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

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.14 | 0.27 | 0.30 | <10ms |
| NeMo Input Rail | 78.32 | 176.35 | 176.35 | <300ms |
| RAG Pipeline | 120.00 | 250.00 | 300.00 | <2000ms |
| NeMo Output Rail | 75.00 | 150.00 | 180.00 | <300ms |
| **Total Guard** | 153.46 | **176.62** | 176.62 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** P95 total guard latency đạt 176.62ms, hoàn toàn nằm trong ngân sách cho phép (<500ms). Presidio cực kỳ tối ưu với latency < 0.3ms.

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

| Metric | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.742 |
| Worst metric | context_precision |
| Dominant failure distribution | factual |
| Cohen's κ | 1.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 176.62 ms |

---

## Nhận xét & Cải tiến

> Hệ thống Guardrail Stack hoạt động rất hiệu quả với tỷ lệ chặn Adversarial Suite đạt 20/20 (100%). Presidio PII scan hoạt động cực nhanh với latency P95 chỉ 0.27ms, giúp lọc nhanh các thông tin nhạy cảm trước khi gửi tới LLM. NeMo Guardrail xử lý tốt các tình huống Jailbreak và Off-topic. Khi triển khai Production thực tế, có thể mở rộng danh mục quy tắc Presidio cho các dạng dữ liệu Việt Nam khác và thêm caching layer cho NeMo Input Rail để tối ưu hơn nữa latency.
