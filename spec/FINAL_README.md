# MARGO روش نهایی — نقشهٔ فایل‌ها

> ⚠️ **منسوخ در بخش روش:** `MARGO-METHOD-v0.2-cavia` (CAVIA-on-z) طبق
> [decisions/ADR-007-adaptation-engine.md](decisions/ADR-007-adaptation-engine.md)
> به‌عنوان موتور adaptation **مرده** است و فقط به‌صورت baseline/ablation منفی
> باقی می‌ماند. هرجا این سند CAVIA را «روش» می‌نامد، ADR-007 مقدم است.

نسخهٔ روش: `MARGO-METHOD-v0.2-cavia`  
پایهٔ فیزیک/داده: `MARGO-SPEC-v0.1` (فریز؛ بازنویسی نمی‌شود)  
`paper_result=false` تا eval پنج‌بذر این روش تمام شود.

این پوشه **روش مقاله** را مشخص می‌کند، نه لایهٔ diagnostic جستجوی جفت.

| فایل | نقش |
|---|---|
| [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md) | ماژول‌ها، تانسورها، قرارداد کنش، فیزیک، CAVIA، آنچه روش نیست |
| [FINAL_DATAFLOW.md](FINAL_DATAFLOW.md) | جریان لحظه‌به‌لحظهٔ train A / train B / meta-test با شکل I/O |
| [FINAL_CONTRIBUTIONS.md](FINAL_CONTRIBUTIONS.md) | سهم نسبت به DAMRL / MRLCO با عدد Kish |
| [FINAL_IMPLEMENTATION_PLAN.md](FINAL_IMPLEMENTATION_PLAN.md) | فاز روی کد فعلی، گیت، فایل‌هایی که باید عوض شوند |
| [CURRENT_ARCHITECTURE.md](CURRENT_ARCHITECTURE.md) | آنچه **الان در ریپو اجرا می‌شود** (لایه A مرده، لایه B BC+جستجو) |

قانون طلایی: خروجی مقاله = خروجی `π(a | h, z)` طول ۲۰.  
`schedule()` فقط فیزیک T و E است. 2-opt / pair refine روش نیست.
