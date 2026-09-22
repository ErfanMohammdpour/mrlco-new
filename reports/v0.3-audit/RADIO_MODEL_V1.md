# ④ RADIO_MODEL_V1 — جداسازی فیزیک رادیو و اجرای مجدد تضمین‌های ③

**تاریخ:** ۲۰۲۶-۰۹-۲۱ · **وضعیت:** پیاده و تست‌شده · **کل تست‌های غیر-TF: ۳۳۸ passed, 5 skipped** (۱۲ تست جدید رادیو)
**فایل اصلی:** `env/mec_offloaing_envs/scheduler/radio.py` · **ابزار audit:** `spec/radio_audit.py` · **خروجی خام:** `reports/energy/radio_model_v1.json`

---

## ۱. مسئله‌ای که ④ حل می‌کند
پیش از این، کل زنجیرهٔ زمان‌بندی یک عدد ضمنی مصرف می‌کرد: `resource_rates.mec_uplink_mbps = 7` → `917504 B/s`. آن «Mbps» یک فرض پنهان **۱ bit/s/Hz** را در خود داشت، در حالی که پهنای‌باند یک بازهٔ فرکانسی است، نه throughput.

```
B [Hz] · η [bit/s/Hz] = R [bit/s]          (eta model)
B [Hz] · log2(1 + SINR_linear) = R [bit/s] (SINR model)
bytes/s = R / 8
```

## ۲. API
- `RadioLinkSpec(bandwidth_hz, rate_model ∈ {eta, sinr}, spectral_efficiency | sinr_db, assumption, source)` با پراپرتی `effective_rate_bps` و `effective_rate_bytes_per_second`.
- `RadioModelSpec(model ∈ {legacy, physical_v1}, links={v2i_ul, v2i_dl, v2v})` + `rate_for_hop(hop)` + `assumptions()` + `as_dict()`.
- `effective_rate_bps(bandwidth_hz, eta=... | sinr_db=...)` و `transfer_time_seconds(bytes, rate)` (rate ≤ 0 ⇒ خطا، نه تقسیم بر صفر).
- `ResourceConfig.hop_rate()` اکنون model-aware است: `physical` ⇒ `radio_model.rate_for_hop(hop)`؛ در غیر این صورت جدول legacy دست‌نخورده.
- `ResourceConfig.from_frozen_yaml(..., energy_model=..., radio_model=...)`: دو سوئیچ **مستقل** (پیش‌فرض legacy برای هر دو) تا audit رادیو-تنها ممکن باشد و با تغییر فیزیک محاسبه قاطی نشود.

## ۳. پارامترها (هر لینک با منبع و پرچم فرض)
| لینک | B | η | نرخ مؤثر | منبع / وضعیت |
|---|---|---|---|---|
| `v2i_ul` | ۱۰ MHz | ۱٫۵ bit/s/Hz | ۱۵ Mbit/s = **۱٫۸۷۵e۶ B/s** | B: Liu2023 Table 1 · η: Michailidis2021 Table 2 (`r_t = 1.5 bps/Hz`) |
| `v2i_dl` | ۱۰ MHz | ۱٫۵ | همان | همان‌ها |
| `v2v` | ۱۰ MHz | ۱٫۵ | همان | ⚠️ **`assumption: true`** — هیچ B/η مخصوص PC5 در ادبیات بررسی‌شده تأیید نشد |

مقایسه با legacy: UL/DL **×۲٫۰۴**، V2V **×۲٫۸۶** (چون legacy برای V2V فقط ۵ Mib/s داشت).

## ۴. اثر عددی (audit، ۶ گراف، dist 1)
میانهٔ makespan پلن‌های خالص (ثانیه):

| مدل | all_UE | all_MEC | all_HELPER |
|---|---|---|---|
| legacy | ۱۴۰۷٫۲ | ۴۸۹٫۷ | ۱۶۵۲٫۰ |
| **radio_only** (فیزیک محاسبه = legacy، رادیو = physical) | ۱۴۰۷٫۲ | **۲۸۲٫۶** | **۱۴۷۶٫۲** |
| physical (هر دو) | ۳۵۴۱٫۴ | ۴۵۳٫۷ | ۲۴۲۹٫۹ |

- رادیو-تنها: UE ×۱٫۰۰ (بدون انتقال)، MEC ×۱٫۷۳۲، HELPER ×۱٫۱۱۹ ⇒ همان‌طور که انتظار می‌رود فقط مسیرهایی که انتقال دارند سریع‌تر می‌شوند.
- زمان انتقال ۱ MiB: UL/DL از ۱٫۱۴۳s به ۰٫۵۵۹s · V2V از ۱٫۶۰۰s به ۰٫۵۵۹s.
- **نکتهٔ مهم:** جدول `radio_only` نشان می‌دهد بهبود MEC مستقل از تغییر فیزیک محاسبه است؛ دو تغییر جدا اندازه‌گیری می‌شوند و به هم نسبت داده نمی‌شوند.

## ۵. اجرای مجدد تضمین‌های ③ (الزام تصویب‌شده)
`test_radio_model.py::TestSuffixGuaranteesUnderPhysicalRadio` — ۵ تست:
| تضمین | نتیجه |
|---|---|
| `transfer_lower_bound` نرخ جدید را می‌گیرد | ۱٫۸۷۵e۶ بایت روی MEC→UE دقیقاً **۱٫۰s**؛ legacy کندتر |
| صدا بودن LB (admissibility) | برای هر سه action × ۹ suffix، `LB ≤ availability` واقعی |
| صدا بودن mask | هر action ماسک‌شده ⇒ هیچ continuation ددلاین hard را نمی‌گیرد |
| شدنی بودن suffix و potential | suffix ساخته می‌شود؛ `Ĵ(s_T) = J_actual` (places=6) |
| سوئیچ رادیو صدا بودن را نمی‌شکند | radio سریع‌تر ممکن است actionهای **بیشتری** باز بگذارد، هرگز کمتر؛ و در هر دو مدل mask صدا می‌ماند |

## ۶. تست‌های واحد رادیو (۷ تست)
- legacy **byte-exact**: `hop_rate` همان ۹۱۷٬۵۰۴ و ۶۵۵٬۳۶۰ B/s.
- `R = B·η` دقیق و `bytes/s = R/8`.
- **تست ضدفرض پنهان:** `rate != B/8`؛ دو برابر کردن η یا B دقیقاً نرخ را دو برابر می‌کند.
- فرم SINR: `B·log2(1+10^(10/10))` تا places=3.
- مونوتونی: B↑⇒R↑ · η↑⇒R↑ · bytes↑⇒t↑.
- اعتبارسنجی: نبود η در eta، نبود sinr_db، B≤۰، مدل نامعتبر، لینک ناقص، لینک ناشناخته.
- YAML: بلوک `radio_model` لود می‌شود، هر لینک `source` دارد، و `v2v` در `assumptions()` هست (فرض پنهان نمی‌ماند).

## ۷. آنچه ④ عمداً تغییر نداد
- **توان/انرژی رادیو** در مدل فیزیکی همان TX-only است (بدون اختراع `p_rx`).
- مسیر legacy دست‌نخورده و byte-exact.
- PPO/constraint/potential: هیچ تغییری.
- فیزیک محاسبه (κ, f, cycles/bit): دست‌نخورده.

## ۸. موارد باز
1. **η برای V2V فرض است** (`assumption: true`). اگر داور بپرسد، یا sweep لازم است (`η ∈ {۱, ۱٫۵, ۲, ۴}`) یا باید منبع PC5 پیدا شود.
2. **SINR form** فقط در schema آماده است (`sinr_db`، با نویز folded). اگر بخواهی SINR/NOMA جدی شود، یک فاز جدا لازم دارد (path-loss + noise + interference).
3. پهنای‌باند ۲ GHz مقالهٔ Zhao (mmWave) در primary استفاده نشد؛ می‌تواند یک محور scenario باشد («mmWave V2I» در برابر «۱۰ MHz sub-6»).
