# قرارداد هدف و متریک — `objective_contract_v1`

این سند تنها تعریف معتبر «هدف» در مسیر primary است. مرجع کد: `spec/objective_contract.py`.

## مسئله‌ای که این قرارداد می‌بندد

reward توکنی primary با `shaping_discount=gamma=0.99` تعریف می‌شود:

```
J_t = L_t / L_scale
r_t = J_{t-1} - gamma * J_t
```

جمع وزنی‌دار (چیزی که PPO بهینه می‌کند) دقیقاً telescope می‌شود:

```
sum_t gamma^(t-1) r_t = J_0 - gamma^N J_N
```

اما متریک قبلی (`validation_query_composite_objective` = `composite_query_objective`) جمع
**بی‌تخفیف** همان rewardها بود:

```
sum_t r_t = J_0 + (1-gamma) * sum_{t<N} J_t - gamma * J_N
```

این یک functional متفاوت است: به prefixهای میانی وزن `1-gamma` می‌دهد. پس scalarی که لاگ
می‌شد و چک‌پوینت را انتخاب می‌کرد، همان کمیتی نبود که بهینه‌ساز بیشینه می‌کرد.

## قرارداد

| مورد | مقدار |
|---|---|
| معیار (بالاتر بهتر) | `validation/objective_discounted_return` = میانگین `sum_t gamma^(t-1) r_t` |
| فرم هم‌ارز | `J_0 - gamma^N J_N` (هدف پلن نهایی، `J_N = L_N / L_scale`) |
| معیار انتخاب چک‌پوینت | همان معیار؛ مقدار در `ckpt/meta_model_best_val.metric.json` ثبت می‌شود |
| reward mode | `latency_only` (`J_t = L_t/L_scale`، بدون ترم انرژی) |
| `gamma` | `0.99` (`shaping_discount`) |
| همراه‌ها (معیار نیستند) | `query_mean_latency` (ثانیه)، `validation/objective_legacy_undiscounted_sum_*` |
| ارزیابی held-out | `query_discounted_return` روی همان rollout، همان معیار |

`J_0` یک ثابت هر گراف است (پلن مرجع pure-location) و هیچ policy آن را تغییر نمی‌دهد؛ `N` هم
ثابت است. بنابراین روی یک مجموعهٔ گراف ثابت:

```
mean(R) = mean(J_0) - gamma^N * mean(J_N)
```

یعنی رتبه‌بندی چک‌پوینت‌ها با `R` و با `J_N` یکی است — و این همان چیزی است که
`spec.evaluate_checkpoint` و `spec.checkpoint_eval_compare` گزارش می‌کنند.

## چرا این تغییر لازم بود (شاهد عددی)

دو چک‌پوینت روی یک گراف با `J_0=1.0`، `N=2`، `gamma=0.99`:

| چک‌پوینت | `J_1` | `J_2` | `R` (قرارداد) | `sum r_t` (قدیمی) |
|---|---|---|---|---|
| A | 0.50 | 0.701 | 0.3129499 | 0.31101 |
| B | 0.00 | 0.700 | 0.3139300 | 0.30700 |

B پلن نهایی بهتری دارد (`J_2` کوچکتر)، پس قرارداد B را انتخاب می‌کند؛ متریک قدیمی A را
انتخاب می‌کرد چون به prefix وزن می‌داد. این تست در
`tests/test_objective_contract.py::TestRankingJustification` قفل شده است.

## قواعد کار

* هر عدد آموزشی/ارزیابی که «بهبود» نامیده می‌شود باید روی همین معیار گزارش شود.
* `query_mean_latency` فقط برای خوانایی انسان است؛ برای انتخاب یا verdict استفاده نمی‌شود.
* `validation_query_composite_objective` صرفاً برای خواندن ران‌های قدیمی نگه داشته شده و
  برچسب `legacy` دارد.
* تغییر در `gamma`، reward mode یا معیار = نسخهٔ جدید قرارداد (`objective_contract_v2`)،
  نه ویرایش خاموش این سند.
