# مشخصات سرور برای اجرای MARGO (نسخهٔ v1 — مبتنی بر محیطی که واقعاً تست شده)

مرجع: سرور `kish-ai` که تمام ران‌های ثبت‌شدهٔ این پروژه روی آن اجرا شده‌اند. هر عدد یا `file:line` دارد یا از خود سرور خوانده شده است.

---

## ۱. خلاصهٔ یک‌خطی

**یک نود x86_64 با Ubuntu 22.04/24.04 + Docker + درایور NVIDIA، و یک یا چند GPU با حداقل ۲۴ گیگ VRAM از نسل Ampere/Ada.** کل استک پایتون/TF داخل ایمیج پروژه است؛ روی هاست چیزی نصب نمی‌شود.

---

## ۲. مشخصات تأییدشدهٔ محیط کارکننده (kish-ai)

| مورد | مقدار واقعی |
|---|---|
| OS | Ubuntu 24.04.4 LTS (kernel 6.8.0-85-generic) |
| CPU | ۳۲ هسته |
| RAM | ۸۸ GB |
| GPU | NVIDIA GeForce RTX 4090، 24564 MiB، compute capability **8.9 (sm_89)** |
| درایور | 580.65.06 |
| Docker | 29.1.3 (overlayfs، nvidia runtime) |
| دیسک | ۶۷۸ GB SSD، ۴۰۸ GB آزاد |
| ایمیج پروژه | `margo-phase4-tf115-nv2212` (۸.۶۹ GB) |
| مخزن | ۲۹۹ MB (دادهٔ گراف‌ها ۱۵ MB) |
| `runs/` بعد از چند ران | **۱۷ GB** (هر ران ۵۰۰-iteration = **۸.۵ GB**) |

---

## ۳. ایمیجی که باید استفاده شود (قطعی)

`spec/Dockerfile.tf115-nv2212` → تگ `margo-phase4-tf115-nv2212`

- پایه: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`
- Python **3.8** (مخزن از `from __future__ import annotations` استفاده می‌کند)
- TensorFlow **1.15.5+nv22.12** (wheel رسمی NVIDIA، `tf.contrib` دست‌نخورده)
- پین‌شده‌ها: `numpy==1.21.6`, `protobuf==3.19.6`, `gast==0.2.2`, `h5py==2.10.0`, `PyYAML==5.4.1`, `pydotplus==2.0.2`, `joblib==0.17.0`, `scipy==1.5.4`, `gym==0.15.7`, `cloudpickle==1.2.2`, `tensorflow-estimator==1.15.1`
- سیستمی: `git`, `graphviz`, `libgomp1`, `curl`, `python3-venv`

### ⛔ هرگز این دو کار را نکنید

1. **`spec/Dockerfile.tf115-gpu` را استفاده نکنید** — کامنت خود فایل: «LEGACY. CUDA 10. Do not use on RTX 4090 (Blas GEMM launch failed / no sm_89)».
2. **TF را روی هاست نصب نکنید.** TF 1.15 به `tf.contrib` نیاز دارد که در TF2 حذف شده؛ و نسخهٔ wheel باید همان nv22.12 باشد. تمام «روی سیستم من کار می‌کند»های این پروژه از همین‌جا می‌آید.

---

## ۴. پیشنهاد کانفیگ (سه سطح)

### سطح A — حداقل قابل‌قبول (تشخیص/دیباگ)
- ۱× GPU با **≥۲۴ GB** (RTX 3090/4090، A5000، L40S)
- ۱۶ هسته CPU، ۶۴ GB RAM، ۱ TB NVMe

### سطح B — پیشنهادی برای کمپین مقاله (توصیهٔ من)
- **۴× GPU مستقل، هر کدام ≥۲۴ GB** (مثلاً ۴× L40S 48GB یا ۴× RTX 4090 یا ۴× A5000)
- ۶۴ هسته CPU، **۲۵۶ GB RAM**، **۲ TB NVMe**
- دلیل: کد **تک‌GPU و تک‌پروسه** است (یک `Session` TF 1.15؛ نه DDP نه multi-GPU). پس «یک GPU قوی‌تر» سرعت یک ران را کم می‌کند ولی «چند GPU» **چند seed را موازی** می‌کند — و کمپین مقاله ۵ seed می‌خواهد (`phase4_campaign.yaml:21`).

### سطح C — تک‌GPU (اگر بودجه محدود است)
- ۱× A100 40/80GB یا L40S 48GB، ۳۲ هسته، ۱۲۸ GB RAM، ۲ TB NVMe
- فقط سریع‌تر است، موازی‌سازی seed ندارد.

### چرا این اعداد (محاسبهٔ مستند)
- **زمان:** نرخ اندازه‌گیری‌شدهٔ runner طولانی = **۳.۵۵۳ دقیقه بر iteration** ⇒ ۱۰۰۰ iteration ≈ **۵۹ ساعت** و ۳۵۰۰ iteration ≈ **۲۰۷ ساعت ≈ ۸.۶ روز برای هر seed**. ۵ seed روی یک GPU ≈ **۴۳ روز**؛ روی ۴ GPU ≈ **۱۰–۱۱ روز**.
- **CPU:** با `parallel=True` (که در Pilot A و ران طولانی فعال است) `MetaParallelEnvExecutor` دقیقاً **`meta_batch_size = 10` پروسهٔ worker** می‌سازد (`samplers/vectorized_env_executor.py:104-114`) به‌علاوهٔ پروسهٔ اصلی ⇒ ≥۱۶ هسته لازم است، ۳۲ راحت.
- **RAM:** روی kish با ۸۸ GB، دو ران هم‌زمان ~۳۳ GB مصرف داشتند ⇒ ~۱۲–۱۶ GB به‌ازای هر ران؛ برای ۴ ران هم‌زمان ۱۲۸–۲۵۶ GB.
- **دیسک:** `audit_writer` هر iteration ≈ **۱۷ MB** می‌نویسد (`trajs_*.jsonl`) ⇒ ۳۵۰۰ iteration ≈ **۶۰ GB برای هر seed**؛ ۵ seed ≈ ۳۰۰ GB. اگر `audit_writer=None` باشد (مثل `pilot_long`) بسیار کمتر. چک‌پوینت‌ها ناچیزند (۳.۴ MB).
- **VRAM:** عدد peak اندازه‌گیری‌شده نداریم، اما کل استک روی ۴۰۹۰ ۲۴GB بدون مشکل اجرا شده است. ۲۴ GB کف تأییدشده است؛ اگر بودجه هست ۴۸ GB برای هم‌زمانی eval/train بهتر است.
- **نسل GPU:** فقط Ampere/Ada. **Hopper (H100) را نگیرید** چون wheel TF 1.15 kernels برای `sm_90` ساخته نشده و ریسک اجرا دارد.

---

## ۵. نرم‌افزار هاست (چیزهایی که باید نصب باشد)

| مورد | حداقل | تأییدشده |
|---|---|---|
| Ubuntu LTS x86_64 | 22.04 | 24.04.4 |
| NVIDIA driver | ≥ 520.61 (CUDA 11.8) | 580.65.06 |
| Docker Engine | ≥ 24 | 29.1.3 |
| nvidia-container-toolkit | لازم (`--gpus all`) | نصب |
| اینترنت در زمان build | برای دانلود wheel از `pypi.nvidia.com` | بود |
| دسترسی root/sudo یا عضویت در گروه `docker` | لازم | root |

نیازی به CUDA toolkit روی هاست، MPI، یا SLURM **نیست** (استک فریز تک‌پروسه است). اگر SLURM دارید، فقط docker را داخل `srun --gres=gpu:1` بگذارید.

### دستورهای راه‌اندازی
```bash
# 1) درایور + داکر + تولکیت
sudo apt-get update && sudo apt-get install -y nvidia-driver-550 docker.io
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# 2) کد و ایمیج
git clone https://github.com/ErfanMohammdpour/mrlco-new.git /opt/margo/mrlco-new-6b
cd /opt/margo/mrlco-new-6b && git checkout phase4-eval
docker build -f spec/Dockerfile.tf115-nv2212 -t margo-phase4-tf115-nv2212 spec
```
اگر نودِ محاسباتی اینترنت ندارد: ایمیج را روی ماشین دیگری build کنید و با `docker save margo-phase4-tf115-nv2212 | gzip > img.tgz` (≈۳ GB فشرده) منتقل و `docker load` کنید.

---

## ۶. چک‌لیست تأیید روی سرور جدید (به ترتیب)

```bash
export MARGO_GPU_IMAGE=margo-phase4-tf115-nv2212
export MARGO_ROOT=/opt/margo/mrlco-new-6b

spec/kish_gpu.sh probe            # python + tf 1.15 + contrib + gpu_available
spec/kish_gpu.sh gate             # phase4_gate.py + phase3_gate.py
spec/kish_gpu.sh energy-tests     # کل سوئیت (در مرجع: ۸۹۸ تست OK)
spec/kish_gpu.sh smoke            # ۱ iteration واقعی روی GPU (gpu-smoke)
spec/kish_gpu.sh energy-consistency   # CPU-only، بدون اشغال GPU
```

انتظار: `probe` باید `tf 1.15.5`، `contrib True`، `gpu_available True` بدهد؛ `energy-tests` باید `OK` بدهد (خطاهای `setUpClass` مربوط به TF فقط در ماشین بدون TF رخ می‌دهد و در ایمیج نیست). سپس اولین ران واقعی:

```bash
spec/kish_gpu.sh pilot-a --root runs/mask_sanity_v3   # ۲۵ iteration (~۱.۶ ساعت)
```

---

## ۷. نکات عملیاتی که به‌دردسر خورده‌اند

1. **GPU اختصاصی:** اسکریپت‌ها فرض می‌کنند GPU آزاد است (`require_gpu_permission` + watchdog). روی سرور اشتراکی، یک job خارجی با ۲۲.۳ GB از ۲۴.۵ GB کل GPU را گرفت و ما مجبور شدیم ران‌های GPU را متوقف کنیم. اگر سرور مشترک است، حتماً سهم GPU را رزرو کنید.
2. **مسیر mount:** `gpu_run` فقط `$ROOT` را روی `/work` سوار می‌کند و launcher مسیرهای زیر `$ROOT` را به `/work` بازنویسی می‌کند. هر مسیر خارج از `$ROOT` (مثل `/tmp` میزبان یا `/opt/margo/...` دیگر) داخل container **دیده نمی‌شود** — این قبلاً یک بار باعث گم‌شدن JSONهای ارزیابی شد (`e3fd3cd`).
3. **`/tmp` داخل container موقتی است.** خروجی‌ها را همیشه زیر `/work` بنویسید.
4. **ران‌های طولانی را detached بزنید:** `setsid nohup … > /tmp/x.log 2>&1 &` + فایل `.exit`؛ قطع ssh یک `docker run` معمولی را می‌کشد (`7fa31f2`).
5. **پروکسی که فراموش نکنید:** `MARGO_ALLOW_GPU=1` برای همهٔ ران‌های GPU، و `MARGO_OBS_VERSION=v3` برای پایلوت/ارزیاب (کد در entrypoint آن را خودش ست می‌کند، اما برای مسیرهای قدیمی‌تر باید صریح باشد).
6. **بکاپ:** `runs/` و `reports/` را جدا بکاپ بگیرید؛ خود مخزن چک‌پوینت‌ها و لاگ‌ها را gitignore می‌کند، پس git جای بکاپ evidence نیست.

---

## ۸. اگر می‌خواهید فقط CPU کار کنید

بدون GPU هم می‌شود: **کل سوئیت تست، probe فیزیک، ساخت sidecar ددلاین، sweep، و merge ارزیابی** روی CPU اجرا می‌شوند (همین الان ارزیابی سه چک‌پوینت روی CPU کیش اجرا شد، هر label ≈۴۴ دقیقه). فقط **train و gpu-smoke** به GPU نیاز دارند. برای این کار: ۱۶ هسته + ۳۲ GB RAM + ۲۰۰ GB دیسک کافی است.

---

## ۹. «آیا TF 1.15 داخل کانتینر روی Ubuntu 24 قطعاً اجرا می‌شود؟» — بله، و اثباتش

نسخهٔ Ubuntu هاست **وارد معادله نمی‌شود**، چون کانتینر userland خودش را دارد. اثبات تجربی روی همان `kish-ai` (هاست Ubuntu 24.04.4 / kernel 6.8 / درایور 580.65.06):

```
container_os = Ubuntu 20.04.6 LTS        <- از خود ایمیج
ldd (Ubuntu GLIBC 2.31-0ubuntu9.12) 2.31 <- glibc کانتینر، نه هاست
python 3.8.10
GPU 0: NVIDIA GeForce RTX 4090
tf 1.15.5
tf.contrib True
gpu_available True
device list: CPU, XLA_CPU, XLA_GPU, GPU:0
matmul_on_gpu_ok [[7.0, 10.0], [15.0, 22.0]]
```

### چرا کار می‌کند (مکانیزم دقیق)

| لایه | از کجا می‌آید | نسخه |
|---|---|---|
| glibc، Python، CUDA user-space (libcudart/cuDNN/cuBLAS)، TF | **داخل ایمیج** | Ubuntu 20.04 / glibc 2.31 / Py3.8 / CUDA 11.8 |
| کرنل | هاست | 6.8.0-85 |
| `libcuda.so` و `/dev/nvidia*` | هاست، توسط nvidia-container-toolkit با `--gpus all` تزریق می‌شود | درایور 580.65.06 |
| سرویس‌دهی کانتینر | هاست | Docker 29.1.3 + cgroup v2 |

تنها کوپلینگ واقعی دو چیز است: (۱) کرنل باید توسط درایور پشتیبانی شود، (۲) درایور باید حداقل نسخهٔ CUDA 11.8 یعنی **≥ 520.61.05** را داشته باشد. درایورهای جدید با CUDA user-space قدیمی‌تر سازگارند (minor-version compatibility)، پس ۵۸۰ با CUDA 11.8 مشکلی ندارد.

### چه چیزی واقعاً می‌شکند (هیچ‌کدام «Ubuntu 24» نیست)

| سناریو | نتیجه |
|---|---|
| ایمیج `margo-phase4-tf115-nv2212` روی هر Ubuntu x86_64 با کرنل ۵.۱۵–۶.۸ و درایور ≥۵۲۰ | ✅ (روی ۲۴.۰۴/۶.۸/۵۸۰ اثبات شد) |
| ایمیج legacy `margo-phase4-tf115-gpu` (CUDA 10) روی Ada/4090 | ❌ `Blas GEMM launch failed / no sm_89` |
| نصب **بومی** TF 1.15 روی Ubuntu 24 (بدون داکر) | ❌ عملاً غیرممکن: پایتون ۳.۱۲، glibc 2.39، و حذف `tf.contrib` در TF2 |
| درایور < 520.61 | ❌ CUDA 11.8 initialize نمی‌شود |
| GPU Hopper (H100، sm_90) | ⚠️ wheel سری nv22.12 برای sm_90 ساخته نشده؛ تست‌نشده |
| داکر بدون `nvidia-container-toolkit` | ❌ `--gpus all` با «could not select device driver» شکست می‌خورد |
| کرنل خیلی جدید (مثلاً ۶.۱۴) با درایور قدیمی | ⚠️ درایور باید هم‌نسل کرنل باشد (مشکل درایور، نه توزیع) |

این probe در حالی اجرا شد که یک job خارجی ~۲۲ GB از ۲۴.۵ GB کارت را گرفته بود و با `TF_FORCE_GPU_ALLOW_GROWTH=true` بدون اختلال در آن job تمام شد (کانتینر با `--rm` هیچ artifactی نگذاشت).
