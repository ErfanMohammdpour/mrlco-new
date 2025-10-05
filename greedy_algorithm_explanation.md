# توضیح الگوریتم Greedy برای محاسبه Latency

## مقدمه
الگوریتم Greedy یک روش ساده و کارآمد برای حل مسئله offloading در محیط MEC (Mobile Edge Computing) است. این الگوریتم در هر مرحله بهترین انتخاب محلی را انجام می‌دهد بدون اینکه به آینده نگاه کند.

## مراحل الگوریتم Greedy

### 1. مقداردهی اولیه (Initialization)
```python
cloud_available_time = 0.0    # زمان در دسترس سرور ابری
ws_available_time = 0.0       # زمان در دسترس لینک بی‌سیم
local_available_time = 0.0    # زمان در دسترس دستگاه محلی

# آرایه‌های نگهداری زمان پایان هر تسک
FT_cloud = [0.0] * task_number    # زمان پایان در سرور ابری
FT_ws = [0.0] * task_number       # زمان پایان انتقال بی‌سیم
FT_locally = [0.0] * task_number  # زمان پایان اجرای محلی
FT_wr = [0.0] * task_number       # زمان پایان دریافت از ابر
```

### 2. اولویت‌بندی تسک‌ها (Task Prioritization)
تسک‌ها بر اساس الگوریتم HEFT اولویت‌بندی می‌شوند:
- محاسبه وزن هر تسک: `w[i] = min(t_locally, t_mec)`
- محاسبه رنک هر تسک با در نظر گیری وابستگی‌ها
- مرتب‌سازی تسک‌ها بر اساس رنک نزولی

### 3. پردازش هر تسک (Task Processing)
برای هر تسک در ترتیب اولویت:

#### 3.1 محاسبه زمان اجرای محلی (Local Execution)
```python
# محاسبه زمان شروع محلی
if len(pre_task_sets[i]) != 0:
    local_start = max(local_available_time, 
                     max([max(FT_locally[j], FT_wr[j]) for j in pre_task_sets[i]]))
else:
    local_start = local_available_time

# محاسبه زمان پایان محلی
local_finish = local_start + locally_execution_cost(processing_data_size)
```

#### 3.2 محاسبه زمان اجرای ابری (Remote Execution)
```python
# مرحله 1: انتقال به سرور ابری (Upload)
if len(pre_task_sets[i]) != 0:
    ws_start = max(ws_available_time, 
                  max([max(FT_locally[j], FT_ws[j]) for j in pre_task_sets[i]]))
else:
    ws_start = ws_available_time

ws_finish = ws_start + up_transmission_cost(processing_data_size)

# مرحله 2: پردازش در سرور ابری
cloud_start = max(cloud_available_time, ws_finish)
cloud_finish = cloud_start + mec_execution_cost(processing_data_size)

# مرحله 3: دریافت نتیجه (Download)
remote_finish = cloud_finish + dl_transmission_cost(transmission_data_size)
```

### 4. انتخاب Greedy (Greedy Choice)
```python
if FT_locally[i] < FT_wr[i]:
    # انتخاب اجرای محلی
    action = 0
    local_available_time = FT_locally[i]
    # پاک کردن مقادیر ابری
    FT_wr[i] = 0.0; FT_cloud[i] = 0.0; FT_ws[i] = 0.0
    task_finish_time = FT_locally[i]
else:
    # انتخاب اجرای ابری
    action = 1
    FT_locally[i] = 0.0
    # به‌روزرسانی زمان‌های در دسترس
    cloud_available_time = FT_cloud[i]
    ws_available_time = FT_ws[i]
    task_finish_time = FT_wr[i]
```

### 5. محاسبه Latency نهایی
```python
greedy_latency = max(max(FT_wr), max(FT_locally))
```

## مثال عملی

فرض کنید یک تسک با مشخصات زیر داریم:
- `processing_data_size = 1000 bytes`
- `transmission_data_size = 500 bytes`
- `mobile_process_capable = 100000 bytes/sec`
- `mec_process_capable = 1000000 bytes/sec`
- `bandwidth_up = 7 Mbps`
- `bandwidth_dl = 7 Mbps`

### محاسبه زمان اجرای محلی:
```
local_time = 1000 / 100000 = 0.01 seconds
```

### محاسبه زمان اجرای ابری:
```
upload_time = 1000 / (7 * 1024 * 1024 / 8) = 0.0011 seconds
mec_time = 1000 / 1000000 = 0.001 seconds  
download_time = 500 / (7 * 1024 * 1024 / 8) = 0.0005 seconds
total_remote_time = 0.0011 + 0.001 + 0.0005 = 0.0026 seconds
```

### انتخاب Greedy:
چون `0.0026 < 0.01`، الگوریتم اجرای ابری را انتخاب می‌کند.

## مزایای الگوریتم Greedy

1. **سادگی**: پیاده‌سازی آسان و درک ساده
2. **سرعت**: پیچیدگی زمانی O(n) که n تعداد تسک‌ها است
3. **کارایی**: در بسیاری از موارد به جواب بهینه نزدیک است
4. **مقیاس‌پذیری**: برای مسائل بزرگ قابل استفاده است

## محدودیت‌های الگوریتم Greedy

1. **عدم تضمین بهینه بودن**: ممکن است به جواب بهینه نرسد
2. **عدم نگاه به آینده**: تصمیمات محلی ممکن است در آینده مشکل ایجاد کند
3. **وابستگی به اولویت‌بندی**: کیفیت جواب به روش اولویت‌بندی بستگی دارد

## کد کامل الگوریتم

```python
def calculate_greedy_latency(task_graph, resource_cluster):
    # مقداردهی اولیه
    cloud_available_time = 0.0
    ws_available_time = 0.0
    local_available_time = 0.0
    
    FT_cloud = [0.0] * task_graph.task_number
    FT_ws = [0.0] * task_graph.task_number
    FT_locally = [0.0] * task_graph.task_number
    FT_wr = [0.0] * task_graph.task_number
    
    greedy_plan = []
    
    # پردازش هر تسک در ترتیب اولویت
    for i in task_graph.prioritize_sequence:
        task = task_graph.task_list[i]
        
        # محاسبه زمان اجرای محلی
        if len(task_graph.pre_task_sets[i]) != 0:
            local_start = max(local_available_time,
                             max([max(FT_locally[j], FT_wr[j]) 
                                  for j in task_graph.pre_task_sets[i]]))
        else:
            local_start = local_available_time
        
        local_finish = local_start + resource_cluster.locally_execution_cost(
            task.processing_data_size)
        FT_locally[i] = local_finish
        
        # محاسبه زمان اجرای ابری
        if len(task_graph.pre_task_sets[i]) != 0:
            ws_start = max(ws_available_time,
                          max([max(FT_locally[j], FT_ws[j]) 
                               for j in task_graph.pre_task_sets[i]]))
            ws_finish = ws_start + resource_cluster.up_transmission_cost(
                task.processing_data_size)
            cloud_start = max(cloud_available_time,
                             max([max(ws_finish, FT_cloud[j]) 
                                  for j in task_graph.pre_task_sets[i]]))
            cloud_finish = cloud_start + resource_cluster.mec_execution_cost(
                task.processing_data_size)
            remote_finish = cloud_finish + resource_cluster.dl_transmission_cost(
                task.transmission_data_size)
        else:
            ws_start = ws_available_time
            ws_finish = ws_start + resource_cluster.up_transmission_cost(
                task.processing_data_size)
            cloud_start = max(cloud_available_time, ws_finish)
            cloud_finish = cloud_start + resource_cluster.mec_execution_cost(
                task.processing_data_size)
            remote_finish = cloud_finish + resource_cluster.dl_transmission_cost(
                task.transmission_data_size)
        
        FT_ws[i] = ws_finish
        FT_cloud[i] = cloud_finish
        FT_wr[i] = remote_finish
        
        # انتخاب Greedy
        if FT_locally[i] < FT_wr[i]:
            action = 0
            local_available_time = FT_locally[i]
            FT_wr[i] = 0.0; FT_cloud[i] = 0.0; FT_ws[i] = 0.0
            task_finish_time = FT_locally[i]
        else:
            action = 1
            FT_locally[i] = 0.0
            cloud_available_time = FT_cloud[i]
            ws_available_time = FT_ws[i]
            task_finish_time = FT_wr[i]
        
        greedy_plan.append((i, action))
    
    # محاسبه Latency نهایی
    greedy_latency = max(max(FT_wr), max(FT_locally))
    
    return greedy_latency, greedy_plan
```

## نتیجه‌گیری

الگوریتم Greedy یک روش ساده و کارآمد برای حل مسئله offloading است که در هر مرحله بهترین انتخاب محلی را انجام می‌دهد. این الگوریتم با محاسبه زمان اجرای محلی و ابری برای هر تسک و انتخاب گزینه بهتر، به یک جواب قابل قبول می‌رسد.


