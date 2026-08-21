# Lab 21 — Evaluation Report

**Họ tên**: Phan Trọng Tiến  **MSSV**: 2A202601095  **Ngày**: 2026-08-21
**Tier**: `CPU` (chạy trên GPU)  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: NVIDIA GeForce RTX 2050 Laptop, 4.0 GB, sm_86 → `precision=bf16`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

**Khai báo lệch chuẩn — đọc trước.** Bốn thay đổi so với lab gốc, không cái nào đụng
vào tập eval hay prompt được chấm:

1. **Tier `CPU` nhưng huấn luyện trên GPU.** Card ở đây chỉ có **4.0 GB** — dưới cả
   mốc 8–12 GB của tier `LAPTOP`, nên không tier nào khớp. Tôi lấy *model* của tier
   `CPU` (`Qwen3.5-0.8B`) và chạy nó trên GPU thật. Vì thế cột `tier` trong
   `runs.csv` ghi `CPU` trong khi `precision` ghi `bf16` và `peak_vram_gb` có số thật:
   đó không phải mâu thuẫn, đó là cấu hình tôi thật sự chạy. Mọi kết luận dưới đây là
   **về model 0.8B**, không được suy ra cho 4B của tier T4.
2. **`bitsandbytes` cài trên Windows** dù `requirements.txt` gắn marker
   `platform_system == "Linux"`. Không cài thì run `qlora` — một trong ba đối chứng
   bắt buộc — không chạy được. Wheel `bitsandbytes-0.50.1-py3-none-win_amd64` hoạt động
   bình thường trên sm_86.
3. **NB2 được thêm một ô** ghi `results/baseline_preds.json` (dự đoán từng mẫu của (a)
   và (b)). §6 của mẫu report yêu cầu đặt cạnh nhau dự đoán của (b) và của bản
   fine-tune trên **cùng** ticket, mà NB2 gốc chỉ giữ điểm tổng hợp. Artefact cộng thêm,
   không sửa một con số nào đã đóng băng.
4. **Thêm `scripts/dump_regression_examples.py`** → `results/regression_examples.json`.
   Lý do ở §6: trên tập target bản fine-tune **không thua mẫu nào**, nên mọi ca thua của
   run này đều nằm ở nhóm regression, thứ NB5 có chấm nhưng không ghi ra từng mẫu.

`OPTIMIZED_PROMPT` **không đổi** — `optimized_prompt_sha = 719e74d3b6232053`, khớp bản
gốc. `EVAL_LIMIT` không đặt (`smoke_mode: false`), chấm đủ 50 mẫu target và 15 mẫu
regression. `EPOCHS=2` (mặc định) cho cả NB3 lẫn NB4.

---

## 1. Setup

| | |
|---|---|
| Dataset | corpus mặc định của lab, 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `512` — p95 đo được là `98` (p99 = 100, max = 101) *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 → **58** optimizer step (batch hiệu dụng 8 = 1 × grad_accum 8) |

**Vì sao `max_length=512` chứ không phải 256 như p95 gợi ý.** NB1 cảnh báo đúng: p95 = 98
làm tròn luỹ thừa 2 ra **256**, còn tier đang đặt 512. Tôi giữ 512 và đây là lý do —
`max` của toàn corpus là **101 token**, tức là ở *cả hai* giá trị đều **không có mẫu nào
bị cắt**, con số này chỉ quyết định mức trần chứ không quyết định dữ liệu. Và cái giá
thường thấy của việc đặt trần quá cao — trả tiền cho padding — ở đây bằng không:
`per_device_train_batch_size=1` và `packing=False`, nên mỗi batch chỉ chứa đúng một chuỗi
~94 token và không có gì để pad tới 512. Đặt 256 sẽ cho **cùng** số token, cùng thời gian,
cùng VRAM. Điều tôi *không* làm là để 512 vì nó là số mặc định: 512 chỉ an toàn ở đây vì
p99 = 100 đã được đo, và nếu đổi corpus thì phải đo lại trước khi giữ nguyên nó.

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json` báo
`"reasoning preserved — safe to train on traces"`, `body_present: true`. Nghĩa là nếu
dataset có reasoning trace thật thì trace đó **sẽ** tới được hàm loss.

Nhưng lưu ý quan trọng cho §5: corpus này **không có trace nào**. Cả 250 câu trả lời đều
là JSON trần, và template Qwen3.5 đóng sẵn khối `<think>\n\n</think>` **bên trong
generation prompt** — tức là phần rỗng đó nằm ở phía *bị mask*, không nằm trong đoạn được
supervise. Hệ quả: trên corpus này `masked-think` và `response-only` cho ra mask **giống
hệt** `assistant-only` (labkit tự cảnh báo điều đó). Nên tôi **không** claim B3: chạy hai
`MASK_MODE` ở đây sẽ chỉ tạo ra hai lần cùng một thí nghiệm.

**Kiến trúc đang fine-tune** (in từ config của chính model, deck §6.4):

```json
{"num_hidden_layers": 24, "full_attention_interval": 4, "linear_num_key_heads": 16,
 "layer_types": {"linear_attention": 18, "full_attention": 6}}
```

24 lớp, trong đó **18 lớp linear attention** xen kẽ **6 lớp full attention** (cứ 4 lớp
một lớp full). `resolve_target_modules(model, "text-linear")` trả về **12** suffix của
text decoder — vision tower bị loại ra đúng như §10.2 sửa lại.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37 / 94 token trên mẫu được giải mã ngược) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Trên toàn tập train: **8564 / 20951 token (40.9%)** được tính loss.

Đoạn được tính loss (`supervised_preview`):

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **không** được tính loss (`masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Hai chi tiết đáng nói. Thứ nhất, `<|im_end|>` **nằm trong** phần được tính loss — đó là
tín hiệu dừng, không dạy nó thì model không biết kết thúc ở đâu. Thứ hai, để đối chứng
NB1 chạy lại cùng mẫu với `mask_mode="everything"`: `supervised 94/94 (100%)`, và đoạn
được tính loss khi đó chứa nguyên văn câu hỏi của khách. Đó chính là bug §16 — model học
cách *viết lại ticket* thay vì phân loại nó — và nó chỉ mất 30 giây để nhìn thấy, so với
40 phút để huấn luyện ra nó.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.6222 | 0.0000 | 9263.3 |
| (b) base + optimized prompt | 0.5000 | 0.6222 | 1.0000 | 2333.9 |
| (c) LoRA fine-tune | **0.9900** | **0.0333** | 1.0000 | 3829.7 |

**(b) có thật sự mạnh hơn (a) không?** **Có, và cách biệt rất lớn: 0.000 → 0.500.**
`make verify` xác nhận `(a)=0.000 -> (b)=0.500`.

Khoảng cách đó không phải "prompt tốt hơn một chút". Với prompt ngây thơ (`"Phân loại
ticket sau."`) model 0.8B trả lời bằng **văn xuôi tiếng Việt**, không phải JSON —
`format = 0.000`, và vì `triage_field_accuracy` không parse được gì nên `target` cũng
bằng 0.000. Thêm schema + enum + một ví dụ vào system prompt là đủ để `format` nhảy thẳng
lên **1.000**: model đã *biết* làm việc này, nó chỉ không biết phải trả lời theo hình
dạng nào. Đây là lý do baseline (b) bắt buộc phải có — nếu chỉ so với (a) thì tôi đã báo
cáo một "chiến thắng" +0.99 mà trong đó **một nửa** thật ra chỉ là công của prompt.

Chi tiết dễ bỏ qua: latency của (a) là **9263 ms**, gấp 4 lần (b). Không phải vì (a) chạy
chậm hơn, mà vì nó **không biết dừng** — nó viết văn xuôi cho tới khi chạm trần 160 token,
trong khi (b) phát ra ~30 token JSON rồi dừng. Cột latency ở đây đo độ dài đầu ra chứ
không đo tốc độ, và tôi đọc nó như vậy.

Tôi **không sửa** `OPTIMIZED_PROMPT` (sha `719e74d3b6232053` khớp bản gốc).

---

## 4. Giải phẫu cấu hình sai (NB4)

Cả bốn run: **58 step**, **cùng ngân sách tham số 10,822,656** (sai lệch 0.00%),
`mask_mode=assistant-only`, seed 42, bf16 (base 4-bit riêng cho `qlora`).

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 10,822,656 | 1e-4 | **0.3941** | **0.990** | 1.000 | 2370.4 | 3.07 |
| `attn_only` | q,v | **271** *(matched)* | 10,822,656 | 1e-4 | 0.4327 | 0.945 | 1.000 | 2196.7 | 3.08 |
| `wrong_lr` | text-linear | 16 | 10,822,656 | 1e-5 | 1.5427 | **0.330** | 0.995 | 2648.6 | 3.08 |
| `qlora` | text-linear | 16 | 10,822,656 | 1e-4 | 0.4242 | 0.955 | 1.000 | 2309.0 | **2.29** |

Bảng vị trí × rank × tham số mà NB4 in ra trước khi train:

| placement | modules | r | trainable |
|---|---|---|---|
| text-linear | 12 | 16 | 10,822,656 |
| attn-only(q,v) | 2 | 16 | 638,976 |
| **attn-only(q,v) matched** | 2 | **271** | **10,822,656** |

Đọc dòng giữa trước: `q,v @ r=16` chỉ có **638,976** tham số — bằng **5.9%** của
`correct`. So hai cái đó với nhau là so *ngân sách chênh 17 lần* rồi gọi kết quả là bằng
chứng về *vị trí*. `matched_rank()` giải ra **r = 271** để đưa attention-only về đúng
10,822,656 tham số. Chỉ từ đó trở đi câu hỏi "vị trí hay rank?" mới có nghĩa.

**4.1 — `attn_only` thắng, thua hay hoà?**
`attn_only` **thua**, nhưng thua sát: **0.945 so với 0.990** trên tập target, tức kém
0.045 (4.5 điểm phần trăm) với **đúng cùng một ngân sách tham số** và cùng 58 step. Thứ
tự này *giống* thứ tự theo train loss (0.4327 so với 0.3941) — hai bảng đồng ý với nhau
trong run này, và tôi nói thẳng điều đó thay vì dựng lên một mâu thuẫn không có. Điều nó
nói về *rank* so với *vị trí*: rank **không** mua lại được vị trí. Để bù cho việc chỉ gắn
vào `q_proj` và `v_proj`, `attn_only` phải đẩy rank từ 16 lên **271** — gấp 17 lần — và
sau tất cả vẫn không đuổi kịp. Nếu rank là đòn bẩy thật thì một adapter r=271 phải thắng
một adapter r=16; nó không thắng. Nhưng tôi cũng không phóng đại: 0.045 trên 50 mẫu là
khoảng 2 mẫu, và tác vụ này hẹp (4 trường, từ vựng đóng) nên phần lớn thứ cần học nằm
gọn trong attention. Kết luận đúng mức là *vị trí quan trọng hơn rank, và trên tác vụ hẹp
thì khoảng cách nhỏ* — không phải "attention-only là vô dụng".

**4.2 — `wrong_lr` khác đúng một con số.**
Chỉ đổi `learning_rate` từ `1e-4` xuống `1e-5` (thang full-FT, ÷10). Train loss cuối:
**1.5427 so với 0.3941** — gần gấp 4 lần, và đường loss của nó tụt rất chậm thay vì rơi
dứt khoát như `correct` (2.81 → 0.32 → 0.075 → 0.0076 qua 58 step). Trên tập target hậu
quả còn nặng hơn tỷ lệ đó: **0.330 so với 0.990**. Đáng chú ý là `format` của nó vẫn
**0.995** — nó *đã* học được hình dạng JSON, chỉ là chưa kịp học **nội dung** nhãn. Nếu
tôi chỉ nhìn loss mà không biết LR, tôi sẽ kết luận sai theo cách rất cụ thể và rất phổ
biến: "1.54 mà chưa hội tụ — chắc LoRA không đủ sức, phải tăng rank hoặc chuyển sang full
fine-tune". Đó đúng là danh tiếng "LoRA học kém hơn full FT" trong tài liệu 2024, và ở
đây nó chỉ là **một con số LR đặt sai một bậc thang**. Cách chữa rẻ hơn 10 lần so với
kết luận sai.

**4.3 — `qlora` tiết kiệm bao nhiêu, trả giá bằng gì?**
VRAM đỉnh **2.29 GB so với 3.07 GB** — tiết kiệm **0.78 GB (−25.4%)**, đúng như kỳ vọng
khi trọng số base 0.8B đi từ bf16 xuống nf4. Giá phải trả trên tập target: **0.955 so với
0.990**, tức mất **0.035**. Thời gian thì gần như hoà (2309 s so với 2370 s), nhưng
**latency suy luận thì tệ hơn: 4017.6 ms so với 3829.7 ms** — dequantize mỗi lần forward
không miễn phí. Số đo của tôi **ủng hộ** khuyến nghị "không dùng QLoRA cho dòng model
này", nhưng ủng hộ một cách có điều kiện: mất 3.5 điểm chất lượng để đổi lấy 0.78 GB là
một cái giá tồi **khi bf16 vẫn vừa card** — mà ở đây nó vừa. Đổi lại, nếu card của tôi
chỉ có 3 GB thì `qlora` là lựa chọn duy nhất còn chạy được, và 0.955 vẫn tốt hơn nhiều so
với 0.500 của baseline (b). Khuyến nghị của nhà cung cấp đúng ở chỗ *đừng mặc định dùng
nó*; nó không nói QLoRA vô dụng.

**Về `final_loss` và thứ hạng.** Trong run này hai cột cho **cùng một thứ tự**
(`correct` > `qlora` > `attn_only` > `wrong_lr`). Tôi vẫn xếp hạng bằng cột target, và
việc chúng trùng nhau không biến train loss thành một thang đo hợp lệ: `wrong_lr` cho
thấy loss 1.54 và loss 0.39 có thể là *hai chế độ khác hẳn nhau*, còn khoảng cách 0.4327
so với 0.3941 giữa `attn_only` và `correct` — 9.8% trên loss — chỉ tương ứng 4.5% trên
target. Loss và năng lực không cùng đơn vị, và cái duy nhất chứng minh được điều đó là
đã chấm cả hai.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.490` · `regression Δ = −0.589` · `valid_trace_rate = 0.00`

Lý do máy đưa ra: *"general capability regressed by 0.589 (tolerance 0.020)"*.

**Diễn giải.** Bản fine-tune thắng tuyệt đối ở đúng thứ nó được huấn luyện: target
0.500 → **0.990**, +0.490, và thắng trên **48/50** mẫu, hoà 2, **thua 0**. Format giữ
1.000. Nếu lab chỉ có một nhóm đo thì đây là một thành công không cần bàn. Nhưng nhóm thứ
hai nói ngược lại: regression **0.622 → 0.033**, mất 0.589 trên ngưỡng cho phép 0.020 —
tức là vượt ngưỡng **29 lần**. Đây không phải "hơi tụt kiến thức phổ thông", đây là **sụp
đổ tác vụ**: hỏi *"Thủ đô của Việt Nam là thành phố nào?"* và model trả lời
`{"intent": "hoi_thong_tin", "urgency": "thap", "product": "hoi_thong_tin"}`. Nó không
quên Hà Nội — nó đã quên rằng **tồn tại một cách trả lời khác ngoài JSON triage**. Sau 58
step trên 225 mẫu mà **100%** câu trả lời là JSON 4 khoá, phân phối đầu ra đã bị nén về
đúng một hình dạng. Chẩn đoán theo đúng thứ tự của NB5: `format` = 1.000 nên không phải
lỗi template/mask; `target` tăng mạnh nên không phải lỗi LR; vậy nguyên nhân nằm ở
**thành phần dữ liệu** — corpus không có một mẫu replay nào (deck §14.3 đề xuất trộn
1–5% dữ liệu phổ thông). `valid_trace_rate = 0.00` là số đo *nhất quán* với chuyện đó
nhưng **không** phải bằng chứng độc lập: corpus vốn không có reasoning trace nào, nên tỷ
lệ này bằng 0 cả trước lẫn sau khi train — tôi báo cáo nó để nó có mặt trong hồ sơ, không
dùng nó để lập luận.

Điều tôi **không** làm: nới `REGRESSION_TOLERANCE`, làm yếu prompt (b), hay đổi tập eval.
Verdict FAILED này là kết quả đúng của một phép so sánh công bằng, và nó có giá trị hơn
một PASS đạt được bằng cách hạ thấp cái thước.

---

## 6. Định tính — bắt buộc có cả ca THUA

**Trên tập target không có ca nào fine-tune thua** (48 thắng / 2 hoà / 0 thua so với (b)
— tính từ `qualitative.json` và `baseline_preds.json`). Mọi ca thua của run này nằm ở
**nhóm regression**, nên hai ca ❌ dưới đây lấy từ đó
(`results/regression_examples.json`), so bản fine-tune với **chính base model** trên cùng
câu hỏi và cùng thang `keyword_recall`.

| # | Ticket / câu hỏi (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `[target 49]` "…ốp lưng điện thoại… **Sai màu**. Sớm nhé." | `san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh` | `van_chuyen / cao / " ốp lưng…" / tich_cuc` — 0.25 | `san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh` — **1.00** | ✅ FT thắng. (b) đoán `van_chuyen` cho một ticket **lỗi sản phẩm**, và còn chép product kèm khoảng trắng thừa. |
| 2 | `[target 48]` "…ốp lưng điện thoại… **Giá bao nhiêu**." | `hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh` | `van_chuyen / cao / " ốp lưng…" / trung_tinh` — 0.50 | `hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh` — **1.00** | ✅ FT thắng. (b) mặc định gán `urgency: cao` gần như mọi nơi; FT học được rằng "giá bao nhiêu" là câu hỏi thông tin, không gấp. |
| 3 | `[regression 0]` "Thủ đô của Việt Nam là thành phố nào?" | chứa `Hà Nội` | base: *"…Thủ đô của Việt Nam là **Hà Nội**…"* — **1.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "hoi_thong_tin"}` — **0.00** | ❌ **FT thua.** Không phải trả lời sai — **trả lời sai định dạng câu hỏi**. Còn nhét cả tên nhãn `hoi_thong_tin` vào ô `product`. |
| 4 | `[regression 3]` "Viết một câu chúc mừng sinh nhật bằng tiếng Việt." | chứa `Chúc mừng sinh nhật` | base: *"Chúc mừng sinh nhật! 🎉 Chúc bạn luôn tràn đầy năng lượng…"* — **1.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "sentiment": "tieu_cuc"}` — **0.00** | ❌ **FT thua.** Yêu cầu sinh văn bản sáng tạo cũng bị ép vào schema triage — và còn thiếu cả khoá `product`. |
| 5 | `[target 18]` "…máy xay sinh tố… **Khi nào có tiền về**." | `hoan_tien / trung_binh / máy xay sinh tố / tieu_cuc` | `hoan_tien / **cao** / máy xay sinh tố / **tich_cuc**` — 0.50 | `hoan_tien / **thap** / máy xay sinh tố / tieu_cuc` — 0.75 | ➖ FT tốt hơn nhưng **cả hai đều sai `urgency`** — và sai về hai phía đối nghịch. Ca tệ nhất của FT trên tập target vẫn là 0.75. |

**Có mẫu chung nào ở các ca FT thua không?** Có, và chỉ có **một** mẫu duy nhất: mọi ca
thua đều là ca mà **câu hỏi không phải là một ticket**. Không có ca thua nào thuộc kiểu
"phân loại sai" — trên miền huấn luyện, ca xấu nhất của bản fine-tune vẫn đạt 0.75 và vẫn
tốt hơn (b). Thất bại của nó **không nằm ở độ chính xác, mà nằm ở phạm vi**: nó đã trở
thành một hàm `ticket → JSON` và mất khả năng nhận ra đầu vào nào **không** phải ticket.
Hai lỗi phụ cùng chỉ về một hướng — `product: "hoi_thong_tin"` (nhét tên nhãn vào ô sản
phẩm) và bỏ hẳn khoá `product` — đều là dấu hiệu model đang cố ép một đầu vào ngoài phân
phối vào khuôn duy nhất nó còn biết. Đó chính xác là hình dạng của quên thảm hoạ khi 100%
dữ liệu huấn luyện có cùng một dạng đầu ra.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** **Không nên deploy bản fine-tune này**, dù nó thắng baseline (b) tới +0.490
trên tác vụ đích. Lý do là nhân quả chứ không phải vì cái cổng báo đỏ: model này không
được huấn luyện để *phân loại ticket tốt hơn*, nó được huấn luyện để *chỉ còn biết phân
loại ticket*. Chừng nào mọi đầu vào production đều là ticket thì 0.990 là con số thật và
dùng được; nhưng một hệ CSKH thực tế luôn nhận cả câu hỏi lạc đề, lời chào, yêu cầu viết
phản hồi cho khách — và với những đầu vào đó, base model **có prompt tử tế** vẫn trả lời
được, còn bản fine-tune trả về JSON vô nghĩa. Tôi đang đánh đổi 0.589 năng lực tổng quát
lấy 0.490 điểm tác vụ, và phép đổi đó chỉ có lãi nếu tôi *chắc chắn* kiểm soát được đầu
vào. Cách sửa đã rõ và rẻ: trộn **1–5% dữ liệu phổ thông** vào tập train (deck §14.3) rồi
train lại đúng 58 step đó, sau đó đọc lại cổng hồi quy — chứ không phải nới ngưỡng 0.020.

**Đâu là đòn bẩy thật sự?** Xếp theo biên độ đo được, thứ tự là: **(1) chất lượng và
thành phần dữ liệu**, **(2) learning rate**, **(3) vị trí gắn adapter**, và **(4) rank —
gần như không phải đòn bẩy gì cả**. Dữ liệu đứng đầu vì nó vừa tạo ra toàn bộ +0.490 vừa
tạo ra toàn bộ −0.589: cùng một quyết định (250 mẫu, 100% cùng một hình dạng đầu ra) sinh
ra cả chiến thắng lẫn thất bại của lab này, và không có nút LoRA nào chạm tới được nó. LR
đứng thứ hai vì lệch **một bậc thang** đã kéo target từ 0.990 xuống 0.330 — biên độ lớn
nhất trong ba đối chứng. Vị trí đứng thứ ba: chuyển từ 12 linear của text decoder về chỉ
`q,v` làm mất 0.045 dù giữ **nguyên** ngân sách tham số. Rank đứng cuối, và nó xứng đáng
đứng cuối: `attn_only` được nâng rank gấp **17 lần** (16 → 271) chỉ để bù cho một quyết
định về vị trí, và vẫn thua. Mask thì không xuất hiện trong bảng xếp hạng này vì tôi đã
chứng minh nó đúng ở NB1 trước khi train — nhưng nếu nó sai, nó sẽ đứng trên tất cả:
`everything` supervise 100% token và biến toàn bộ 4 tiếng chạy pipeline thành rác.

**Ba điều tôi học được:**

1. **Baseline (b) là thứ đắt giá nhất trong cả lab, và nó tốn 4 phút.** Nếu chỉ có (a),
   tôi đã báo cáo "fine-tune từ 0.000 lên 0.990" — một câu đúng về số học và sai về bản
   chất, vì **một nửa** khoảng đó (0.000 → 0.500) mua được bằng cách viết schema vào
   system prompt. Đo (b) **trước** khi train là thứ ngăn tôi tự lừa mình, và nó phải đo
   trước vì đo sau thì tôi đã biết mình cần con số nào.
2. **Quên thảm hoạ không trông giống "kém đi", nó trông giống "đổi nghề".** Tôi đã tưởng
   regression tụt nghĩa là câu trả lời nhạt hơn, sai vặt hơn. Thực tế model trả lời câu
   hỏi thủ đô Việt Nam bằng một object JSON có khoá `urgency`. Nếu tôi chỉ theo dõi
   `target` và `format` — hai cột đẹp nhất bảng, 0.990 và 1.000 — thì **không có tín hiệu
   nào** báo động. Nhóm regression không phải thủ tục cho đủ bốn cột; nó là cột duy nhất
   nhìn thấy được thứ hỏng.
3. **"Cùng rank" không phải là công bằng, "cùng số tham số" mới là.** `q,v @ r=16` chỉ có
   638,976 tham số — 5.9% của `correct`. Nếu tôi so hai cái đó rồi kết luận "attention-only
   kém hơn", tôi đã chứng minh một điều tầm thường (ít tham số hơn thì học ít hơn) và
   tưởng mình chứng minh được điều thú vị (vị trí quan trọng). Phải đẩy rank lên **271**
   cho bằng ngân sách thì câu hỏi mới trở thành câu hỏi — và câu trả lời (thua 0.045) mới
   có nghĩa.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 10–12 mẫu hỏi-đáp phổ thông (~5% của 225) vào
`train_seed.jsonl`, train lại đúng 58 step và cùng seed, rồi đọc lại cổng hồi quy — đây là
thí nghiệm một-biến trực tiếp trả lời câu hỏi mở duy nhất còn lại của report này: khoản
−0.589 kia là **cái giá bắt buộc** của fine-tuning hay chỉ là **hệ quả của một tập train
đơn dạng**. Nếu còn dư thời gian, tôi chạy B4 (quét rank có kiểm soát ở `text-linear`,
r ∈ {8, 16, 64}) để kiểm tra xem r=16 có thật sự là điểm bão hoà trên tác vụ hẹp này không.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap** — `results/merge_check.json`: trước merge **0.9900**,
  sau merge **0.9900**, **Δ = +0.0000** (ngưỡng 0.01) trên đủ 50 mẫu. Hot-swap **3**
  adapter (`correct`, `attn_only`, `qlora`) trên **cùng một** base đang nạp; ba adapter
  cho ba câu trả lời khác nhau trên cùng ticket, trong đó `qlora` đoán `hoan_tien` còn
  hai adapter kia đoán `doi_tra`.
  *Merge được gì, mất gì:* cùng 50 prompt đó, bản chưa merge sinh xong trong **181 s**,
  bản đã merge trong **103 s** — nhanh hơn ~1.75× vì đồ thị phục vụ trở lại y hệt base,
  không còn nhánh adapter. Cái mất là **tính linh hoạt**: một checkpoint merge phục vụ
  đúng một tác vụ, và trọng số merge nặng bằng cả base (~1.7 GB) so với ~21 MB của
  adapter. Khi nào nên giữ adapter riêng dù chậm hơn: khi một base phải phục vụ **nhiều**
  khách hàng/tác vụ cùng lúc — chi phí VRAM khi đó là *một* base cộng vài chục MB mỗi
  adapter, thay vì một bản sao model đầy đủ cho mỗi tác vụ.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`) — không làm; đã chạy corpus mặc định
- [ ] B3 reasoning-trace collapse — **không claim, và lý do là một kết quả**: corpus này có
  0/250 câu trả lời chứa `<think>`, còn template Qwen3.5 đóng khối `<think></think>` rỗng
  **bên trong generation prompt**, tức là ở phía bị mask. Nên `assistant-only`,
  `masked-think` và `response-only` sinh ra **mask giống hệt nhau** (labkit cảnh báo đúng
  điều này). Chạy hai `MASK_MODE` ở đây sẽ chỉ là chạy cùng một thí nghiệm hai lần và báo
  cáo hai con số giống nhau như thể chúng là bằng chứng.
- [ ] B4 quét rank có kiểm soát — không đủ thời gian (mỗi run ~40 phút trên card 4 GB này)
- [ ] B5 HuggingFace Hub — không push

---

## Phụ lục — tái lập

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install bitsandbytes            # Windows: bỏ qua marker Linux

COMPUTE_TIER=CPU EPOCHS=2 MASK_MODE=assistant-only \
  .venv/Scripts/python notebooks/01_data_and_mask.py      # 25 s
  .venv/Scripts/python notebooks/02_baselines.py          # 4 ph
  .venv/Scripts/python notebooks/03_train_correct.py      # 39.5 ph
  .venv/Scripts/python notebooks/04_misconfig_autopsy.py  # 118 ph (3 run)
  .venv/Scripts/python notebooks/05_evaluate_and_verdict.py  # 13 ph
  .venv/Scripts/python notebooks/06_merge_and_serve.py    # 6 ph
  .venv/Scripts/python scripts/dump_regression_examples.py   # 2 ph
.venv/Scripts/python scripts/verify.py
```

Tổng thời gian đo thật trên RTX 2050 4 GB: **~3 giờ 5 phút** cho NB1→NB6. Huấn luyện
chậm hơn nhiều so với con số/step của một T4 không phải vì model lớn — 0.8B là model nhỏ
nhất trong lab — mà vì `batch=1` × chuỗi ~94 token khiến GPU chạy ở mức sử dụng 8–22%:
run bị chặn bởi chi phí phóng kernel chứ không phải bởi phép tính.
