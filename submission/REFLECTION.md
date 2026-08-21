# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hình dạng của quên thảm hoạ. Tôi vẫn hình dung "regression tụt" nghĩa là model trả lời
kiến thức phổ thông kém đi — nhạt hơn, sai vặt hơn. Thực tế: hỏi *"Thủ đô của Việt Nam là
thành phố nào?"*, bản fine-tune trả lời `{"intent": "hoi_thong_tin", "urgency": "thap",
"product": "hoi_thong_tin"}`. Nó không quên Hà Nội, nó quên rằng **có tồn tại cách trả
lời khác ngoài JSON triage**. Chỉ 58 step trên 225 mẫu mà 100% cùng một hình dạng đầu ra
là đủ để nén phân phối đầu ra về đúng một khuôn. Và điều làm tôi lạnh gáy là hai cột đẹp
nhất bảng lúc đó — `target` 0.990 và `format` 1.000 — **không hề báo động gì**.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán, và cũng không phải chỗ nào liên quan tới LoRA. Mất nhiều
nhất là **tải `torch` bản CUDA**: 2.75 GB trên đường truyền ~0.4 MB/s, gần 1 giờ 40 phút,
lâu hơn cả NB3. Về phần chạy thật thì NB4 chiếm ~118 phút cho ba run — đúng như rubric
cảnh báo, đây là notebook dài nhất. Điều tôi *không* lường trước: card 4 GB này chạy ở
mức sử dụng GPU chỉ **8–22%** trong lúc train, vì `batch=1` với chuỗi ~94 token thì thời
gian trôi vào chi phí phóng kernel chứ không vào phép tính. Bài học vận hành: "GPU chậm"
và "GPU rảnh" là hai chẩn đoán khác nhau, và `nvidia-smi` phân biệt được chúng trong 3
giây — tôi đã đoán mò khá lâu trước khi thật sự nhìn vào nó.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tin rằng **rank là nút chính**. Cách hiểu cũ của tôi gần như là "muốn model học nhiều hơn
thì tăng r". Run `attn_only` giết niềm tin đó một cách rất cụ thể: để bù cho việc chỉ gắn
adapter vào `q_proj, v_proj`, nó phải đẩy rank từ 16 lên **271** — gấp 17 lần — chỉ để
*ngang* ngân sách tham số, và sau tất cả vẫn thua 0.945 so với 0.990. Rank mua được sức
chứa; nó không mua được **vị trí**. Niềm tin thứ hai bị bỏ: rằng loss thấp nghĩa là học
tốt. `wrong_lr` cho thấy khoảng cách loss 1.54 so với 0.39 và khoảng cách loss 0.4327 so
với 0.3941 không cùng một loại thông tin — cái đầu là hai chế độ khác hẳn nhau, cái sau
9.8% trên loss chỉ đổi được 4.5% trên tác vụ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để dựng môi trường (chọn wheel CUDA, xử lý chuyện `bitsandbytes` bị gắn marker
`platform_system == "Linux"` trong khi vẫn có wheel `win_amd64` chạy tốt), để tải song
song torch và trọng số model cho đỡ nghẽn băng thông, và để đọc chéo `labkit/` trước khi
chạy. Chỗ nó sai — và tôi phải tự bắt: nó ước tính đường CPU "khả thi" cho toàn bộ
pipeline. Tôi đo thật thì một lượt sinh 50 mẫu trên CPU mất ~16 phút, cả pipeline sẽ là
~3 giờ **và** run `qlora` không chạy được vì 4-bit cần CUDA — tức là hỏng mất một trong
ba đối chứng bắt buộc. Một ước lượng nghe hợp lý nhưng chưa đo là đúng loại lỗi mà lab
này dạy cách bắt; tôi chỉ đổi cách áp dụng nó từ mask sang môi trường.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Viết baseline (b) cho tử tế và đo nó **trước**, trước cả khi cài `peft`. Trên chính lab
này, nửa khoảng thắng của tôi (0.000 → 0.500) mua được bằng cách viết schema vào system
prompt, không cần một step huấn luyện nào. Nếu prompt tốt đã đủ vượt yêu cầu của khách,
kết luận đúng là **đừng fine-tune** — rẻ hơn, không có gì để bảo trì, không có
regression để canh. Bước thứ hai, ngay sau đó và trước khi train: chốt tập eval, đóng
băng nó, và **trộn sẵn 1–5% dữ liệu phổ thông** vào tập train. Ở lab này tôi bỏ qua bước
đó và phải trả bằng −0.589 regression cùng một verdict FAILED.
