# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary

- **Total examples loaded**: 24 mẫu
- **Validation issues found**: Dòng 1 có cặp dấu ngoặc kép quanh `self-attention` chưa được escape nên JSON không hợp lệ.
- **Cleaning steps taken**: Escape cặp dấu ngoặc kép ở dòng 1. Loader hiện báo kèm số dòng khi gặp lỗi JSON hoặc schema và từ chối prompt trùng sau khi chuẩn hóa chữ hoa, chữ thường và khoảng trắng.

### Split Strategy

- **Train/Val Ratio**: 19 train/5 val mẫu, theo tỷ lệ mặc định 80/20.
- **Leakage Prevention**: Các mẫu được nhóm theo prompt rồi mới xáo trộn cố định với seed 42. Mỗi nhóm prompt chỉ được đưa vào một tập, vì vậy cùng một prompt không thể xuất hiện ở cả train và val.

## 2. Implementation: DPO (có triển khai thêm ORPO)

### Objective Selection

- **Why this method?**: Chọn DPO làm mục tiêu chính vì độ chênh giữa policy và reference thể hiện trực tiếp tín hiệu preference, nhờ đó dễ kiểm thử bằng giá trị số. Cũng có triển khai một phiên bản ORPO đơn giản để đối chiếu.
- **Key Hyperparameters**:
  - `beta`: 0.1
  - `lambda_orpo` (if applicable): 0.1

### Numerical Stability

- **Challenges**: Tính sigmoid và log trực tiếp có thể gây tràn số khi preference margin quá lớn. ORPO còn cần tính log(1 - p), vốn thiếu ổn định khi p tiến tới 1.
- **Solutions**: Phần DPO và preference penalty của ORPO dùng numpy.logaddexp để tính softplus ổn định hơn. ORPO tính log-odds bằng hai nhánh log1 và expm1. Nếu log-probability bằng đúng 0, giá trị này được đưa về số hữu hạn gần 0 nhất ở phía âm.

## 3. Evaluation Results

### Metrics

| Metric                                       | Value  |
| -------------------------------------------- | ------ |
| Pairwise Accuracy                            | 16.67% |
| DPO loss trên bộ dữ liệu test cố định | 0.6636 |

### Qualitative Review

- **Prompt**: Giải thích khái niệm "self-attention" trong Transformer.
- **Chosen Response**: Self-attention gán trọng số cho mức độ quan trọng của các từ khác nhau trong đầu vào khi xử lý từng từ, nhờ đó nắm bắt được các quan hệ phụ thuộc xa.
- **Rejected Response**: Self-attention là một phiên bản RNN đơn giản hơn, dùng ít bộ nhớ và huấn luyện nhanh hơn.
- **Model Preference**: Sai. Unigram baseline chấm chosen là -5.2764 và rejected là -5.1762.

## 4. Discussion & Failure Modes

- **What went well?**: Loader báo lỗi rõ ràng, cách chia dữ liệu không làm rò rỉ prompt, và các hàm loss vẫn cho kết quả hữu hạn với đầu vào cực trị. Metric có thể chạy lại cho cùng kết quả và toàn bộ test đều pass.
- **Observed Bias**: Unigram baseline có add-one smoothing thường ưu tiên câu ngắn, chung chung và chứa nhiều token phổ biến. Nó đo xác suất từ vựng chứ không kiểm tra độ đúng về nội dung, nên pairwise accuracy chỉ đạt 16.67%.
- **Safety**: CPU baseline chỉ chấm điểm các cặp câu trả lời có sẵn, không sinh câu trả lời mới. Vì vậy, bốn regression prompt chưa thể được đánh giá một cách có ý nghĩa. Cần chạy lại chúng khi thay baseline bằng một policy model có khả năng sinh văn bản; báo cáo này không đưa ra kết luận về độ an toàn.

## 5. Extended Work: thử fine-tune một mô hình thật

Phần này nằm ngoài yêu cầu chính của lab. Sau khi hoàn thành các hàm loss và CPU baseline, tôi muốn kiểm tra xem file preference có thực sự dùng để cập nhật trọng số mô hình được không. Mục tiêu của phần mở rộng là đi hết pipeline một lần và ghi lại những vấn đề gặp phải, không phải chứng minh mô hình đã tốt lên trên một benchmark lớn.

### Mở rộng và kiểm tra dữ liệu

Bộ dữ liệu ban đầu có 24 cặp preference, khá ít cho một lần fine-tune. Nên em đã tạo thêm 40 cặp synthetic rồi ghép hai nguồn thành 64 cặp. Các prompt được chuẩn hóa để phát hiện trùng lặp. Trong quá trình rà soát, đã bỏ 12 mẫu có nội dung lặp hoặc quá giống câu hỏi giáo khoa và tạo lại chúng. Hai mẫu thay thế vẫn còn yếu và được đổi sang các tình huống thực tế hơn về thay đổi categorical encoding và feedback loop trong hệ gợi ý.

Sau bước kiểm tra, dữ liệu được chia theo nhóm prompt với seed 42 thành 51 mẫu train và 13 mẫu validation. Cách chia này giữ nguyên nguyên tắc chống leakage đã dùng trong phần lab. File `synthetic_preferences.jsonl` vì vậy không chỉ là dữ liệu tham khảo; 40 cặp trong file này đã được đưa vào tập dùng cho fine-tune.

### Cấu hình chạy thử

Dùng `HuggingFaceTB/SmolLM2-135M-Instruct` để phù hợp với Quadro P2000 4 GB. DPO được chạy bằng TRL, còn LoRA chỉ cập nhật các module `q_proj` và `v_proj` với rank 8, alpha 16 và dropout 0.05. Các tham số còn lại gồm `beta = 0.1`, learning rate `1e-5`, sequence length tối đa 256, batch size 1, gradient accumulation 4 và 3 epoch. Reference log-probability được tính trước để giảm bộ nhớ cần dùng trong lúc train.

Ban đầu muốn thử FP16 vì muốn tiết kiệm bộ nhớ. Forward pass chạy, nhưng gradient norm là `NaN`. Adapter được lưu ra có giá trị hữu hạn, tuy nhiên toàn bộ tensor LoRA-B vẫn bằng 0. Điều này cho thấy gradient scaler đã bỏ qua optimizer step, nghĩa là lần chạy đó chưa cập nhật trọng số. Chuyển sang FP32 giải quyết được vấn đề: gradient norm của smoke test là 1.456 và toàn bộ 184,320 phần tử LoRA-B trở thành khác 0. Với mô hình 135M, FP32 vẫn vừa trong 4 GB VRAM nên đây là lựa chọn ổn định hơn cho GPU Pascal này.

### Kết quả của lần fine-tune mở rộng

| Metric                             |       Value |
| ---------------------------------- | ----------: |
| Số optimizer step                 |          39 |
| Thời gian train                   | 60.72 giây |
| Train loss                         |      0.6897 |
| Held-out DPO loss                  |      0.6907 |
| Held-out reward accuracy           |        100% |
| Held-out reward margin trung bình |     0.00496 |

Adapter cuối có kích thước khoảng 1.86 MB. Cả 120 tensor đều hữu hạn và mọi phần tử LoRA-B đều đã thay đổi, nên lần chạy này thực sự tạo ra weight delta thay vì chỉ tính loss. Bộ test của project vẫn có 18/18 test pass sau khi thêm training entrypoint.

Kết quả 100% reward accuracy cần được đọc theo đúng phạm vi. Metric này cho biết policy sau fine-tune tạo reward margin dương cho cả 13 cặp validation khi so với reference policy. Tập validation nhỏ và cùng kiểu dữ liệu với tập train, nên con số đó chưa chứng minh mô hình sinh câu trả lời tốt hơn trong sử dụng thực tế. Bước tiếp theo hợp lý là so sánh câu trả lời sinh ra trước và sau fine-tune trên regression prompts, đồng thời tăng số lượng và độ đa dạng của preference pairs trước khi thử một base model lớn hơn.
