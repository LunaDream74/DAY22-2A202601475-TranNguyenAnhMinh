# Data Card

- Dataset name: Bộ dữ liệu preference mẫu cho lab Preference Alignment
- Source: File `data/sample_preferences.jsonl` do khóa học cung cấp. Repo không ghi rõ nguồn gốc ban đầu của dữ liệu.
- License/permission: Repo không nêu giấy phép sử dụng. Trước khi xác minh được nguồn và quyền tái sử dụng, dữ liệu chỉ nên dùng trong phạm vi lab này.
- Schema: Mỗi dòng JSONL gồm ba chuỗi không rỗng là `prompt`, `chosen`, `rejected` và trường `metadata` không bắt buộc. Bộ dữ liệu có 24 mẫu.
- Labeling rubric: Nhãn thể hiện preference tương đối giữa hai câu trả lời. Metadata ghi domain và rubric; dữ liệu mẫu chủ yếu xoay quanh kiến thức AI/ML và độ chính xác của nội dung.
- Known biases: Dữ liệu chỉ tập trung vào giáo dục AI, số mẫu ít và mỗi prompt chỉ có một cặp preference. Các câu chosen thường chi tiết hơn. Vì vậy, bộ dữ liệu không đại diện cho nhiều loại yêu cầu của người dùng trong thực tế.
- Safety/PII checks: Pipeline dùng Pydantic để kiểm tra schema, chuẩn hóa trước khi so sánh chosen với rejected, báo lỗi kèm số dòng và từ chối prompt trùng. Chưa có bước phát hiện PII tự động nên vẫn cần kiểm tra thủ công.
- Train/validation/test split method: Dữ liệu được chia theo nhóm prompt với seed 42 để kết quả không thay đổi giữa các lần chạy. Tỷ lệ validation mặc định 20% tạo ra 19 mẫu train và 5 mẫu validation. Repo không cung cấp tập test riêng.
