"""Gold evaluation dataset for the rag_server.

Bộ dữ liệu vàng dùng để đánh giá chất lượng retrieval + answer generation của
rag_server bằng RAGAS. Gồm 2 phần:

1. ``DOCS_CORPUS`` — tài liệu (Phase 1): 3 tài liệu tiếng Việt dài ~5–8 đoạn
   mỗi cái, trộn lẫn thuật ngữ tài chính / nhân sự / sản phẩm.

2. ``TRANSCRIPT_CORPUS`` — transcript cuộc họp (Phase 2): 12 câu thoại
   theo chuẩn ``meeting-{uuid}``, có 2–3 người nói, dùng tiền tố ``meeting-``
   đúng quy ước v2.

3. ``DOCS_QA`` / ``TRANSCRIPT_QA`` — danh sách Q&A vàng. Mỗi item gồm
   ``question``, ``ground_truth_answer``, và (chỉ ở transcript)
   ``ground_truth_sequence_ids`` (dùng để kiểm tra window fetch).

Bộ Q&A được thiết kế cố ý để phủ 4 loại câu hỏi điển hình của RAG:
- Factual: "Quyết định về ngân sách Q4 là gì?"
- Lookup người nói / con số cụ thể
- Multi-hop: cần ghép nhiều đoạn
- Negative: không có thông tin trong corpus (để đo precision)
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Phase 1 — Tài liệu
# --------------------------------------------------------------------------- #

DOCS_CORPUS: list[dict] = [
    {
        "source": "chinh_sach_nghi_phep_2026.md",
        "chunks": [
            (
                "Chính sách nghỉ phép năm 2026\n\n"
                "Mỗi nhân viên chính thức được hưởng 12 ngày nghỉ phép có lương mỗi năm. "
                "Số ngày này được cộng dồn tối đa 5 ngày sang năm tiếp theo, "
                "tổng tích lũy không vượt quá 17 ngày trong một năm tài chính."
            ),
            (
                "Nhân viên làm việc trên 5 năm được cộng thêm 2 ngày phép thưởng trung thành. "
                "Phòng Nhân sự sẽ tự động cập nhật vào ngày kỷ niệm làm việc thứ 5 của nhân viên."
            ),
            (
                "Đơn xin nghỉ phép phải được gửi trước ít nhất 3 ngày làm việc đối với nghỉ từ 1 đến 3 ngày. "
                "Nghỉ từ 4 ngày trở lên phải gửi trước 7 ngày và cần sự phê duyệt của quản lý cấp trưởng phòng."
            ),
            (
                "Nghỉ phép không lương (unpaid leave) được chấp thuận trong trường hợp cá nhân đặc biệt "
                "và phải có sự đồng ý của Ban Giám đốc. Thời gian nghỉ không lương không tính vào thâm niên."
            ),
        ],
    },
    {
        "source": "quy_dinh_cong_tac_phi_2026.md",
        "chunks": [
            (
                "Quy định công tác phí 2026\n\n"
                "Mức công tác phí trong nước là 250.000 đồng/ngày cho nhân viên cấp nhân viên, "
                "350.000 đồng/ngày cho cấp quản lý và 500.000 đồng/ngày cho cấp trưởng phòng trở lên."
            ),
            (
                "Tiền khách sạn được thanh toán theo hóa đơn thực tế, tối đa 1.200.000 đồng/đêm đối với "
                "thành phố Hà Nội, TP. HCM, Đà Nẵng; các tỉnh khác tối đa 800.000 đồng/đêm."
            ),
            (
                "Vé máy bay hạng economy cho chuyến dưới 4 tiếng, hạng business cho chuyến từ 4 tiếng trở lên. "
                "Cần đặt vé trước ít nhất 14 ngày để được áp dụng mức giá công vụ."
            ),
            (
                "Báo cáo công tác phải được nộp trong vòng 5 ngày làm việc sau khi kết thúc chuyến công tác, "
                "kèm đầy đủ hóa đơn gốc. Quá thời hạn sẽ không được hoàn trả."
            ),
        ],
    },
    {
        "source": "san_pham_ai_meeting_box.md",
        "chunks": [
            (
                "AI Meeting Box — Thông số kỹ thuật\n\n"
                "AI Meeting Box là thiết bị phòng họp thông minh chạy chip Qualcomm QCS8550, "
                "hỗ trợ nhận diện khuôn mặt theo thời gian thực với độ trễ dưới 200ms. "
                "Thiết bị hỗ trợ đa ngôn ngữ, trong đó tiếng Việt là ngôn ngữ mặc định."
            ),
            (
                "Màn hình cảm ứng 10.1 inch độ phân giải 1920x1200, "
                "tích hợp 4 microphone đa hướng phạm vi thu 5 mét, "
                "loa ngoài 2x5W và camera 4K góc rộng 120 độ."
            ),
            (
                "Hệ điều hành: bản fork Android 13 tối ưu cho workstation. "
                "Bộ nhớ trong 128GB, RAM 8GB LPDDR5. Kết nối Wi-Fi 6E, Bluetooth 5.3, "
                "USB-C 3.2 và HDMI 2.1 output."
            ),
            (
                "Mô hình ngôn ngữ nhỏ (SLM) chạy local trên thiết bị: Llama-3.1-8B-Instruct "
                "phiên bản lượng tử hóa 4-bit, đủ sức sinh câu trả lời ngắn cho người dùng "
                "trong phòng họp với độ trễ dưới 1.5 giây."
            ),
        ],
    },
]


# --------------------------------------------------------------------------- #
# Phase 2 — Transcript cuộc họp
# --------------------------------------------------------------------------- #

TRANSCRIPT_CORPUS: list[dict] = [
    {
        "meeting_id": "meeting-bt2-eval",
        "utterances": [
            {
                "sequence_id": 1,
                "speaker": "Đoàn Sỹ Nguyên",
                "text": (
                    "Chào mọi người, hôm nay chúng ta họp về ngân sách quý 4 năm 2026. "
                    "Tôi muốn mời Mai Xuân Ngọc trình bày tổng quan trước."
                ),
            },
            {
                "sequence_id": 2,
                "speaker": "Mai Xuân Ngọc",
                "text": (
                    "Cảm ơn anh Nguyên. Tổng ngân sách quý 4 năm 2026 là 18 tỷ đồng, "
                    "tăng 12% so với cùng kỳ năm ngoái. Trong đó chi phí vận hành chiếm 9 tỷ, "
                    "chi phí nhân sự chiếm 6 tỷ, còn lại 3 tỷ dành cho marketing và R&D."
                ),
            },
            {
                "sequence_id": 3,
                "speaker": "Đoàn Sỹ Nguyên",
                "text": (
                    "Con số này khá cao. Tôi đề xuất cắt giảm 15% chi phí vận hành, "
                    "tương đương khoảng 1.35 tỷ đồng. Số tiền này chuyển sang quỹ dự phòng."
                ),
            },
            {
                "sequence_id": 4,
                "speaker": "Mai Xuân Ngọc",
                "text": (
                    "Tôi đồng ý về mặt nguyên tắc. Cụ thể, phòng vận hành sẽ rà soát lại "
                    "3 hợp đồng thuê ngoài lớn nhất: dịch vụ vệ sinh, IT helpdesk, và bảo trì."
                ),
            },
            {
                "sequence_id": 5,
                "speaker": "Lê Hoàng Anh",
                "text": (
                    "Bổ sung thêm: phòng IT có thể tự vận hành helpdesk thay vì thuê ngoài, "
                    "tiết kiệm khoảng 80 triệu mỗi tháng, tức gần 1 tỷ một năm."
                ),
            },
            {
                "sequence_id": 6,
                "speaker": "Đoàn Sỹ Nguyên",
                "text": (
                    "Tốt. Vậy chốt lại: cắt giảm 15% chi phí vận hành, "
                    "phòng IT tự vận hành helpdesk từ tháng 11, giao Mai Xuân Ngọc "
                    "lên kế hoạch chi tiết trong tuần này."
                ),
            },
            {
                "sequence_id": 7,
                "speaker": "Mai Xuân Ngọc",
                "text": (
                    "Đồng ý. Tôi sẽ trình kế hoạch chi tiết trước thứ Sáu tuần này, "
                    "có phân bổ rõ theo từng hạng mục và timeline cụ thể."
                ),
            },
            {
                "sequence_id": 8,
                "speaker": "Đoàn Sỹ Nguyên",
                "text": (
                    "Sang phần tiếp theo, bàn về kế hoạch tuyển dụng quý 4. "
                    "Phòng R&D đề xuất tuyển 3 kỹ sư AI mới."
                ),
            },
            {
                "sequence_id": 9,
                "speaker": "Trần Minh Quân",
                "text": (
                    "Phòng R&D cần 3 vị trí kỹ sư AI: 1 chuyên về NLP tiếng Việt, "
                    "1 chuyên về computer vision, và 1 chuyên về MLOps. "
                    "Mức lương kỳ vọng từ 35 đến 50 triệu đồng mỗi người."
                ),
            },
            {
                "sequence_id": 10,
                "speaker": "Lê Hoàng Anh",
                "text": (
                    "Cảnh báo rủi ro: với budget hiện tại, nếu tuyển đủ 3 người ở mức lương "
                    "trung bình 40 triệu thì quỹ lương quý 4 vượt 6% so với dự toán."
                ),
            },
            {
                "sequence_id": 11,
                "speaker": "Đoàn Sỹ Nguyên",
                "text": (
                    "Chốt: tuyển 2 người trước (NLP và MLOps), CV sẽ tuyển đầu năm 2027 "
                    "khi budget nhân sự được mở rộng. Phòng HR làm JD trong tuần này."
                ),
            },
            {
                "sequence_id": 12,
                "speaker": "Mai Xuân Ngọc",
                "text": (
                    "Cuộc họp kết thúc. Cảm ơn mọi người, biên bản sẽ được gửi qua email "
                    "trong vòng 24 giờ."
                ),
            },
        ],
    },
]


# --------------------------------------------------------------------------- #
# Q&A vàng
# --------------------------------------------------------------------------- #
#
# Schema:
#   question:                câu hỏi tiếng Việt
#   ground_truth_answer:    câu trả lời mẫu (dùng để tính context_recall và làm ref)
#   gold_chunks:            (chỉ docs) chỉ số chunk vàng trong DOCS_CORPUS[doc_idx]
#   gold_sequence_ids:      (chỉ transcript) sequence_id liên quan trong transcript
#   kind:                   loại câu hỏi — dùng để phân tích theo nhóm

DOCS_QA: list[dict] = [
    {
        "question": "Nhân viên làm việc trên 5 năm được cộng thêm bao nhiêu ngày phép?",
        "ground_truth_answer": (
            "Nhân viên làm trên 5 năm được cộng thêm 2 ngày phép thưởng trung thành, "
            "phòng nhân sự tự động cập nhật vào ngày kỷ niệm làm việc thứ 5."
        ),
        "doc_idx": 0,
        "gold_chunks": [1],
        "kind": "factual",
    },
    {
        "question": "Đơn nghỉ phép 5 ngày phải gửi trước bao nhiêu ngày?",
        "ground_truth_answer": (
            "Đơn nghỉ từ 4 ngày trở lên phải gửi trước 7 ngày và cần phê duyệt của "
            "quản lý cấp trưởng phòng."
        ),
        "doc_idx": 0,
        "gold_chunks": [2],
        "kind": "factual",
    },
    {
        "question": "Công tác phí ở Hà Nội tối đa bao nhiêu mỗi đêm khách sạn?",
        "ground_truth_answer": (
            "Tiền khách sạn tối đa 1.200.000 đồng/đêm tại Hà Nội, TP. HCM và Đà Nẵng."
        ),
        "doc_idx": 1,
        "gold_chunks": [1],
        "kind": "factual",
    },
    {
        "question": "Hạng vé máy bay cho chuyến công tác dài 5 tiếng?",
        "ground_truth_answer": (
            "Chuyến từ 4 tiếng trở lên được bay hạng business. Vé phải đặt trước "
            "ít nhất 14 ngày để áp dụng mức giá công vụ."
        ),
        "doc_idx": 1,
        "gold_chunks": [2],
        "kind": "lookup_number",
    },
    {
        "question": "AI Meeting Box dùng chip gì?",
        "ground_truth_answer": (
            "AI Meeting Box chạy chip Qualcomm QCS8550, hỗ trợ nhận diện khuôn mặt "
            "thời gian thực với độ trễ dưới 200ms."
        ),
        "doc_idx": 2,
        "gold_chunks": [0],
        "kind": "factual",
    },
    {
        "question": "Mô hình ngôn ngữ chạy trên thiết bị là gì?",
        "ground_truth_answer": (
            "Thiết bị chạy Llama-3.1-8B-Instruct lượng tử 4-bit, đủ sức sinh câu trả lời "
            "ngắn với độ trễ dưới 1.5 giây."
        ),
        "doc_idx": 2,
        "gold_chunks": [3],
        "kind": "factual",
    },
    {
        "question": "Cấu hình màn hình và camera của Meeting Box?",
        "ground_truth_answer": (
            "Màn hình cảm ứng 10.1 inch 1920x1200, camera 4K góc rộng 120 độ, "
            "4 micro đa hướng phạm vi 5 mét, loa 2x5W."
        ),
        "doc_idx": 2,
        "gold_chunks": [1],
        "kind": "lookup_spec",
    },
    {
        "question": "Mức công tác phí cho nhân viên cấp quản lý mỗi ngày?",
        "ground_truth_answer": (
            "Cấp quản lý được 350.000 đồng/ngày công tác phí trong nước."
        ),
        "doc_idx": 1,
        "gold_chunks": [0],
        "kind": "lookup_number",
    },
    {
        "question": "Sau chuyến công tác bao lâu phải nộp báo cáo?",
        "ground_truth_answer": (
            "Báo cáo công tác phải nộp trong vòng 5 ngày làm việc sau khi kết thúc "
            "chuyến công tác, kèm hóa đơn gốc. Quá hạn không được hoàn trả."
        ),
        "doc_idx": 1,
        "gold_chunks": [3],
        "kind": "factual",
    },
    {
        "question": "Chính sách nghỉ phép năm cho phép cộng dồn tối đa bao nhiêu ngày?",
        "ground_truth_answer": (
            "Được cộng dồn tối đa 5 ngày sang năm tiếp theo, tổng tích lũy không "
            "vượt quá 17 ngày trong một năm tài chính."
        ),
        "doc_idx": 0,
        "gold_chunks": [0],
        "kind": "factual",
    },
    {
        "question": "Hệ điều hành và bộ nhớ của AI Meeting Box?",
        "ground_truth_answer": (
            "Chạy bản fork Android 13, 128GB bộ nhớ trong, RAM 8GB LPDDR5, "
            "hỗ trợ Wi-Fi 6E, Bluetooth 5.3, USB-C 3.2 và HDMI 2.1."
        ),
        "doc_idx": 2,
        "gold_chunks": [2],
        "kind": "lookup_spec",
    },
    {
        "question": "Trưởng phòng được công tác phí bao nhiêu một ngày?",
        "ground_truth_answer": (
            "Cấp trưởng phòng trở lên được 500.000 đồng/ngày công tác phí trong nước."
        ),
        "doc_idx": 1,
        "gold_chunks": [0],
        "kind": "lookup_number",
    },
]


TRANSCRIPT_QA: list[dict] = [
    {
        "question": "Tổng ngân sách quý 4 năm 2026 là bao nhiêu?",
        "ground_truth_answer": (
            "Tổng ngân sách quý 4 năm 2026 là 18 tỷ đồng, tăng 12% so với cùng kỳ năm ngoái."
        ),
        "gold_sequence_ids": [2],
        "kind": "factual",
    },
    {
        "question": "Đoàn Sỹ Nguyên đề xuất cắt giảm bao nhiêu phần trăm chi phí vận hành?",
        "ground_truth_answer": (
            "Đoàn Sỹ Nguyên đề xuất cắt giảm 15% chi phí vận hành, tương đương khoảng 1.35 tỷ đồng."
        ),
        "gold_sequence_ids": [3, 6],
        "kind": "lookup_speaker",
    },
    {
        "question": "Phòng nào sẽ tự vận hành helpdesk thay vì thuê ngoài?",
        "ground_truth_answer": (
            "Phòng IT sẽ tự vận hành helpdesk từ tháng 11, tiết kiệm khoảng 80 triệu mỗi tháng."
        ),
        "gold_sequence_ids": [5, 6],
        "kind": "factual",
    },
    {
        "question": "Quyết định cuối cùng về kế hoạch tuyển dụng là gì?",
        "ground_truth_answer": (
            "Chốt tuyển 2 người trước (NLP và MLOps), vị trí computer vision sẽ tuyển đầu năm 2027. "
            "Phòng HR làm JD trong tuần này."
        ),
        "gold_sequence_ids": [11],
        "kind": "multi_hop",
    },
    {
        "question": "Phòng R&D đề xuất tuyển bao nhiêu kỹ sư AI?",
        "ground_truth_answer": (
            "Phòng R&D đề xuất tuyển 3 kỹ sư AI: NLP tiếng Việt, computer vision, và MLOps. "
            "Mức lương kỳ vọng 35 đến 50 triệu đồng mỗi người."
        ),
        "gold_sequence_ids": [8, 9],
        "kind": "factual",
    },
    {
        "question": "Ai sẽ trình kế hoạch chi tiết về cắt giảm chi phí?",
        "ground_truth_answer": (
            "Mai Xuân Ngọc được giao lên kế hoạch chi tiết về cắt giảm chi phí vận hành, "
            "trình trước thứ Sáu tuần này."
        ),
        "gold_sequence_ids": [6, 7],
        "kind": "lookup_speaker",
    },
    {
        "question": "Lê Hoàng Anh cảnh báo rủi ro gì về tuyển dụng?",
        "ground_truth_answer": (
            "Lê Hoàng Anh cảnh báo rằng nếu tuyển đủ 3 người ở mức lương trung bình 40 triệu "
            "thì quỹ lương quý 4 vượt 6% so với dự toán."
        ),
        "gold_sequence_ids": [10],
        "kind": "lookup_speaker",
    },
    {
        "question": "Cuộc họp kết thúc như thế nào?",
        "ground_truth_answer": (
            "Cuộc họp kết thúc bằng phát biểu của Mai Xuân Ngọc, biên bản sẽ được gửi "
            "qua email trong vòng 24 giờ."
        ),
        "gold_sequence_ids": [12],
        "kind": "factual",
    },
    {
        "question": "Có phương án nào khác để tiết kiệm chi phí ngoài cắt giảm 15% không?",
        "ground_truth_answer": (
            "Phòng IT tự vận hành helpdesk, tiết kiệm khoảng 80 triệu mỗi tháng, "
            "gần 1 tỷ đồng mỗi năm — đây là đề xuất bổ sung của Lê Hoàng Anh."
        ),
        "gold_sequence_ids": [5],
        "kind": "multi_hop",
    },
    {
        "question": "Phân bổ ngân sách giữa vận hành, nhân sự và marketing như thế nào?",
        "ground_truth_answer": (
            "Chi phí vận hành 9 tỷ, nhân sự 6 tỷ, marketing và R&D 3 tỷ đồng, "
            "tổng cộng 18 tỷ đồng."
        ),
        "gold_sequence_ids": [2],
        "kind": "lookup_number",
    },
]
