# 🚦 Traffic Monitor Desktop — YOLOv8

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-111F68)
![Supervision](https://img.shields.io/badge/Roboflow-Supervision%200.26%2B-6706CE)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/GUI-PySide6%20(Qt%206)-41CD52?logo=qt&logoColor=white)
![PyTorch CUDA](https://img.shields.io/badge/GPU-PyTorch%20CUDA%20FP16-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D6?logo=windows&logoColor=white)
![Tests](https://img.shields.io/badge/tests-60%20unit%20%2B%2034%20E2E-2EA44F)

Ứng dụng **desktop giám sát giao thông bằng Computer Vision**. Phần mềm phát hiện và phân loại phương tiện (ô tô, xe máy, xe buýt, xe tải) bằng **YOLOv8n**, theo dõi từng xe bằng **ByteTrack**, đếm lượt **vào/ra (IN/OUT)** qua vạch đếm, **đo tốc độ km/h** bằng biến đổi phối cảnh (perspective transform), **cảnh báo vượt tốc độ realtime**, rồi xuất **video đã gắn nhãn, báo cáo JSON, CSV và ảnh vi phạm**. Mọi chức năng được điều khiển trên giao diện PySide6. Inference chạy trên **GPU NVIDIA (CUDA, FP16)** và tự chuyển sang CPU khi máy không có GPU.

![Dashboard](docs/images/dashboard_finished.png)

---

## Mục lục

1. [Mô tả dự án](#1-mô-tả-dự-án)
2. [Công nghệ sử dụng](#2-công-nghệ-sử-dụng)
3. [Tính năng](#3-tính-năng)
4. [Giao diện](#4-giao-diện)
5. [Kiến trúc](#5-kiến-trúc)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Cài đặt](#7-cài-đặt)
8. [Chạy ứng dụng](#8-chạy-ứng-dụng)
9. [Cấu hình](#9-cấu-hình)
10. [Nguồn video: file, webcam, RTSP](#10-nguồn-video-file-webcam-rtsp)
11. [Hiệu chỉnh đo tốc độ](#11-hiệu-chỉnh-đo-tốc-độ)
12. [Kết quả đầu ra](#12-kết-quả-đầu-ra)
13. [Hiệu năng](#13-hiệu-năng)
14. [Kiểm thử](#14-kiểm-thử)
15. [Xử lý sự cố](#15-xử-lý-sự-cố)
16. [Hướng phát triển](#16-hướng-phát-triển)

---

## 1. Mô tả dự án

Đồ án cuối khoá **Bài 9 — Traffic Monitor Desktop**. Mục tiêu là xây dựng một phần mềm giám sát giao thông chạy trên Windows, có thể dùng cho camera đặt trên cầu vượt hoặc cột đèn nhìn xuống đường:

| Bài toán | Cách giải quyết |
|---|---|
| Có bao nhiêu xe, loại gì? | YOLOv8n phát hiện, chỉ giữ 4 lớp COCO: `car`, `motorcycle`, `bus`, `truck` |
| Xe nào là xe nào qua các frame? | ByteTrack gán `tracker_id` ổn định, DetectionsSmoother giảm rung khung |
| Bao nhiêu xe đi vào / đi ra? | Supervision `LineZone`, mỗi `tracker_id` chỉ được đếm **một lần** |
| Xe chạy nhanh bao nhiêu? | Homography 4 điểm từ pixel sang mét, vận tốc hồi quy tuyến tính trong cửa sổ 1 giây |
| Xe nào vượt tốc độ? | `speed > speed_limit` tạo sự kiện `SPEED_VIOLATION` (1 lần/xe), box đỏ, ảnh chụp |
| Lưu lại kết quả? | Video MP4 đã gắn nhãn, báo cáo JSON, danh sách sự kiện CSV, log |

**Nguyên tắc thiết kế:** không có dữ liệu giả và không có nút “trang trí”; mọi con số trên giao diện đều đến từ pipeline thật. Giao diện không bao giờ bị treo vì inference chạy trong `QThread`. Khi xảy ra lỗi, ứng dụng hiện hộp thoại rõ ràng thay vì crash.

📄 **Tài liệu đặc tả kỹ thuật đầy đủ** (kiến trúc, đặc tả module, mô hình dữ liệu, xử lý lỗi, kế hoạch kiểm thử, UI demo tương tác): [`docs/traffic-monitor-spec.html`](docs/traffic-monitor-spec.html). Tải về và mở bằng trình duyệt, hoặc xem nhanh qua [htmlpreview](https://htmlpreview.github.io/?https://github.com/TruongTanNghia/Traffic-Monitor-YoloV8/blob/main/docs/traffic-monitor-spec.html). Đề bài gốc: [`docs/original-requirements.md`](docs/original-requirements.md).

## 2. Công nghệ sử dụng

| Công nghệ | Phiên bản | Vai trò trong dự án |
|---|---|---|
| **Python** | 3.10+ (đã kiểm thử 3.11) | Ngôn ngữ chính; dùng type hints, dataclass, pathlib |
| **Ultralytics YOLOv8n** | ultralytics ≥ 8.1 (đã kiểm thử 8.4) | Mô hình phát hiện đối tượng (`models/yolov8n.pt`, huấn luyện trên COCO) |
| **PyTorch + CUDA** | torch 2.x, CUDA 12.x | Chạy inference trên GPU NVIDIA với độ chính xác **FP16** |
| **Roboflow Supervision** | ≥ 0.26 (đã kiểm thử 0.27) | `sv.Detections`, **ByteTrack**, `DetectionsSmoother`, **LineZone**, các annotator (box, label, trace) |
| **OpenCV** | opencv-python ≥ 4.8 | Đọc/ghi video, webcam (DirectShow), RTSP (FFmpeg có timeout), `getPerspectiveTransform` |
| **NumPy** | ≥ 1.24 | Xử lý toạ độ, hồi quy vận tốc |
| **PySide6 (Qt 6)** | ≥ 6.6 | Giao diện desktop, `QThread` + Signal/Slot, model/view cho bảng sự kiện, theme QSS |
| **PyYAML** | ≥ 6.0 | Lưu/đọc cấu hình `config.yaml` |
| **python-dotenv** | ≥ 1.0 | Nạp biến môi trường từ `.env` (device, log level, RTSP URL có mật khẩu) |
| **pytest** | ≥ 8.0 | 60 unit test (YOLO được mock, không cần GPU) |

**Kỹ thuật chính:**

- **Phát hiện:** YOLOv8n (FP16 trên GPU), frame 4K được thu nhỏ còn ≤ 1280 px trước inference rồi scale box về toạ độ gốc.
- **Tracking:** ByteTrack + làm mượt box 5 frame.
- **Đếm:** LineZone ngang với anchor `BOTTOM_CENTER`, chống đếm trùng theo `tracker_id`.
- **Đo tốc độ:** homography 4 điểm từ pixel sang mặt đường theo mét. Vận tốc tính bằng hồi quy tuyến tính (least squares) trên cửa sổ 1 giây. Mẫu có cạnh dưới bị xe phía trước che khuất được loại bỏ, và giá trị phi thực tế (> 250 km/h) bị bỏ qua.
- **Đa luồng:** GUI thread → worker `QThread` → thread giải mã video (prefetch) + thread mã hoá MP4, có cơ chế back-pressure cho preview.
- **Kiến trúc phân lớp:** chỉ `app/gui` import Qt; lõi Computer Vision thuần Python để test độc lập với giao diện.

## 3. Tính năng

| Tính năng | Quản lý trên UI |
|---|---|
| Phát hiện, phân loại, tracking ID | Dashboard (video realtime) |
| Đếm IN / OUT theo từng loại xe | Dashboard, Reports |
| Đo tốc độ km/h, cảnh báo vượt tốc độ realtime | Dashboard (box đỏ `OVER`), Event Logs (badge ⚠) |
| 8 thẻ thống kê + FPS, GPU, model, trạng thái | Dashboard |
| START / PAUSE / RESUME / STOP, thanh tiến độ | Dashboard |
| Nguồn: video file, webcam, RTSP (TEST / APPLY) | Video Source |
| Chỉnh model, GPU/CPU, confidence, giới hạn tốc độ, vạch đếm, kích thước đường | Configuration |
| Hiệu chỉnh 4 điểm phối cảnh bằng click chuột trên frame thật | Configuration → *Calibrate from video…* |
| SAVE CONFIG (có kiểm tra hợp lệ) / RESET DEFAULT | Configuration |
| Xuất JSON, CSV, mở thư mục output | Reports |
| Bảng sự kiện realtime, lọc theo loại / ID | Event Logs |
| Xuất video đã gắn nhãn, ảnh chụp mỗi vi phạm | Tự động khi kết thúc phiên |

## 4. Giao diện

| Configuration | Reports |
|---|---|
| ![Configuration](docs/images/configuration.png) | ![Reports](docs/images/reports.png) |
| **Event Logs** | **Video Source** |
| ![Event Logs](docs/images/event_logs.png) | ![Video Source](docs/images/video_source.png) |

## 5. Kiến trúc

```
Presentation   app/gui        MainWindow · Dashboard · Source · Settings · Reports · Logs · CalibrationDialog
      │ Signal/Slot (queued)
Worker         app/gui/processing_worker.py      ProcessingWorker (QThread)
      ▼
Orchestration  app/core · app/video    TrafficMonitor · ProcessingPipeline · FrameProcessor · VideoReader · VideoWriter
Domain CV      app/core                VehicleDetector · VehicleTracker · TrafficCounter · SpeedEstimator · ZoneManager · AlertManager
Infrastructure app/video · vision · reports
Foundation     app/config · app/utils
```

Pipeline cho mỗi frame:

```
detect (YOLOv8n) → filter (confidence + class) → track (ByteTrack + Smoother) → count (LineZone)
      → speed (homography + regression) → alert (SPEED_VIOLATION) → annotate → statistics
```

Luồng xử lý:

```
MainThread (GUI) ──start / pause / stop──▶ ProcessingWorker (QThread)
      ▲                                     ├─ VideoReader   (thread giải mã, prefetch 4 frame)
      │ frame_ready(QImage)                 ├─ YOLOv8n       (GPU FP16)
      │ statistics_updated(dict)            ├─ ByteTrack · SpeedEstimator · AlertManager
      │ event_created(dict)                 ├─ FrameProcessor (annotate)
      │ processing_started / finished       └─ VideoWriter   (thread mã hoá MP4)
      │ error_occurred(title, msg, critical)
```

Trạng thái phiên: `IDLE → STARTING → RUNNING ⇄ PAUSED → STOPPING → FINISHED`, hoặc `ERROR`.

## 6. Cấu trúc thư mục

```
traffic_monitor/
├── main.py                  # chỉ khởi động app: load config → MainWindow → QApplication
├── requirements.txt · .env.example · .gitignore · pytest.ini
├── app/
│   ├── gui/                 # main_window, dashboard, video_panel, statistics_panel, settings_panel,
│   │   │                    # report_panel, source_panel, log_panel, calibration_dialog,
│   │   │                    # processing_worker, background, ui_helpers
│   │   └── widgets/         # stat_card, video_view, log_view
│   ├── core/                # traffic_monitor, detector, tracker, counter, speed_estimator,
│   │                        # zone_manager, alert_manager, pipeline
│   ├── video/               # video_source (File/Camera/RTSP), video_reader, video_writer, frame_processor
│   ├── vision/              # annotations, transformer (ViewTransformer), geometry, coordinates
│   ├── reports/             # report_generator, json_exporter, csv_exporter
│   ├── config/              # settings (AppConfig dataclass), default_config
│   └── utils/               # logger, file_utils, time_utils, constants, exceptions
├── models/                  # yolov8n.pt (tải bằng script)
├── data/{input, samples, output/{videos, reports, snapshots}}
├── logs/                    # app.log
├── tests/                   # test_detector, test_tracker, test_counter, test_speed, test_report, test_config
├── scripts/                 # download_model, test_camera, calibrate_perspective, smoke_test_gui
├── assets/styles/app.qss    # theme tối kiểu dashboard giám sát công nghiệp
└── docs/                    # đặc tả kỹ thuật HTML, đề bài gốc, ảnh giao diện
```

## 7. Cài đặt

```powershell
git clone https://github.com/TruongTanNghia/Traffic-Monitor-YoloV8.git
cd Traffic-Monitor-YoloV8
python -m venv .venv
.venv\Scripts\activate

# GPU NVIDIA: cài PyTorch bản CUDA TRƯỚC (bỏ qua bước này nếu chỉ chạy CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# Tải model YOLOv8n + video mẫu (4K, 538 frame)
python scripts/download_model.py --sample
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Biến môi trường (tuỳ chọn): sao chép `.env.example` thành `.env`.

| Biến | Ý nghĩa |
|---|---|
| `TM_DEVICE` | `auto` (GPU nếu có), `cuda:0`, `cpu` |
| `TM_LOG_LEVEL` | `INFO` (mặc định), `DEBUG`, `WARNING` |
| `TM_RTSP_URL` | URL RTSP có mật khẩu (không đưa vào `config.yaml`) |

## 8. Chạy ứng dụng

```powershell
python main.py
```

1. **Video Source**: chọn *Local video* → *Browse…* → *TEST SOURCE* → *APPLY SOURCE*. Lần đầu chạy, `data/samples/vehicles.mp4` đã được chọn sẵn.
2. **Dashboard**: bấm **START**. Trạng thái chuyển `LOADING… → RUNNING`. Dùng **PAUSE / RESUME**, **STOP**.
3. Khi kết thúc, hộp thoại hiển thị đường dẫn output; xem chi tiết ở **Reports** và **Event Logs**.

Phím tắt: `Ctrl+1 … Ctrl+5` để chuyển trang.

## 9. Cấu hình

Trang **Configuration** chỉnh mọi tham số. **SAVE CONFIG** kiểm tra hợp lệ rồi ghi `config.yaml`. Trường sai được tô đỏ và cấu hình cũ được giữ nguyên. Cấu hình bị khoá khi đang xử lý và áp dụng cho lần START tiếp theo.

| Tham số | Mặc định | Miền hợp lệ |
|---|---|---|
| `model_path` | `models/yolov8n.pt` | file `.pt` tồn tại |
| `device` | `auto` | `auto` / `cuda:0` / `cpu` |
| `confidence_threshold` | 0.30 | 0.05 – 0.95 |
| `speed_limit_kmh` | 80 | 10 – 250 |
| `line_y_ratio` | 0.55 | 0.10 – 0.90 |
| `road_width_m` / `road_length_m` | 25 / 250 | > 0 |
| `perspective_source` + `perspective_reference_size` | 4 điểm cho video mẫu 3840×2160 | tứ giác lồi, tự scale theo độ phân giải |
| `min_track_seconds` | 0.5 | 0.2 – 3.0 |
| `show_preview`, `save_output_video`, `save_report`, `save_snapshots` | `true` | |
| `source_type` / `source_uri` / `rtsp_timeout_ms` | `file` / `data/samples/vehicles.mp4` / 5000 | |
| `output_dir` | `data/output` | |

## 10. Nguồn video: file, webcam, RTSP

- **Video file:** `.mp4 .avi .mkv .mov .m4v .wmv`. Thời gian được tính theo `frame_index / fps`, nên tốc độ đo đúng kể cả khi máy xử lý chậm hơn realtime.
- **Webcam:** chọn camera index (backend DirectShow trên Windows).
- **RTSP:** `rtsp://user:pass@host:554/path`, có timeout. Khi mất kết nối, ứng dụng thử lại 3 lần (1 s, 2 s, 4 s) rồi dừng phiên và vẫn lưu báo cáo phần đã xử lý.

Kiểm tra nhanh bằng dòng lệnh: `python scripts/test_camera.py --camera 0`, `--rtsp <url>` hoặc `--video <file>`.

## 11. Hiệu chỉnh đo tốc độ

1. Chọn trên mặt đường một hình chữ nhật biết kích thước thật (ví dụ 25 m × 250 m).
2. **Configuration** → nhập *Road width / Road length* → *Calibrate from video…* → click lần lượt A (trên-trái), B (trên-phải), C (dưới-phải), D (dưới-trái). Có thể click ra ngoài khung hình; chuột phải để xoá điểm cuối.
3. Bấm **SAVE CONFIG**.

Có thể dùng công cụ dòng lệnh thay thế: `python scripts/calibrate_perspective.py --video <file> --width 25 --length 250`.

> Điểm mặc định chỉ khớp với video mẫu. Với camera khác, **bắt buộc phải hiệu chỉnh lại** thì tốc độ mới chính xác.

## 12. Kết quả đầu ra

Mọi file của cùng một phiên dùng chung `session_id` (`YYYYMMDD_HHMMSS`):

```
data/output/videos/traffic_output_YYYYMMDD_HHMMSS.mp4
data/output/reports/traffic_report_YYYYMMDD_HHMMSS.json   # session, source, config, total_in/out,
                                                          # vehicle_counts, speed_violations, maximum_speed, events
data/output/reports/traffic_report_YYYYMMDD_HHMMSS.csv    # timestamp, frame_index, video_time_s, tracker_id,
                                                          # vehicle_type, speed_kmh, speed_limit_kmh, event_type
data/output/snapshots/violation_<session>_id<ID>_f<frame>.jpg
logs/app.log
```

Khi bấm STOP giữa chừng, ứng dụng vẫn ghi đủ output của phần đã xử lý.

## 13. Hiệu năng

Đo trên RTX 3050 Laptop 4 GB, video mẫu 3840×2160:

| Hạng mục | Tốc độ |
|---|---|
| Chỉ YOLOv8n detect (GPU FP16) | ~28 FPS |
| Toàn bộ pipeline, không ghi video | ~18 FPS |
| Toàn bộ pipeline, có ghi MP4 4K | ~12.6 FPS (giới hạn bởi bộ mã hoá mp4v) |
| Webcam 640×480 | theo tốc độ camera |

Với video 1080p hoặc thấp hơn, tốc độ cao hơn đáng kể.

## 14. Kiểm thử

```powershell
python -m pytest                      # 60 unit test — YOLO được mock, không cần GPU
python scripts/smoke_test_gui.py      # 34 kiểm tra end-to-end trên GUI với pipeline thật
```

`smoke_test_gui.py` điều khiển giao diện như người dùng. Các luồng được kiểm tra: START → xử lý hết video → FINISHED, PAUSE / RESUME, STOP giữa chừng, xuất JSON / CSV, lọc log, SAVE / RESET / kiểm tra cấu hình, TEST / APPLY nguồn video, các hộp thoại lỗi (thiếu model, thiếu video, camera không kết nối) và đóng cửa sổ khi đang xử lý.

## 15. Xử lý sự cố

| Thông báo | Nguyên nhân / cách xử lý |
|---|---|
| *Không load được model* | Chạy `python scripts/download_model.py` hoặc chọn model khác |
| *Video không tồn tại* / *Không mở được video* | Sai đường dẫn, file hỏng hoặc codec lạ → chuyển sang MP4 H.264 |
| *Camera không kết nối* | Camera đang bị ứng dụng khác dùng, hoặc sai index |
| *RTSP timeout* | Kiểm tra mạng / URL, tăng timeout |
| *Perspective không hợp lệ* | 4 điểm phải theo thứ tự A-B-C-D và tạo thành tứ giác lồi |
| *Cấu hình không hợp lệ* | Trường tô đỏ ghi rõ miền hợp lệ; `config.yaml` hỏng khi khởi động thì ứng dụng dùng mặc định |
| *Lỗi inference* | Frame lỗi bị bỏ qua; 10 frame lỗi liên tiếp thì dừng phiên. Hết VRAM → `TM_DEVICE=cpu` |
| *Không ghi được video* | Vẫn xử lý tiếp nhưng không lưu video; kiểm tra quyền ghi thư mục output |
| Device hiển thị `cpu` dù có GPU | Đang dùng PyTorch bản CPU → cài lại torch CUDA (mục 7) |

Chi tiết lỗi xem trong `logs/app.log`.

## 16. Hướng phát triển

- Nhiều vạch đếm / vùng polygon, đếm theo làn.
- Nhận dạng biển số (ANPR) cho xe vi phạm.
- Lưu sự kiện vào SQLite, dashboard thống kê theo ngày.
- Nhiều camera đồng thời; mã hoá video bằng GPU (NVENC).
- Xuất báo cáo PDF.
