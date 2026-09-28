Bạn là Senior Python Computer Vision Engineer + Senior Desktop Application Engineer.

Hãy xây dựng hoàn chỉnh đồ án cuối khóa BÀI 9: TRAFFIC MONITOR DESKTOP.

## 1. MỤC TIÊU

Xây dựng một ứng dụng desktop giám sát giao thông bằng Computer Vision.

Ứng dụng phải sử dụng:

* Python
* OpenCV
* Ultralytics YOLO
* Roboflow Supervision
* PySide6
* NumPy

Hệ thống phải chạy được trên Windows.

Mục tiêu của ứng dụng:

1. Phát hiện phương tiện.
2. Phân loại phương tiện.
3. Tracking từng phương tiện bằng ID.
4. Đếm phương tiện đi vào / đi ra qua LineZone.
5. Đo tốc độ phương tiện bằng Perspective Transform / ViewTransformer.
6. Phát hiện phương tiện vượt quá tốc độ.
7. Hiển thị cảnh báo realtime.
8. Hiển thị video realtime trên desktop.
9. Hiển thị dashboard thống kê realtime.
10. Xuất video đã annotation.
11. Xuất báo cáo JSON.
12. Xuất báo cáo CSV.
13. Lưu event log.
14. Hỗ trợ video file.
15. Thiết kế kiến trúc có thể mở rộng sang webcam và RTSP.

---

# 2. CÔNG NGHỆ

Python 3.10+

PySide6
OpenCV
NumPy
Supervision >= 0.26
Ultralytics
PyYAML

Không viết toàn bộ hệ thống thành một file.

Không hard-code toàn bộ logic vào GUI.

Tách rõ GUI, Computer Vision, video processing, report và configuration.

---

# 3. KIẾN TRÚC THƯ MỤC BẮT BUỘC

traffic_monitor/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── app/
│   ├── **init**.py
│   │
│   ├── gui/
│   │   ├── **init**.py
│   │   ├── main_window.py
│   │   ├── dashboard.py
│   │   ├── video_panel.py
│   │   ├── statistics_panel.py
│   │   ├── settings_panel.py
│   │   ├── report_panel.py
│   │   └── widgets/
│   │       ├── **init**.py
│   │       ├── stat_card.py
│   │       ├── video_view.py
│   │       └── log_view.py
│   │
│   ├── core/
│   │   ├── **init**.py
│   │   ├── traffic_monitor.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── counter.py
│   │   ├── speed_estimator.py
│   │   ├── zone_manager.py
│   │   ├── alert_manager.py
│   │   └── pipeline.py
│   │
│   ├── video/
│   │   ├── **init**.py
│   │   ├── video_source.py
│   │   ├── video_reader.py
│   │   ├── video_writer.py
│   │   └── frame_processor.py
│   │
│   ├── vision/
│   │   ├── **init**.py
│   │   ├── annotations.py
│   │   ├── transformer.py
│   │   ├── geometry.py
│   │   └── coordinates.py
│   │
│   ├── reports/
│   │   ├── **init**.py
│   │   ├── report_generator.py
│   │   ├── json_exporter.py
│   │   └── csv_exporter.py
│   │
│   ├── config/
│   │   ├── **init**.py
│   │   ├── settings.py
│   │   └── default_config.py
│   │
│   └── utils/
│       ├── **init**.py
│       ├── logger.py
│       ├── file_utils.py
│       ├── time_utils.py
│       └── constants.py
│
├── models/
│   └── yolov8n.pt
│
├── data/
│   ├── input/
│   ├── output/
│   │   ├── videos/
│   │   ├── reports/
│   │   └── snapshots/
│   └── samples/
│
├── logs/
│   └── app.log
│
├── tests/
│   ├── **init**.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_counter.py
│   ├── test_speed.py
│   ├── test_report.py
│   └── test_config.py
│
├── scripts/
│   ├── download_model.py
│   ├── test_camera.py
│   └── calibrate_perspective.py
│
└── assets/
├── icons/
└── styles/
└── app.qss

---

# 4. VEHICLE DETECTION

Tạo:

app/core/detector.py

Class:

VehicleDetector

Các method bắt buộc:

load_model()
detect(frame)
filter_detections(detections)
get_vehicle_classes()

Model mặc định:

yolov8n.pt

Có thể thay model thông qua configuration.

Vehicle classes:

car
motorcycle
bus
truck

Confidence threshold mặc định:

0.30

Pipeline:

frame
→ YOLO
→ sv.Detections
→ confidence filter
→ vehicle class filter

---

# 5. TRACKING

Tạo:

app/core/tracker.py

Class:

VehicleTracker

Method:

initialize()
update(detections)
reset()
get_track_ids()
smooth(detections)

Sử dụng Supervision ByteTrack.

Sử dụng DetectionsSmoother để giảm jitter.

Mỗi vehicle phải có tracker_id.

---

# 6. COUNTING

Tạo:

app/core/counter.py

Class:

TrafficCounter

Sử dụng Supervision LineZone.

Method:

initialize_line()
process(detections)
get_total_in()
get_total_out()
get_class_counts()
reset()

Đếm:

car
motorcycle
bus
truck

Theo:

IN
OUT

Không được đếm trùng cùng một tracker ID.

---

# 7. SPEED ESTIMATION

Tạo:

app/core/speed_estimator.py

Class:

SpeedEstimator

Sử dụng perspective transform.

Method:

initialize_transformer()
transform_points()
update_track_position()
calculate_speed()
get_speed()
get_speed_violations()
reset()

Sử dụng bottom-center anchor của detection.

Lưu lịch sử tọa độ theo tracker ID bằng deque.

Tính tốc độ theo km/h.

Không tính tốc độ ngay frame đầu tiên.

Phải có khoảng thời gian tối thiểu trước khi hiển thị speed.

---

# 8. ALERT MANAGER

Tạo:

app/core/alert_manager.py

Class:

AlertManager

Method:

check_speed()
create_speed_alert()
register_alert()
get_active_alerts()
get_alert_history()
clear_alerts()

Speed limit mặc định:

80 km/h

Nếu:

speed > speed_limit

thì tạo SPEED_VIOLATION.

Alert phải chứa:

timestamp
tracker_id
vehicle_type
speed
speed_limit
event_type

---

# 9. VIDEO PROCESSING

Tạo:

app/video/frame_processor.py

Class:

FrameProcessor

Method quan trọng:

process_frame(frame)

Pipeline:

detect
→ filter
→ track
→ count
→ speed
→ alert
→ annotate
→ dashboard

Phải trả về:

annotated_frame

---

# 10. VIDEO SOURCE

Tạo:

app/video/video_source.py

Hỗ trợ:

1. Local video
2. Webcam
3. RTSP

Thiết kế interface thống nhất.

Ví dụ:

VideoSource
├── FileVideoSource
├── CameraVideoSource
└── RTSPVideoSource

Không để GUI trực tiếp xử lý cv2.VideoCapture.

---

# 11. GUI DESKTOP

Sử dụng PySide6.

Không dùng Tkinter.

Thiết kế giao diện hiện đại nhưng KHÔNG được có phong cách "AI generated".

Không lạm dụng gradient.

Không làm UI quá màu mè.

Ưu tiên:

dark professional dashboard
industrial monitoring
clean layout
clear typography
compact information density

Main window gồm:

Dashboard
Video Source
Configuration
Reports
Event Logs

---

# 12. DASHBOARD

Dashboard phải hiển thị:

LIVE VIDEO

Stat cards:

Total Vehicles
Vehicles IN
Vehicles OUT
Cars
Motorcycles
Buses
Trucks
Speed Violations

Thông tin hệ thống:

FPS
Processing status
Current source
Model
Confidence threshold

---

# 13. VIDEO VIEW

Video phải được hiển thị trực tiếp trong PySide6.

Không mở cửa sổ cv2.imshow riêng khi chạy GUI.

Frame OpenCV phải chuyển sang QImage/QPixmap.

Video processing phải chạy trong QThread hoặc worker thread.

Không được block GUI thread.

Có nút:

START
STOP
PAUSE

---

# 14. ANNOTATION

Video phải hiển thị:

Bounding box
Vehicle class
Confidence
Tracker ID
Speed km/h
Speed violation
Tracking trace
Counting line
IN / OUT count
Dashboard overlay

Màu tracker có thể khác nhau theo ID.

---

# 15. REPORT

Tạo:

app/reports/report_generator.py

Class:

ReportGenerator

Method:

generate()
save_json()
save_csv()

JSON phải chứa:

session information
source information
total_in
total_out
vehicle counts by class
speed violations
maximum speed
event history

CSV chứa danh sách event.

---

# 16. CONFIGURATION

Không hard-code configuration trong business logic.

Tạo AppConfig bằng dataclass.

Các tham số:

model_path
confidence_threshold
speed_limit_kmh
line_y_ratio
road_width_m
road_length_m
perspective_source
show_preview
save_output_video
save_report

GUI phải cho phép thay đổi những tham số phù hợp.

---

# 17. SETTINGS PAGE

Cho phép user:

Select model
Select video
Set confidence
Set speed limit
Set line position
Set road width
Set road length
Configure output directory

Có:

SAVE CONFIG
RESET DEFAULT

---

# 18. REPORT PAGE

Hiển thị:

Total vehicles
IN
OUT
Car
Motorcycle
Bus
Truck
Speed violations
Maximum speed

Có button:

Export JSON
Export CSV
Open Output Folder

---

# 19. EVENT LOG

Mỗi event:

timestamp
track ID
vehicle type
speed
event type

Ví dụ:

12:31:02
Vehicle #21
Car
92 km/h
SPEED_VIOLATION

Event log phải hiển thị realtime.

---

# 20. LOGGING

Tạo:

app/utils/logger.py

Sử dụng Python logging.

Log:

INFO
WARNING
ERROR

Log file:

logs/app.log

---

# 21. ERROR HANDLING

Phải xử lý:

Model không tồn tại
Video không tồn tại
Video không mở được
Camera không kết nối
RTSP timeout
Invalid perspective points
Invalid configuration
Model inference error
Video writer error

GUI không được crash khi xảy ra lỗi.

Hiển thị QMessageBox phù hợp.

---

# 22. THREADING

Không chạy inference trong GUI thread.

Kiến trúc:

MainThread
│
├── GUI
│
└── ProcessingWorker
│
├── VideoReader
├── YOLO
├── ByteTrack
├── SpeedEstimator
└── FrameProcessor

Worker phát signal:

frame_ready
statistics_updated
event_created
processing_started
processing_finished
error_occurred

---

# 23. OUTPUT

Sau khi xử lý:

data/output/videos/

traffic_output_YYYYMMDD_HHMMSS.mp4

data/output/reports/

traffic_report_YYYYMMDD_HHMMSS.json

traffic_report_YYYYMMDD_HHMMSS.csv

---

# 24. TEST

Viết unit test cho:

VehicleDetector
VehicleTracker
TrafficCounter
SpeedEstimator
ReportGenerator
AppConfig

Không yêu cầu test inference YOLO thật trong unit test.

Có thể mock model.

---

# 25. REQUIREMENTS.TXT

Tạo requirements.txt phù hợp.

Tối thiểu:

PySide6
opencv-python
numpy
supervision
ultralytics
PyYAML

Nếu cần thêm thư viện thì giải thích lý do.

---

# 26. MAIN.PY

main.py chỉ có nhiệm vụ:

1. Khởi động application.
2. Load configuration.
3. Tạo MainWindow.
4. Chạy QApplication.

Không đặt logic Computer Vision trong main.py.

---

# 27. CODE QUALITY

Yêu cầu:

PEP8
Type hints
Docstrings
Tên biến rõ ràng
Tên class rõ ràng
Không viết code lặp
Không viết function quá dài
Không dùng global state tùy tiện
Không hard-code đường dẫn Windows
Sử dụng pathlib
Có logging
Có exception handling

Mỗi class chỉ nên chịu trách nhiệm cho một nhóm nhiệm vụ.

---

# 28. README

Tạo README.md đầy đủ:

1. Giới thiệu
2. Features
3. Architecture
4. Folder structure
5. Installation
6. Environment setup
7. Download model
8. Run application
9. Configuration
10. Video input
11. RTSP input
12. Speed calibration
13. Output
14. Troubleshooting
15. Testing
16. Future improvements

---

# 29. CHƯƠNG TRÌNH PHẢI CHẠY ĐƯỢC

Sau khi hoàn thành source code, kiểm tra:

python main.py

Ứng dụng phải:

1. Mở GUI.
2. Cho chọn video.
3. Load model.
4. Start processing.
5. Hiển thị video realtime.
6. Hiển thị bounding boxes.
7. Hiển thị tracker IDs.
8. Đếm IN / OUT.
9. Hiển thị speed.
10. Cảnh báo speed violation.
11. Cập nhật dashboard.
12. Ghi event log.
13. Xuất video.
14. Xuất JSON.
15. Xuất CSV.

---

# 30. QUAN TRỌNG

Không tạo demo giả.

Không fake dữ liệu.

Không tạo nút chức năng nhưng không có logic phía sau.

Mọi button trên GUI phải thực sự hoạt động.

Mọi chức năng trong dashboard phải kết nối với pipeline thật.

Không sử dụng dữ liệu hard-code để giả lập số lượng xe.

Nếu model/video không tồn tại thì phải báo lỗi rõ ràng.

---

# 31. CÁCH TRIỂN KHAI

Không viết tất cả code trong một file.

Hãy tạo từng file theo đúng cấu trúc thư mục.

Trước tiên hãy tạo project structure.

Sau đó triển khai:

PHASE 1
Configuration + Utils

PHASE 2
Video Source

PHASE 3
YOLO Detector

PHASE 4
ByteTrack

PHASE 5
Traffic Counter

PHASE 6
Speed Estimator

PHASE 7
Alert Manager

PHASE 8
Frame Processor

PHASE 9
Report Generator

PHASE 10
PySide6 GUI

PHASE 11
Threading

PHASE 12
Testing

PHASE 13
README

Sau mỗi phase phải kiểm tra import và lỗi syntax.

Cuối cùng chạy:

python main.py

và kiểm tra end-to-end.

Khi trả code, phải ghi rõ:

FILE PATH

sau đó mới đưa code của file đó.

Không được bỏ qua file quan trọng.

Không viết "...rest of code".

Không dùng placeholder cho những phần bắt buộc.

Hãy triển khai đầy đủ source code có thể chạy được.
