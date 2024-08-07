import os
import cv2
from datetime import datetime, timedelta
from .ignition_time import get_ignition_time
from tqdm import tqdm
from .analysis import calculate_detection_metrics, save_metrics_as_txt, plot_metrics

DATE_TIME_LEN = 19

def process_files_in_folder(folder_path, output_folder, model, ignition_times, time_step, confidence_threshold):
    folder_name = os.path.basename(folder_path)
    ignition_time = get_ignition_time(folder_name, ignition_times)

    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.jpg')],
        key=lambda filename: datetime.strptime(filename[-(DATE_TIME_LEN+4):-4], '%Y_%m_%dT%H_%M_%S')  # Ordenar archivos por fecha y hora en el nombre.
    )

    if not file_list:
        return None
    first_filename = file_list[0]
    date_time_str = first_filename[-(DATE_TIME_LEN+4):-4]
    initial_time = datetime.strptime(date_time_str, '%Y_%m_%dT%H_%M_%S')
    current_time = initial_time

    detection_data = {
        "before_ignition_not_detected": 0,
        "before_ignition_detected": 0,
        "after_ignition_not_detected": 0,
        "after_ignition_detected": 0,
        "detection_delay": None,
        "total_frames": len(file_list)
    }

    detected_after_ignition = False

    detections_folder = os.path.join(output_folder, 'detections')
    results_folder = os.path.join(output_folder, 'results')
    os.makedirs(detections_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    for filename in tqdm(file_list, desc=f"Processing images in {folder_name}"):
        file_path = os.path.join(folder_path, filename)
        time_elapsed = current_time - initial_time
        state = "UNKNOWN"
        if ignition_time:
            current_time_time = current_time.time()
            if current_time_time < ignition_time:
                state = "BEFORE IGNITION"
            else:
                state = "AFTER IGNITION"

        results = model(file_path, verbose=False)

        if results and len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            detected_classes = [results[0].names[int(cls)] for cls in classes]
            detected = any(cls in ['fire', 'smoke'] and conf >= confidence_threshold for cls, conf in zip(detected_classes, confidences))

            img = cv2.imread(file_path)
            for box, cls, conf in zip(boxes, classes, confidences):
                class_name = results[0].names[int(cls)]
                if class_name in ['fire', 'smoke'] and conf >= confidence_threshold:
                    x_min, y_min, x_max, y_max = map(int, box)
                    color = (0, 255, 0) if class_name == 'fire' else (255, 0, 0)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    label = f"{class_name} {conf:.0%}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    label_y_min = max(y_min, label_size[1] + 10)
                    cv2.putText(img, label, (x_max - label_size[0], y_min + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_file_path = os.path.join(detections_folder, filename)
            cv2.imwrite(output_file_path, img)
        else:
            detected = False

        if state == "BEFORE IGNITION":
            if detected:
                detection_data["before_ignition_detected"] += 1
            else:
                detection_data["before_ignition_not_detected"] += 1
        elif state == "AFTER IGNITION":
            if detected:
                detection_data["after_ignition_detected"] += 1
                if not detected_after_ignition:
                    detection_data["detection_delay"] = time_elapsed.total_seconds()
                    detected_after_ignition = True
            else:
                detection_data["after_ignition_not_detected"] += 1

        current_time += timedelta(seconds=time_step)

    # Calcular métricas y guardarlas
    metrics = calculate_detection_metrics(detection_data)
    save_metrics_as_txt(metrics, results_folder)
    plot_metrics(metrics, results_folder)

    return detection_data
