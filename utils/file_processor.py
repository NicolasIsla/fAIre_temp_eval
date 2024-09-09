import os
import cv2
from datetime import datetime, timedelta, time
from .ignition_time import get_ignition_time
from tqdm import tqdm
from .analysis import calculate_detection_metrics, save_metrics_as_txt, plot_metrics

DATE_TIME_LEN = 19

def process_files_in_folder(folder_path, output_folder, yolo_model, lstm_resnet_model, ignition_times, time_step, confidence_threshold, frames_back):
    folder_name = os.path.basename(folder_path)
    ignition_time = get_ignition_time(folder_name, ignition_times)

    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.jpg')],
        key=lambda filename: datetime.strptime(filename[-(DATE_TIME_LEN+4):-4], '%Y_%m_%dT%H_%M_%S')
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
        "total_frames": len(file_list),
        "bounding_boxes": []
    }
    video_state_detection = [folder_name]

    detected_after_ignition = False
    buffer_bounding_boxes = []

    detections_folder = os.path.join(output_folder, 'detections')
    results_folder = os.path.join(output_folder, 'results')
    zoom_folder = os.path.join(output_folder, 'data/buffer')
    os.makedirs(detections_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(zoom_folder, exist_ok=True)

    def clear_buffer_folder(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for frame_idx, filename in enumerate(tqdm(file_list, desc=f"Processing images in {folder_name}")):
        file_path = os.path.join(folder_path, filename)
        
        time_elapsed = None
        if ignition_time:
            if isinstance(ignition_time, datetime):
                time_elapsed = current_time - ignition_time
            elif isinstance(ignition_time, time):
                ignition_time = datetime.combine(current_time.date(), ignition_time)
                time_elapsed = current_time - ignition_time
            else:
                print(f"Warning: Unsupported ignition_time type: {type(ignition_time)}")

        state = "UNKNOWN"
        if ignition_time and isinstance(ignition_time, datetime):
            if current_time < ignition_time:
                state = "BEFORE IGNITION"
            else:
                state = "AFTER IGNITION"

        results = yolo_model(file_path, verbose=False)
        detected = False
        if results and len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            detected_classes = [results[0].names[int(cls)] for cls in classes]
            detected = any(cls in ['fire', 'smoke'] and conf >= confidence_threshold for cls, conf in zip(detected_classes, confidences))

        print(f"Frame index: {frame_idx}, Image name: {filename}, Detected: {detected}, State: {state}")

        cropped_image_paths = []
        fire_detected = False

        if detected:
            print(f"Incendio detectado en frame {filename}. Guardando los últimos {frames_back} frames.")

            start_idx = max(0, frame_idx - frames_back)
            buffer_bounding_boxes = [(file_list[i], os.path.join(folder_path, file_list[i])) for i in range(start_idx, frame_idx + 1)]

            def enlarge_bounding_box(x_min, y_min, x_max, y_max, img_width, img_height, x_pixels):
                x_min_enlarged = max(0, x_min - x_pixels)
                y_min_enlarged = max(0, y_min - x_pixels)
                x_max_enlarged = min(img_width, x_max + x_pixels)
                y_max_enlarged = min(img_height, y_max + x_pixels)

                return x_min_enlarged, y_min_enlarged, x_max_enlarged, y_max_enlarged

            for (buffer_filename, buffer_filepath) in buffer_bounding_boxes:
                print(f"Guardando recorte del frame: {buffer_filename}")
                img = cv2.imread(buffer_filepath)
                
                img_with_boxes = img.copy()

                img_height, img_width = img.shape[:2]
                
                x_pixels = 50

                if len(confidences) > 0:
                    max_conf_index = confidences.argmax()
                    x_min, y_min, x_max, y_max = map(int, boxes[max_conf_index])
                    conf = confidences[max_conf_index]
                    class_name = results[0].names[int(classes[max_conf_index])]
                    
                    if class_name in ['fire', 'smoke'] and conf >= confidence_threshold:
                        x_min_enlarged, y_min_enlarged, x_max_enlarged, y_max_enlarged = enlarge_bounding_box(
                            x_min, y_min, x_max, y_max, img_width, img_height, x_pixels
                        )
                        # fire_detected = True # solo yolo
                        
                        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        
                        cv2.putText(img_with_boxes, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        cropped_img = img[y_min_enlarged:y_max_enlarged, x_min_enlarged:x_max_enlarged]
                        zoom_image_path = os.path.join(zoom_folder, f'{buffer_filename[:-4]}_zoom_{class_name}.jpg')
                        cv2.imwrite(zoom_image_path, cropped_img)
                        print(buffer_filename[:-4])

                        cropped_image_paths.append(zoom_image_path)
                        
                if buffer_filename == filename:
                    detection_image_path = os.path.join(detections_folder, f'{buffer_filename}')
                    cv2.imwrite(detection_image_path, img_with_boxes)

            print(f"Buffer (last {frames_back} frames): {[f[0] for f in buffer_bounding_boxes]}")

            if len(cropped_image_paths) == 4:
                lstm_prediccion = lstm_resnet_model.infer_4_frames(cropped_image_paths)
                print(f"Predicción del LSTM para los últimos 4 frames: {lstm_prediccion}")
                fire_detected = lstm_prediccion > 0.8
        
        print(f"¿Se detectó incendio? {fire_detected}")

        if state == "BEFORE IGNITION":
            if fire_detected:
                detection_data["before_ignition_detected"] += 1
                video_state_detection.append(-1)
            else:
                detection_data["before_ignition_not_detected"] += 1
                video_state_detection.append(1)
        elif state == "AFTER IGNITION":
            # añadir un 0 en la primera vez que se cambia el estado a "AFTER IGNITION"
            # si no hay 0 en la lista de video_state_detection, se añade un 0
            if 0 not in video_state_detection:
                video_state_detection.append(0)
            if fire_detected:
                detection_data["after_ignition_detected"] += 1
                video_state_detection.append(1)
                if not detected_after_ignition and time_elapsed is not None:
                    detection_data["detection_delay"] = time_elapsed.total_seconds() // 60
                    print(f"Detection delay: {detection_data['detection_delay']} seconds")
                    detected_after_ignition = True
            else:
                detection_data["after_ignition_not_detected"] += 1
                video_state_detection.append(-1)
            buffer_bounding_boxes.clear()
            clear_buffer_folder(zoom_folder)

        current_time += timedelta(seconds=time_step)
        print(f"Tiempo actual: {current_time}, Tiempo de ignición: {ignition_time}")
        print()
    metrics = calculate_detection_metrics(detection_data)
    save_metrics_as_txt(metrics, results_folder)
    plot_metrics(metrics, results_folder)

    return detection_data, video_state_detection