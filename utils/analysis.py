import matplotlib.pyplot as plt
import json
import os

def calculate_detection_metrics(detection_data):
    before_ignition_total = detection_data["before_ignition_not_detected"] + detection_data["before_ignition_detected"]
    after_ignition_total = detection_data["after_ignition_not_detected"] + detection_data["after_ignition_detected"]

    metrics = {
        "before_ignition_not_detected_percentage": (detection_data["before_ignition_not_detected"] / before_ignition_total) * 100 if before_ignition_total else 0,
        "before_ignition_detected_percentage": (detection_data["before_ignition_detected"] / before_ignition_total) * 100 if before_ignition_total else 0,
        "after_ignition_not_detected_percentage": (detection_data["after_ignition_not_detected"] / after_ignition_total) * 100 if after_ignition_total else 0,
        "after_ignition_detected_percentage": (detection_data["after_ignition_detected"] / after_ignition_total) * 100 if after_ignition_total else 0,
        "detection_delay_seconds": detection_data["detection_delay"] if detection_data["detection_delay"] is not None else 0
    }

    return metrics

def save_metrics_as_txt(metrics, output_folder, filename='results.txt'):
    with open(os.path.join(output_folder, filename), 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_metrics(metrics, output_folder, filename='results.png'):
    labels = list(metrics.keys())
    values = [0 if v is None else v for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color='skyblue')
    ax.set_xlabel('Percentage / Seconds')
    ax.set_title('Detection Metrics')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()
