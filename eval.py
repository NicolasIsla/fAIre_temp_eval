import os
import argparse
import json
from datetime import datetime
from models.model_loader import load_model
from utils.ignition_time import load_ignition_times
from utils.file_processor import process_files_in_folder

CONFIG_PATH = 'config'

def main():
    parser = argparse.ArgumentParser(description="Evaluación de detección de incendios")
    parser.add_argument('--config', type=str, default= CONFIG_PATH + '/example.json', help='Ruta del archivo de configuración')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    main_path = config['main_path']
    output_path = config['output_path']
    time_step = config['time_step']
    model_type = config['model_type']
    model_version = config['model_version']
    ignition_time_path = config['ignition_time_path']
    video_folder = config.get('video_folder')
    confidence_threshold = config['confidence_threshold']

    model = load_model(model_type, model_version)
    ignition_times = load_ignition_times(ignition_time_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    evaluation_id = f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    evaluation_folder = os.path.join(output_path, evaluation_id)
    os.makedirs(evaluation_folder)

    all_detection_data = []

    if video_folder:
        folder_path = os.path.join(main_path, video_folder)
        if os.path.exists(folder_path):
            print(f'Processing folder: {folder_path}')
            output_folder = os.path.join(evaluation_folder, video_folder)
            os.makedirs(output_folder)
            detection_data = process_files_in_folder(folder_path, output_folder, model, ignition_times, time_step, confidence_threshold)
            all_detection_data.append(detection_data)
        else:
            print(f'Video folder {video_folder} does not exist.')
    else:
        for root, dirs, files in os.walk(main_path):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                print(f'Processing folder: {folder_path}')
                output_folder = os.path.join(evaluation_folder, dir_name)
                os.makedirs(output_folder)
                detection_data = process_files_in_folder(folder_path, output_folder, model, ignition_times, time_step, confidence_threshold)
                all_detection_data.append(detection_data)

    print("Processing completed.")

if __name__ == "__main__":
    main()
