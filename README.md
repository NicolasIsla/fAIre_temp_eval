# Fire Detection Evaluation

This project is a tool for evaluating fire detection in images using computer vision models. The script processes images in folders, applies a fire detection model, and saves the results and evaluation metrics in organized folders.

## Requirements

- Python 3.x
- Necessary libraries (can be installed using `requirements.txt`):
- matplotlib
- opencv-python
- Pillow
- ultralytics
- torch
- tqdm

## Configuration

The configuration file `config/example.json` defines the necessary parameters to run the evaluation.

### Example Configuration

```json
{
    "main_path": "path/to/main",
    "output_path": "path/to/output",
    "time_step": 5,
    "model_type": "your_model_type",
    "model_version": "your_model_version",
    "ignition_time_path": "path/to/ignition_time",
    "video_folder": "optional_video_folder",
    "confidence_threshold": 0.5
}
```

### Parameter Descriptions

- `main_path`: Path to the main folder containing the image folders to be processed.

- `output_path`: Path where the evaluation results will be saved.

- `time_step`: Time interval (in seconds) between the processed images.

- `model_type`: Type of detection model to use.

- `model_version`: Version of the detection model to use.

- `ignition_time_path`: Path to the file containing the ignition times.

- `video_folder`: (Optional) Name of a specific video folder to process. If not specified, all folders in main_path will be processed.

- `confidence_threshold`: Confidence threshold to consider a detection as valid.

## Execution
1. Ensure all necessary libraries are installed.
2. Modify the configuration file config/example.json with the appropriate parameters or introduce your configuration file.
3. Run the script eval.py with the name of your configuration as parameter:
```batch
python eval.py --config config/example.json
```