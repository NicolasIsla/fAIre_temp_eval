import os
import json
from datetime import datetime

def load_ignition_times(ignition_time_path):
    with open(ignition_time_path, 'r') as f:
        return json.load(f)

def get_ignition_time(folder_name, ignition_times):
    for entry in ignition_times:
        if entry['folder'] == folder_name:
            return datetime.strptime(entry['ignition_time'], '%H:%M').time()
    return None
