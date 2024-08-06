import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime

class ImageMarkerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Marker")
        
        self.image_index = 0
        self.image_paths = []
        self.folder_path = ""
        self.json_path = 'data/fire_truth.json'
        self.date_time_format = '%Y_%m_%dT%H_%M_%S'
        self.date_time_len = 19

        self.label = tk.Label(root)
        self.label.pack()

        self.mark_button = tk.Button(root, text="Mark Image", command=self.mark_image)
        self.mark_button.pack()

        self.next_button = tk.Button(root, text="Next Image", command=self.show_next_image)
        self.next_button.pack()

        self.prev_button = tk.Button(root, text="Previous Image", command=self.show_prev_image)
        self.prev_button.pack()

        self.open_folder_button = tk.Button(root, text="Open Folder", command=self.open_folder)
        self.open_folder_button.pack()

        self.progress_label = tk.Label(root, text="")
        self.progress_label.pack()

    def open_folder(self):
        base_folder = 'data'
        folder_path = filedialog.askdirectory(initialdir=base_folder)
        print(f"Selected folder: {folder_path}")
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ]
        print(f"Unsorted image paths: {self.image_paths}")
        try:
            self.image_paths = sorted(
                self.image_paths,
                key=lambda filename: datetime.strptime(os.path.basename(filename)[-(self.date_time_len+4):-4], self.date_time_format)
            )
        except Exception as e:
            print(f"Error sorting images: {e}")
            for filename in self.image_paths:
                try:
                    date_str = os.path.basename(filename)[-(self.date_time_len+4):-4]
                    print(f"Parsing date from {filename}: {date_str}")
                    datetime.strptime(date_str, self.date_time_format)
                except Exception as parse_e:
                    print(f"Failed to parse date from {filename}: {parse_e}")

        print(f"Sorted image paths: {self.image_paths}")
        if self.image_paths:
            self.image_index = 0
            self.show_image()

    def show_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.image_index]
            image = Image.open(image_path)
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            self.label.config(image=photo)
            self.label.image = photo
            self.update_progress()
            self.root.title(f"Viewing: {os.path.basename(image_path)}")

    def mark_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.image_index]
            filename = os.path.basename(image_path)
            ignition_time_str = filename[-(self.date_time_len+4):-4]
            folder_name = os.path.basename(self.folder_path)
            
            try:
                ignition_time = datetime.strptime(ignition_time_str, self.date_time_format)
                ignition_time_formatted = ignition_time.strftime('%H:%M')
            except Exception as e:
                print(f"Error parsing time from {filename}: {e}")
                ignition_time_formatted = "Invalid"

            new_entry = {
                "folder": folder_name,
                "ignition_time": ignition_time_formatted
            }

            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as file:
                    data = json.load(file)
            else:
                data = []

            data.append(new_entry)

            with open(self.json_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Marked: {new_entry}")

    def show_next_image(self):
        if self.image_paths:
            self.image_index = (self.image_index + 1) % len(self.image_paths)
            self.show_image()

    def show_prev_image(self):
        if self.image_paths:
            self.image_index = (self.image_index - 1) % len(self.image_paths)
            self.show_image()

    def update_progress(self):
        total_images = len(self.image_paths)
        current_image = self.image_index + 1
        self.progress_label.config(text=f"Image {current_image} of {total_images}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMarkerApp(root)
    root.mainloop()
