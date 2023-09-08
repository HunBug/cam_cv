from pathlib import Path
import cv2

class VideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.reader = None

    def __enter__(self):
        self.reader = cv2.VideoCapture(str(self.video_path))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.reader.release()