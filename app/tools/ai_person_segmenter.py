from pathlib import Path
import yaml
import numpy as np
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenterOptions


class AiPersonSegmenter:
    def __init__(self, model_path: Path, labels_path: Path, confidence_threshold: float = 0.5):
        self.MODEL_INPUT_HEIGHT = 257
        self.MODEL_INPUT_WIDTH = 257
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.MODEL_PATH = model_path
        self.LABELS_PATH = labels_path
        self.labels = yaml.safe_load(self.LABELS_PATH .read_text())
        self.person_label = int(self.labels.get("labels").get("person"))

        self._base_options = BaseOptions(model_asset_path=str(model_path))
        self._options = ImageSegmenterOptions(base_options=self._base_options,
                                            output_category_mask=True)



        self.segmenter = None

    def __enter__(self):
        self.segmenter = vision.ImageSegmenter.create_from_options(self._options)
        return self

    def __exit__(self, type, value, traceback):
        self.segmenter.close()

    def segment_background(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_mp = Image(image_format=ImageFormat.SRGB, data=image_rgb)

        # Retrieve the masks for the segmented image
        segmentation_result = self.segmenter.segment(image_mp)
        confidence_mask = segmentation_result.confidence_masks[self.person_label]

        condition = np.stack((confidence_mask.numpy_view(),) * 3, axis=-1) > self.CONFIDENCE_THRESHOLD
        output_image = np.where(condition, 0, image)

        return output_image