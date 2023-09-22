from pathlib import Path
import cv2
import numpy as np
from app.third_party.image_stitcher import stitch_inpainting
from app.tools.ai_person_segmenter import AiPersonSegmenter
from app.tools.video_reader import VideoReader
from app.tools.segmenters import (
    segment_by_homography,
    segment_by_optical_flow,
    average_segmentation_over_time,
    segmentation_refinement,
    AveragingMethod
)
from enum import Enum

class SegmentationMethod(Enum):
    OPTICAL_FLOW = 1
    HOMOGRAPHY = 2
    DEEPLABV3 = 3


if __name__ == "__main__":
    segmentation_method = SegmentationMethod.DEEPLABV3

    model_path = Path(__file__).parent / "models" / "deeplabv3.tflite"
    labels_path = Path(__file__).parent / "models" / "deeplabv3_labels.yaml"
    image_path = Path(__file__).parent.parent / "data" / "selfie.png"
    video_path = Path(__file__).parent.parent / "data" / "yt" / "01.mp4"

    backgournd_images = []
    with VideoReader(video_path) as reader:
        if segmentation_method == SegmentationMethod.DEEPLABV3:
            with AiPersonSegmenter(model_path, labels_path) as segmenter:
                while True:
                    success, image = reader.reader.read()
                    if not success:
                        break
                    output_image = segmenter.segment_background(image)
                    backgournd_images.append(output_image)
        else:
            segmenter = None
            if segmentation_method == SegmentationMethod.OPTICAL_FLOW:
                segmenter = segment_by_optical_flow
            elif segmentation_method == SegmentationMethod.HOMOGRAPHY:
                segmenter = segment_by_homography
            else:
                raise ValueError("Unknown segmentation method")
            raw_backgrounds = []
            raw_frames = []
            while True:
                success, image = reader.reader.read()
                if not success:
                    break
                raw_frames.append(image)
                if len(raw_frames) > 1:
                    _, background = segmenter(raw_frames[-2], raw_frames[-1])
                    # check the background area is not too big
                    if np.count_nonzero(background) > 0.7 * background.shape[0] * background.shape[1]* background.shape[2]:
                        continue
                    raw_backgrounds.append(background)
            for i, background in enumerate(raw_backgrounds):
                raw_background_window_start = max(0, i - 5)
                raw_background_window_end = min(len(raw_backgrounds), i + 5)
                raw_background_window = raw_backgrounds[raw_background_window_start:raw_background_window_end]
                average_mask = average_segmentation_over_time(raw_background_window, AveragingMethod.WEIGHTED_MEAN)
                refined_mask = segmentation_refinement(average_mask)
                refined_mask = refined_mask.astype(np.uint8)
                masked_image = cv2.bitwise_and(raw_frames[i], raw_frames[i], mask=refined_mask)
                backgournd_images.append(masked_image)  

    backgournd_images = backgournd_images[::5]
    print(f"Found {len(backgournd_images)} background images")
    inpainted_images = stitch_inpainting(backgournd_images)
    for i, image in enumerate(inpainted_images):
        cv2.imwrite(f"{video_path.name}_result_{i}.png", image)
    cv2.imwrite(f"{video_path.name}_background.png", backgournd_images[0])
