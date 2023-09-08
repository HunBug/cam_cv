from pathlib import Path
import cv2
from app.third_party.image_stitcher import stitch_inpainting
from app.tools.ai_person_segmenter import AiPersonSegmenter
from app.tools.video_reader import VideoReader


if __name__ == "__main__":
    model_path = Path(__file__).parent / "models" / "deeplabv3.tflite"
    labels_path = Path(__file__).parent / "models" / "deeplabv3_labels.yaml"
    image_path = Path(__file__).parent.parent / "data" / "selfie.png"
    video_path = Path(__file__).parent.parent / "data" / "yt" / "04.mp4"

    backgournd_images = []
    # with AiPersonSegmenter(model_path, labels_path) as segmenter:
    #     image = cv2.imread(str(image_path))
    #     output_image = segmenter.segment_background(image)
    #     backgournd_images.append(output_image)

    with VideoReader(video_path) as reader, AiPersonSegmenter(model_path, labels_path) as segmenter:
        while True:
            success, image = reader.reader.read()
            if not success:
                break
            output_image = segmenter.segment_background(image)
            backgournd_images.append(output_image)

    backgournd_images = backgournd_images[::5]
    print(f"Found {len(backgournd_images)} background images")
    inpainted_images = stitch_inpainting(backgournd_images)
    for i, image in enumerate(inpainted_images):
        cv2.imwrite(f"{video_path.name}_result_{i}.png", image)
    cv2.imwrite(f"{video_path.name}_background.png", backgournd_images[0])
        