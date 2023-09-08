class AiPersonSegmenter:
    def __init__(self, model_path):
        self.MODEL_INPUT_HEIGHT = 257
        self.MODEL_INPUT_WIDTH = 257
        self.CONFIDENCE_THRESHOLD = 0.5
        self.MODEL_PATH = Path("models/deeplabv3.tflite")
        self.LABELS_PATH = Path("models/deeplabv3_labels.yaml")

        self.base_options = python.BaseOptions(model_asset_path=str(DATA_ROOT / MODEL_PATH))
        self.options = vision.ImageSegmenterOptions(base_options=base_options,
                                            output_category_mask=True)

        self.labels = yaml.safe_load((DATA_ROOT / LABELS_PATH).read_text())
        self.person_label = int(labels.get("labels").get("person"))