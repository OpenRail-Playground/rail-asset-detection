"""Convert coco annotations to yolo train format."""

from ultralytics.data.converter import convert_coco

COCO_ANNOTATIONS_DIR = "tmp/train/annotations/"
    
convert_coco(labels_dir=COCO_ANNOTATIONS_DIR, use_segments=True)