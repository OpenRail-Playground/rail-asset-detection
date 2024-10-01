"""Predict."""

import json
from ultralytics import YOLO

from src.coordinate_conversion import dbref_to_wgs84, pixel_to_coordinates

MODEL = "yolo11n.pt"
FOLDER_OR_IMAGE_PATH = "tmp/a_34358_55129_200_DB_REF_20140425.tif"

# Load a model
model = YOLO(MODEL)
# Perform object detection on an image
detection_results = model(FOLDER_OR_IMAGE_PATH)


def calculate_center(a: float, b: float) -> float:
    """Calculate the center of two points."""
    return float((a + b) / 2)


result_dict = {}

for result in detection_results:
    # print(result.path)
    # result.show()
    box_results = {}
    for i, box in enumerate(result.boxes):
        box_class = int(box.cls[0])
        box_class_name = result.names[box_class]
        # print(f"result {i} with class {box_class_name}.")
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        x_pixel = calculate_center(x1, x2)
        y_pixel = calculate_center(y1, y2)

        x_coordinate, y_coordinate = pixel_to_coordinates(
            path=result.path, x_pixel=x_pixel, y_pixel=y_pixel
        )
        # print(f"DB_REF coordinates: {x_coordinate}, {y_coordinate}")
        latitute, longitute, _ = dbref_to_wgs84(
            x_coordinate=x_coordinate, y_coordinate=y_coordinate
        )
        # print(f"latitute and longitute: {latitute}, {longitute}")

        box_results[i] = {
            "box_class": box_class,
            "box_class_name": box_class_name,
            "coordinates": {
                "DB_REF": [x_coordinate, y_coordinate],
                "WGS84": [latitute, longitute],
            },
        }

    result_dict[str(result.path)] = box_results
    
print(json.dumps(result_dict, sort_keys=True, indent=3))
