"""Predict."""

from ultralytics import YOLO

from src.coordinate_conversion import dbref_to_wgs84, pixel_to_coordinates

# Load a model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model("tmp/a_34358_55129_200_DB_REF_20140425.tif")


def calculate_center(a: float, b: float) -> float:
    """Calculate the center of two points."""
    return float((a + b) / 2)


for result in results:
    print(result.path)
    # result.show()
    for i, box in enumerate(result.boxes):
        print(f"result {i} with class {result.names[int(box.cls[0])]}.")
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        x_pixel = calculate_center(x1, x2)
        y_pixel = calculate_center(y1, y2)
        
        x_coordinate, y_coordinate = pixel_to_coordinates(path=result.path, x_pixel=x_pixel, y_pixel=y_pixel)
        
        latitute, longitute, _ = dbref_to_wgs84(x_coordinate=x_coordinate, y_coordinate=y_coordinate)
        print(latitute, longitute)
        
