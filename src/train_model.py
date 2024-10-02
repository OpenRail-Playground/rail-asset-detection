"""Train model script.
Copied from https://github.com/ultralytics/ultralytics
"""
from ultralytics import YOLO

full_image_dataset="/scratch/full-image-90-dataset/full_coco/data.yaml"
tiled_dataset="/scratch/full-image-90-dataset/tiled_yolo/data.yaml"

# Load a model
model = YOLO("runs/detect/tiled_images_220_epoch/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=tiled_dataset, epochs=100, imgsz=640, device=[1])

# Perform object detection on an image
#results = model("tmp/a_34358_55129_200_DB_REF_20140425.tif")
#results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
