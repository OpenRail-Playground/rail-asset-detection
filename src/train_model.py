"""Train model script.
Copied from https://github.com/ultralytics/ultralytics
"""

from ultralytics import YOLO

BLUE_BOX_TRAIN_YAML = "/Users/danielringler/git/3lh/datasets/carl_manual_images_test_v1/data.yaml" # "coco8.yaml"  # "tmp/train/carl_wo_images_test_v1/data.yaml"

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data=BLUE_BOX_TRAIN_YAML,  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size
    device="mps",  # "cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("tmp/a_34358_55129_200_DB_REF_20140425.tif")
results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
