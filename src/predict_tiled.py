from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

tiled_model="runs/detect/tiled_images_220_epoch/weights/best.pt"
image_path = "/scratch/cvat_share/per_station/Allensbach/0382ae63-ac1e-45fb-8f00-6491e66c738c/35052_52862_300_DB_REF2016_20210328.tif"

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=tiled_model,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

result.export_visuals(export_dir="demo_data/")