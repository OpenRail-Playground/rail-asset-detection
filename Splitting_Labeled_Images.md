# Splitting Labeled Images

Labeling the original large images and then splitting them afterwards has the benefit, that the splitting can be prformed multiple times to try out different resolutions. It can also give the labeler more context when labeling.

The steps are:
- export in CVAT as *COCO 1.0*
- split images with the [sahi library](https://github.com/obss/sahi)
    
    ``` bash
    sahi coco slice  --image_dir path/to/images/ --dataset_json_path path/toannotations.json --slice_size 640 --output_dir tiled_coco_dataset/
    ```
    This will split the images and also transform the labels for the new images.
- convert to a YOLO dataset with [convert_coco](https://docs.ultralytics.com/reference/data/converter/?h=convert_coc#ultralytics.data.converter.convert_coco)

    ``` python
        from ultralytics.data.converter import convert_coco
        convert_coco("/path/to/annotations.json/parent/dir/")
    ```
    This will create a .txt file for each image in your dataset.  Create a new directory for and move the images and labels into it. eg:
    - `yolo_dataset/images`: for all the images.
    - `yolo_dataset/labels`: for all the label .txt files.
- split into train val dataset with [autosplit](https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.autosplit)
    ``` python
    from ultralytics.data.utils import autosplit
    autosplit( path="./yolo_dataset", weights=(0.85, 0.15, 0.0), annotated_only=False )
    ```
    This will create a *autosplit_train.txt*  and *autosplit_val.txt* in your dataset folder defining your train/val split.

The final dataset then looks like this:
- Folderstructure

    ```
        yolo_dataset/
        |-images/
        |... all images
        |-labels/
        |... all labels .txt, same name as the image
        - autosplit_train.txt
        - autosplit_val.txt
        - data.yaml
    ```
- data.yaml

    ```yaml
    names:
      0: blue box
      1: light pole
    path: /path/to/yolo_dataset
    train: autosplit_train.txt
    val: autosplit_val.txt
    ```
- autosplit.txt

    ```txt
    ./images/imagename.jpg
    ...
    ```
    The path should be relative to `path: /path/to/yolo_dataset` and start with `./`

## Inferencing with a sliding window

When inferencing is done on the large image directly the image is scaled down. This can cause bad results if the model was trained on splitted image, because the inferencing resoluton is now a lot lower than the training resolution.

The [sahi library](https://github.com/obss/sahi) can automatically perform sliding window inferencing on larger images:
- [documentation](https://docs.ultralytics.com/guides/sahi-tiled-inference/)
- example src/predict_tiled.py