# labelme2yolo-pose

A tool to convert Labelme annotations to YOLO format for pose estimation tasks.  
Supports bounding boxes and keypoint annotations with customizable configurations.

## Features
- Convert Labelme JSON annotations to YOLO pose estimation format.
- Associate keypoints with corresponding bounding boxes.
- Handle 2D or 3D keypoint visibility flags.
- Filter unsupported classes and auto-normalize coordinates.

## Requirements
- Python 3.6+
- Required libraries: `json`, `os`, `collections` (all included in Python standard library).

## Usage

### 1. Configure Parameters
Modify the following variables in `labelme2yolo-pose.py` according to your dataset:
```python
CLASS_MAPPING = {"tomato": 0}  # Map object classes to YOLO indices
KEYPOINT_LABELS = ["head", "tail"]  # Ordered list of keypoint labels
DIM = 2  # 2 for (x, y), 3 for (x, y, visibility)
```

### 2. Folder Structure
- **Input**: Folder containing Labelme JSON files.
- **Output**: Empty folder for YOLO-formatted TXT files.

Example:
```
Input (JSON files):
D:\dataT\train\labels\image1.json
...

Output (TXT files):
D:\datat\labels\train\image1.txt
...
```

### 3. Run the Script
Update the paths in the `__main__` block and execute:
```python
if __name__ == "__main__":
    convert_labelme_to_yolo(
        json_folder="PATH/TO/JSON_FOLDER",
        output_folder="PATH/TO/OUTPUT_FOLDER"
    )
```

## YOLO Output Format
Each line in the TXT file represents an object:  
`<class_id> <x_center> <y_center> <width> <height> [kp1_x kp1_y ... kpN_x kpN_y]`

Example (2D keypoints):
```
0 0.453125 0.621093 0.125000 0.234375 0.45 0.62 0.48 0.60
```

## Notes
- Keypoints outside bounding boxes are ignored (filled with `0` coordinates).
- Only rectangles and points are processed; other shapes (polygons, circles) are skipped.
- Ensure all JSON files follow Labelme's format (with `imageWidth` and `imageHeight` fields).

    
