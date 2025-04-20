import json
import os
from collections import defaultdict

# Configuration parameters (modify according to actual needs)
CLASS_MAPPING = {"object": 0}  # Object category to index mapping
KEYPOINT_LABELS = ["p1", "p2"]  # Keypoint labels in order
DIM = 2  # Dimension format: 2 or 3

def preprocess_shapes(data):
    """
    Preprocess annotation shapes to separate rectangles and keypoints
    
    Parameters:
    data: Labelme JSON data
    
    Returns:
    rectangles: List of rectangle annotations
    keypoints: List of keypoint annotations
    img_w: Image width
    img_h: Image height
    """
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    
    rectangles = []
    keypoints = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            rectangles.append(shape)
        elif shape["shape_type"] == "point":
            keypoints.append(shape)
    
    return rectangles, keypoints, img_w, img_h

def process_rectangle(rect, img_w, img_h, keypoints, used_kp_indices):
    """
    Process a single rectangle and its associated keypoints
    
    Parameters:
    rect: Rectangle annotation
    img_w: Image width
    img_h: Image height
    keypoints: List of keypoint annotations
    used_kp_indices: Set of used keypoint indices
    
    Returns:
    line_data: YOLO-formatted line data list
    updated_used_kp_indices: Updated set of used keypoint indices
    """
    # Validate category
    class_label = rect["label"]
    if class_label not in CLASS_MAPPING:
        return None, used_kp_indices
    
    # Parse rectangle boundaries
    points = rect["points"]
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)
    
    # Calculate YOLO-formatted bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalization
    x_center_n = round(x_center / img_w, 6)
    y_center_n = round(y_center / img_h, 6)
    width_n = round(width / img_w, 6)
    height_n = round(height / img_h, 6)
    
    # Collect keypoints
    kp_values = []
    found_kp = False
    for label in KEYPOINT_LABELS:
        found = False
        # Iterate through all keypoints to find matches
        for idx, kp in enumerate(keypoints):
            if idx in used_kp_indices:
                continue
            if kp["label"] != label:
                continue
            
            # Check if keypoint is inside the rectangle
            kp_x, kp_y = kp["points"][0]
            if (x_min <= kp_x <= x_max) and (y_min <= kp_y <= y_max):
                # Normalized coordinates
                kp_x_n = round(kp_x / img_w, 6)
                kp_y_n = round(kp_y / img_h, 6)
                
                # Handle visibility
                if DIM == 2:
                    kp_values.extend([kp_x_n, kp_y_n])
                elif DIM == 3:
                    kp_values.extend([kp_x_n, kp_y_n, 2])  # 2 indicates visible
                
                used_kp_indices.add(idx)
                found = True
                found_kp = True
                break
        
        # Handle missing keypoints
        if not found:
            if DIM == 2:
                kp_values.extend([0.0, 0.0])
            elif DIM == 3:
                kp_values.extend([0.0, 0.0, 0])  # 0 indicates invisible
    
    # Build YOLO line
    line_data = [
        str(CLASS_MAPPING[class_label]),
        str(x_center_n),
        str(y_center_n),
        str(width_n),
        str(height_n)
    ]
    
    # Add keypoint data only if found
    if found_kp:
        line_data.extend(map(str, kp_values))
    
    return line_data, used_kp_indices

def convert_labelme_to_yolo(json_folder, output_folder):
    """
    Convert Labelme annotations to YOLO pose estimation format
    
    Parameters:
    json_folder: Input folder path containing Labelme JSON files
    output_folder: Output path for saving TXT files
    """
    os.makedirs(output_folder, exist_ok=True)

    for json_name in os.listdir(json_folder):
        if not json_name.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, json_name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Preprocess annotation shapes
        rectangles, keypoints, img_w, img_h = preprocess_shapes(data)
        used_kp_indices = set()
        yolo_lines = []

        # Process all rectangles
        for rect in rectangles:
            line_data, used_kp_indices = process_rectangle(
                rect, img_w, img_h, keypoints, used_kp_indices
            )
            if line_data is not None:
                yolo_lines.append(" ".join(line_data))

        # Write to file
        if yolo_lines:
            base_name = os.path.splitext(json_name)[0]
            txt_path = os.path.join(output_folder, base_name + ".txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    convert_labelme_to_yolo(
        json_folder="/path/to/labelme_json/labels/train/",
        output_folder="/path/to/YoloDataset/labels/train/"
    )
