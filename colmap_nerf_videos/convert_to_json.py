import json
import math
import numpy as np
import os
import shutil
import argparse

def w2c_to_c2w(qw, qx, qy, qz, tx, ty, tz):
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    t = np.array([tx, ty, tz])
    
    # Invert to get Camera-to-World (C2W)
    R_c2w = R.T
    t_c2w = -R_c2w @ t
    
    # Convert from OpenCV to OpenGL coordinate system (flip Y and Z axes)
    R_c2w[:, 1:3] *= -1
    t_c2w[1:3] *= -1
    
    # Construct the 4x4 transformation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    return c2w.tolist()


def read_camera_angle_x(cameras_txt_path):
    with open(cameras_txt_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            width = float(parts[2])
            focal_x = float(parts[4])
            return 2.0 * math.atan(width / (2.0 * focal_x))
    raise RuntimeError(f"No valid camera line found in {cameras_txt_path}")


def read_frames_from_images(images_txt_path, image_prefix):
    frames = []
    with open(images_txt_path, "r") as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]

        output_name = f"{image_prefix}_{image_name}" if image_prefix else image_name
        matrix = w2c_to_c2w(qw, qx, qy, qz, tx, ty, tz)
        frames.append(
            {
                "input_image_name": image_name,
                "output_image_name": output_name,
                "matrix": matrix,
            }
        )

    frames.sort(key=lambda item: item["output_image_name"])
    return frames


def split_name(index):
    if index % 10 == 0:
        return "val"
    if index % 10 == 5:
        return "test"
    return "train"


def clean_split_dirs(scene_dir):
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(scene_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for name in os.listdir(split_dir):
            path = os.path.join(split_dir, name)
            if os.path.isfile(path):
                os.remove(path)


def process_single_model(base_dir, output_root, model_id, frame_folder_name, scene_suffix):
    cameras_txt = os.path.join(base_dir, "sparse", model_id, "cameras.txt")
    images_txt = os.path.join(base_dir, "sparse", model_id, "images.txt")
    frames_dir = os.path.join(base_dir, frame_folder_name)

    if not (os.path.exists(cameras_txt) and os.path.exists(images_txt) and os.path.isdir(frames_dir)):
        print(f"Skipping model {model_id}: missing cameras/images/frame folder.")
        return

    scene_dir = os.path.join(output_root, f"{frame_folder_name}{scene_suffix}")
    os.makedirs(scene_dir, exist_ok=True)
    clean_split_dirs(scene_dir)

    camera_angle_x = read_camera_angle_x(cameras_txt)
    all_frames_data = read_frames_from_images(images_txt, image_prefix="")

    splits_data = {"train": [], "val": [], "test": []}

    for i, frame in enumerate(all_frames_data):
        input_img_name = frame["input_image_name"]
        matrix = frame["matrix"]
        split = split_name(i)

        src_path = os.path.join(frames_dir, input_img_name)
        dst_path = os.path.join(scene_dir, split, input_img_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {src_path} not found.")
            continue

        output_name_without_ext = os.path.splitext(input_img_name)[0]
        splits_data[split].append(
            {
                "file_path": f"./{split}/{output_name_without_ext}",
                "rotation": 0.012566370614359171,
                "transform_matrix": matrix,
            }
        )

    for split in ["train", "val", "test"]:
        output = {
            "camera_angle_x": camera_angle_x,
            "frames": splits_data[split],
        }
        json_filename = f"transforms_{split}.json"
        with open(os.path.join(scene_dir, json_filename), "w") as file:
            json.dump(output, file, indent=4)

        print(
            f"[{frame_folder_name}] Created {split}/ and {json_filename} with {len(splits_data[split])} frames."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP sparse outputs to NeRF transforms and split images into train/val/test."
    )
    parser.add_argument("--base_dir", default=".", help="Directory containing sparse/, frames_3306/, frames_3308/.")
    parser.add_argument(
        "--output_dir",
        default="./generated_nerf_scenes",
        help="Root directory where copied scene folders are created (e.g., output/frames_3306_copy, output/frames_3308_copy).",
    )
    parser.add_argument(
        "--scene_suffix",
        default="_copy",
        help="Suffix appended to each output scene folder name.",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir)

    model_frame_map = {
        "0": "frames_3306",
        "1": "frames_3308",
    }

    for model_id, frame_folder_name in model_frame_map.items():
        process_single_model(base_dir, output_dir, model_id, frame_folder_name, args.scene_suffix)

if __name__ == "__main__":
    main()