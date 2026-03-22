This directory contains a dataset of synthetically rendered images that were used in "NeRF: Representing Scenes as
Neural Radiance Fields for View Synthesis".

Converting COLMAP Output to NeRF Format
---------------------------------------
Use `colmap_nerf_videos/convert_to_json.py` to convert COLMAP sparse reconstruction outputs into the NeRF dataset format.

Usage:
  python colmap_nerf_videos/convert_to_json.py --base_dir <colmap_project_dir> --output_dir <output_path>

Arguments:
  --base_dir     Directory containing sparse/ and frame folders (default: .)
  --output_dir   Where to create the NeRF scene folders (default: ./generated_nerf_scenes)
  --scene_suffix Suffix for output folder names (default: _copy)

The script will:
  - Read camera intrinsics from sparse/*/cameras.txt
  - Read poses from sparse/*/images.txt
  - Split images into train/val/test (every 10th frame to val, every 10+5th to test)
  - Generate transforms_{train,val,test}.json files

Stats:
+ 8 Scenes
+ 100 Training images
+ 100 Validation images
+ 200 Test images
+ Images are 800x800

Structure:
  SCENE_NAME
    -train
      r_*.png
    -val
      r_*.png
    -test
      r_*.png
      r_*_depth_0000.png
      r_*_normal_0000.png
    transforms_train.json
    transforms_val.json
    transforms_test.json

Transform json details:
camera_angle_x: The FOV in x dimension
frames: List of dictionaries that contain the camera transform matrices for each image.

Attribution:
The renders are from modified blender models located on blendswap.com
chair by 1DInc (CC-0): https://www.blendswap.com/blend/8261
drums by bryanajones (CC-BY): https://www.blendswap.com/blend/13383
ficus by Herberhold (CC-0): https://www.blendswap.com/blend/23125
hotdog by erickfree (CC-0): https://www.blendswap.com/blend/23962
lego by Heinzelnisse (CC-BY-NC): https://www.blendswap.com/blend/11490
materials by elbrujodelatribu (CC-0): https://www.blendswap.com/blend/10120
mic by up3d.de (CC-0): https://www.blendswap.com/blend/23295
ship by gregzaal (CC-BY-SA): https://www.blendswap.com/blend/8167
