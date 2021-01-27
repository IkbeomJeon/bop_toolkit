import os
import numpy as np
import matplotlib.pyplot as plt

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility
from bop_toolkit_lib import visualization

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'lm',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # Tolerance used in the visibility test [mm].
  'delta': 15,  # 5 for ITODD, 15 for the other datasets.

  # Type of the renderer.
  'renderer_type': 'python',  # Options: 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = None
if p['dataset'] == 'tless':
  model_type = 'cad'
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)


# Load colors.
colors_path = os.path.join(
  os.path.dirname(visualization.__file__), 'colors.json')

colors = inout.load_json(colors_path)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
scene_id = scene_ids[0]
gt_id = 0
#for scene_id in scene_ids:

# Load scene info and ground-truth poses.
scene_camera = inout.load_scene_camera(
  dp_split['scene_camera_tpath'].format(scene_id=scene_id))

scene_gt_info = inout.load_json(
    dp_split['scene_gt_info_tpath'].format(scene_id=scene_id), keys_to_int=True)

im_ids = sorted(scene_gt_info.keys())


plt.figure()
for im_counter, im_id in enumerate(im_ids):
  rgb = inout.load_im(dp_split['rgb_tpath'].format(
    scene_id=scene_id, im_id=im_id))[:, :, :3]

  mask_path = dp_split['mask_tpath'].format(
    scene_id=scene_id, im_id=im_id, gt_id=gt_id)
  mask = inout.load_im(mask_path)
  mask_rect = visualization.draw_rect(rgb, scene_gt_info[im_id][0]['bbox_obj'], colors[gt_id])

  plt.imshow(mask_rect)
  plt.show()


