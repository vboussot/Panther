#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
The following is the inference script for the baseline algorithm for Task 1 of the PANTHER challenge.

It is meant to run within a container.

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the GC platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

from pathlib import Path
import time
from mrsegmentator import inference
import glob
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import shutil
from scipy import ndimage
from data_utils import *
from scipy.ndimage import label
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import warnings
warnings.filterwarnings("ignore")


class PancreaticTumorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/nnUNet_results")
        self.mrsegmentator_input_dir = Path("/opt/algorithm/mrsegmentator/input")
        self.mrsegmentator_output_dir = Path("/opt/algorithm/mrsegmentator/output")

        # input / output paths for predictions-model
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print("Folder containing eval image", mr_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_mri))
            mr_ip_dir_name = 'abdominal-t1-mri' #default value
        
        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}") #abdominal-t2-mri
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "pancreatic-tumor-segmentation"))
        self.segmentation_mask = self.output_dir_seg_mask / "tumor_seg.mha"
        self.mrsegmentator_weights = "/opt/algorithm/Model/"
        os.environ["MRSEG_WEIGHTS_PATH"] = self.mrsegmentator_weights

        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_input_dir.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_output_dir.mkdir(exist_ok=True, parents=True)

        mha_files = glob.glob(os.path.join(self.mr_ip_dir, '*.mha'))
        # Check if any .mha files were found
        if mha_files:
            self.mr_image = mha_files[0]
        else:
            print('No mha images found in input directory')

    def run(self):
        """
        Load T1 MRI and generate segmentation of the tumor 
        """
        _show_torch_cuda_info()
        start_time = time.perf_counter()

        #1. copy the input image to nnUNet input directory adding _0000 suffix
        shutil.copyfile(self.mr_image, self.nnunet_input_dir / "mri_0000.mha")
        #   print shape and spacing
        itk_image = sitk.ReadImage(self.mr_image)
        print(f"Original image shape: {itk_image.GetSize()}, spacing: {itk_image.GetSpacing()}")

        sitk.WriteImage(itk_image, str(self.mrsegmentator_input_dir / "mri_0000.nii.gz"))
        mrseg_image = os.path.join(self.mrsegmentator_input_dir, "mri_0000.nii.gz")

        inference.infer([mrseg_image], self.mrsegmentator_output_dir, [0])
        mrseg_mask = sitk.ReadImage(self.mrsegmentator_output_dir / "mri_0000_seg.nii.gz")
        cropped_mri, cropped_mask, crop_origin = crop_image_by_label(itk_image, mrseg_mask, label=7, margin_mm=(15, 50, 100))
        cropped_mri_np = sitk.GetArrayFromImage(cropped_mri).astype(np.float32)

        #5. predict with nnUNet
        print(f"Input image nnunet:{os.listdir(self.nnunet_input_dir)}")
        predictor = nnUNetPredictor(
                    tile_step_size=0.33,
                    use_gaussian=True,
                    use_mirroring=True,
                    perform_everything_on_device=True,
                    device=torch.device('cuda'),
                    verbose=True,
                    verbose_preprocessing=False,
                    allow_tqdm=True)
        
        list_segmentation = []
        list_segmentation_sitk = []

        for model_id in ["0","1","2","3","4"]:
            
            predictor.initialize_from_trained_model_folder(
                    "/opt/algorithm/nnUNet_results/nnUNetTrainer_nnUNetResEncUNetLPlans_{}_3d_fullres/".format(model_id),
                    use_folds=("all",),
                    checkpoint_name="checkpoint_final.pth",
            )
            
            output_segmentation = predictor.predict_single_npy_array(input_image=np.expand_dims(cropped_mri_np, axis=0), 
                                                                                image_properties={'spacing': itk_image.GetSpacing()},
                                                                                segmentation_previous_stage=None, 
                                                                                output_file_truncated=None, 
                                                                                save_or_return_probabilities=False)
            
            output_segmentation[output_segmentation>1] = 0
            output_segmentation_sitk = sitk.GetImageFromArray(output_segmentation)
            output_segmentation_sitk.CopyInformation(cropped_mri)
            list_segmentation.append(output_segmentation)
            list_segmentation_sitk.append(output_segmentation_sitk)

            
        foregroundValue = 1
        threshold = 0.5
        reference_segmentation_STAPLE_probabilities = sitk.STAPLE(list_segmentation_sitk, foregroundValue)
        output_segmentation = reference_segmentation_STAPLE_probabilities > threshold
        output_segmentation = sitk.GetArrayFromImage(output_segmentation)

        restored_seg = uncrop_to_original_shape(output_segmentation, itk_image, crop_origin, fill_value=0)
        restored_seg_sitk = sitk.GetImageFromArray(restored_seg)
        restored_seg_sitk.CopyInformation(itk_image)
        sitk.WriteImage(restored_seg_sitk, self.segmentation_mask)

        self.predict()
        end_time = time.perf_counter()
        print(f"Prediction time: {end_time - start_time:.3f} seconds")

    def predict(self):
        return 0
    
def _show_torch_cuda_info():
    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

    
def keep_largest_component(mask, target_label=7):
    binary = (mask == target_label).astype(np.int32)
    labeled_array, num_features = label(binary)

    if num_features == 0:
        return np.zeros_like(mask)

    # Utilise np.bincount pour compter les voxels de chaque composante
    # Index 0 est le fond, on ignore
    counts = np.bincount(labeled_array.ravel())
    counts[0] = 0  # ignore background
    largest_component = counts.argmax()

    # Génère le masque final
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labeled_array == largest_component] = target_label

    return cleaned_mask


def crop_image_by_label(image, mask, label=4, margin_mm=(15, 50, 100)):
    """
    Crop autour du label donné en utilisant une bounding box + marge (Z, Y, X) en mm.
    """
    image_np = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    mask_np = sitk.GetArrayFromImage(mask)

    spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()    # (x, y, z)

    indices = np.argwhere(mask_np == label)
    if indices.size == 0:
        return None, None

    # Bounding box min/max
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)

    # Convert margin in mm to voxels (Z, Y, X)
    margin_vox = [
        int(round(margin_mm[0] / spacing[2])),  # Z
        int(round(margin_mm[1] / spacing[1])),  # Y
        int(round(margin_mm[2] / spacing[0]))   # X
    ]

    # Apply margins
    min_idx = min_idx - margin_vox
    max_idx = max_idx + margin_vox

    # Clamp to image bounds
    shape = mask_np.shape  # (Z, Y, X)
    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, shape)

    # Crop arrays
    cropped_image_np = image_np[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
    cropped_mask_np = mask_np[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]

    # Convert back to SimpleITK images
    cropped_image = sitk.GetImageFromArray(cropped_image_np)
    cropped_mask = sitk.GetImageFromArray(cropped_mask_np)

    # Set new origin
    new_origin = [
        origin[0] + min_idx[2] * spacing[0],
        origin[1] + min_idx[1] * spacing[1],
        origin[2] + min_idx[0] * spacing[2]
    ]

    # Set metadata
    cropped_image.SetSpacing(spacing)
    cropped_image.SetOrigin(new_origin)
    cropped_image.SetDirection(image.GetDirection())

    cropped_mask.SetSpacing(spacing)
    cropped_mask.SetOrigin(new_origin)
    cropped_mask.SetDirection(mask.GetDirection())

    return cropped_image, cropped_mask, new_origin

def uncrop_to_original_shape(cropped_array, original_image, crop_origin, fill_value=0):
    original_shape = sitk.GetArrayFromImage(original_image).shape
    spacing = original_image.GetSpacing()
    origin = original_image.GetOrigin()

    offset_vox = [
        int(round((crop_origin[2] - origin[2]) / spacing[2])),  # Z
        int(round((crop_origin[1] - origin[1]) / spacing[1])),  # Y
        int(round((crop_origin[0] - origin[0]) / spacing[0]))   # X
    ]

    restored = np.full(original_shape, fill_value, dtype=cropped_array.dtype)

    z, y, x = cropped_array.shape
    oz, oy, ox = offset_vox

    end_z = min(oz + z, original_shape[0])
    end_y = min(oy + y, original_shape[1])
    end_x = min(ox + x, original_shape[2])

    restored[oz:end_z, oy:end_y, ox:end_x] = cropped_array[:end_z - oz, :end_y - oy, :end_x - ox]
    return restored


if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()
