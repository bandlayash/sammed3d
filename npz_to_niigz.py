import numpy as np
import nibabel as nib
import os
from pathlib import Path
import SimpleITK as sitk
import torchio as tio
from torchio.data.io import sitk_to_nib

def verify_conversion_with_sitk(file_path):
    """
    Verify that SimpleITK and torchio can read the converted file
    """
    try:
        # Test if SimpleITK can read it
        sitk_img = sitk.ReadImage(str(file_path))
        
        # Test if torchio can process it
        sitk_arr, _ = sitk_to_nib(sitk_img)
        
        # Convert to float32 to avoid PyTorch norm errors
        sitk_arr = sitk_arr.astype(np.float32)
        
        # Check if we can create a torchio subject
        if 'label' in str(file_path):
            tio_obj = tio.LabelMap(tensor=sitk_arr)
        else:
            tio_obj = tio.ScalarImage(tensor=sitk_arr)
            
        return True, f"Successfully verified {file_path}"
    except Exception as e:
        return False, f"Error verifying {file_path}: {str(e)}"


def convert_npz_to_nifti(npz_path, output_path, is_ct=True):
    """
    Convert .npz file to .nii.gz format for CT images
    """
    # Load the .npz file
    data = np.load(npz_path)
    
    # Check available keys
    print(f"Keys in {npz_path}: {list(data.keys())}")
    
    # Extract the volume data
    if 'data' in data.keys():
        volume = data['data']
    elif 'array' in data.keys():
        volume = data['array']
    elif len(data.keys()) == 1:
        key = list(data.keys())[0]
        volume = data[key]
    else:
        key = list(data.keys())[0]
        volume = data[key]
        print(f"Using key: {key}")
    
    # Print original data info
    print(f"Original dtype: {volume.dtype}")
    print(f"Original shape: {volume.shape}")
    print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Convert to float32 (standard for medical images)
    volume = volume.astype(np.float32)
    
    # For CT scans, optionally pre-clamp to HU range (SAM-Med3D does this anyway)
    if is_ct:
        # Standard CT Hounsfield unit range
        volume = np.clip(volume, -1000, 1000)
        print(f"CT values clamped to [-1000, 1000] HU")
    
    # Create NIfTI image with identity affine (no spatial info)
    # Note: This assumes 1mm isotropic spacing
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
    
    # Set the data type in the header
    nifti_img.header.set_data_dtype(np.float32)
    
    # Save as .nii.gz
    nib.save(nifti_img, output_path)
    print(f"Converted {npz_path} -> {output_path}")
    

    # Verify the conversion
    success, msg = verify_conversion_with_sitk(output_path)
    if success:
        print(f"✓ Verification passed")
    else:
        print(f"✗ Verification failed: {msg}")
    
    return success

def convert_mask_to_binary(mask_npz_path, output_path):
    """
    Convert mask .npz file to binary NIfTI format
    """
    # Load the mask .npz file
    data = np.load(mask_npz_path)
    
    # Check available keys
    print(f"Keys in {mask_npz_path}: {list(data.keys())}")
    
    # Extract mask data
    if 'data' in data.keys():
        mask = data['data']
    elif 'mask' in data.keys():
        mask = data['mask']
    elif 'array' in data.keys():
        mask = data['array']
    elif len(data.keys()) == 1:
        key = list(data.keys())[0]
        mask = data[key]
    else:
        key = list(data.keys())[0]
        mask = data[key]
        print(f"Using key: {key}")
    
    # Print original mask info
    print(f"Original dtype: {mask.dtype}")
    print(f"Original shape: {mask.shape}")
    print(f"Unique values before conversion: {np.unique(mask)}")
    
    # Convert to binary (0 and 1) - using uint8 for masks
    # This is more memory efficient and standard for segmentation masks
    binary_mask = (mask > 0).astype(np.uint8)
    
    print(f"Mask shape: {binary_mask.shape}")
    print(f"Unique values after conversion: {np.unique(binary_mask)}")
    print(f"Number of positive voxels: {binary_mask.sum()}")
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(binary_mask, affine=np.eye(4))
    
    # Set the data type in the header
    nifti_img.header.set_data_dtype(np.uint8)
    
    # Save as .nii.gz
    nib.save(nifti_img, output_path)
    print(f"Converted mask: {mask_npz_path} -> {output_path}")
    
    # Verify the conversion
    verify_img = nib.Nifti1Image(binary_mask.astype(np.float32), affine=np.eye(4))
    nib.save(verify_img, output_path.with_name(output_path.stem + "_verify.nii.gz"))
    
    success, msg = verify_conversion_with_sitk(output_path.with_name(output_path.stem + "_verify.nii.gz"))
    
    # Clean up temporary verify file
    output_path.with_name(output_path.stem + "_verify.nii.gz").unlink(missing_ok=True)

    return success

def validate_pair(image_path, mask_path, min_volume_mm3=10):
    """
    Validate that an image-mask pair is suitable for training
    """
    try:
        # Load both files
        img_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        # Check shapes match
        if img_nii.shape != mask_nii.shape:
            return False, f"Shape mismatch: image {img_nii.shape} vs mask {mask_nii.shape}"
        
        # Check mask has sufficient volume
        mask_data = mask_nii.get_fdata()
        # Assuming 1mm³ voxels (since we use identity affine)
        volume_mm3 = mask_data.sum()
        
        if volume_mm3 < min_volume_mm3:
            return False, f"Mask volume too small: {volume_mm3:.1f} mm³ < {min_volume_mm3} mm³"
        
        return True, f"Valid pair (volume: {volume_mm3:.1f} mm³)"
        
    except Exception as e:
        return False, f"Error validating pair: {str(e)}"

def convert_dataset(ct_scans_dir, annotations_dir, output_dir, target='veins', 
                   min_volume_mm3=10, is_ct=True):
    """
    Convert entire dataset from .npz to SAM-Med3D format
    
    Args:
        ct_scans_dir: Directory containing CT scan .npz files
        annotations_dir: Directory containing annotation .npz files
        output_dir: Output directory for converted files
        target: 'veins' or 'arteries' - which masks to use
        min_volume_mm3: Minimum mask volume in mm³ to include
        is_ct: Whether the images are CT scans (applies HU clamping)
    """
    ct_scans_path = Path(ct_scans_dir)
    annotations_path = Path(annotations_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / "imagesTr"
    labels_dir = output_path / "labelsTr"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of CT scan files
    ct_files = sorted(list(ct_scans_path.glob("*.npz")))
    print(f"Found {len(ct_files)} CT scan files")
    
    successful_conversions = 0
    skipped_files = []
    
    for i, ct_file in enumerate(ct_files):
        base_name = ct_file.stem
        print(f"\n[{i+1}/{len(ct_files)}] Processing: {base_name}")
        print("-" * 50)
        
        try:
            # Convert CT scan
            ct_output = images_dir / f"{base_name}.nii.gz"
            ct_success = convert_npz_to_nifti(ct_file, ct_output, is_ct=is_ct)
            
            if not ct_success:
                print(f"⚠️  Failed to convert CT scan, skipping pair")
                if ct_output.exists():
                    ct_output.unlink()
                continue
            
            # Find corresponding mask file
            mask_file = annotations_path / target / f"{base_name}.npz"
            
            if mask_file.exists():
                # Convert mask
                mask_output = labels_dir / f"{base_name}.nii.gz"
                mask_success = convert_mask_to_binary(mask_file, mask_output)
                
                if not mask_success:
                    print(f"⚠️  Failed to convert mask, removing CT scan")
                    if ct_output.exists():
                        ct_output.unlink()
                    continue
                
                # Validate the pair
                valid, msg = validate_pair(ct_output, mask_output, min_volume_mm3)
                
                if valid:
                    print(f"✅ {msg}")
                    successful_conversions += 1
                else:
                    print(f"⚠️  Invalid pair: {msg}")
                    skipped_files.append((base_name, msg))
                    # Remove both files
                    if ct_output.exists():
                        ct_output.unlink()
                    if mask_output.exists():
                        mask_output.unlink()
            else:
                print(f"⚠️  No {target} mask found for {base_name}")
                skipped_files.append((base_name, "No mask found"))
                # Remove the CT scan since we don't have a corresponding mask
                if ct_output.exists():
                    ct_output.unlink()
                    
        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}")
            skipped_files.append((base_name, str(e)))
            # Clean up any partial files
            ct_output = images_dir / f"{base_name}.nii.gz"
            mask_output = labels_dir / f"{base_name}.nii.gz"
            if ct_output.exists():
                ct_output.unlink()
            if mask_output.exists():
                mask_output.unlink()
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully converted: {successful_conversions}/{len(ct_files)} pairs")
    
    if skipped_files:
        print(f"\n⚠️  Skipped {len(skipped_files)} files:")
        for name, reason in skipped_files[:10]:  # Show first 10
            print(f"   - {name}: {reason}")
        if len(skipped_files) > 10:
            print(f"   ... and {len(skipped_files) - 10} more")
    
    return successful_conversions

# Example usage
if __name__ == "__main__":
    # Set your paths here
    CT_SCANS_DIR = "data/ct_scan"
    ANNOTATIONS_DIR = "data/annotation" 
    
    # Choose what you want to segment: 'vein' or 'artery'
    TARGET = 'artery'  # Change to 'artery' to segment arteries
    
    # Output directory
    OUTPUT_DIR = f"data/medical_preprocessed/{TARGET}_segmentation/ct_scans"
    
    # Minimum mask volume in mm³ (assuming 1mm³ voxels)
    MIN_VOLUME_MM3 = 10
    
    print(f"🎯 Converting dataset for {TARGET} segmentation...")
    print(f"📂 CT scans from: {CT_SCANS_DIR}")
    print(f"📂 Using {TARGET} masks from: {ANNOTATIONS_DIR}/{TARGET}/")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"🔍 Minimum mask volume: {MIN_VOLUME_MM3} mm³")
    
    # Convert the dataset
    successful = convert_dataset(
        CT_SCANS_DIR, 
        ANNOTATIONS_DIR, 
        OUTPUT_DIR, 
        TARGET,
        min_volume_mm3=MIN_VOLUME_MM3,
        is_ct=True  # Set to False if not CT data
    )
    
    if successful > 0:
        print(f"\n✅ SUCCESS! Converted {successful} image-mask pairs")
        print(f"📁 Images saved to: {OUTPUT_DIR}/imagesTr/")
        print(f"📁 Masks saved to: {OUTPUT_DIR}/labelsTr/")
        print(f"\n📋 Next steps:")
        print(f"1. Update utils/data_paths.py with: '{OUTPUT_DIR}'")
        print(f"2. Download pre-trained weights to SAM-Med3D/ckpt/")
        print(f"3. Run training with: python train.py --task_name {TARGET}_segmentation")
    else:
        print(f"\n❌ No successful conversions. Please check:")
        print(f"   - File paths are correct")
        print(f"   - .npz files contain expected data")
        print(f"   - Masks have sufficient volume (>{MIN_VOLUME_MM3} mm³)")