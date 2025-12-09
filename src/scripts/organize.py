import os
import shutil
import glob
from pathlib import Path

# Config
# Get the Project Root (Assuming this script is in src/data/)
# .../LeafModel/src/data/organize.py -> parents[2] = .../LeafModel
ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "data"
DEST_DIR = ROOT_DIR / "data_processed"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def clean_name(name):
    """Removes special chars and spaces from names."""
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("__", "_")

def organize_dataset():
    print(f"Scanning '{SOURCE_DIR}' for image folders...")
    
    if os.path.exists(DEST_DIR):
        print(f"Destination '{DEST_DIR}' already exists. Skipping to avoid duplicates.")
        print("Delete it manually if you want to rebuild.")
        return

    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Walk through all directories
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Check if this folder contains images
        images = [f for f in files if Path(f).suffix.lower() in ALLOWED_EXTENSIONS]
        
        if len(images) > 10: # Threshold to ignore noise folders
            # We found a folder with images!
            path = Path(root)
            
            # Heuristic to find Crop and Disease names
            # Example Path: data/Mandua/blast -> Crop: Mandua, Disease: blast
            # Example Path: data/Mango/Augmented Mango/Anthracnose -> Crop: Mango, Disease: Anthracnose
            
            parts = path.parts
            # Assume last part is Disease, second to last is Crop
            # But if second to last is 'data', then... wait.
            
            # Improved logic:
            # Find the index of 'data' or the SOURCE_DIR name in the parts
            try:
                # We identify where 'data' is and take the next folder as Crop
                # path.parts looks like ('C:', 'Project', 'CDD', 'LeafModel', 'data', 'Mandua', 'blast')
                # We want 'Mandua' (index of 'data' + 1)
                
                # Search for the source_dir name (e.g. 'data')
                source_dir_name = SOURCE_DIR.name 
                idx = parts.index(source_dir_name)
                
                crop_name = clean_name(parts[idx + 1])
                disease_name = clean_name(parts[-1])
                
                # If the folder structure is 'data/Augmented Mango Dataset/Augmented Mango Dataset/Anthracnose'
                # crop_name = Augmented_Mango_Dataset
                # disease_name = Anthracnose
                
                # If simple: 'data/Mandua/blast' -> Mandua, blast
            except ValueError:
                # 'data' not in path? Should not happen if we walked strict
                continue
            except IndexError:
                continue

            # Skip if crop name is same as disease (e.g. data/Pigeon_Pea/Pigeon_Pea)
            if crop_name.lower() == disease_name.lower():
                 continue

            final_class_name = f"{crop_name}_{disease_name}"
            
            # Handle "Healthy" generically? No, keep it specific: "Mango_Healthy"
            
            target_folder = os.path.join(DEST_DIR, final_class_name)
            os.makedirs(target_folder, exist_ok=True)
            
            print(f"Processing: {final_class_name} ({len(images)} images)")
            
            for img in images:
                src_path = os.path.join(root, img)
                dst_path = os.path.join(target_folder, img)
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying {img}: {e}")

    print("="*50)
    print(f"Organization Complete! Data is now in '{DEST_DIR}'")
    print("Update your config.yaml to point here.")
    print("="*50)

if __name__ == "__main__":
    organize_dataset()
