import os
import shutil
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

# Config
ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "data"
DEST_DIR = ROOT_DIR / "data_processed"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def clean_name(name):
    """Removes special chars and spaces from names."""
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("__", "_")

def organize_dataset():
    logger.info(f"Scanning '{SOURCE_DIR}' for image folders...")
    
    if os.path.exists(DEST_DIR):
        logger.warning(f"Destination '{DEST_DIR}' already exists. Skipping to avoid duplicates.")
        return

    os.makedirs(DEST_DIR, exist_ok=True)
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        images = [f for f in files if Path(f).suffix.lower() in ALLOWED_EXTENSIONS]
        
        if len(images) > 10:
            path = Path(root)
            parts = path.parts
            
            try:
                # Robustly find 'data' in the path
                if 'data' in parts:
                    idx = parts.index('data')
                elif SOURCE_DIR.name in parts:
                    idx = parts.index(SOURCE_DIR.name)
                else:
                    continue

                if idx + 1 >= len(parts): 
                    continue
                    
                crop_name = clean_name(parts[idx + 1])
                disease_name = clean_name(parts[-1])
                
            except (ValueError, IndexError):
                continue

            if crop_name.lower() == disease_name.lower():
                 continue

            final_class_name = f"{crop_name}_{disease_name}"
            target_folder = os.path.join(DEST_DIR, final_class_name)
            os.makedirs(target_folder, exist_ok=True)
            
            logger.info(f"Processing: {final_class_name} ({len(images)} images)")
            
            for img in images:
                src_path = os.path.join(root, img)
                dst_path = os.path.join(target_folder, img)
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    logger.error(f"Error copying {img}: {e}")

    logger.info(f"Organization Complete! Data is in '{DEST_DIR}'")

if __name__ == "__main__":
    organize_dataset()
