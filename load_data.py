#Functions for data loading.
from pathlib import Path
import os
from collections import defaultdict
from PIL import Image
import pandas as pd



class Dataloader:

    """
    Loads and prepares data from a given file or directory path.

    This class provides a simple interface to load input data from disk,
    which can then be used for further processing, training, or evaluation.

    Args:
        data_path (str): Path to the input data. Can be a file or directory.

    Attributes:
        path (str): Stores the provided path to the input data.
    """

    def __init__(self, data_path):
        self.path = data_path
        self.structure = self._scan_structure()
        self.shape = DataShape(self.structure)

    def _scan_structure(self) -> dict:
        """
        Scans the data directory and builds a dictionary of the structure.
        Keys are folder names (e.g., classes), values are lists of file paths.
        """
        
        structure = defaultdict(lambda: defaultdict(dict))

        for root, dirs, files in os.walk(self.path):
            root_path = Path(root)
            relative = root_path.relative_to(self.path)

            if len(relative.parts) < 2:
                continue

            class_name = relative.parts[0]
            subtype = relative.parts[1]

            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    structure[class_name][subtype].setdefault("images", []).append(root_path / file)
                elif file.lower().endswith((".csv")):
                    structure[class_name][subtype]["parts"]=root_path / file

        return {k: dict(v) for k, v in structure.items()}

    def load_product(self, category: str, product_name:str):
        product_images = self.structure[category][product_name]["images"]
        product_parts = self.structure[category][product_name]["parts"]
        loaded_Images = [
            Image.open(path) for path in product_images
        ]
        parts = pd.read_csv(product_parts)
        loaded_parts = pd.DataFrame(parts)
        print(f"For Product: {category}; {product_name} Images found: {len(loaded_Images)}")      
        return loaded_Images, loaded_parts




class DataShape:
    def __init__(self, structure: dict):
        self.structure = structure
        self.num_classes = len(structure)
        self.total_files = sum(
            len(v["images"]) for sub in structure.values() for v in sub.values()
        )
        self.files_per_class = {
            cls: {
                subtype: len(files)
                for subtype, files in subdict.items()
            }
            for cls, subdict in structure.items()
        }

    def __repr__(self):
        return f"<DataShape classes={self.num_classes}, total_files={self.total_files}>"
