"""Feature bank dataset for parallel data loading with PyTorch DataLoader."""
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset

from configs.config import MIN_MASK_AREA
from src.region_extractors import get_regions_from_pseudo


class FeatureBankDataset(Dataset):
    """Load training regions from pre-generated SAM3 pseudo masks."""

    def __init__(
        self, 
        image_paths: List[Path],
        strict_errors: bool = False,
    ):
        self.image_paths = image_paths
        self.strict_errors = strict_errors

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, list]:
        img_path = self.image_paths[idx]
        regions = []
        try:
            regions = get_regions_from_pseudo(str(img_path), min_area=MIN_MASK_AREA)
        except Exception as e:
            if self.strict_errors:
                raise
            print(f"\nWarning: error processing {img_path.name}: {e}")
            regions = []
        return str(img_path), regions


def collate_fn(batch):
    """Filter out samples with no regions."""
    return [item for item in batch if item[1]]


__all__ = ['FeatureBankDataset', 'collate_fn']
