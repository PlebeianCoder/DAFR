from typing import List

import numpy as np


def load_mask(mask_index: int) -> np.ndarray:
    return np.genfromtxt(
        f"advfaceutil/masks/mask_{mask_index}.csv", dtype=np.int32, delimiter=","
    )


def load_all_masks() -> List[np.ndarray]:
    return [load_mask(index) for index in range(24)]
