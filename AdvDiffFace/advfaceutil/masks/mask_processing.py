import numpy as np
from PIL import Image


# Load the glasses image
glasses_masks = np.asarray(Image.open("frames.png").convert("L"))

# Convert the image to a binary mask
masks = np.zeros_like(glasses_masks)
masks[glasses_masks >= 128] = 1

# Separate the mask into 6 rows and 4 columns
masks = np.split(masks, 6)

# Separate each row into 4 masks
masks = [np.split(mask, 4, 1) for mask in masks]

# Flatten the masks into one list
masks = [mask for row in masks for mask in row]

# Crop the masks to remove the space around the glasses
cropped_masks = []

for mask in masks:
    # Find where the rows and columns are non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # Find the minimum and maximum indices of the rows and columns
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop the mask
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_masks.append(cropped_mask)


# Pad the masks to 128x352, placing the mask in the centre
padded_masks = []

for mask in cropped_masks:
    # Calculate the padding needed
    y_pad = 128 - mask.shape[0]
    x_pad = 352 - mask.shape[1]

    # Calculate the padding before and after
    y_before = y_pad // 2
    y_after = y_pad - y_before
    x_before = x_pad // 2
    x_after = x_pad - x_before

    # Pad the mask
    padded_mask = np.pad(mask, ((y_before, y_after), (x_before, x_after)))
    padded_masks.append(padded_mask)


# Shrink the masks to 64x176 by averaging 2x2 blocks
shrunken_masks = []

for mask in padded_masks:
    # Split the mask into 64 2x2 blocks
    blocks = mask.reshape(64, 2, 176, 2)

    # Calculate the mean of each block
    shrunken_mask = np.mean(blocks, axis=(1, 3))

    # Convert the mask to binary
    shrunken_mask[shrunken_mask >= 0.5] = 1
    shrunken_mask[shrunken_mask < 0.5] = 0

    shrunken_masks.append(shrunken_mask)

# Save the masks to CSV files
for i, mask in enumerate(shrunken_masks):
    np.savetxt(f"masks/mask_{i}.csv", mask, fmt="%d", delimiter=",")
