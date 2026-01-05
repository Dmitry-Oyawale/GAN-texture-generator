import base64
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import os

# Path to one of the parquet files
parquet_file_path = "data/minecraft-skins-captioned-900k/data/train-00000-of-00007.parquet"

# Check if the file exists
if not os.path.exists(parquet_file_path):
    print(f"Error: Parquet file not found at {parquet_file_path}")
    exit()

print(f"Reading parquet file: {parquet_file_path}")
df = pd.read_parquet(parquet_file_path)
print(f"Successfully read {len(df)} rows.")

# Select a random data point
random_index = random.randint(0, len(df) - 1)
data_point = df.iloc[random_index]

# Attempt to extract image and caption
image_data = None
caption = "No caption found"

# Common column names for image and text
image_col_names = ["image", "pixel_values"] # Add other possible image column names
caption_col_names = ["text", "caption", "label"] # Add other possible caption column names

for col in image_col_names:
    if col in data_point:
        image_data = data_point[col]
        break

for col in caption_col_names:
    if col in data_point:
        caption = data_point[col]
        break

if image_data is None:
    print("Error: Could not find image data in common column names.")
    print("Available columns:", df.columns.tolist())
    exit()

# Try to open the image
try:
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
except Exception as e:
    print(f"Error opening image: {e}")
    print("Assuming image data might be in a different format. Displaying raw data type for debugging:")
    print(type(image_data))
    print(image_data[:50]) # Print first 50 bytes for inspection
    exit()


# Display the image and caption
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Random Data Point\nCaption: {caption}")
plt.axis('off')
plt.show()

print("Display complete.")
