import pandas as pd
import base64
from PIL import Image
import io
import os
import glob

# Define input and output paths
parquet_files_pattern = "data/minecraft-skins-captioned-900k/data/*.parquet"
output_image_dir = "output_images"

# Ensure output directory exists
os.makedirs(output_image_dir, exist_ok=True)

# Common column names for image and text
image_col_names = ["image", "pixel_values"]
caption_col_names = ["text", "caption", "label"]

# Get all parquet files
parquet_files = glob.glob(parquet_files_pattern)
if not parquet_files:
    print(f"No parquet files found matching {parquet_files_pattern}")
    exit()

total_images_saved = 0

for parquet_file in parquet_files:
    print(f"Processing {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    for index, row in df.iterrows():
        image_data = None
        caption = None

        # Extract image data
        for col in image_col_names:
            if col in row:
                image_data = row[col]
                break

        # Extract caption
        for col in caption_col_names:
            if col in row:
                caption = row[col]
                break

        if image_data is None:
            print(f"Warning: No image data found for row {index} in {parquet_file}. Skipping.")
            continue

        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Generate a unique filename
            # Using original parquet file name and row index for uniqueness
            base_filename = os.path.basename(parquet_file).replace(".parquet", "")
            image_filename = f"{base_filename}_{index}.png"
            image_path = os.path.join(output_image_dir, image_filename)

            # Save the image
            image.save(image_path)
            total_images_saved += 1

            # Save caption if available
            if caption:
                caption_filename = f"{base_filename}_{index}.txt"
                caption_path = os.path.join(output_image_dir, caption_filename)
                with open(caption_path, "w") as f:
                    f.write(str(caption)) # Ensure caption is a string

        except Exception as e:
            print(f"Error processing row {index} in {parquet_file}: {e}. Skipping.")
            continue

    print(f"Finished processing {parquet_file}.")

print(f"\nSuccessfully saved {total_images_saved} images and their captions to '{output_image_dir}' directory.")
