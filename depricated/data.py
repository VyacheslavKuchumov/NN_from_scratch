import os
import random
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# Download the dataset and print its path
dataset_path = kagglehub.dataset_download("juniorbueno/neural-networks-homer-and-bart-classification")
print("Path to dataset files:", dataset_path)

# Define the folder where the images are stored
image_folder = os.path.join(dataset_path, "homer_bart_1")

# Set the desired image size (lower resolution)
resize_dim = (64, 64)

# List to store each image's data (including pixel data and label fields)
data_rows = []

# Process each image and store the converted data
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        file_path = os.path.join(image_folder, filename)
        with Image.open(file_path) as img:
            # Resize the image to lower resolution
            img_resized = img.resize(resize_dim)
            # Convert the image to grayscale
            img_gray = img_resized.convert("L")
            # Flatten the image into a 1D list of pixel values
            pixel_data = np.array(img_gray).flatten().tolist()
            
            # Determine the label based on the filename:
            # For Homer images, label = [1, 0]; for Bart images, label = [0, 1]
            if "homer" in filename.lower():
                label = [1, 0]
            elif "bart" in filename.lower():
                label = [0, 1]
            else:
                # Skip images that do not clearly match Homer or Bart
                continue
            
            # Append the pixel data and the label to the list
            data_rows.append(pixel_data + label)

# (Optional) Create a Pandas DataFrame and save as CSV
header = [f"pixel_{i}" for i in range(resize_dim[0] * resize_dim[1])] + ["homer", "bart"]
df = pd.DataFrame(data_rows, columns=header)
output_csv = "homer_bart_images_pandas.csv"
df.to_csv(output_csv, index=False)
print("CSV file created:", output_csv)

# -------------------------------
# Now, let's display a random converted image
# -------------------------------

# Choose a random row from the processed data_rows list
random_row = random.choice(data_rows)

# Extract pixel data (all columns except the last two, which are labels)
pixel_data = random_row[:resize_dim[0] * resize_dim[1]]

# Convert the pixel data back into a NumPy array and reshape it to the original image dimensions
img_array = np.array(pixel_data, dtype=np.uint8).reshape(resize_dim)

# Display the image using matplotlib
plt.imshow(img_array, cmap="gray")
plt.title("Random Converted Image")
plt.axis("off")
plt.show()
