import os
import csv
import random

dataset_path = "dataset"

# Define friction ranges per class
friction_ranges = {
    "clear": (0.85, 1.00),
    "plowed": (0.65, 0.85),
    "light": (0.40, 0.65),
    "medium": (0.15, 0.40)
}

output_file = "labels.csv"

rows = []

for folder in os.listdir(dataset_path):
    if folder not in friction_ranges:
        continue

    folder_path = os.path.join(dataset_path, folder)
    min_fric, max_fric = friction_ranges[folder]

    for file in os.listdir(folder_path):
        image_path = os.path.join(folder, file)  # relative path
        friction_value = round(random.uniform(min_fric, max_fric), 4)

        rows.append([image_path, friction_value])

# Write CSV
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "friction"])
    writer.writerows(rows)

print(f"labels.csv created with {len(rows)} entries.")