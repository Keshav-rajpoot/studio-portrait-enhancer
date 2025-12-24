import cv2
import os

from src.pipeline import StudioPortraitPipeline

INPUT_DIR = "samples/input"
OUTPUT_DIR = "samples/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = StudioPortraitPipeline()

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    output = pipeline.process(image)

    out_path = os.path.join(
        OUTPUT_DIR, img_name.replace(".jpg", "_studio.jpg")
    )
    cv2.imwrite(out_path, output)

print("✅ Studio portrait pipeline completed.")
