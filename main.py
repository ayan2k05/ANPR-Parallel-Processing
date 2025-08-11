import os
import re
import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
from pathlib import Path
import threading
from queue import Queue
import time


"""
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀⠀⠀⠙⠋⢟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⣱⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢚⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠫⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢟⠉⡀⠆⡡⢒⠰⣜⢧⣻⢴⣤⡤⣤⣀⡄⠀⠀⠀⠺⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⠂⠤⠑⡤⣩⢞⡭⢿⡹⣞⢧⢿⣱⣏⢾⡱⢂⠔⡀⠊⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠂⠌⡀⢏⡴⢣⡞⣽⣧⣿⣞⣯⣷⣳⣎⡧⣝⡬⢒⠠⢁⠨⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠂⠀⠈⡔⡘⠀⣈⠁⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⢮⠳⡍⠂⠄⠂⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡷⠀⠀⢁⢒⡠⠙⠘⠛⠆⠀⠠⢟⣿⡿⡿⠏⠁⠀⣤⣤⠄⠁⠈⠄⠀⠈⠪⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⡯⠃⠁⠀⠀⠀⠀⠀⠀⢮⠐⡀⠘⣄⣠⣆⡳⢈⣿⡇⢡⢔⡌⠁⢠⠈⠘⡄⠀⠂⠀⠀⠀⠈⠻⢿⣿⣿⣿⣿⣿⣿
⣿⣿⡿⡿⠿⣿⡏⠁⠀⠀⠀⠀⠀⠀⠀⠸⠸⡎⣰⢿⣸⣇⣶⢁⠾⣷⣉⠀⣎⢷⡱⣏⠀⢆⡸⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿
⣿⡿⠁⠀⠀⠙⠃⠀⠀⠀⠀⣰⢟⣖⠀⢂⠱⣭⢻⣿⣿⣿⡿⢌⣹⣷⣯⡘⣬⣷⣷⡽⣚⢬⠓⡀⠀⢀⡀⠀⠀⠀⠀⠈⠙⢻⣿⣿⣿
⡿⠋⠀ ⠀⠀⠀⠀⠀⠀⠀⢹⣸⣟⡀⠠⢓⡌⣿⣿⣿⣿⢣⡜⣿⣿⣿⡇⡸⣿⣿⣿⡙⢎⡁⠀⣿⣿⣿⡆⠀⠀⠀⠀⢘⣿⣿⣿⣿
⣷⠀⠀⠀⠀⠀⠀⠀⠀⣀⡀⠨⠿⣿⡁⠐⣡⠚⡵⣿⣿⣿⣀⣂⠻⠿⠟⡁⢢⣿⣿⡷⣍⠲⠀⣼⣮⣿⣿⠏⣠⣤⣤⣰⣿⣿⣿⣿⣿
⣿⣧⣠⣀⣀⡀⠀⠀⢠⣹⣿⣦⡲⠬⠷⠀⡜⡇⢹⡿⣛⢲⡹⣾⢿⣷⣿⡹⢏⣿⣿⡝⢢⡑⢀⣿⣿⢿⠍⣺⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣦⣤⣾⣿⣿⣿⣿⣿⣿⣷⠀⢸⢳⢰⡓⣇⠈⠛⠉⠛⣛⠙⠧⢣⣸⢿⠘⢠⠇⠘⣻⠖⣈⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠂⠀⢫⠘⣽⣟⣎⠱⣞⣷⣶⢆⡄⣲⣾⡣⢨⡟⢀⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⠆⠀⠃⢼⣻⣾⣷⣖⣦⣙⣮⡶⣿⢯⠁⠚⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⡐⢎⡄⡀⠈⢿⣻⣿⣿⣿⣿⣿⣿⡝⠂⢀⠄⣳⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠠⣙⠦⡜⡤⣁⠈⠓⠛⠿⡿⢿⡻⠌⢁⠰⣈⠒⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣛⣿⠟⠡⣑⢮⡻⣵⣓⢦⡱⢌⠲⣤⢐⠠⢄⠒⣬⠳⠄⢼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⠟⣭⣾⣿⣿⣿⠀⡜⡰⣭⢾⣽⣷⣿⢧⣛⣮⢷⣹⣎⢱⢪⡝⡶⣍⠌⡛⢺⣟⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⡿⣫⣵⣶⣿⣿⣿⣿⣿⣿⡜⣵⣹⢮⣟⣾⣿⣿⣿⡽⣮⣳⠻⣌⢷⣫⢞⡷⣌⠢⢵⢸⣿⣿⣷⣾⣭⣛⣻⢿⣿⣿⣿⣿⣿⣿⣿⡿
⡧⠾⠿⠿⠿⠿⠿⠿⠿⠿⡿⠿⠞⠵⠻⠞⠯⠿⠿⠿⠿⠷⠯⠟⠮⠷⠏⠿⠰⠢⠱⠎⠾⠿⠿⣿⠿⣿⠿⠿⠿⠶⠽⢻⣿⣿⣿⠃⠀"""
#~~~~~~~~~~~~~~~~~~~~~~Hakla~Was~Here~~~~~~~~~~~~~~~~~~~~~~~


images_folder = '/content/hehe'
output_folder = '/content/output'
incomplete_folder = os.path.join(output_folder, 'Incomplete_plate')
Path(output_folder).mkdir(exist_ok=True)
Path(incomplete_folder).mkdir(exist_ok=True)

device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Available GPUs: {device_count}")

models = {}
ocrs = {}
CORRECTIONS = {
    "MB": "WB",
    "W8": "WB",
    "18": "WB",
    "IB": "WB",
    "0D": "OD",
    "DD": "OD",
    "CD": "OD",
    "H8": "HR",
    "G7": "GJ",
    "K4": "KA",
    "R7": "RJ",
    "R1": "RJ",
    "U9": "UP",
    "C6": "CG",
    "T5": "TS"
}

VALID_PATTERNS = [re.compile(p) for p in [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',
    r'^[A-Z]{2}[0-9]{2}[A-Z][0-9]{4}$',
    r'^[A-Z]{2}[0-9][A-Z]{3}[0-9]{4}$'
]]

BANNED_WORDS = frozenset({'IND', 'SPEED', 'SS', 'INO', 'ND'})
VALID_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png'})
YOLO_BATCH_SIZE = 16
OCR_BATCH_SIZE = 8
CHAR_FILTER = str.maketrans('', '', ''.join(chr(i) for i in range(256)
                                           if chr(i) not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

def init_gpu_resources():
    """Initialize models on all available GPUs"""
    global models, ocrs

    if device_count == 0:
        print("No GPUs available, using CPU")
        models[0] = YOLO('/content/license_plate_detector.pt').to('cpu')
        ocrs[0] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        return

    print(f"Initializing models on {device_count} GPUs...")

    for gpu_id in range(device_count):
        device = f'cuda:{gpu_id}'
        print(f"Loading YOLO on GPU {gpu_id}")
        models[gpu_id] = YOLO('/content/license_plate_detector.pt').to(device)

        ocrs[gpu_id] = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            use_gpu=True,
            gpu_mem=4000,#4 gig per gpu
        )

    print("GPU initialization complete!")

@lru_cache(maxsize=256)
def apply_corrections(prefix):
    """Cached correction lookup"""
    return CORRECTIONS.get(prefix, prefix)

def is_valid_extension(filename):
    """Fast extension check"""
    return Path(filename).suffix.lower() in VALID_EXTENSIONS

def clean_text_fast(text):
    """Optimized text cleaning using translation table"""
    return text.upper().translate(CHAR_FILTER)[:12]

def validate_plate_pattern(text):
    """Fast pattern validation"""
    return any(pattern.fullmatch(text) for pattern in VALID_PATTERNS)

def load_images_batch(image_paths, max_batch_size=12):
    """Load images in batches to utilize 12GB memory"""
    images = []
    valid_paths = []

    for path in image_paths[:max_batch_size]:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
            valid_paths.append(path)

    return images, valid_paths

def process_yolo_batch(images, gpu_id=0):
    """Process YOLO inference on batch of images using specific GPU"""
    if not images:
        return []

    device = f'cuda:{gpu_id}' if device_count > 0 else 'cpu'
    model = models[gpu_id]

    try:
        with torch.no_grad():
            torch.backends.cudnn.benchmark = True
            results = model.predict(
                images,
                verbose=False,
                half=True,
                device=device,
                batch=len(images)
            )

        return results
    except Exception as e:
        print(f"YOLO batch processing error on GPU {gpu_id}: {e}")
        return []

def process_ocr_batch(crops, gpu_id=0):
    if not crops:
        return []

    ocr = ocrs[gpu_id]
    results = []

    try:
        for i in range(0, len(crops), OCR_BATCH_SIZE):
            batch_crops = crops[i:i + OCR_BATCH_SIZE]

            for crop in batch_crops:
                result = ocr.predict(crop)
                results.append(result)

    except Exception as e:
        print(f"OCR batch processing error on GPU {gpu_id}: {e}")
        results = [[] for _ in crops]

    return results

def extract_plates_from_results(images, results, image_paths):
    crops = []
    crop_info = []

    for idx, (image, result, path) in enumerate(zip(images, results, image_paths)):
        if result is None or not hasattr(result, 'boxes'):
            continue

        boxes = result.boxes
        if len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = xyxy[0].astype(int)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop = image[y1:y2, x1:x2]
        crops.append(plate_crop)
        crop_info.append({
            'image': image,
            'path': path,
            'crop_idx': idx
        })

    return crops, crop_info

def process_batch_on_gpu(batch_paths, gpu_id):
    """Process a batch of images on a specific GPU"""
    results = []

    try:
        # Load images
        images, valid_paths = load_images_batch(batch_paths, YOLO_BATCH_SIZE)

        if not images:
            return []

        yolo_results = process_yolo_batch(images, gpu_id)


        crops, crop_info = extract_plates_from_results(images, yolo_results, valid_paths)
        ocr_results = process_ocr_batch(crops, gpu_id)
        for crop_data, ocr_result in zip(crop_info, ocr_results):
            result = process_single_result(crop_data, ocr_result)
            results.append(result)

    except Exception as e:
        print(f"Batch processing error on GPU {gpu_id}: {e}")

    return results

def process_single_result(crop_data, ocr_result):
    try:
        image = crop_data['image']
        path = crop_data['path']
        image_name = os.path.basename(path)

        rec_texts = []
        if ocr_result:
            for r in ocr_result:
                if 'rec_texts' in r:
                    rec_texts.extend(
                        t for t in r['rec_texts']
                        if t.strip().upper() not in BANNED_WORDS
                    )

        if not rec_texts:
            file_base_name = "UNKNOWN"
            is_valid = False
        else:
            full_text = ''.join(rec_texts)
            full_text_cleaned = clean_text_fast(full_text)

            if len(full_text_cleaned) >= 2:
                prefix = full_text_cleaned[:2]
                corrected_prefix = apply_corrections(prefix)
                full_text_cleaned = corrected_prefix + full_text_cleaned[2:]

            is_valid = validate_plate_pattern(full_text_cleaned)
            file_base_name = full_text_cleaned if full_text_cleaned else "UNKNOWN"

        target_folder = output_folder if is_valid else incomplete_folder
        save_path = os.path.join(target_folder, f"{file_base_name}.jpg")

        cv2.imwrite(save_path, image, [
            cv2.IMWRITE_JPEG_QUALITY, 95,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])

        status = 'Valid' if is_valid else 'Invalid'
        return f"Processed: {file_base_name} → {status}"

    except Exception as e:
        return f"Error processing result: {str(e)}"

def get_image_files():
    """Get list of image files efficiently"""
    try:
        all_files = os.listdir(images_folder)
        image_files = [f for f in all_files if is_valid_extension(f)]
        # Return full paths
        return [os.path.join(images_folder, f) for f in image_files]
    except OSError as e:
        print(f"Error reading directory {images_folder}: {e}")
        return []

def distribute_work(image_paths, num_gpus):
    """Distribute work across available GPUs"""
    if num_gpus <= 1:
        return [image_paths]

    chunk_size = len(image_paths) // num_gpus
    chunks = []

    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1: 
            end_idx = len(image_paths)
        else:
            end_idx = start_idx + chunk_size

        chunks.append(image_paths[start_idx:end_idx])

    return chunks

def main():
    """Main processing function optimized for T4x2"""
    print("🚀 T4x2 Optimized License Plate Detection")
    print("=" * 50)
    init_gpu_resources()
    image_paths = get_image_files()

    if not image_paths:
        print("No valid image files found!")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Using {device_count if device_count > 0 else 1} processing units")

    start_time = time.time()

    if device_count <= 1:
        print("Processing on single device...")
        all_results = []
        for i in range(0, len(image_paths), YOLO_BATCH_SIZE):
            batch = image_paths[i:i + YOLO_BATCH_SIZE]
            batch_results = process_batch_on_gpu(batch, 0)
            all_results.extend(batch_results)

            print(f"Processed batch {i//YOLO_BATCH_SIZE + 1}/{(len(image_paths)-1)//YOLO_BATCH_SIZE + 1}")

    else:
        print(f"Processing on {device_count} GPUs...")
        work_chunks = distribute_work(image_paths, device_count)

        all_results = []
        with ThreadPoolExecutor(max_workers=device_count) as executor:
            futures = []
            for gpu_id, chunk in enumerate(work_chunks):
                if chunk:
                    for i in range(0, len(chunk), YOLO_BATCH_SIZE):
                        batch = chunk[i:i + YOLO_BATCH_SIZE]
                        future = executor.submit(process_batch_on_gpu, batch, gpu_id)
                        futures.append(future)
            for i, future in enumerate(as_completed(futures)):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    print(f"Completed batch {i+1}/{len(futures)}")
                except Exception as e:
                    print(f"Batch failed: {e}")
    print("\n" + "=" * 50)
    print("PROCESSING RESULTS:")
    print("=" * 50)

    for result in all_results:
        print(result)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n Completed processing {len(image_paths)} images")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f" Average: {processing_time/len(image_paths):.3f} seconds per image")

    if device_count > 1:
        print(f"Performance: ~{len(image_paths)/processing_time:.1f} images/second")

if __name__ == "__main__":
    main()
