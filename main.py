import sys
import os
import json
import time
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import easyocr
from collections import defaultdict
import concurrent.futures
import traceback

os.environ["YOLO_VERBOSE"] = "False"
sys.path.append('/app')

try:
    from doclayout_yolo.models.yolov10.model import YOLOv10
except ModuleNotFoundError:
    print("FATAL ERROR: Could not import YOLOv10.")
    sys.exit(1)

DPI = 120
MODEL_PATH = "models/doclayout_yolo.pt"
SUPPORTED_LANGUAGES = ['en', 'fr']

def extract_title_and_headings(elements):
    if not elements:
        return "Untitled Document", []
    page1 = sorted([el for el in elements if el['page'] == 1], key=lambda e: e['y_coord'])
    if not page1:
        return "Untitled Document", sorted(elements, key=lambda e: (e['page'], e['y_coord']))
    document_title = page1[0]['text']
    title_id = (page1[0]['page'], page1[0]['y_coord'])
    remaining_elements = [el for el in elements if (el['page'], el['y_coord']) != title_id]
    sorted_headings = sorted(remaining_elements, key=lambda e: (e['page'], e['y_coord']))
    return document_title, sorted_headings

def analyze_and_classify_headings_with_clustering(headings_data, page_height=842):
    if not headings_data:
        return []

    df = pd.DataFrame(headings_data)
    df['y_center'] = df['y_coord'] + df['box_height'] / 2

    mu_h = df['box_height'].mean()
    sigma_h = df['box_height'].std(ddof=0) or 1.0
    df['height_z'] = (df['box_height'] - mu_h) / sigma_h

    df['pos_norm'] = 1 - (df['y_center'] / page_height)
    df['score'] = df['height_z'] + df['pos_norm']

    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    N = len(df)

    def assign_level(i):
        p = (i + 1) / N
        if p <= 0.25:
            return 'H1'
        elif p <= 0.45:
            return 'H2'
        elif p <= 0.75:
            return 'H3'
        else:
            return 'H4'

    df['level'] = [assign_level(i) for i in range(N)]

    # ❗️Filter out H4 before returning
    df = df[df['level'].isin(['H1', 'H2', 'H3'])]

    df_sorted = df.sort_values(by=['page', 'y_coord'])
    return df_sorted[['level', 'text', 'page']].to_dict('records')


def process_pdf_to_json(pdf_path):
    try:
        yolo_model = YOLOv10(MODEL_PATH)
        ocr_reader = easyocr.Reader(SUPPORTED_LANGUAGES, gpu=False, verbose=False)

        doc = fitz.open(pdf_path)
        class_names = yolo_model.names
        all_elements = []

        page_images = [page.get_pixmap(dpi=DPI) for page in doc]
        page_images_np = [np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) for pix in page_images]

        if not page_images_np:
            doc.close()
            return {'title': "Empty or Unreadable Document", 'outline': []}

        results_batch = yolo_model.predict(page_images_np, conf=0.3, verbose=False)

        for page_num, (results, pix) in enumerate(zip(results_batch, page_images_np), 1):
            for box in results.boxes:
                label = class_names[int(box.cls)]

                is_heading = label in ["title", "list"]
                if not is_heading and label == "text":
                    box_width = box.xyxy[0][2] - box.xyxy[0][0]
                    if box_width > 100:
                        is_heading = True

                if is_heading:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cropped_img = pix[y1:y2, x1:x2]

                    if cropped_img.size > 0:
                        try:
                            text = " ".join(ocr_reader.readtext(cropped_img, detail=0, paragraph=True))
                            if len(text.split()) > 1 and len(text) > 5:
                                all_elements.append({
                                    "text": text,
                                    "page": page_num,
                                    "box_height": y2 - y1,
                                    "y_coord": y1,
                                    "label": label,
                                    "confidence": float(box.conf[0])
                                })
                        except Exception:
                            continue

        doc.close()
        title, sorted_elems = extract_title_and_headings(all_elements)
        page_height = page_images[0].height if page_images else 842
        outline = analyze_and_classify_headings_with_clustering(sorted_elems, page_height)

        return {'title': title, 'outline': outline}
    except Exception as e:
        traceback_str = traceback.format_exc()
        return {'error': f"{str(e)}\n{traceback_str}", 'file': os.path.basename(pdf_path)}

def process_file_worker(pdf_path, output_folder):
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
    print(f"\n--- Process {os.getpid()} starting on: {filename} ---")
    start_time = time.time()

    json_data = process_pdf_to_json(pdf_path)

    if 'error' in json_data:
        print(f"❌ Error processing {filename}:\n{json_data['error']}")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    elapsed_time = time.time() - start_time
    print(f"  ✅ Process {os.getpid()} finished {filename}. Saved to: {output_path} (Time: {elapsed_time:.2f}s)")

def batch_process_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
        futures = [executor.submit(process_file_worker, path, output_folder) for path in pdf_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A worker process task failed: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    INPUT_FOLDER = "/app/input"
    OUTPUT_FOLDER = "/app/output"

    print(f"Starting batch processing from '{INPUT_FOLDER}'...")
    batch_start_time = time.time()
    batch_process_pdfs(INPUT_FOLDER, OUTPUT_FOLDER)
    total_time = time.time() - batch_start_time

    print(f"\n--- Batch processing complete. ---")
    print(f"Total time for all files: {total_time:.2f} seconds.")
