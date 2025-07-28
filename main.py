import sys
import os
import json
import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import easyocr
from sklearn.cluster import KMeans
from huggingface_hub import hf_hub_download

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from doclayout_yolo.models.yolov10.model import YOLOv10
except ModuleNotFoundError:
    print("FATAL ERROR: Could not import YOLOv10.")
    sys.exit(1)


def extract_title_and_headings(elements):
    """
    Given a list of detected elements with 'page' and 'y_coord',
    returns the document title (topmost on page 1) and a list of
    remaining headings sorted by page, then vertical position.
    """
    page1 = [el for el in elements if el['page'] == 1]
    if not page1:
        return None, sorted(elements, key=lambda e: (e['page'], e['y_coord']))

    page1_sorted = sorted(page1, key=lambda e: e['y_coord'])
    document_title = page1_sorted[0]['text']
    remaining = [el for el in elements if el is not page1_sorted[0]]
    sorted_headings = sorted(remaining, key=lambda e: (e['page'], e['y_coord']))
    return document_title, sorted_headings


def analyze_and_classify_headings_with_clustering(headings_data, page_height=842):
    if not headings_data:
        return []

    df = pd.DataFrame(headings_data)
    df['y_center'] = df['y_coord'] + df['box_height'] / 2

    # Compute mean and std of box heights
    mu_h = df['box_height'].mean()
    sigma_h = df['box_height'].std(ddof=0) or 1.0  # Avoid zero division

    # Compute heading score
    df['height_z'] = (df['box_height'] - mu_h) / sigma_h
    df['pos_norm'] = 1 - (df['y_center'] / page_height)
    df['score'] = df['height_z'] + df['pos_norm']

    # Sort by score and assign levels
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

    df_sorted = df[df['level'].isin(['H1', 'H2', 'H3'])].sort_values(by=['page', 'y_coord'])

    return df_sorted[['level', 'text', 'page']].to_dict('records')


def process_pdf_to_json(pdf_path, yolo_model, ocr_reader):
    doc = fitz.open(pdf_path)
    all_elements = []
    class_names = yolo_model.names

    for i, page in enumerate(doc):
        page_num = i + 1
        pix = page.get_pixmap(dpi=200)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        results = yolo_model.predict(img_np, conf=0.25, verbose=False)
        for box in results[0].boxes:
            label = class_names[int(box.cls)]
            score = float(box.conf[0])

            is_heading = False
            if label in ["title", "list"]:
                is_heading = True
            elif label == "text":
                box_width = box.xyxy[0][2] - box.xyxy[0][0]
                if box_width > 100:
                    is_heading = True

            if is_heading:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                cropped_img = img_np[y1:y2, x1:x2]
                if cropped_img.size > 0:
                    ocr_result = ocr_reader.readtext(cropped_img, detail=0, paragraph=True)
                    if ocr_result:
                        text = " ".join(ocr_result)
                        if len(text.split()) > 1 and len(text) > 5:
                            all_elements.append({
                                "text": text,
                                "page": page_num,
                                "box_height": y2 - y1,
                                "y_coord": y1,
                                "label": label,
                                "confidence": score
                            })

    doc.close()
    title, sorted_elems = extract_title_and_headings(all_elements)
    outline = analyze_and_classify_headings_with_clustering(sorted_elems)

    return { 'title': title or "Untitled Document", 'outline': outline }


if __name__ == "__main__":
    print("Loading models...")
    try:
        model_path = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
        yolo_model = YOLOv10(model_path)
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("All models loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load models. Error: {e}")
        sys.exit(1)

    INPUT_FOLDER = "1a/temp"
    OUTPUT_FOLDER = "final_json_outputs"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ No PDFs found in folder '{os.path.abspath(INPUT_FOLDER)}'. Please check the path.")
    else:
        for filename in pdf_files:
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.json")
            
            print(f"\n--- Creating JSON for: {filename} ---")
            
            try:
                json_data = process_pdf_to_json(input_path, yolo_model, ocr_reader)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"  ✅ JSON output saved to: {output_path}")

            except Exception as e:
                print(f"❌ An unexpected error occurred while processing {filename}: {e}")

