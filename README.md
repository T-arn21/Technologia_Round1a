
# 📄 PDF Heading Extractor - Adobe “Connecting the Dots” Challenge (Round 1A)

## 🔍 Task Overview

As part of Round 1A of the [Adobe India Hackathon - Connecting the Dots Challenge](https://github.com/jhaaj08/Adobe-India-Hackathon25.git), our goal is to **extract structured outlines** from raw PDF files.

This includes identifying:

- **Document Title**
- **Headings** at three levels: `H1`, `H2`, and `H3`
- For each heading: text, level, and page number

## 🧠 Our Approach

We designed a CPU-only, offline, parallelized system that extracts a structured outline from a PDF using the following key components:

### 1. **Document Layout Detection with `doclayout-yolov10`**
- We use `YOLOv10`, a lightweight and fast object detection model, trained for document layout analysis.
- The model identifies **text box bounding boxes** (with coordinates) across PDF pages.

### 2. **Text Recognition with `EasyOCR`**
- Once boxes are detected, `EasyOCR` is applied on each box to extract **text content**.
- It supports multilingual PDFs and works offline — complying with the hackathon constraints.

### 3. **Heading Classification**
To classify text boxes into headings:
- We use **box height** and **average Y coordinate** as features.
- Larger boxes higher on the page are likely to be `H1`, followed by `H2` and `H3`.

### 4. **Multithreaded File Processing**
- We use **thread pools** (`ThreadPoolExecutor`) to process multiple PDF files in parallel.
- This improves throughput significantly while staying within the 10-second execution time constraint.

### 5. **Output Format**
The output is a structured JSON in the format:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## 🛠️ Setup Instructions

### 🔧 Prerequisites
Ensure you have Docker installed on your system. The solution runs in a CPU-only AMD64 container with:

- No internet access
- ≤ 200MB model size
- ≤ 10s for 50-page PDFs

---

## 🚀 How to Build and Run

### 🐳 Build Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### 📂 Expected Folder Structure

```
project/
├── input/
│   ├── document1.pdf
│   └── document2.pdf
├── output/
│   └── (Generated JSON files will be saved here)
├── Dockerfile
├── main.py
├── models/
├── README.md
└── requirements.txt
```

### ▶️ Run the Container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor:latest
```

This will:
- Process all PDFs from `/app/input`
- Generate structured outlines as JSON in `/app/output`

---

## 📚 Libraries and Models Used

| Component         | Tool Used         | Purpose                             |
|------------------|-------------------|-------------------------------------|
| Layout Detection | YOLOv10 (doclayout) | Detects text regions on PDF pages   |
| OCR              | EasyOCR           | Extracts text from image regions    |
| Parsing Engine   | PyMuPDF (`fitz`)  | Reads and renders PDF pages         |
| Threading        | ThreadPoolExecutor| Parallel file processing            |
| Clustering       | KMeans (optional) | For heading level clustering        |

---

## ✅ Compliance Summary

| Constraint              | Status         |
|-------------------------|----------------|
| CPU-only                | ✅ Met          |
| ≤ 200MB model           | ✅ Met          |
| Execution Time ≤ 10s    | ✅ Met (for 50-page PDFs) |
| Works Offline           | ✅ Met          |
| No Hardcoded Logic      | ✅ Met          |

---

## 🌐 Multilingual Support

- EasyOCR supports multilingual text recognition.
- You can preload languages for faster OCR performance.

---

## ✍️ Final Notes

- The heading classification logic adapts to PDF variability using geometric features (not font size alone).
- The entire pipeline is modular, scalable, and optimized for Round 1B as well.

---

## 📎 Sample Output

```json
{
  "title": "Connecting the Dots",
  "outline": [
    { "level": "H1", "text": "Round 1A: Understand Your Document", "page": 1 },
    { "level": "H2", "text": "Your Mission", "page": 1 },
    { "level": "H3", "text": "Why This Matters", "page": 2 }
  ]
}
```
