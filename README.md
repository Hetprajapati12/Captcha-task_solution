# CAPTCHA Solver

An automated CAPTCHA solver for 2captcha.com demo using Python, Playwright, and Tesseract OCR.

## Overview

This project demonstrates an automated approach to solving text-based CAPTCHAs using:
- **Playwright** for browser automation
- **Tesseract OCR** for optical character recognition
- **OpenCV** for image preprocessing
- **Smart candidate generation** with character substitution algorithms

## Features

- Automated browser navigation to 2captcha.com demo page
- CAPTCHA image extraction and preprocessing
- Red text extraction from noisy backgrounds
- Intelligent OCR with multiple PSM modes
- Character substitution based on common OCR confusions
- Smart candidate prioritization using pattern matching
- Automated form submission and result verification

## Requirements

- Python 3.9+
- Tesseract OCR 5.0+
- Chromium browser (installed via Playwright)

## Installation

### 1. Clone/Download the project
```bash
cd captcha_solver
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Install Playwright browsers
```bash
playwright install chromium
```

### 6. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to: `C:\Program Files\Tesseract-OCR\`
- The path is already configured in the script

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Usage

### Run the solver
```bash
python captcha3.py
```

### Configuration

Edit the following variables in `captcha3.py`:
```python
HEADLESS = False  # Set to True to run browser in background
```

## How It Works

### 1. Image Preprocessing
- Extracts red text from the CAPTCHA image using color masking
- Converts to binary (black text on white background)
- Upscales image for better OCR accuracy

### 2. OCR Processing
- Uses Tesseract with multiple Page Segmentation Modes (PSM)
- Applies character whitelist (A-Z, 0-9)
- Collects all possible OCR readings

### 3. Candidate Generation
- Normalizes OCR results to 5-character strings
- Applies character substitution based on common OCR confusions:
  - `O` ↔ `0`, `9`
  - `S` ↔ `5`, `H`, `K`
  - `W` ↔ `V`, `9`
  - `R` ↔ `H`, `K`
  - And many more...

### 4. Smart Prioritization
- Prioritizes candidates matching typical CAPTCHA patterns
- Pattern LNLNL (Letter-Number-Letter-Number-Letter) scored highest
- Further sub-scoring based on likely character positions

### 5. Automated Testing
- Submits candidates to the form automatically
- Detects success/failure messages
- Continues until correct answer found

## Project Structure
```
captcha_solver/
├── captcha3.py          # Main solver script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── output/
    └── debug/
        └── extracted.png  # Debug image output
```

## Output Example
```
[OCR] Tesseract: 5.5.0.20241111
[Browser] Started
==================================================
SOLVING CAPTCHA
==================================================
[Browser] CAPTCHA loaded (28830 bytes)
[OCR] Raw results: ['VWSRSIK', 'WWSRSK', 'WWSS']
[OCR] Generated 25 5-char bases
[OCR] Generated 34940 total candidates
[OCR] 'W9H5K' found at position 0
[Solver] Testing 34940 candidates...
  Progress: 0/34940 (0%)
  SUCCESS: 'W9H5K' (#1 of 34940)
============================================================
 SUCCESS! Answer: 'W9H5K'
============================================================
[Browser] Stopped
```

## Technical Details

### Character Substitution Map

The solver uses a comprehensive substitution map to handle common OCR misreadings:

| OCR Reads | Might Actually Be |
|-----------|-------------------|
| O | 0, 9, Q, D |
| S | 5, 8, H, K |
| W | V, M, U, 9 |
| R | H, K, A, P, 5 |
| I | 1, L, J, 5, K |
| G | 9, 6, C, Q |
| B | 8, 3, D, R |

### Scoring Algorithm

Candidates are scored based on pattern matching:
1. **Score 0**: LNLNL pattern (e.g., W9H5K) with optimal characters
2. **Score 1**: 3 letters + 2 numbers
3. **Score 2**: Letters at start/end with numbers in middle
4. **Score 3**: Mixed letters and numbers (2-4 letters, 1-3 numbers)
5. **Score 4**: Any mix of letters and numbers
6. **Score 5**: All letters or all numbers

## Limitations

- Designed specifically for 2captcha.com demo CAPTCHAs
- Requires visible browser window for best results (non-headless)
- Success rate depends on CAPTCHA complexity
- Server-side caching may result in same CAPTCHA being served

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| playwright | 1.49.1 | Browser automation |
| pytesseract | 0.3.13 | OCR interface |
| opencv-python | 4.10.0.84 | Image processing |
| numpy | >=1.24.0,<2.0.0 | Array operations |
| Pillow | 11.0.0 | Image handling |

