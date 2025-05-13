---

# Vietnamese Sentence Spelling Correction

This project implements a deep learning-based model for Vietnamese sentence-level spelling correction. The system detects and corrects spelling mistakes in Vietnamese text using xlm-roberta-base architectures, ensuring high accuracy and fluency in the corrected output. It is suitable for applications in document processing, chatbots, and language education tools.

## Overview

The pipeline consist of 2 parts:
- TASK 1: Error Detection
  - Train a Token Classification model to identify and label spelling errors in sentences.

- TASK 2: Spelling Correction
  - Train Transformer-based model for MLM task to extract semantic meaning. Take top 10 predictions and select the best match for the misspelled word.

---

## Dataset

- Paragraphs with random errors (remove spaces, change punctuation, lowercase, uppercase, insert symbols, etc.).

---

## Requirements

The following libraries and frameworks are required for running the project:

* Python 3.10
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* h5py
* tqdm
* wandb
  ...

To install these dependencies, you can use the provided `requirements.txt` file.

### Example:

```
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/sonho4ng/Sentence-Spelling-Correction.git](https://github.com/sonho4ng/Sentence-Spelling-Correction
   cd Sentence-Spelling-Correction
   ```

2. **Install dependencies:**

   If you're using a `requirements.txt` file, run:

rewrite the description and overview so that it is suitable for this project'
