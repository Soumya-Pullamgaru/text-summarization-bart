# Abstractive Text Summarization using BART

## Overview
This project implements abstractive text summarization using 
Facebook's BART model, fine-tuned on CNN Daily Mail dataset.

## Model
- **Architecture**: BART (Bidirectional Encoder + Autoregressive Decoder)
- **Pre-trained on**: English language
- **Fine-tuned on**: CNN Daily Mail

## Results
| Metric   | F-Score |
|----------|---------|
| ROUGE-1  | 77%     |
| ROUGE-2  | 64%     |
| ROUGE-L  | 77%     |

## Installation
pip install -r requirements.txt

## How to Run
python src/summarizer.py
python src/evaluate.py

## Technologies Used
- Python
- HuggingFace Transformers
- ROUGE
- Matplotlib
- Google Colab