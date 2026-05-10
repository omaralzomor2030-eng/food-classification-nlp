#  Food Classification using NLP (DistilBERT)

##  Overview

This project builds an intelligent Natural Language Processing (NLP) system that classifies food products based on their ingredient text.

Given a list of ingredients, the model predicts the correct food category such as:

* Snacks
* Beverages
* Dairies
* Meals
* Condiments
* And more...

The system is powered by a fine-tuned DistilBERT model using the HuggingFace Transformers library.

---

##  Objectives

* Automate food product categorization
* Handle noisy real-world ingredient data
* Build a scalable NLP classification pipeline
* Deploy a reusable and testable model

---

## 📊 Dataset

* Source: Open Food Facts
* Raw records: ~300,000
* Cleaned dataset: ~98,000 samples
* Final classes: 11–15 categories

### 🧾 Data Format

| Column | Description      |
| ------ | ---------------- |
| text   | Ingredients list |
| label  | Food category    |

---

##  Data Preprocessing

The dataset required extensive cleaning due to noise and inconsistencies.

### Key Steps:

* Convert text to lowercase
* Remove numbers, symbols, and extra spaces
* Remove invalid entries (e.g., "add the ingredients")
* Filter non-food text
* Remove `Unknown` and `Null` labels
* Remove rare categories (< 30 samples)
* Validate ingredient-like content using keyword filtering

---

## ⚙️ Model Architecture

* Model: `distilbert-base-uncased`
* Task: Multi-class text classification
* Framework: HuggingFace Transformers
* Backend: PyTorch

---

##  Training Setup

* Epochs: 3
* Batch size: 16
* Learning rate: 2e-5
* Weight decay: 0.01
* Loss: CrossEntropy (with class weights)
* Train/Validation/Test split:

  * 70% Training
  * 10% Validation
  * 20% Testing

---

## 📈 Results

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.87  |
| F1 Score | 0.87  |

### 🔍 Notes:

* Strong performance on major categories (Snacks, Dairies, Beverages)
* Lower performance on rare classes
* Class imbalance handled using weighted loss

---

## 🤖 Model on HuggingFace

👉 https://huggingface.co/Omarrs11/food-classifier-model

---

##  Quick Test

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Omarrs11/food-classifier-model"
)

classifier("milk, sugar, cocoa")
```

---

## 🧪 Example

**Input:**

```
milk, sugar, cocoa butter
```

**Output:**

```
Dairies (Confidence: ~0.87)
```

---

## 🖥️ Web Interface (Gradio)

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Omarrs11/food-classifier-model"
)

def predict(text):
    result = classifier(text)
    return f"{result[0]['label']} ({result[0]['score']:.2f})"

gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="🍔 Food Classifier",
    description="Enter ingredients to classify food category"
).launch()
```

---

## 📁 Project Structure

```
food-classification-nlp/
│
├── README.md
├── requirements.txt
│
├── notebook/
│   └── project.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── scraping.py
│
├── data/
│   └── sample_dataset.csv
│
├── app.py
```

---

## ⚡ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Preprocess data

```
python src/preprocess.py
```

### 3. Train the model

```
python src/train.py
```

### 4. Evaluate the model

```
python src/evaluate.py
```

### 5. Run inference

```
python src/inference.py
```

---

## ⚠️ Challenges

* Noisy and inconsistent ingredient data
* Multi-language entries
* Severe class imbalance
* Data cleaning complexity

---

## 🔮 Future Improvements

* Improve minority class performance
* Use larger transformer models (RoBERTa, DeBERTa)
* Deploy as API (FastAPI / Flask)
* Add real-time web interface
* Expand dataset size

---

## 🛠 Technologies Used

* Python
* HuggingFace Transformers
* PyTorch
* Pandas
* Scikit-learn
* Gradio

---

## 👨‍💻 Author

Omar

---

## ⭐ If you found this project useful, consider giving it a star!
