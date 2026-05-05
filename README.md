# 🧠 Food Classification using NLP (DistilBERT)

## 📌 Project Overview

This project focuses on building a deep learning model to automatically classify food products based on their ingredient text using Natural Language Processing (NLP).

The system reads raw ingredient lists and predicts the correct food category such as:

* Snacks
* Beverages
* Dairies
* Meals
* Condiments
* And more...

---

## 🎯 Objective

The goal of this project is to:

* Automate food product classification
* Improve data organization for food databases
* Enable smart applications in nutrition and food analysis

---

## 📊 Dataset

* Source: Open Food Facts
* Raw data collected: ~300,000 records
* Cleaned dataset: ~98,000 records
* Final dataset used for training: ~98,349 samples
* Number of categories: 11–15 classes

### 🧾 Data Fields

* `text`: Ingredient list
* `label`: Food category

---

## 🧹 Data Cleaning & Preprocessing

Several preprocessing steps were applied:

* Removed non-food text such as:

  * "Add the ingredients"
  * "Composition minérale"
* Removed Unknown and Null categories
* Removed rare classes (less than 30 samples)
* Removed numbers, symbols, and noise
* Converted all text to lowercase
* Filtered only valid ingredient-like entries

---

## 📈 Exploratory Data Analysis

* Total categories: 11–15
* Most common category: Snacks (~31%)
* Dataset is **imbalanced**
* Addressed using **class weights during training**

---

## ⚙️ Model Architecture

* Model: DistilBERT (`distilbert-base-uncased`)
* Task: Multi-class text classification
* Framework: HuggingFace Transformers

---

## 🏋️ Training Details

* Epochs: 3
* Batch size: 16
* Learning rate: 2e-5
* Loss Function: CrossEntropyLoss with class weights
* Train/Validation/Test split:

  * 70% Training
  * 10% Validation
  * 20% Testing

---

## 📊 Results

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.87  |
| F1 Score | 0.87  |

### 🔍 Observations:

* Strong performance on major classes (Snacks, Dairies, Beverages)
* Lower performance on rare categories
* Model generalizes well on unseen data

---

## 🤖 Model on HuggingFace

👉 https://huggingface.co/Omarrs11/food-classifier-model

---

## 🚀 How to Use the Model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Omarrs11/food-classifier-model"
)

result = classifier("milk, sugar, cocoa butter")
print(result)
```

---

## 🧪 Example

Input:
milk, sugar, cocoa butter

Output:
Dairies (Confidence: ~0.87)

---

## 🖥️ Web Interface (Gradio)

You can run a simple interactive interface:

```python
import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Omarrs11/food-classifier-model"
)

def predict(text):
    result = classifier(text)
    return f"Prediction: {result[0]['label']} | Confidence: {result[0]['score']:.2f}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter ingredients..."),
    outputs="text",
    title="🍔 Food Classification AI",
    description="Enter ingredient text and the model will predict the category."
)

interface.launch()
```

---

## 🛠 Technologies Used

* Python
* HuggingFace Transformers
* PyTorch
* Pandas
* Scikit-learn
* Gradio

---

## 📁 Project Structure

```
food-classifier/
│
├── notebook/
│   └── project.ipynb
│
├── src/
│   ├── cleaning.py
│   ├── training.py
│
├── app.py
├── README.md
├── requirements.txt
```

---

## ⚠️ Challenges

* Noisy and inconsistent data
* Multiple languages in dataset
* Class imbalance
* Data cleaning complexity

---

## 🔮 Future Work

* Improve performance on minority classes
* Use larger models (RoBERTa, DeBERTa)
* Deploy as API or web application
* Expand dataset

---

## 👨‍💻 Author

Omar
AI & NLP Project

---
