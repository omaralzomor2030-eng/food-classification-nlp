import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
import evaluate

def train_model(data_path, output_dir):
    df = pd.read_csv(data_path)
    
    # تجهيز التصنيفات
    labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # تقسيم البيانات
    X = df['text']
    Y = df['label']
    
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=0.125, random_state=42, stratify=Y_temp
    )

    # التحميل
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # تجهيز Dataset
    train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": [label2id[y] for y in Y_train]})
    val_dataset = Dataset.from_dict({"text": X_val.tolist(), "label": [label2id[y] for y in Y_val]})

    def tokenize_function(example):
        return tokenizer(example["text"], padding=False, truncation=True)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # حساب أوزان الفئات (Class Weights)
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=Y_train)
    weights = torch.tensor([class_weights[label2id[c]] for c in classes]).to(torch.float)

    # مقاييس التقييم
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # إعدادات التدريب
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        report_to="none"
    )

    # إنشاء المدرب
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # بدء التدريب
    trainer.train()
    trainer.save_model(f"{output_dir}/final_model")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_model("final_cleaned_ready.csv", "./results")
