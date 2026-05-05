import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
import evaluate
from sklearn.model_selection import train_test_split

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)
    
    # استخراج مجموعة الاختبار (نفس التقسيم المستخدم في التدريب)
    labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    
    _, X_test, _, Y_test = train_test_split(
        df['text'], df['label'], test_size=0.20, random_state=42, stratify=df['label']
    )

    # تحميل النموذج والـ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # تجهيز البيانات
    test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": [label2id[y] for y in Y_test]})
    
    def tokenize_function(example):
        return tokenizer(example["text"], padding=False, truncation=True)

    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # التقييم
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(test_tokenized)
    print("\nEvaluation Results on Test Set:")
    print(results)

if __name__ == "__main__":
    evaluate_model("final_cleaned_ready.csv", "./results/final_model")
