from transformers import pipeline

def predict(text, model_path="./results/final_model"):
    """
    توقع فئة المنتج بناءً على نص المكونات.
    """
    # إنشاء الـ pipeline
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    
    # الحصول على التوقع
    prediction = classifier(text)[0]
    
    return prediction

if __name__ == "__main__":
    # مثال للاستخدام
    sample_text = "whole milk, sugar, cocoa, natural flavor, vanilla extract"
    result = predict(sample_text)
    
    print(f"\nInput Text: {sample_text}")
    print(f"Predicted Category: {result['label']}")
    print(f"Confidence Score: {result['score']:.4f}")
