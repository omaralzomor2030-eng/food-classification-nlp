import pandas as pd
import re

def clean_text(text):
    """تنظيف نص المكونات من الرموز والأرقام والأقواس والمسافات."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)        # إزالة الأقواس وما داخلها
    text = re.sub(r"[\d%*:\-]", "", text)        # إزالة الأرقام والرموز
    text = re.sub(r"\s*,\s*", ", ", text)        # توحيد الفواصل
    text = re.sub(r"\s+", " ", text).strip()     # إزالة المسافات الزائدة
    return text

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # 1. حذف السجلات غير الغذائية
    bad_keywords = [
        "composition minérale", "résidu sec", "fabrique par",
        "no eco", "dalaa distribue", "bottled by", "non food"
    ]
    mask_non_food = df["text"].str.contains("|".join(bad_keywords), case=False, na=False)
    df = df[~mask_non_food]

    # 2. حذف النصوص القصيرة جدًا
    mask_too_short = df["text"].str.len() < 15
    df = df[~mask_too_short]

    # 3. حذف النصوص غير المفيدة
    bad_texts = [
        "could you add the ingredients",
        "add the ingredients",
        "no ingredients found",
        "ingredients to be added"
    ]
    mask_bad_text = df["text"].str.contains("|".join(bad_texts), case=False, na=False)
    df = df[~mask_bad_text]

    # 4. حذف التصنيفات غير المعروفة
    mask_unknown = df["label"].isin(["Undefined", "Null", "unknown", "none", "Unknown"])
    df = df[~mask_unknown]

    # 5. حذف الفئات النادرة (أقل من 2000 سجل)
    label_counts = df["label"].value_counts()
    rare_labels = label_counts[label_counts < 2000].index
    df = df[~df["label"].isin(rare_labels)]

    # 6. تنظيف النصوص
    df["text"] = df["text"].apply(clean_text)

    # 7. التحقق من وجود كلمات مكونات غذائية أساسية
    keywords = [
        "milk", "sugar", "oil", "flour", "salt", "water", "butter", "cheese",
        "wheat", "egg", "corn", "yeast", "cocoa", "honey", "soy", "tomato",
        "garlic", "onion", "meat", "fish", "pepper", "cream", "fruit", "vegetable"
    ]
    mask_has_ingredients = df["text"].str.contains("|".join(keywords), case=False, na=False)
    df = df[mask_has_ingredients]

    # حفظ البيانات النهائية
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Final records: {len(df)}")

if __name__ == "__main__":
    preprocess_data("merged_file.csv", "final_cleaned_ready.csv")
