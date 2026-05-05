import time, re, os, requests, pandas as pd
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from IPython.display import display

# -------- إعدادات --------
BASE_URL = "https://world.openfoodfacts.org/facets/countries/united-states"
OUTPUT_CSV = "/content/drive/MyDrive/M/usa_products_auto.csv"
ALL_LINKS_FILE = "/content/all_links_auto.txt"
PROCESSED_LINKS_FILE = "/content/processed_links_auto.txt"
MAX_PRODUCTS_PER_RUN = 3000   # كم منتج لكل تشغيل قبل الحفظ التلقائي
MAX_WORKERS = 10              # عدد الاتصالات المتوازية (كلما زاد، أسرع)
SAVE_BATCH = 200              # حجم الدفعة للحفظ المرحلي
START_PAGE = 3451                # من أين يبدأ (سيتحدث تلقائياً لاحقاً)
# --------------------------

session = requests.Session()
session.headers.update({"User-Agent": "AutoScraper/2.0"})

# --- تحميل التقدم المحفوظ ---
def load_progress():
    all_links, processed_links = set(), set()
    if os.path.exists(ALL_LINKS_FILE):
        with open(ALL_LINKS_FILE, "r") as f:
            all_links = set(line.strip() for line in f if line.strip())
    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, "r") as f:
            processed_links = set(line.strip() for line in f if line.strip())
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=["text", "label"]).to_csv(OUTPUT_CSV, index=False)
    return all_links, processed_links

# --- إعداد Selenium ---
def setup_selenium():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--blink-settings=imagesEnabled=false")
    driver = webdriver.Chrome(options=opts)
    return driver

# --- استخراج بيانات المنتج ---
def parse_product_html(html):
    soup = BeautifulSoup(html, "lxml")

    ingredients = ""
    panel = soup.select_one('#panel_ingredients_content') or soup.select_one('#field_ingredients_value')
    if panel:
        text_element = panel.select_one('div.panel_text') or panel
        for tag in text_element.find_all(['font', 'strong', 'b', 'i']):
            tag.unwrap()
        ingredients = text_element.get_text(" ", strip=True)
        ingredients = re.sub(r'^(Ingredients|Traces)\s*[:\-]?\s*', '', ingredients, flags=re.I).strip()

    if len(ingredients.split()) < 3:
        ingredients = ""

    cats = [a.get_text(" ", strip=True) for a in soup.select("#field_categories_value a")]
    label = cats[0].strip() if cats else "Unknown"

    return {"ingredients": ingredients, "label": label}

# --- جلب رابط منتج (requests) ---
def fetch_and_parse(link):
    try:
        r = session.get(link, timeout=(5, 15))
        if r.status_code != 200:
            return None
        parsed = parse_product_html(r.text)
        if not parsed["ingredients"]:
            return None
        return {"text": parsed["ingredients"], "label": parsed["label"], "url": link}
    except:
        return None

# --- جمع روابط المنتجات من صفحة معينة ---
def collect_links_from_page(driver, page_number):
    page_url = f"{BASE_URL}/{page_number}" if page_number > 1 else BASE_URL
    driver.get(page_url)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, "lxml")
    links = {
        urljoin(BASE_URL, a["href"].split("#")[0])
        for a in soup.select("a[href*='/product/']")
        if a.get("href")
    }
    return links

# --- المرحلة الرئيسية ---
def main():
    all_links, processed_links = load_progress()
    driver = setup_selenium()

    collected_total = len(pd.read_csv(OUTPUT_CSV))
    current_page = START_PAGE

    print(f"🚀 بدء الجمع من الصفحة {current_page}, عدد المنتجات الحالية: {collected_total}")

    while collected_total < 200000:  # الهدف النهائي الكبير
        print(f"\n🟢 جمع روابط الصفحة {current_page}")
        new_links = collect_links_from_page(driver, current_page)
        if not new_links:
            print("⚠️ لا توجد منتجات إضافية، يتم الإيقاف.")
            break

        all_links.update(new_links)
        with open(ALL_LINKS_FILE, "w") as f:
            for link in all_links:
                f.write(f"{link}\n")

        links_to_process = list(all_links - processed_links)
        print(f"🔗 سيتم معالجة {len(links_to_process)} رابط جديد من الصفحة {current_page}")

        data_to_append = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_and_parse, link): link for link in links_to_process}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                link = futures[future]
                processed_links.add(link)
                with open(PROCESSED_LINKS_FILE, "a") as f:
                    f.write(f"{link}\n")
                if result:
                    data_to_append.append({"text": result["text"], "label": result["label"]})
                    collected_total += 1
                    if collected_total % 100 == 0:
                        print(f"✅ {collected_total} منتج حتى الآن.")
                if len(data_to_append) >= SAVE_BATCH:
                    pd.DataFrame(data_to_append).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                    data_to_append = []

                if collected_total % MAX_PRODUCTS_PER_RUN == 0:
                    print("💾 حفظ دفعة تشغيل كاملة.")
                    pd.DataFrame(data_to_append).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                    data_to_append = []
                    break

        if data_to_append:
            pd.DataFrame(data_to_append).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

        current_page += 1
        print(f"➡️ الانتقال للصفحة التالية ({current_page})")

    driver.quit()
    print(f"\n🎯 تم الانتهاء. مجموع المنتجات المجمعة: {collected_total}")

main()