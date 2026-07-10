import streamlit as st
import cv2
import os
import json
from PIL import Image

# --- تنظیمات مسیرها ---
# آدرس پوشه تصاویر تست روبوفلو را اینجا ست کن
IMAGE_DIR = "./data/detection/raw/test/images" 
LABEL_FILE = "./data/detection/raw/test/test_labels.json"

def load_images():
    return [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def save_label(data):
    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    st.title("🏷️ ابزار برچسب‌گذاری سریع پلاک")
    
    images = load_images()
    
    # لود کردن دیتاهای قبلی برای اینکه از ادامه کار شروع کنی
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    else:
        labels = {}

    # پیدا کردن اولین عکسی که هنوز برچسب نخورده
    current_idx = 0
    for i, img_name in enumerate(images):
        if img_name not in labels:
            current_idx = i
            break
    else:
        st.success("🎉 ایول! تمام ۷۶ عکس برچسب‌گذاری شد.")
        current_idx = len(images) - 1

    # نمایش پیشرفت
    progress = (len(labels) / len(images))
    st.progress(progress)
    st.write(f"پیشرفت: {len(labels)} از {len(images)}")

    if len(labels) < len(images):
        img_name = images[current_idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # نمایش تصویر
        image = Image.open(img_path)
        st.image(image, caption=f"File: {img_name}", use_column_width=True)

        # فرم دریافت متن پلاک
        with st.form("label_form", clear_on_submit=True):
            plate_text = st.text_input("متن پلاک را وارد کنید (مثلا: 12B34567):")
            submitted = st.form_submit_button("ثبت و بعدی ➡️")

            if submitted and plate_text:
                labels[img_name] = plate_text
                save_label(labels)
                st.rerun() # رفرش صفحه برای عکس بعدی

    # دکمه دانلود فایل نهایی
    if labels:
        st.divider()
        st.download_button(
            label="📥 دانلود فایل Label نهایی",
            data=json.dumps(labels, indent=4),
            file_name="test_labels.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()