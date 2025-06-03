# xóa ảnh lỗi
from PIL import Image
import os

def clean_folder(folder):
    error_count = 0
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                try:
                    img = Image.open(img_path)
                    img.verify()
                except:
                    try:
                        os.remove(img_path)
                        error_count += 1
                    except:
                        pass
    print(f"Đã xóa {error_count} ảnh lỗi.")
