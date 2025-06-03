# Mã này sử dụng mô hình YOLO để phát hiện lỗi trên sản phẩm từ webcam. 
from ultralytics import YOLO
import cv2

# Đặt đường dẫn đến mô hình YOLO đã huấn luyện
model = YOLO("C:/project_root/cnn/dataset/dataset_yolo/runs/detect/train/weights/best.pt")

# Capture ảnh từ webcam
def capture_image(filename="captured.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam không mở được.")
    print("Nhấn phím SPACE để chụp ảnh.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(filename, frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã lưu ảnh thành {filename}")

# Detect lỗi trên sản phẩm
def detect(image_path):
    print("Bắt đầu detect...")
    results = model(image_path)
    print("Detect xong.")
    if len(results[0].boxes) == 0:
        print("Không phát hiện lỗi nào trên sản phẩm.")
    else:
        results[0].show()
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Lỗi: {model.names[cls_id]}, độ tin cậy: {conf:.2f}")

if __name__ == "__main__":
    capture_image("captured.jpg")
    detect("captured.jpg")
