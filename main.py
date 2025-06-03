from utils.webcam import capture_image
from cnn.predict_cnn import predict_image
from yolo.detect_yolo import detect
# from web3.send_to_contract import report_defect  ← Bỏ comment nếu dùng smart contract

def main():
    capture_image("captured.jpg")
    label, conf = predict_image("captured.jpg")
    print(f"Sản phẩm nhận dạng: {label} — Độ tin cậy: {conf:.2f}")
    detect("captured.jpg")
    # report_defect(label, "Trầy xước") ← Gửi lỗi nếu phát hiện

if __name__ == "__main__":
    main()
