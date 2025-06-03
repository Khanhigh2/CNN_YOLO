from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from web3 import Web3
import json
import os
import time

app = Flask(__name__)
CORS(app)

# Đảm bảo thư mục uploads tồn tại
os.makedirs("uploads", exist_ok=True)

# Cho phép truy cập file ảnh tĩnh trong thư mục uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Kết nối blockchain
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
with open("web3/abi.json", "r") as f:
    abi = json.load(f)
contract_address = "0xf2F778356Ca3aCA52acD0275fD2803b195F1829c"
contract = w3.eth.contract(address=contract_address, abi=abi)

# Model YOLO
model = YOLO(r"C:\project_root\cnn\dataset\dataset_yolo\runs\detect\train\weights\best.pt")

OWNER_PRIVATE_KEY = "0xb2d8fcd900eb1cf7a57ce30ded41de794b2838c5a496740c17dc67684db7aa3e"
OWNER_ADDRESS = "0x6911A50D5268639E72B847B230a77432F8683E61"

@app.route("/detect_and_report", methods=["POST"])
def detect_and_report():
    if "image" not in request.files or "supplier_address" not in request.form:
        return jsonify({"error": "Thiếu ảnh hoặc địa chỉ supplier"}), 400

    file = request.files["image"]
    supplier_address = request.form["supplier_address"]
    timestamp = int(time.time() * 1000)
    safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
    save_path = os.path.join("uploads", safe_filename)
    file.save(save_path)

    # Detect lỗi
    results = model(save_path)
    errors = []
    boxes = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            error_type = model.names[cls_id]
            errors.append(error_type)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            boxes.append([x1, y1, w, h])

        # Gửi lỗi lên smart contract
        tx = contract.functions.reportFault(
            supplier_address, errors, f"/uploads/result_{safe_filename}"
        ).build_transaction({
            'from': OWNER_ADDRESS,
            'nonce': w3.eth.get_transaction_count(OWNER_ADDRESS),
            'gas': 3000000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=OWNER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        w3.eth.wait_for_transaction_receipt(tx_hash)

        # Lưu ảnh có bounding box
        result_filename = f"result_{safe_filename}"
        results[0].save(filename=os.path.join("uploads", result_filename))
        result_image_url = f"/uploads/{result_filename}"
    else:
        # Không có lỗi, chỉ lưu ảnh gốc
        result_image_url = f"/uploads/{safe_filename}"

    return jsonify({
        "errors": errors,
        "image_url": result_image_url,
        "boxes": boxes
    })

@app.route("/faults")
def get_faults():
    supplier = request.args.get("supplier")
    if not supplier:
        return jsonify([])

    events = contract.events.ProductFaultReported().get_logs(fromBlock=0, toBlock='latest', argument_filters={"supplier": supplier})
    result = []
    for e in events:
        args = e['args']
        result.append({
            "timestamp": args['timestamp'],
            "errors": args['detectedErrors'],
            "image_url": args['imageUrl']
        })
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
