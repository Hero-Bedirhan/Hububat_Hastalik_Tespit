from ultralytics import YOLO

model = YOLO("/home/bedirhan/Desktop/Hububat_Hastalik_Tespit/runs/detect/train/weights/best.pt")
print("Model names:", model.names)

test_image = "/home/bedirhan/Desktop/Hububat_Hastalik_Tespit/Wheat-Disease.v1i.yolov8/test/images/Black-Rust_102_jpg.rf.68f852cee0e38c0dd9990759c26948da.jpg"
results = model(test_image)

for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} boxes")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f"Class: {name} (ID: {cls_id}), Conf: {conf:.2f}")
