from ultralytics import YOLO
import pprint
model = YOLO('models/detection_model.pt')
pprint.pprint(model.names)
