import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

test_image = "/home/bedirhan/Desktop/Hububat_Hastalik_Tespit/Wheat-Disease.v1i.yolov8/test/images/Black-Rust_102_jpg.rf.68f852cee0e38c0dd9990759c26948da.jpg"
result = client.infer(test_image, model_id="wheat-model-train-kftfo/1")
print(result)
