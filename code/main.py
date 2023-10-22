from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='lidc-idri.yaml', imgsz=512, epochs=10)
results = model.val()
success = model.export(format='onnx')