import torch
import pdb
from ultralytics import YOLO



#model_t = YOLO('yolov8l.pt')
model_t = YOLO('/runs/detect/train9/weights/best.pt')
data = 'apple.yaml'
#model_t.model.model[-1].set_Distillation = True
#model_t.train(data=data, epochs=10, device='1', batch=8, imgsz=640, Distillation=None)

model_s = YOLO('yolov8n.pt')
# modelL.val(data=data)
model_s.train(data=data, epochs=10, device='1', batch=8,imgsz=640, Distillation=model_t.model)

#model_s.train(data=data, epochs=10, device='1', batch=8,imgsz=640, Distillation=None)


