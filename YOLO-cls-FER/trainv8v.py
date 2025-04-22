
if __name__ == '__main__':

 from ultralytics import YOLO

# Load a model
 model = YOLO('D:/gaijin/ultralytics-main1/ultralytics/cfg/models/v8/yolov8-cls.yaml')  # build a new model from YAML

 #加入预训练权重
 # model = YOLO('D:/gaijin/ultralytics/yolov8n-cls')  # load a pretrained model (recommended for training)
 # model = YOLO('D:/gaijin/ultralytics/ultralytics/cfg/models/v8/yolov8-cls.yaml').load('D:/gaijin/ultralytics/yolov8n-cls')  # build from YAML and transfer weights

# Train the model
 results = model.train(data='D:/gaijin/ultralytics-main1/datasets/fer2013_64', epochs=1000, batch=32, imgsz=64, workers=6)

