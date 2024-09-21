import cv2
import numpy as np
import onnxruntime as ort

from nms import non_max_suppression
from processing import letterbox, scale_coords


class YOLOv5:
    
    def __init__(self, path, conf_thres=0.65, iou_thres=0.2, stride=32):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.session = ort.InferenceSession(path,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])
        
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.stride = stride
        
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = letterbox(img, self.input_size, stride=self.stride, auto=False)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = img[np.newaxis, :, :, :].astype(np.float32)
        return img
    
    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def postprocess(self, outputs):
        predictions = outputs[0]
        predictions = non_max_suppression(predictions, self.conf_thres, self.iou_thres, classes=0, agnostic=True)
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        for idx, det in enumerate(predictions):
            if len(det):
                det[:, :4] = scale_coords(self.input_size, det[:, :4], self.img_size).round()
                scores = det[:, 4]
                class_ids = det[:, 5]
                all_boxes.append(det[:, :4].astype("int"))
                all_scores.append(scores)
                all_class_ids.append(class_ids.astype("int"))
            else:
                all_boxes.append(np.array([]))
                all_scores.append(np.array([]))
                all_class_ids.append(np.array([]))
        return all_boxes, all_scores, all_class_ids
    
    def __call__(self, image):
        self.img_size = (image.shape[0], image.shape[1])
        
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        boxes, scores, class_ids = self.postprocess(outputs)
        
        # bs = 1
        boxes = boxes[0].tolist()
        scores = scores[0].tolist()
        class_ids = class_ids[0].tolist()
        return boxes, scores, class_ids
