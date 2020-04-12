import cv2
from keras.models import load_model
import numpy as np
from functools import reduce
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf

from keras.regularizers import l2
from functools import wraps

tf.control_flow_ops = tf


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


class ML_Model():
    def __init__(self):

        self.model = load_model('myapp/yolo/model_data/yolo.h5')

        self.input_w, self.input_h = 416, 416
        self.input_shape = self.input_w, self.input_h
        self.labels = [

            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",

        ]

    @staticmethod
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
            y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union

    def prepare_image(self, frame):
        height, width = frame.shape[0], frame.shape[1]

        return np.expand_dims(cv2.resize(frame, (self.input_w, self.input_h)), 0) / 255, height, width

    @staticmethod
    def get_boxes(boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(labels)):
                # check if the threshold for this label is high enough
                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(labels[i])
                    v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def decode_netout(self, netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])
        netout[..., 4:] = self._sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h * grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                if (objectness.all() <= obj_thresh): continue
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w  # center position, unit: image width
                y = (row + y) / grid_h  # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
        return boxes

    def get_boxes_from_inference(self, image):

        model_input, h, w = self.prepare_image(image)
        yhat = self.model.predict(model_input)

        # summarize the shape of the list of arrays
        # define the anchors
        anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        # define the probability threshold for detected objects

        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += self.decode_netout(yhat[i][0], anchors[i], class_threshold, self.input_h, self.input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        self.correct_yolo_boxes(boxes, h, w, self.input_h, self.input_w)
        # suppress non-maximal boxes
        self.do_nms(boxes, 0.5)
        # define the labels

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = self.get_boxes(boxes, self.labels, class_threshold)

        return v_boxes, v_labels, v_scores

    def draw_results_on_frame(self, frame, v_boxes, v_labels, v_scores):
        for box, label, score in zip(v_boxes, v_labels, v_scores):
            print(box.xmin, box.ymin, box.xmax, box.ymax)
            print(label)
            frame = cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), (100, 100, 0), 5)
            # cv2.imshow('', frame)
            # cv2.waitKey(500)
            frame = cv2.putText(frame, label, (box.xmin, box.ymin), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                                color=(255, 255, 255))
            # frame = cv2.putText(frame,score,(box.xmax,box.ymin),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255,255,255))

        return frame

    def test_camera(self):
        # N.B. : need to run as root in order for camera to open
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        rval, frame = vc.read()
        # frame_count = 850
        while True:

            if frame is not None:
                print(frame.shape)
                boxes, labels, scores = self.get_boxes_from_inference(frame)
                frame = self.draw_results_on_frame(frame, boxes, labels, scores)

                cv2.imshow("preview", frame)

            rval, frame = vc.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict_on_image(self, image):
        boxes, labels, scores = self.get_boxes_from_inference(image)
        frame = self.draw_results_on_frame(image, boxes, labels, scores)
        return frame

# ml = ML_Model()
# img = cv2.imread('image.png')
# out = ml.predict_on_image(img)
# cv2.imwrite('out.png', out)
