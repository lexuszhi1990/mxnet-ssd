# -*- coding: utf-8 -*-

import sys
import cv2
import mxnet as mx
import numpy as np
import random
from pathlib import Path
import time

millisecond = lambda x: int(round(x * 1000))

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    img_path : str
        image path
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    threshold: float
        thresh for scores
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape=300, mean_pixels=(123, 117, 104), threshold=0.2, batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.data_shape = data_shape
        self.threshold = threshold
        self.mean_pixels = mean_pixels
        self.batch_size = batch_size
        self.dets = None
        self.load_symbol, self.args, self.auxs = mx.model.load_checkpoint(
            model_prefix, epoch)
        self.args, self.auxs = self.ch_dev(self.args, self.auxs, self.ctx)

    def ch_dev(self, arg_params, aux_params, ctx):
      new_args = dict()
      new_auxs = dict()
      for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
      for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
      return new_args, new_auxs

    def make_input(self, input_image):
      img = cv2.resize(input_image, (self.data_shape, self.data_shape))
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      img = np.swapaxes(img, 0, 2)
      img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
      img = img[np.newaxis, :]
      return img

    def nms(self, boxes, overlap_threshold, mode='Union'):
        """non max suppression

        Paremeters:
        ----------
        box: numpy array n x 5
            input bbox array, x1,y1,x2,y2,score
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
        Returns:
        -------
            index array of the selected bbox
        """
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        score, x1, y1, x2, y2 = [boxes[:, i+1] for i in range(5)]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(score)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            if mode == 'Min':
                overlap = inter / np.minimum(area[i], area[idxs[:last]])
            else:
                overlap = inter / (area[i] + area[idxs[:last]] - inter)
            idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_threshold)[0])))
        return pick

    def im_detect(self, input_image):
        """
        wrapper for detecting multiple images

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        start = time.clock()
        im_data = self.make_input(input_image)
        print("make inputs costs: %dms" % millisecond(time.clock()-start))

        start = time.clock()
        self.args["data"] = mx.nd.array(im_data, self.ctx)
        exe = self.load_symbol.bind(self.ctx, self.args, args_grad=None,
                                    grad_req="null", aux_states=self.auxs)
        print("bind data  costs: %dms" % millisecond(time.clock()-start))

        start = time.clock()
        exe.forward()
        total_dets = exe.outputs[0][0]
        # https://github.com/apache/incubator-mxnet/issues/6974
        total_dets.wait_to_read()
        print("network forward costs: %dms" % millisecond(time.clock()-start))

        start = time.clock()
        total_dets_np = total_dets.asnumpy()
        selected_dets = total_dets_np[total_dets_np[:, 0] == 1]
        picked_ids = self.nms(selected_dets, overlap_threshold=0.5)
        self.dets = selected_dets[picked_ids]
        print("results post-processing costs: %dms" % millisecond(time.clock()-start))

        return self.dets

    def save_results(self, input_img, frame_num=0, save_path="./", color='red'):

        if len(self.dets) == 0:
            return

        height, width, _ = input_img.shape
        colors = dict()

        for det in self.dets:
            cls_id, score, box = int(det[0]), det[1], det[2:]
            if cls_id not in colors:
                colors[cls_id] = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            left, top = int(box[0] * width), int(box[1] * height)
            right, bottom = int(box[2] * width), int(box[3] * height)
            cv2.rectangle(input_img, (left, top), (right, bottom), colors[cls_id], 1)
            cv2.putText(input_img, '%d:%.3f'%(cls_id,score), (left, top+30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 1)

        det_img_path = Path(save_path, "frame_det_%d.png" % (frame_num))
        if not det_img_path.parent.exists():
            det_img_path.parent.mkdir()
        cv2.imwrite(det_img_path.as_posix(), input_img)
        print("save results at %s" % det_img_path)

def main(*args, **kwargs):

    video_path = args[0]
    # video_path = '../../data/videos/ch01_20180508113155.mp4'
    assert Path(video_path).exists(), "%s not exists" % video_path

    frame_num = 0
    epoch_num = 128
    threshold = 0.65
    data_shape = 300
    ctx = mx.gpu(0)
    # model_prefix = '/app/model/deploy_ssd-densenet-tiny-ebike-detection'
    # model_prefix = '/app/model/deploy_ssd-densenet-two-bikes'
    model_prefix = '/app/model/deploy_deploy_ssd-densenet-tiny-ebike-detection-nms'
    ped_detector = Detector(symbol=None, model_prefix=model_prefix, epoch=epoch_num, threshold=threshold, data_shape=data_shape, ctx=ctx)
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame_num += 1
        if frame_num % 30 == 0:
            # img = cv2.imread(self.img_path)
            start = time.clock()
            ped_detector.im_detect(frame)
            print("total time used: %.4fs" % (time.clock()-start))
            ped_detector.save_results(frame, frame_num, 'noon-video4-test1')

if __name__ == '__main__':
    print("load video from %s" % sys.argv[1])
    main(sys.argv[1])
