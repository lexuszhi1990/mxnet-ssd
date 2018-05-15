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
    def __init__(self, symbol, model_prefix, epoch, data_shape=300, img_path=None, mean_pixels=(123, 117, 104), threshold=0.2, batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.data_shape = data_shape
        self.threshold = threshold
        self.mean_pixels = mean_pixels
        self.batch_size = batch_size
        self.img_path = img_path
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

    def make_input(self):
      img = cv2.imread(self.img_path)
      img = cv2.resize(img, (self.data_shape, self.data_shape))
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
        x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
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

    def im_detect(self):
        """
        wrapper for detecting multiple images

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        start = time.clock()
        im_data = self.make_input()
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

        # self.dets = [output.asnumpy() for output in total_dets if output[0] == 0]
        start = time.clock()
        self.dets =  [{"bbox": det[2:].asnumpy().tolist(),
                      "score": det[1].asnumpy()[0].astype(float),
                      "category_id": det[0].asscalar()
                      } for det in total_dets if det[1] >= self.threshold and det[0] >= 0]
        print("results post-processing costs: %dms" % millisecond(time.clock()-start))

        return self.dets

    def save_results(self, save_path="./", color='red'):
      draw = cv2.imread(self.img_path)
      height, width, _ = draw.shape
      colors = dict()

      for det in self.dets:
        box = det["bbox"]
        score = det["score"]
        cls_id = det["category_id"]
        if cls_id not in colors:
          colors[cls_id] = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
        left, top = int(box[0] * width), int(box[1] * height)
        right, bottom = int(box[2] * width), int(box[3] * height)
        cv2.rectangle(draw, (left, top), (right, bottom), colors[cls_id], 1)
        cv2.putText(draw, '%.3f'%score, (left, top+30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 1)

      img_path = Path(self.img_path)
      save_path = img_path.parent.joinpath(img_path.stem + '_result.png')
      cv2.imwrite(save_path.as_posix(), draw)
      print("save results at %s" % save_path)

def main(*args, **kwargs):
    img_path = args[0]
    if not Path(img_path).exists():
      print(img_path+' image not exists')
      return

    epoch_num = 128
    threshold = 0.65
    data_shape = 300
    ctx = mx.gpu(0)
    # model_prefix = '/app/model/deploy_ssd-densenet-tiny-ebike-detection'
    # model_prefix = '/app/model/deploy_ssd-densenet-two-bikes'
    model_prefix = '/app/model/deploy_deploy_ssd-densenet-tiny-ebike-detection-nms'

    start = time.clock()
    ped_detector = Detector(symbol=None, model_prefix=model_prefix, epoch=epoch_num, threshold=threshold, img_path=img_path, data_shape=data_shape, ctx=ctx)
    ped_detector.im_detect()
    print("total time used: %.4fs" % (time.clock()-start))
    ped_detector.save_results()

if __name__ == '__main__':
    print("argv[]="+sys.argv[0]+" "+sys.argv[1])
    main(sys.argv[1])
