# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
DNTextSpotter model object, 
to be unit-tested via TT-Testing-Outputs-DNTextSpotter.py
OR used alongside OCRInferencer-DNTextSpotter for Weaveforge
'''
import argparse
import atexit
import bisect
import sys
import multiprocessing as mp
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.data.transforms import Augmentation, PadTransform
from adet.utils.visualizer import TextVisualizer
from adet.config import get_cfg
from adet.modeling import vitae_v2
from pathlib import Path

sys.path.insert(0, '../adet')

class Pad(Augmentation):
    '''
    Class taken from elsewhere to siam the imports
    '''
    def __init__(self, divisible_size = 32):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        ori_h, ori_w = img.shape[:2]  # h, w
        if ori_h % 32 == 0:
            pad_h = 0
        else:
            pad_h = 32 - ori_h % 32
        if ori_w % 32 == 0:
            pad_w = 0
        else:
            pad_w = 32 - ori_w % 32
        # pad_h, pad_w = 32 - ori_h % 32, 32 - ori_w % 32
        return PadTransform(
            0, 0, pad_w, pad_h, pad_value=0
        )

class TextModel():
    """
    Args:
        cfg (CfgNode):
        instance_mode (ColorMode):
        parallel (bool): whether to run the model in different 
            processes from visualization.
            Useful since the visualization logic can be slow.
    """
    def __init__(self, instance_mode=ColorMode.IMAGE, parallel=False):
        self.args = self.get_parser().parse_args()
        self.cfg = self.setup_cfg(self.args)
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if self.cfg.DATASETS.TEST else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = self.cfg.MODEL.TRANSFORMER.ENABLED

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(self.cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(self.cfg)
        if self.cfg.MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
            self.predictor = ViTAEPredictor(self.cfg)

    def get_parser(self):
        '''
        Prepare standard parser
        '''
        parser = argparse.ArgumentParser(description="Detectron2 Demo")
        parser.add_argument(
            "--config-file",
            default="./src/configs/ViTAEv2_S/pretrain/150k_tt_mlt_13_15_textocr.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--webcam", 
            action="store_true",
            help="Take inputs from webcam.")
        parser.add_argument(
            "--output",
            help="A file or directory to save output visualizations. "
            "If not given, will show output in an OpenCV window.",
            default="./results/"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.3,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[
                'MODEL.WEIGHTS',
                './assets/vitaev2_pretrain_tt_model_final.pth'],
            nargs=argparse.REMAINDER,
        )
        return parser

    def setup_cfg(self, args):
        '''
        load config from file and command-line arguments
        '''
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg

    @classmethod
    def _ctc_decode_recognition(cls, rec, voc_size=37):
        if voc_size == 37:
            ct_labels = [
                'a','b','c','d','e','f','g','h','i',
                'j','k','l','m','n','o','p','q','r',
                's','t','u','v','w','x','y','z','0',
                '1','2','3','4','5','6','7','8','9']
            last_char = '-'
            s = ''
            for c in rec:
                c = int(c)
                if c < voc_size - 1:
                    if last_char != c:
                        s += ct_labels[c]
                        last_char = c
                else:
                    last_char = '-'
            s = s.replace('-', '')
        elif voc_size == 96:
            ct_labels = [
                ' ','!','"','#','$','%','&','\'',
                '(',')','*','+',',','-','.','/',
                '0','1','2','3','4','5','6','7','8',
                '9',':',';','<','=','>','?','@','A',
                'B','C','D','E','F','G','H','I','J',
                'K','L','M','N','O','P','Q','R','S',
                'T','U','V','W','X','Y','Z','[','\\',
                ']','^','_','`','a','b','c','d','e','f',
                'g','h','i','j','k','l','m','n','o','p',
                'q','r','s','t','u','v','w','x','y','z',
                '{','|','}','~']
            last_char = '###'
            s = ''
            for c in rec:
                c = int(c)
                if c < voc_size - 1:
                    if last_char != c:
                        s += ct_labels[c]
                        last_char = c
                else:
                    last_char = '###'

        else:
            raise NotImplementedError
        return s

    def single_image_inference(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # image = read_image(image, format="RGB")
        predictions = self.predictor(image)["instances"].to(self.cpu_device)
        recs = predictions.recs
        scores = predictions.scores.tolist()
        bd_pnts = predictions.bd.numpy()
        results = []
        for bd, rec, score in zip(bd_pnts, recs, scores):
            box_res = {
                'x': None,
                'y': None,
                'w': None,
                'h': None,
                'Text': None,
                'score': None
            }
            text = [self._ctc_decode_recognition(rec)][0]
            bd = np.hsplit(bd,2)
            bd = np.vstack([bd[0], bd[1][::-1]])
            box_res['polygon'] = [tuple(int(p) for p in pair) for pair in bd]
            bd_x = bd[:,0]
            bd_y = bd[:,1]
            width = int(max(bd_x) - min(bd_x))
            height = int(max(bd_y) - min(bd_y))
            box_res['x'] = int(min(bd_x))
            box_res['y'] = int(min(bd_y))
            box_res['w'] = int(width)
            box_res['h'] = int(height)
            box_res['Text'] = text
            box_res['bbox'] = [box_res['x'], box_res['y'], box_res['w'], box_res['h']]
            box_res['score'] = score
            results.append(box_res)
        return results

    def video_inference(self, vid_path):
        cap = cv2.VideoCapture(Path(vid_path))
        frame_gen = self._frame_from_video(cap)
        res = []
        for frame in frame_gen:
            res.append(self.single_image_inference(frame))
        return res

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(
                image, self.metadata, instance_mode=self.instance_mode,
                cfg=self.cfg)
        else:
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)

        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def vis_bases(self, bases):
        '''
        some visualization function
        '''
        basis_colors = [
            [2, 200, 255], [107, 220, 255],
            [30, 200, 255], [60, 220, 255]
            ]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        _, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros(
                (basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object,
                whose source can be either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vis_frame = None
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = f"cuda:{gpuid}" if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue,
                self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        '''
        self-explanatory but pylint wants me to write shiet
        '''
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        '''
        self-explanatory but pylint wants me to write shiet
        '''
        self.get_idx += 1  # the index needed for this request
        if self.result_rank and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        '''
        self-explanatory but pylint wants me to write shiet
        '''
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        '''
        self-explanatory but pylint wants me to write shiet
        '''
        return len(self.procs) * 5


class ViTAEPredictor:
    '''
    ViTAEPredictor is a Predictor that uses ViTAE
    '''
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): 
            an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = self.pad.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
