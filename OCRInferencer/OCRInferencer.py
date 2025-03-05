
'''
Weaveforge component
For use with DNTextSpotter's OCR Inferencing
'''
import logging

import numpy as np
from src.dntextspotter_model_object import TextModel
from weaveforge import WFEventGenerator

logging.basicConfig(level=logging.INFO)

class OCRInferencerComponent(WFEventGenerator):
    '''
    Behavior:
    - Processes one frame at a time
    - For each frame in the video, yields 
    In: 
        Frame - np.ndarray
    Out: 
        x - int (a.k.a. the L in LTRB)
        y - int (a.k.a. the T in LTRB)
        w - int 
        h - int
        Text - str
        class - str
        confidence - float
        current_bb_number - int (basically start from 1 for each new frame coming in)
        max_bb_number - int (total number of bb inferences)
    '''
    def __init__(self):
        self.model = TextModel()

    def process(self, frame: np.ndarray):
        logging.info("Received frame.")
        assert isinstance(frame, np.ndarray)
        frame = np.array(frame).astype(np.uint8)
        frame_result = self.model.single_image_inference(frame)
        max_bb_number = len(frame_result)
        current_bb_number = 0
        class_name = "Text"
        for f in frame_result:
            current_bb_number += 1 # Treat this for loop like a while loop
            x = f['x']
            y = f['y']
            w = f['w']
            h = f['h']
            text = f['Text']
            confidence = f['score']
            yield x, y, w, h, text, class_name, confidence, current_bb_number, max_bb_number
