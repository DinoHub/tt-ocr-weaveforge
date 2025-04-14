'''
WeaveForge VideoToFrame component
'''
import cv2
import numpy as np
from minio import Minio
from weaveforge import WFEventGenerator

class VideoToFrame(WFEventGenerator):
    '''
    Inputs:
        bucket name: String
        object key: String
    Outputs:
        Frame: nparray
        Current Frame Number (for future keyframe indication use & also for json id tracking): Int
        Max Frame Number (Because BY wants this): Int
        Width: Int
        Height: Int
    '''
    def __init__(self):
        self.client = Minio(
            self.get_component_configurations()['MINIO_URL'],
            access_key = self.get_component_configurations()['MINIO_ACCESS_KEY'],
            secret_key = self.get_component_configurations()['MINIO_SECRET_KEY'],
            secure = False
        )
        self.batch_size = self.get_component_configurations()['BATCH_SIZE']

    def batch_1_routine(self, cap):
        keyframe_ids = []
        current_frame_number = 0
        max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_number += 1
            ### INSERT YOUR KEYFRAME SHIT HERE IN THE FUTURE
            keyframe_ids.append(current_frame_number)
            yield frame, [current_frame_number], max_frame_number, width, height
    
    def batch_max_routine(self, cap):
        frames = []
        max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        while cap.isOpened():
            ret, frame = cap.read()
            frames.append(frame)
            if not ret:
                break
        frames = np.stack(frames)
        keyframe_ids = list(range(1,len(frames)+1,1))
        return frames, keyframe_ids, max_frame_number, width, height
    
    def custom_batch_routine(self, cap, batch_size):
        frames = []
        frame_ids = []
        max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        while cap.isOpened():
            ret, frame = cap.read()
            frames.append(frame)
            current_frame_number += 1
            frame_ids.append(current_frame_number)
            if len(frames) == batch_size:
                frames = np.stack(frames)
                yield frames, frame_ids, max_frame_number, width, height
                frames = []
                frame_ids = []
            if not ret:
                if frames:
                    yield frames, frame_ids, max_frame_number, width, height
                    frames = []
                    frame_ids = []
                break

    def generate(self, bucket:str, object_key:str):
        '''
        client code is for testing
        '''
        # self.client = Minio(
        #     'localhost:9000', 
        #     access_key = 'EftVSRmijXRsItFf2zcb',
        #     secret_key = 'CMuwmyhht7O766VsbpY68WTbystVhJ43AF6CqQnt',
        #     secure=False
        # )
        obj = self.client.presigned_get_object(bucket, object_key)
        cap = cv2.VideoCapture(obj)
        if self.batch_size == 1:
            for output in self.batch_1_routine(cap):
                yield output
        elif self.batch_size == -1:
            return self.batch_max_routine(cap)
        else:
            for output in self.custom_batch_routine(cap, self.batch_size):
                yield output
        cap.release()
