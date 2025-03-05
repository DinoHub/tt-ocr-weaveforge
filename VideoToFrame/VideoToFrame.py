'''
WeaveForge VideoToFrame component
'''
import cv2
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
        current_frame_number = 0

        max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frames = []
        keyframe_ids = []
        # file_path = bucket + '/' + object_key
        while cap.isOpened():
            ret, frame = cap.read()
            frames.append(frame)
            if not ret:
                break
            current_frame_number = current_frame_number + 1
            ### INSERT YOUR KEYFRAME SHIT HERE IN THE FUTURE
            keyframe_ids.append(current_frame_number)
            yield frame, current_frame_number, max_frame_number, width, height
        cap.release()
