import redis
import logging

class RedisTest():
    def __init__(self) -> None:
        with redis.Redis(decode_responses=True,
                        host='127.0.0.1',
                        port='9001',
                        retry=redis.retry.Retry(backoff=redis.backoff.ConstantBackoff(1),
                        retries=3),
                        retry_on_error=[ConnectionError]) as r:
            self.r = r

    def process_image(self,
                bucket_name: str,
                object_key: str,
                width: int,
                height: int,
                current_frame_number: int,
                max_frame_number: int):
        '''Event loop to write sapphire data to Redis hash.

        Input Arguments:
            bucket_name: str
                the name of the bucket where the S3 object is located
            object_key: str
                the key of the S3 object
            AAAAAAAAA

        Output Arguments:
            -
        '''
        key = ':'.join([bucket_name, object_key, 'i'+str(current_frame_number), 'i'+str(max_frame_number)])
        img_anno = {
            'file_name': f'frame_{current_frame_number:06}.jpg',
            'height': int(height),
            'width': int(width),
            'id': int(current_frame_number)
        }
        self.r.hmset(key, mapping=img_anno)
        logging.info(f"{key} {str(key)}")
        
    def process_anno(self,
                    bucket_name: str,
                    object_key: str,
                    x: int,
                    y: int,
                    w: int,
                    h: int,
                    text: str,
                    score: float,
                    category_id: int,
                    current_frame_number: int,
                    max_frame_number: int,
                    current_bb_number: int,
                    max_bb_number: int):
        '''
        Event loop to write COCO annotation data to Redis hash.

        Input Arguments:
            bucket_name: str
                the name of the bucket where the S3 object is located
            object_key: str
                the key of the S3 object
            x: int
            y: int
            w: int
            h: int
            text: str
            score: float
            category_id: int
            current_frame_number: int
            max_frame_number: int
            current_bb_number: int
            max_bb_number: int

        Output Arguments:
            -
        '''
        key = ':'.join([
            bucket_name, object_key, 'i'+str(current_frame_number), 
            'i'+str(max_frame_number), 'a'+str(current_bb_number), 'a'+str(max_bb_number)
        ])
        anno = {
            'image_id': current_frame_number,
            'bbox': [x,x+w,y,y+h],
            'attributes': {
                'Text': text
            },
            'score': score,
            'category_id': category_id,
            'id': None
        }
        self.r.hmset(key, mapping=anno)
        logging.info(f"{key} {str(key)}")

test = RedisTest()
test.process_image(bucket_name='test-json',object_key='1',width=69,height=420,current_frame_number=1,max_frame_number=2)
test.process_image(bucket_name='test-json',object_key='1',width=69,height=420,current_frame_number=2,max_frame_number=2)
test.process_anno(bucket_name='test-json',object_key='1',x=1,y=2,w=10,h=5,text='Meme1',score=0.94,category_id=1,current_frame_number=1,max_frame_number=2,current_bb_number=1,max_bb_number=2)
test.process_anno(bucket_name='test-json',object_key='1',x=2,y=3,w=10,h=6,text='Meme2',score=0.93,category_id=1,current_frame_number=1,max_frame_number=2,current_bb_number=2,max_bb_number=2)
test.process_anno(bucket_name='test-json',object_key='1',x=3,y=4,w=11,h=7,text='Meme3',score=0.92,category_id=1,current_frame_number=2,max_frame_number=2,current_bb_number=1,max_bb_number=2)
test.process_anno(bucket_name='test-json',object_key='1',x=4,y=5,w=11,h=8,text='Meme4',score=0.91,category_id=1,current_frame_number=2,max_frame_number=2,current_bb_number=2,max_bb_number=2)