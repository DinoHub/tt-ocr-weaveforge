from weaveforge import WFEventProcessor

import logging
import redis

class WriteCocoImageToRedisHash(WFEventProcessor):
    '''AAAAAAAAAAA
    
    WeaveForge Component Configurations:
        redis_host: str
            the host of the Redis service endpoint
        redis_port: int
            the port of the Redis service endpoint
    '''
    def __init__(self) -> None:
        '''Establish the Redis database connection.
        '''
        logging.info(f"Waiting For Redis Database @ {self.get_component_configurations()['redis_host']}:{self.get_component_configurations()['redis_port']}")
        with redis.Redis(decode_responses=True,
                         host=self.get_component_configurations()["redis_host"],
                         port=self.get_component_configurations()["redis_port"],
                         retry=redis.retry.Retry(backoff=redis.backoff.ConstantBackoff(1),
                                                 retries=-1),
                         retry_on_error=[ConnectionError]) as r:
            self.r = r
        logging.info(f"Connected To Redis Database @ {self.get_component_configurations()['redis_host']}:{self.get_component_configurations()['redis_port']}")

    def process(self,
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