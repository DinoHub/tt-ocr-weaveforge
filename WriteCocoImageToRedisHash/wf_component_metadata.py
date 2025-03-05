from weaveforge import WFComponentMetadata
from weaveforge import WFComponentTypes
from weaveforge import WFDataTypes
from weaveforge import WFDeviceTypes

wf_component_metadata = WFComponentMetadata(
    component_class_name="WriteCocoImageToRedisHash",
    component_configurations=[
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "the url of the Redis service endpoint",
            "name": "redis_host",
            "value": ""
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "the port of the Redis service endpoint",
            "name": "redis_port",
            "value": 6379
        }
    ],
    component_description="This component writes intermediate COCO data (image) to a Redis hash.",
    component_device=WFDeviceTypes.CPU,
    component_image_url_reference="",
    component_inputs=[
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "the name of the bucket where the S3 object is located",
            "name": "bucket name",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "the key of the S3 object",
            "name": "object key",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "width of video",
            "name": "width",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "height of video",
            "name": "height",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "the current frame number of the video being processed",
            "name": "current_frame_number",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "the maximum amount of frames for the video being processed",
            "name": "max_frame_number",
            "sub_data_types": []
        }
    ],
    component_name="WriteCocoImageToRedisHash",
    component_outputs=[
    ],
    component_tags=[
        "default",
        "redis",
        "coco"
    ],
    component_type=WFComponentTypes.WFEVENTPROCESSOR)

from weaveforge import register_component_to_weaveforge_component_store

register_component_to_weaveforge_component_store("http://public-api-server.weaveforge-beta.apps-crc.testing/component", wf_component_metadata)
