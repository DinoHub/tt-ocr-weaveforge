from weaveforge import WFComponentMetadata
from weaveforge import WFComponentTypes
from weaveforge import WFDataTypes
from weaveforge import WFDeviceTypes

wf_component_metadata = WFComponentMetadata(
    component_class_name="",
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
    component_description="This component writes COCO annotations to a Redis hash.",
    component_device=WFDeviceTypes.CPU,
    component_image_url_reference="",
    component_inputs=[
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "the name of the bucket where the S3 object is located",
            "name": "bucket_name",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "the key of the S3 object",
            "name": "object_key",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "x-coordinate of the top left of the inferred bounding box",
            "name": "x",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "y-coordinate of the top left of the inferred bounding box",
            "name": "y",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "width of the processed frame in pixels",
            "name": "w",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "height of the processed frame in pixels",
            "name": "h",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFSTRING,
            "description": "Inferred text within the bounding box",
            "name": "text",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFFLOAT,
            "description": "end-to-end(e2e) confidence score of the model for the inferred bounding box",
            "name": "score",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "COCO-required category id",
            "name": "category_id",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "number of the current processed frame",
            "name": "current_frame_number",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "max number of frames for the current processed video",
            "name": "max_frame_number",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "number of the current bounding box being processed for the current frame",
            "name": "current_bb_number",
            "sub_data_types": []
        },
        {
            "data_type": WFDataTypes.WFINT,
            "description": "maximum number of bounding boxes for the current processed frame",
            "name": "max_bb_number",
            "sub_data_types": []
        }
    ],
    component_name="WriteCocoAnnoToRedisHash",
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
