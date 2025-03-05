from weaveforge import WFComponentMetadata, WFComponentTypes, WFDataTypes, WFDeviceTypes

wf_component_metadata = WFComponentMetadata(
    component_description = 'Extracts frames from a video file and generates new events for every frame.',
    component_device = WFDeviceTypes.CPU,
    component_inputs = [
        {
            'data_type': WFDataTypes.WFSTRING,
            'description': 's3 bucket name where target file resides',
            'name': 'bucket',
            "sub_data_types": []
        },
        {
            'data_type': WFDataTypes.WFSTRING,
            'description': 'Name of target file residing in the s3 bucket',
            'name': 'object_key',
            "sub_data_types": []
        }
    ],
    component_name = 'VideoToFrame',
    component_class_name = '',
    component_image_url_reference = '',
    component_outputs = [
        [
            {
                'data_type': WFDataTypes.WFNUMPYNDARRAY,
                'description': 'Frame data in numpy',
                'name': 'frame',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'output frame number',
                'name': 'current_frame_number',
                'sub_data_types': []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'maximum number of frames for the video',
                'name': 'max_frame_number',
                'sub_data_types': []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'width of the video in pixels',
                'name': 'width',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'height of the video in pixels',
                'name': 'height',
                "sub_data_types": []
            }
        ]
    ],
    component_tags = [
        'Computer Vision',
        'Frame Generation'
    ],
    component_type = WFComponentTypes.WFEVENTGENERATOR
)

from weaveforge import register_component_to_weaveforge_component_store

register_component_to_weaveforge_component_store("http://public-api-server.weaveforge-beta.apps-crc.testing/component", wf_component_metadata)
