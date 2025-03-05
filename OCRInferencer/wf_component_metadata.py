from weaveforge import WFComponentMetadata, WFComponentTypes, WFDataTypes, WFDeviceTypes

wf_component_metadata = WFComponentMetadata(
    component_description = 'Processes a singular frame and generates each individual bounding-box inference as an event.',
    component_device = WFDeviceTypes.CUDA,
    component_inputs = [
        {
            'data_type': WFDataTypes.WFNUMPYNDARRAY,
            'description': 'Input video frame in ndarray form',
            'name': 'input_numpyndarray',
            "sub_data_types": []
        }
    ],
    component_name = 'OCRInferencer',
    component_class_name = '',
    component_image_url_reference = '',
    component_outputs = [
        '''
        x - int (a.k.a. the L in LTRB)
        y - int (a.k.a. the T in LTRB)
        w - int 
        h - int
        text - str
        class - str
        confidence - float
        current_bb_number - int (basically start from 1 for each new frame coming in)
        max_bb_number - int (total number of bb inferences)
        '''
        [
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'x-coordinate of the top left of the inferred bounding box',
                'name': 'x',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'y-coordinate of the top left of the inferred bounding box',
                'name': 'y',
                'sub_data_types': []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'video width',
                'name': 'w',
                'sub_data_types': []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'video height',
                'name': 'h',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFSTRING,
                'description': 'Inferred text within the bounding box',
                'name': 'text',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFSTRING,
                'description': 'Class of bounding box (Text)',
                'name': 'class',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFFLOAT,
                'description': 'Confidence score of the inference (e2e)',
                'name': 'confidence',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'Number X of Y maximum number of bounding boxes for this frame',
                'name': 'current_bb_number',
                "sub_data_types": []
            },
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'Maximum number of bounding boxes for this frame',
                'name': 'text',
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
