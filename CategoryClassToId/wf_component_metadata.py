from weaveforge import WFComponentMetadata, WFComponentTypes, WFDataTypes, WFDeviceTypes

wf_component_metadata = WFComponentMetadata(
    component_configurations = [
        {
            'data_type': WFDataTypes.WFSTRING,
            'description': '',
            'name': 'MAPPINGS',
            'value': ''
        }
    ],
    component_description = 'Converts category class from text to an integer id',
    component_device = WFDeviceTypes.CPU,
    component_inputs = [
        {
            'data_type': WFDataTypes.WFSTRING,
            'description': 'string of the class to be converted',
            'name': 'class_name',
            "sub_data_types": []
        }
    ],
    component_name = 'CategoryClassToId',
    component_class_name = '',
    component_image_url_reference = '',
    component_outputs = [
        [
            {
                'data_type': WFDataTypes.WFINT,
                'description': 'category id matching the class name defined in the MAPPINGS configuration',
                'name': 'category_id',
                "sub_data_types": []
            }
        ]
    ],
    component_tags = [
        'default',
        'coco'
    ],
    component_type = WFComponentTypes.WFEVENTGENERATOR
)

from weaveforge import register_component_to_weaveforge_component_store

register_component_to_weaveforge_component_store("http://public-api-server.weaveforge-beta.apps-crc.testing/component", wf_component_metadata)
