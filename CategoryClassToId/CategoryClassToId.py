'''
WeaveForge CategoryClassToId Component
'''
from weaveforge import WFEventGenerator

class CategoryClassToId(WFEventGenerator):
    '''
    Converts class name to matching ID defined in MAPPINGS
    
    Inputs:
        class_name: str
    Outputs:
        id: int
    '''
    def __init__(self):
        self.mappings = self.get_component_configurations()['MAPPINGS']

    def generate(self, class_name: str):
        if class_name in self.mappings:
            return list(self.mappings.values()).index(class_name) + 1
