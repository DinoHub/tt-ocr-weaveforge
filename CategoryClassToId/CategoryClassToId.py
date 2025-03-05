'''
WeaveForge VideoToFrame component
'''
from weaveforge import WFEventGenerator

class CategoryClassToId(WFEventGenerator):
    '''
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
