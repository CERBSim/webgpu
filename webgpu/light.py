from .utils import read_shader_file


class Light:
    def __init__(self, device):
        self.device = device

    def get_bindings(self):
        return []

    def get_shader_code(self):
        return read_shader_file("light.wgsl", __file__)
