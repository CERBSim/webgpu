from .uniforms import BaseBinding, CameraUniforms
from .utils import SamplerBinding, UniformBinding, read_shader_file


class Camera:
    def __init__(self, device):
        self.device = device
        self.uniforms = CameraUniforms(device)

    def get_bindings(self) -> list[BaseBinding]:
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("camera.wgsl", __file__)

    def __del__(self):
        del self.uniforms
