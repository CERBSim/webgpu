import typing

import js

from .gpu import WebGPU
from .mesh import RenderObject
from .uniforms import Binding, LineIntegralConvolutionUniforms
from .utils import ShaderStage, StorageTextureBinding, TextureBinding


class LineIntegralConvolutionRenderObject(RenderObject):
    """Line Integral Convolution (LIC) render object"""

    texture_input: typing.Any
    texture_output: typing.Any
    uniforms: LineIntegralConvolutionUniforms
    gpu: WebGPU
    _buffers: dict = {}

    def __init__(self, gpu, width, height):
        self.gpu = gpu
        self.uniforms = LineIntegralConvolutionUniforms(gpu.device)
        self.uniforms.width = width
        self.uniforms.height = height
        self.uniforms.update_buffer()
        self.texture_input = self.device.create_texture(
            {
                "label": "LIC input texture",
                "size": [width, height],
                "format": "rg32float",
                "usage": js.GPUTextureUsage.TEXTURE_BINDING
                | js.GPUTextureUsage.COPY_DST
                | js.GPUTextureUsage.RENDER_ATTACHMENT,
            }
        )
        self.texture_output = self.device.create_texture(
            {
                "label": "LIC output texture",
                "size": [width, height],
                "format": "r32float",
                "usage": js.GPUTextureUsage.STORAGE_BINDING,
            }
        )

        self._create_pipelines()

    def _create_evaluate_pipeline(self):
        device = self.device
        layout, self.bind_group = device.create_bind_group(
            self.get_bindings(), "LIC evaluate"
        )

        self.evaluate_pipeline = device.create_compute_pipeline(
            layout,
            {
                "label": "create_test_mesh",
                "layout": device.create_pipeline_layout(layout, "LIC evaluate"),
                "compute": {
                    "module": device.shader_module,
                    "entryPoint": "evaluateLineIntegralConvolution",
                },
            },
        )

    def _create_compute_pipeline(self):
        device = self.device
        layout, self.bind_group = device.create_bind_group(
            self.get_bindings(), "LIC compute"
        )

        self.compute_pipeline = device.create_compute_pipeline(
            layout,
            {
                "label": "create_test_mesh",
                "layout": device.create_pipeline_layout(layout, "LIC compute"),
                "compute": {
                    "module": device.shader_module,
                    "entryPoint": "computeLineIntegralConvolution",
                },
            },
        )

    def render(self, encoder):
        pass_encoder = encoder.beginComputePass()
        pass_encoder.setPipeline(self.compute_pipeline)
        pass_encoder.setBindGroup(0, self.bind_group)
        pass_encoder.dispatchWorkgroups(
            (self.uniforms.width + 15 // 16), (self.uniforms.height + 15) // 16, 1
        )
        pass_encoder.end()

    @property
    def device(self):
        return self.gpu.device

    def get_bindings(self):
        return (
            self.uniforms.get_bindings()
            + self.gpu.get_bindings()
            + [
                TextureBinding(
                    Binding.LINE_INTEGRAL_CONVOLUTION_INPUT_TEXTURE,
                    self.texture_input,
                    dim=2,
                    visibility=ShaderStage.COMPUTE,
                    sample_type="unfilterable-float",
                ),
                StorageTextureBinding(
                    Binding.LINE_INTEGRAL_CONVOLUTION_OUTPUT_TEXTURE,
                    self.texture_output,
                    dim=2,
                    access="write-only",
                ),
            ]
        )
