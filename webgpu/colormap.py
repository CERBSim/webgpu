from webgpu.webgpu_api import (
    TexelCopyBufferLayout,
    TexelCopyTextureInfo,
    TextureUsage,
    TextureFormat,
)

from .uniforms import Binding
from .utils import SamplerBinding, TextureBinding, read_shader_file


class Colormap:
    def __init__(self, device):
        self.device = device
        n = 5
        data = [0] * n * 4
        data[3::4] = [1] * n
        data[0 * 4 : 1 * 4 - 1] = [0, 0, 1]
        data[1 * 4 : 2 * 4 - 1] = [0, 1, 1]
        data[2 * 4 : 3 * 4 - 1] = [0, 1, 0]
        data[3 * 4 : 4 * 4 - 1] = [1, 1, 0]
        data[4 * 4 : 5 * 4 - 1] = [1, 0, 0]
        data = [255 * x for x in data]

        self.texture = device.createTexture(
            size=[n, 1, 1],
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
            format=TextureFormat.rgba8unorm,
            dimension="1d",
        )

        device.queue.writeTexture(
            TexelCopyTextureInfo(self.texture),
            data,
            TexelCopyBufferLayout(bytesPerRow=n * 4),
            [n, 1, 1],
        )

        self.sampler = device.createSampler(
            magFilter="linear",
            minFilter="linear",
        )

    def get_bindings(self):
        return [
            TextureBinding(Binding.COLORMAP_TEXTURE, self.texture),
            SamplerBinding(Binding.COLORMAP_SAMPLER, self.sampler),
        ]

    def get_shader_code(self):
        return read_shader_file("colormap.wgsl", __file__)


    def __del__(self):
        del self.texture
