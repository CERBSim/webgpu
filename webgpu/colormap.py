from .webgpu_api import (
    TexelCopyBufferLayout,
    TexelCopyTextureInfo,
    Texture,
    TextureFormat,
    TextureUsage,
)

from .labels import Labels
from .render_object import MultipleRenderObject, RenderObject
from .uniforms import Binding, UniformBase, ct
from .utils import SamplerBinding, TextureBinding, format_number, read_shader_file


class ColormapUniforms(UniformBase):
    _binding = Binding.COLORMAP
    _fields_ = [
        ("min", ct.c_float),
        ("max", ct.c_float),
        ("position_x", ct.c_float),
        ("position_y", ct.c_float),
        ("discrete", ct.c_uint32),
        ("n_colors", ct.c_uint32),
        ("width", ct.c_float),
        ("height", ct.c_float),
    ]


class Colorbar(RenderObject):
    texture: Texture
    vertex_entry_point: str = "colormap_vertex"
    fragment_entry_point: str = "colormap_fragment"
    n_vertices: int = 3

    def __init__(self, minval=0, maxval=1):
        self.texture = None
        self.minval = minval
        self.maxval = maxval
        self.position_x = -0.9
        self.position_y = 0.9
        self.discrete = 0
        self.n_colors = 8
        self.width = 1.0
        self.height = 0.05
        self.uniforms = None
        self.sampler = None
        self.autoupdate = True

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        if self.uniforms is None:
            self.uniforms = ColormapUniforms(self.device)
        self.uniforms.min = self.minval
        self.uniforms.max = self.maxval
        self.uniforms.position_x = self.position_x
        self.uniforms.position_y = self.position_y
        self.uniforms.discrete = self.discrete
        self.uniforms.n_colors = self.n_colors
        self.uniforms.width = self.width
        self.uniforms.height = self.height
        self.n_instances = 2 * self.n_colors
        self.uniforms.update_buffer()

        if self.sampler is None:
            self.sampler = self.device.createSampler(
                magFilter="linear",
                minFilter="linear",
            )

        if self.texture is None:
            self.set_colormap("matlab:jet")
        self.create_render_pipeline()

    def get_bounding_box(self):
        return None

    def set_n_colors(self, n_colors):
        self.n_instances = 2 * n_colors
        if self.uniforms is not None:
            self.uniforms.n_colors = n_colors
            self.uniforms.update_buffer()

    def set_min_max(self, minval, maxval, set_autoupdate=True):
        self.minval = minval
        self.maxval = maxval
        if set_autoupdate:
            self.autoupdate = False
        if self.uniforms is not None:
            self.uniforms.min = minval
            self.uniforms.max = maxval
            self.uniforms.update_buffer()

    def get_bindings(self):
        return [
            TextureBinding(Binding.COLORMAP_TEXTURE, self.texture),
            SamplerBinding(Binding.COLORMAP_SAMPLER, self.sampler),
            *self.uniforms.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("colormap.wgsl", __file__)

    def set_colormap(self, name: str):
        if self.texture is not None:
            self.texture.destroy()

        data = _colormaps[name]
        n = len(data)
        v4 = [v + [255] for v in data]
        data = sum(v4, [])

        self.texture = self.device.createTexture(
            size=[n, 1, 1],
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
            format=TextureFormat.rgba8unorm,
            dimension="1d",
        )
        self.device.queue.writeTexture(
            TexelCopyTextureInfo(self.texture),
            data,
            TexelCopyBufferLayout(bytesPerRow=n * 4),
            [n, 1, 1],
        )


class Colormap(MultipleRenderObject):
    def __init__(self):
        self.colorbar = Colorbar()
        self.labels = Labels([], [], font_size=14, h_align="center", v_align="top")
        self.update_labels()
        super().__init__([self.colorbar, self.labels])

    @property
    def autoupdate(self):
        return self.colorbar.autoupdate

    def get_shader_code(self):
        return self.colorbar.get_shader_code()

    def get_bindings(self):
        return self.colorbar.get_bindings()

    @autoupdate.setter
    def autoupdate(self, value):
        self.colorbar.autoupdate = value

    def set_min_max(self, min, max, set_autoupdate=True):
        self.colorbar.set_min_max(min, max, set_autoupdate)
        self.update_labels()

    def update_labels(self):
        self.labels.labels = [
            format_number(v)
            for v in [
                self.colorbar.minval
                + i / 4 * (self.colorbar.maxval - self.colorbar.minval)
                for i in range(6)
            ]
        ]
        self.labels.positions = [
            (
                self.colorbar.position_x + i * self.colorbar.width / 4,
                self.colorbar.position_y - 0.01,
                0,
            )
            for i in range(5)
        ]

    def get_bounding_box(self):
        return None


_colormaps = {
    "viridis": [
        [68, 1, 84],
        [71, 13, 96],
        [72, 24, 106],
        [72, 35, 116],
        [71, 45, 123],
        [69, 55, 129],
        [66, 64, 134],
        [62, 73, 137],
        [59, 82, 139],
        [55, 91, 141],
        [51, 99, 141],
        [47, 107, 142],
        [44, 114, 142],
        [41, 122, 142],
        [38, 130, 142],
        [35, 137, 142],
        [33, 145, 140],
        [31, 152, 139],
        [31, 160, 136],
        [34, 167, 133],
        [40, 174, 128],
        [50, 182, 122],
        [63, 188, 115],
        [78, 195, 107],
        [94, 201, 98],
        [112, 207, 87],
        [132, 212, 75],
        [152, 216, 62],
        [173, 220, 48],
        [194, 223, 35],
        [216, 226, 25],
        [236, 229, 27],
    ],
    "plasma": [
        [13, 8, 135],
        [34, 6, 144],
        [49, 5, 151],
        [63, 4, 156],
        [76, 2, 161],
        [89, 1, 165],
        [102, 0, 167],
        [114, 1, 168],
        [126, 3, 168],
        [138, 9, 165],
        [149, 17, 161],
        [160, 26, 156],
        [170, 35, 149],
        [179, 44, 142],
        [188, 53, 135],
        [196, 62, 127],
        [204, 71, 120],
        [211, 81, 113],
        [218, 90, 106],
        [224, 99, 99],
        [230, 108, 92],
        [235, 118, 85],
        [240, 128, 78],
        [245, 139, 71],
        [248, 149, 64],
        [251, 161, 57],
        [253, 172, 51],
        [254, 184, 44],
        [253, 197, 39],
        [252, 210, 37],
        [248, 223, 37],
        [244, 237, 39],
    ],
    "cet_l20": [
        [48, 48, 48],
        [55, 51, 69],
        [60, 54, 89],
        [64, 57, 108],
        [66, 61, 127],
        [67, 65, 145],
        [67, 69, 162],
        [65, 75, 176],
        [63, 81, 188],
        [59, 88, 197],
        [55, 97, 201],
        [50, 107, 197],
        [41, 119, 183],
        [34, 130, 166],
        [37, 139, 149],
        [49, 147, 133],
        [66, 154, 118],
        [85, 160, 103],
        [108, 165, 87],
        [130, 169, 72],
        [150, 173, 58],
        [170, 176, 43],
        [190, 179, 29],
        [211, 181, 19],
        [230, 183, 19],
        [241, 188, 20],
        [248, 194, 20],
        [252, 202, 20],
        [254, 211, 19],
        [255, 220, 17],
        [254, 230, 15],
        [252, 240, 13],
    ],
    "matlab:jet": [
        [0, 0, 128],
        [0, 0, 164],
        [0, 0, 200],
        [0, 0, 237],
        [0, 1, 255],
        [0, 33, 255],
        [0, 65, 255],
        [0, 96, 255],
        [0, 129, 255],
        [0, 161, 255],
        [0, 193, 255],
        [0, 225, 251],
        [22, 255, 225],
        [48, 255, 199],
        [73, 255, 173],
        [99, 255, 148],
        [125, 255, 122],
        [151, 255, 96],
        [177, 255, 70],
        [202, 255, 44],
        [228, 255, 19],
        [254, 237, 0],
        [255, 208, 0],
        [255, 178, 0],
        [255, 148, 0],
        [255, 119, 0],
        [255, 89, 0],
        [255, 59, 0],
        [255, 30, 0],
        [232, 0, 0],
        [196, 0, 0],
        [159, 0, 0],
    ],
}


if __name__ == "__main__":
    from cmap import Colormap

    print("_colormaps = {")
    for name in ["viridis", "plasma", "cet_l20", "matlab:jet"]:
        print(f"  '{name}' : [")
        cm = Colormap(name)
        for i in range(32):
            c = cm(i / 32)
            r, g, b = [int(255 * c[i] + 0.5) for i in range(3)]
            print(f"    [{r}, {g}, {b}],")
        print("  ],")
    print("}")
