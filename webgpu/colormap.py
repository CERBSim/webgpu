from .labels import Labels
from .renderer import BaseRenderer, Renderer, RenderOptions
from .uniforms import Binding, UniformBase, ct
from .utils import (
    SamplerBinding,
    TextureBinding,
    format_number,
    get_device,
    read_shader_file,
)
from .webgpu_api import (
    TexelCopyBufferLayout,
    TexelCopyTextureInfo,
    Texture,
    TextureFormat,
    TextureUsage,
)


class ColormapUniforms(UniformBase):
    _binding = Binding.COLORMAP
    _fields_ = [
        ("min", ct.c_float),
        ("max", ct.c_float),
        ("discrete", ct.c_uint32),
        ("n_colors", ct.c_uint32),
    ]


class ColorbarUniforms(UniformBase):
    _binding = Binding.COLORBAR
    _fields_ = [
        ("position", ct.c_float * 2),
        ("width", ct.c_float),
        ("height", ct.c_float),
    ]


class Colormap(BaseRenderer):
    texture: Texture

    def __init__(self, minval=None, maxval=None, colormap: list | str = "matlab:jet"):
        self.texture = None
        self.autoscale = minval is None or maxval is None
        self.minval = minval if minval is not None else 0
        self.maxval = maxval if maxval is not None else 1
        self.discrete = 0
        self.n_colors = 8
        self.uniforms = None
        self.sampler = None

        self.set_colormap(colormap)

    def update(self, options: RenderOptions):
        if self.uniforms is None:
            self.uniforms = ColormapUniforms()
        self.uniforms.min = self.minval
        self.uniforms.max = self.maxval
        self.uniforms.discrete = self.discrete
        self.uniforms.n_colors = self.n_colors
        self.uniforms.update_buffer()

        if self.sampler is None:
            self.sampler = get_device().createSampler(
                magFilter="linear",
                minFilter="linear",
            )

        if self.texture is None:
            self._create_texture()

    def set_colormap(self, colormap: list | str):
        if self.texture is not None:
            self.texture.destroy()
            self.texture = None
        if isinstance(colormap, str):
            if colormap in _colormaps:
                colormap = _colormaps[colormap]
            else:
                colormap = create_colormap(colormap, 32)

        self.colors = colormap
        self.set_needs_update()

    def set_n_colors(self, n_colors):
        self.n_instances = 2 * n_colors
        if self.uniforms is not None:
            self.uniforms.n_colors = n_colors
            self.uniforms.update_buffer()

    def set_min_max(self, minval, maxval, set_autoscale=True):
        self.minval = minval
        self.maxval = maxval
        if set_autoscale:
            self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.min = minval
            self.uniforms.max = maxval
            self.uniforms.update_buffer()

    def set_min(self, minval):
        self.minval = minval
        self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.min = minval
            self.uniforms.update_buffer()

    def set_max(self, maxval):
        self.maxval = maxval
        self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.max = maxval
            self.uniforms.update_buffer()

    def get_bindings(self):
        return [
            TextureBinding(Binding.COLORMAP_TEXTURE, self.texture),
            SamplerBinding(Binding.COLORMAP_SAMPLER, self.sampler),
            *self.uniforms.get_bindings(),
        ]

    def _create_texture(self):
        if self.texture is not None:
            self.texture.destroy()

        data = self.colors
        n = len(data)
        v4 = [v + [255] for v in data]
        data = sum(v4, [])

        device = get_device()
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


class Colorbar(Renderer):
    vertex_entry_point: str = "colormap_vertex"
    fragment_entry_point: str = "colormap_fragment"
    n_vertices: int = 3

    def __init__(
        self, colormap: Colormap | None = None, position=(-0.9, 0.9), width=1, height=0.05
    ):
        super().__init__()
        self.colormap = colormap or Colormap()
        self.labels = Labels([], [], font_size=14, h_align="center", v_align="top")
        self.uniforms = None

        self._position = position
        self._width = width
        self._height = height

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        if self.uniforms is not None:
            self.uniforms.position = value
        self.set_needs_update()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        if self.uniforms is not None:
            self.uniforms.width = value
        self.set_needs_update()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        if self.uniforms is not None:
            self.uniforms.height = value
        self.set_needs_update()

    def get_shader_code(self):
        return read_shader_file("colormap.wgsl")

    def get_bindings(self):
        return (
            self.colormap.get_bindings() + self.labels.get_bindings() + self.uniforms.get_bindings()
        )

    def update(self, options: RenderOptions):
        if self.uniforms is None:
            self.uniforms = ColorbarUniforms()
            self.uniforms.position = self.position
            self.uniforms.width = self.width
            self.uniforms.height = self.height

        self.uniforms.update_buffer()

        self.n_instances = 2 * self.colormap.n_colors

        self.labels.labels = [
            format_number(v)
            for v in [
                self.colormap.minval + i / 4 * (self.colormap.maxval - self.colormap.minval)
                for i in range(6)
            ]
        ]
        self.labels.positions = [
            (
                self.position[0] + i * self.width / 4,
                self.position[1] - 0.01,
                0,
            )
            for i in range(5)
        ]
        self.labels.set_needs_update()
        self.labels._update_and_create_render_pipeline(options)

    def render(self, options: RenderOptions):
        super().render(options)
        self.labels.render(options)

    def add_options_to_gui(self, gui):
        if gui is None:
            return
        folder = gui.folder("Colormap", closed=True)
        folder.value("min", self.colormap.minval, self.set_min)
        folder.value("max", self.colormap.maxval, self.set_max)

    def set_min(self, minval):
        self.colormap.set_min(minval)
        self.set_needs_update()

    def set_max(self, maxval):
        self.colormap.set_max(maxval)
        self.set_needs_update()


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


def create_colormap(name: str, n_colors: int = 32):
    """Create a colormap with the given name and number of colors."""
    from cmap import Colormap

    cm = Colormap(name)
    colors = []
    for i in range(n_colors):
        c = cm(i / n_colors)
        colors.append([int(255 * c[i] + 0.5) for i in range(3)])
    return colors


if __name__ == "__main__":
    print("_colormaps = {")
    for name in ["viridis", "plasma", "cet_l20", "matlab:jet"]:
        colors = create_colormap(name, n_colors=32)
        print(f"  '{name}' : [")
        for i in range(32):
            print(f"    {colors[i]},")
        print("  ],")
    print("}")
