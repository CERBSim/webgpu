import numpy as np

from .background import Background
from .labels import Labels
from .renderer import BaseRenderer, MultipleRenderer, Renderer, RenderOptions
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


def _clean_autoscale_range(minval, maxval):
    """Clean up autoscale min/max values for display.

    - Handles invalid/uninitialized ranges (NaN, inf, inverted sentinels).
    - Rounds values that are negligibly small relative to the range to zero
      (e.g. 1e-13 to 1 becomes 0 to 1).
    - Handles constant functions (min == max) by providing a sensible range
      to avoid numerical flickering.
    """
    import math

    # Handle NaN, inf, or inverted ranges (e.g. from uninitialized sentinels)
    if (math.isnan(minval) or math.isnan(maxval)
            or math.isinf(minval) or math.isinf(maxval)
            or minval > maxval):
        return 0.0, 1.0

    range_size = maxval - minval
    abs_max_val = max(abs(minval), abs(maxval))

    # Handle constant function (min == max or very close)
    # Also treat as constant if both values are negligibly small (near-zero noise)
    is_constant = (
        range_size == 0
        or abs(range_size) < 1e-10 * max(abs_max_val, 1e-300)
        or abs_max_val < 1e-12  # both values essentially zero
    )
    if is_constant:
        # Use a representative value (midpoint)
        value = (minval + maxval) * 0.5
        if abs(value) < 1e-12:
            # Constant zero: use 0 to 1
            return 0.0, 1.0
        elif value > 0:
            # Positive constant: use 0 to value
            return 0.0, value
        else:
            # Negative constant: use value to 0
            return value, 0.0

    # Round near-zero values relative to the range
    # If a boundary is less than 1e-8 of the range away from zero,
    # it's likely a floating-point artifact and should be snapped to zero.
    threshold = abs(range_size) * 1e-8

    if abs(minval) < threshold:
        minval = 0.0
    if abs(maxval) < threshold:
        maxval = 0.0

    return minval, maxval


class Colormap(BaseRenderer):
    texture: Texture

    def __init__(self, minval=None, maxval=None, colormap: list | str = "matlab:jet", n_colors=8):
        self.texture = None
        self.autoscale = minval is None or maxval is None
        self.minval = minval if minval is not None else 0
        self.maxval = maxval if maxval is not None else 1
        self.discrete = 0
        self.n_colors = n_colors
        self.uniforms = None
        self.sampler = None
        self._callbacks = []
        self.set_colormap(colormap)
        self._needs_new_texture = True
        super().__init__("Colormap")

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

        if self.texture is None or self._needs_new_texture:
            self._create_texture()

    def set_colormap(self, colormap: list | str):
        if isinstance(colormap, str):
            if colormap in _colormaps:
                colormap = _colormaps[colormap]
            else:
                colormap = create_colormap(colormap, 32)

        self.colors = colormap
        self.set_needs_update()
        for callback in self._callbacks:
            callback()

        if self.texture is not None and self._texture_dims(len(self.colors)) == (
            self.texture.width, self.texture.height
        ):
            self._create_texture()
            self._needs_new_texture = False
        else:
            self._needs_new_texture = True

    @staticmethod
    def _texture_dims(n_colors):
        w = min(n_colors, 1024)
        h = (n_colors + w - 1) // w
        return w, h

    def set_n_colors(self, n_colors):
        self.n_instances = 2 * n_colors
        self.n_colors = n_colors
        if self.uniforms is not None:
            self.uniforms.n_colors = n_colors
            self.uniforms.update_buffer()
            self.set_needs_update()
        for callback in self._callbacks:
            callback()

    def widen_range(self, minval, maxval, timestamp=None):
        """Extend the autoscale range and apply it.

        On the first call with a new timestamp, resets to inverted range so
        each frame starts fresh. Subsequent calls in the same frame widen.
        Ignores invalid values (NaN, inf, or sentinel 1e99/-1e99 pairs).
        """
        import math

        # Skip invalid values that come from uninitialized/failed data
        if (math.isnan(minval) or math.isnan(maxval)
                or math.isinf(minval) or math.isinf(maxval)
                or minval > maxval):
            return

        ts = getattr(self, '_autoscale_ts', None)
        if timestamp is not None and timestamp != ts:
            self._autoscale_ts = timestamp
            self._autoscale_min = 1e99
            self._autoscale_max = -1e99
        self._autoscale_min = min(getattr(self, '_autoscale_min', minval), minval)
        self._autoscale_max = max(getattr(self, '_autoscale_max', maxval), maxval)
        clean_min, clean_max = _clean_autoscale_range(
            self._autoscale_min, self._autoscale_max
        )
        self.set_min_max(clean_min, clean_max, set_autoscale=False)

    def set_min_max(self, minval, maxval, set_autoscale=True):
        self.minval = minval
        self.maxval = maxval
        if set_autoscale:
            self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.min = minval
            self.uniforms.max = maxval
            self.uniforms.update_buffer()
            self.set_needs_update()
        for callback in self._callbacks:
            callback()

    def set_min(self, minval):
        self.minval = minval
        self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.min = minval
            self.uniforms.update_buffer()
            self.set_needs_update()
        for callback in self._callbacks:
            callback()

    def set_max(self, maxval):
        self.maxval = maxval
        self.autoscale = False
        if self.uniforms is not None:
            self.uniforms.max = maxval
            self.uniforms.update_buffer()
            self.set_needs_update()
        for callback in self._callbacks:
            callback()

    def set_discrete(self, discrete: bool):
        self.discrete = 1 if discrete else 0
        if self.uniforms is not None:
            self.uniforms.discrete = self.discrete
            self.uniforms.update_buffer()
            self.set_needs_update()
        for callback in self._callbacks:
            callback()

    def get_bindings(self):
        return [
            TextureBinding(Binding.COLORMAP_TEXTURE, self.texture, dim=2),
            SamplerBinding(Binding.COLORMAP_SAMPLER, self.sampler),
            *self.uniforms.get_bindings(),
        ]

    def _create_texture(self):
        data = self.colors
        if len(data[0]) == 4:
            v4 = data
        else:
            v4 = [v + [255] for v in data]
        data = sum(v4, [])
        n = len(data) // 4
        w = min(n, 1024)
        h = (n + w - 1) // w
        data = data + [255] * ((w * h - n) * 4)

        device = get_device()
        if self.texture is None or self.texture.width != w or self.texture.height != h:
            import os
            extra = TextureUsage.COPY_SRC if os.environ.get("WEBGPU_EXPORTING") else 0
            self.texture = device.createTexture(
                size=[w, h, 1],
                usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST | extra,
                format=TextureFormat.rgba8unorm,
                dimension="2d",
                label="colormap_texture",
            )

        device.queue.writeTexture(
            TexelCopyTextureInfo(self.texture),
            np.array(data, dtype=np.uint8).tobytes(),
            TexelCopyBufferLayout(bytesPerRow=w * 4),
            [w, h, 1],
        )


class ColorbarStrip(Renderer):
    """Renders the colored strip of the colorbar."""
    vertex_entry_point: str = "colormap_vertex"
    fragment_entry_point: str = "colormap_fragment"
    select_entry_point: str = ""
    n_vertices: int = 3

    def __init__(self, get_bindings_fn):
        super().__init__()
        self._get_bindings_fn = get_bindings_fn

    def get_shader_code(self):
        return read_shader_file("colormap.wgsl")

    def get_bindings(self):
        return self._get_bindings_fn()

    def update(self, options: RenderOptions):
        pass

    def get_export_descriptor(self, options, buffer_registry):
        desc = super().get_export_descriptor(options, buffer_registry)
        desc.pass_type = "transparent"
        return desc


class Colorbar(MultipleRenderer):

    def __init__(
        self,
        colormap: Colormap | None = None,
        position=(-0.9, 0.9),
        width=1,
        height=0.05,
        number_format=None,
    ):
        self.colormap = colormap or Colormap()
        self.number_format = number_format
        self.uniforms = None

        self._position = position
        self._width = width
        self._height = height

        self._bg = Background(position=position, width=width, height=height)
        self._strip = ColorbarStrip(lambda: self._get_all_bindings())
        self._labels = Labels([], [], font_size=14, h_align="center", v_align="top")

        super().__init__([self._bg, self._strip, self._labels])
        self.colormap._callbacks.append(self.set_needs_update)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        if self.uniforms is not None:
            self.uniforms.position = value
        self._bg.position = value
        self.set_needs_update()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        if self.uniforms is not None:
            self.uniforms.width = value
        self._bg.width = value
        self.set_needs_update()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        if self.uniforms is not None:
            self.uniforms.height = value
        self._bg.height = value
        self.set_needs_update()

    def _get_all_bindings(self):
        return (
            self.colormap.get_bindings() + self._labels.get_bindings() + self.uniforms.get_bindings()
        )

    def update(self, options: RenderOptions):
        if self.uniforms is None:
            self.uniforms = ColorbarUniforms()
            self.uniforms.position = self.position
            self.uniforms.width = self.width
            self.uniforms.height = self.height

        self.uniforms.update_buffer()
        self.colormap.update(options)

        self._strip.n_instances = 2 * self.colormap.n_colors

        self._labels.labels = [
            format_number(v, self.number_format)
            for v in [
                self.colormap.minval + i / 4 * (self.colormap.maxval - self.colormap.minval)
                for i in range(6)
            ]
        ]
        self._labels.positions = [
            (
                self.position[0] + i * self.width / 4,
                self.position[1] - 0.01,
                0,
            )
            for i in range(5)
        ]
        super().update(options)

    def set_min(self, minval):
        self.colormap.set_min(minval)
        self.set_needs_update()

    def set_max(self, maxval):
        self.colormap.set_max(maxval)
        self.set_needs_update()


_colormaps = {
    "turbo": [
        [48, 18, 59],
        [57, 42, 115],
        [64, 64, 162],
        [68, 86, 199],
        [71, 110, 230],
        [70, 130, 248],
        [65, 150, 255],
        [53, 171, 248],
        [37, 192, 231],
        [26, 210, 210],
        [24, 224, 189],
        [34, 235, 170],
        [60, 245, 142],
        [89, 251, 115],
        [121, 254, 89],
        [150, 254, 68],
        [175, 250, 55],
        [195, 241, 52],
        [215, 229, 53],
        [231, 215, 57],
        [245, 197, 58],
        [252, 179, 54],
        [254, 158, 47],
        [252, 135, 37],
        [246, 108, 25],
        [237, 85, 16],
        [226, 67, 10],
        [212, 51, 5],
        [193, 35, 2],
        [172, 23, 1],
        [149, 13, 1],
        [122, 4, 3],
    ],
    "rainbow": [
        [13, 59, 221],
        [11, 76, 222],
        [9, 94, 224],
        [7, 111, 225],
        [5, 128, 227],
        [3, 145, 228],
        [0, 163, 230],
        [4, 170, 214],
        [9, 175, 194],
        [14, 180, 174],
        [19, 184, 154],
        [24, 189, 134],
        [29, 194, 114],
        [51, 198, 96],
        [83, 200, 79],
        [116, 203, 62],
        [148, 205, 44],
        [181, 208, 27],
        [213, 210, 10],
        [234, 207, 0],
        [238, 195, 0],
        [242, 183, 0],
        [245, 171, 0],
        [249, 159, 0],
        [252, 148, 0],
        [254, 135, 1],
        [249, 117, 6],
        [245, 100, 11],
        [240, 83, 16],
        [235, 66, 21],
        [231, 48, 26],
        [226, 31, 31],
    ],
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
    "matplotlib:coolwarm": [
        [59, 76, 192],
        [68, 90, 204],
        [78, 104, 216],
        [88, 117, 225],
        [98, 130, 234],
        [108, 143, 241],
        [119, 154, 247],
        [130, 166, 251],
        [141, 176, 254],
        [152, 185, 255],
        [163, 194, 254],
        [174, 201, 252],
        [185, 208, 249],
        [195, 213, 244],
        [204, 217, 237],
        [213, 219, 229],
        [221, 220, 220],
        [229, 216, 209],
        [236, 211, 197],
        [241, 204, 184],
        [245, 196, 172],
        [247, 186, 159],
        [247, 176, 147],
        [246, 165, 134],
        [244, 152, 122],
        [240, 139, 110],
        [235, 125, 98],
        [228, 110, 86],
        [221, 95, 75],
        [212, 78, 65],
        [202, 59, 55],
        [190, 36, 46],
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
    for name in ["viridis", "plasma", "cet_l20", "matlab:jet", "matplotlib:coolwarm"]:
        colors = create_colormap(name, n_colors=32)
        print(f"  '{name}' : [")
        for i in range(32):
            print(f"    {colors[i]},")
        print("  ],")
    print("}")
