import base64
import json
import os
import zlib

from .uniforms import Binding, UniformBase, ct
from .utils import Device, TextureBinding, read_shader_file
from .webgpu_api import *


def create_font_texture(device: Device, size: int = 15):
    fonts = json.load(open(os.path.join(os.path.dirname(__file__), "fonts.json")))

    dist = 0
    while str(size) not in fonts:
        # try to find the closest available font size
        dist += 1
        if dist > 20:
            raise ValueError(f"Font size {size} not found")

        if str(size + dist) in fonts:
            size += dist
            break

        if str(size - dist) in fonts:
            size -= dist
            break

    font = fonts[str(size)]
    data = zlib.decompress(base64.b64decode(font["data"]))
    w = font["width"]
    h = font["height"]

    tex_width = w * (127 - 32)

    texture = device.createTexture(
        size=[tex_width, h, 1],
        usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
        format=TextureFormat.r8unorm,
        label="font",
    )
    device.queue.writeTexture(
        TexelCopyTextureInfo(texture),
        data,
        TexelCopyBufferLayout(bytesPerRow=tex_width),
        size=[tex_width, h, 1],
    )

    return texture


def _get_default_font():
    import os

    # font = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"
    font = ""
    if not os.path.exists(font):
        from matplotlib import font_manager

        for f in font_manager.fontManager.ttflist:
            if "mono" in f.name.lower():
                font = f.fname
            if f.fname.lower().endswith("DejaVuSansMono.ttf".lower()):
                break

    return font


def create_font_data(size: int = 15, font_file: str = ""):
    from PIL import Image, ImageDraw, ImageFont

    font_file = font_file or _get_default_font()
    text = "".join([chr(i) for i in range(32, 127)])  # printable ascii characters

    # disable ligatures and other features, because they are merging characters
    # this is not desired when using the rendered image as a texture
    features = [
        "-liga",
        "-kern",
        "-calt",
        "-clig",
        "-ccmp",
        "-locl",
        "-mark",
        "-mkmk",
        "-rlig",
    ]

    font = ImageFont.truetype(font_file, size)
    x0, y0, x1, y1 = font.getbbox("$", features=features)

    # the actual height is usually a few pixels less than the font size
    h = round(y1 - y0)
    w = round(x1 - x0)

    # create an image with the text (greyscale, will be used as alpha channel on the gpu)
    image = Image.new("L", (len(text) * w, h), (0))
    draw = ImageDraw.Draw(image)
    for i, c in enumerate(text):
        draw.text((i * w, -y0), c, font=font, fill=(255), features=features)

    # image.save(f"out_{size}.png")
    return image.tobytes(), w, h


class FontUniforms(UniformBase):
    _binding = Binding.FONT
    _fields_ = [
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("width_normalized", ct.c_float),
        ("height_normalized", ct.c_float),
    ]


class Font:
    def __init__(self, canvas, size=15):
        self.canvas = canvas
        self.uniforms = FontUniforms(canvas.device)
        self.set_font_size(size)

        self.canvas.on_resize(self.update)

    def get_bindings(self):
        return [
            TextureBinding(Binding.FONT_TEXTURE, self._texture, dim=2),
            *self.uniforms.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("font.wgsl", __file__)

    def set_font_size(self, font_size: int):
        from .font import create_font_texture

        self._texture = create_font_texture(self.canvas.device, font_size)
        char_width = self._texture.width // (127 - 32)
        char_height = self._texture.height
        self.uniforms.width = char_width
        self.uniforms.height = char_height
        self.update()

    def update(self):
        self.uniforms.width_normalized = 2.0 * self.uniforms.width / self.canvas.width
        self.uniforms.height_normalized = (
            2.0 * self.uniforms.height / self.canvas.height
        )
        self.uniforms.update_buffer()


if __name__ == "__main__":
    # create font data and store it as json because we cannot generate this in pyodide

    fonts = {}

    for size in list(range(8, 21, 2)) + [25, 30, 40]:
        data, w, h = create_font_data(size)
        fonts[size] = {
            "data": base64.b64encode(zlib.compress(data)).decode("utf-8"),
            "width": w,
            "height": h,
        }

    json.dump(fonts, open("fonts.json", "w"), indent=2)
