import base64
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .uniforms import Binding, UniformBase, ct
from .utils import (
    SamplerBinding,
    TextureBinding,
    get_device,
    read_shader_file,
    texture_from_data,
)
from .webgpu_api import *


class FontUniforms(UniformBase):
    _binding = Binding.FONT
    _fields_ = [
        ("size", ct.c_float),
        ("aspect", ct.c_float),
        ("chars_per_row", ct.c_uint32),
        ("n_rows", ct.c_uint32),
        ("x_shift", ct.c_float),
        ("y_shift", ct.c_float),
        ("size_scaling", ct.c_float),
        ("advance", ct.c_float),
        ("font_size", ct.c_float),
        ("padding", ct.c_float * 3),
    ]


@dataclass
class FontAtlas:
    char_width: int
    char_height: int
    x_shift: int
    y_shift: int
    chars_per_row: int
    n_rows: int
    font_size: float
    charset: str
    image: np.ndarray
    texture: Texture | None = None
    texture_sampler: Sampler | None = None
    char_map: dict[int, int] = None
    advance: float = 1.0

    _struct_format = "ffffiiiiii"

    def save(self, filepath: str | Path):
        filepath = Path(filepath)
        utf8_bytes = self.charset.encode("utf-8")

        image_data = self.image.astype(np.uint8).tobytes()

        data = struct.pack(
            self._struct_format,
            self.font_size,
            self.x_shift,
            self.y_shift,
            self.advance,
            self.char_width,
            self.char_height,
            self.chars_per_row,
            self.n_rows,
            len(utf8_bytes),
            len(image_data),
        )
        data += utf8_bytes
        data += image_data
        filepath.write_bytes(zlib.compress(data))

    @classmethod
    def load(cls, filepath: str | Path):
        data = zlib.decompress(Path(filepath).read_bytes())
        header_size = struct.calcsize(cls._struct_format)
        header = data[:header_size]
        data = data[header_size:]

        (
            font_size,
            x_shift,
            y_shift,
            advance,
            char_width,
            char_height,
            chars_per_row,
            n_rows,
            utf8_len,
            image_len,
        ) = struct.unpack(cls._struct_format, header)

        charset = data[:utf8_len].decode("utf-8")
        img_data = data[utf8_len : utf8_len + image_len]
        width = char_width * chars_per_row
        height = char_height * n_rows
        image = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 4))

        return cls(
            char_width=char_width,
            char_height=char_height,
            x_shift=x_shift,
            y_shift=y_shift,
            advance=advance,
            chars_per_row=chars_per_row,
            n_rows=n_rows,
            font_size=font_size,
            charset=charset,
            image=image,
            texture=None,
            texture_sampler=None,
            char_map={ord(c): i for i, c in enumerate(charset)},
        )

    def update(self):
        if self.texture is not None:
            return

        self.texture = texture_from_data(
            self.image.shape[1],
            self.image.shape[0],
            self.image.tobytes(),
            format=TextureFormat.rgba8unorm,
            label="font",
        )

        self.texture_sampler = get_device().createSampler(
            addressModeU=AddressMode.clamp_to_edge,
            addressModeV=AddressMode.clamp_to_edge,
            magFilter=FilterMode.linear,
            minFilter=FilterMode.linear,
            label="font_sampler",
        )

    def get_uniform_settings(self):
        print("image shape", self.image.shape)
        s = {
            "x_shift": self.x_shift,
            "y_shift": self.y_shift,
            "font_size": self.font_size,
            "size_scaling": (self.char_height - 2 * self.y_shift * self.image.shape[0])
            / self.font_size,
            "aspect": self.char_width / self.char_height,
            "advance": self.advance,
        }
        print("FontAtlas.get_uniform_settings", s)
        return s


_default_font_atlas: FontAtlas | None = None


def set_default_font_atlas(atlas: FontAtlas):
    global _default_font_atlas
    _default_font_atlas = atlas


def get_font_atlas():
    global _default_font_atlas
    if _default_font_atlas is None:
        font_file = Path(__file__).parent / "font.bin"
        try:
            import webgpu_fonts

            font_file = webgpu_fonts.get_default_font_file()
        except ImportError:
            pass

        print("loading font", font_file)

        _default_font_atlas = FontAtlas.load(font_file)
    return _default_font_atlas


class Font:
    def __init__(self, canvas, size=15):
        self.canvas = canvas
        self.atlas = get_font_atlas()
        self.uniforms = FontUniforms(
            size=size,
            chars_per_row=self.atlas.chars_per_row,
            n_rows=self.atlas.n_rows,
            **self.atlas.get_uniform_settings(),
        )
        self.update()

    def get_bindings(self):
        return [
            TextureBinding(Binding.FONT_TEXTURE, self.atlas.texture, dim=2),
            SamplerBinding(Binding.FONT_SAMPLER, self.atlas.texture_sampler),
            *self.uniforms.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("font.wgsl")

    def set_font_size(self, font_size: float):
        self.uniforms.size = font_size
        self.update()

    def update(self):
        self.atlas.update()
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
