import base64
import json
import os
import zlib


def create_font_texture(device, size: int = 15):
    import js

    from .utils import to_js

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
    print("Font size:", tex_width, h)
    print("data size:", len(data))
    print("data / tex_width:", len(data) / tex_width)
    print("Font texture size:", tex_width, h)
    texture = device.create_texture(
        {
            "dimension": "2d",
            "size": [tex_width, h, 1],
            "format": "r8unorm",
            "usage": js.GPUTextureUsage.TEXTURE_BINDING | js.GPUTextureUsage.COPY_DST,
        }
    )
    device.write_texture(
        texture,
        data,
        bytes_per_row=tex_width,
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
