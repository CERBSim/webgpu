struct FontUniforms {
  width: u32,
  height: u32,

  width_normalized: f32,
  height_normalized: f32,
};

@group(0) @binding(2) var<uniform> u_font : FontUniforms;
@group(0) @binding(3) var u_font_texture : texture_2d<f32>;

fn fontGetTexCoord(char: u32, vertexId: u32) -> vec2<f32> {
    return vec2<f32>(
        f32((char - 32) * u_font.width),
        f32(u_font.height)
    );
}

struct FontFragmentInput {
    @builtin(position) fragPosition: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@fragment
fn fragmentFont(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let alpha: f32 = textureLoad(
        u_font_texture,
        vec2i(floor(tex_coord)),
        0
    ).x;

    if alpha < 0.01 {
      discard;
    }

    return vec4(0., 0., 0., alpha);
}

fn fontCalc(char: u32, position: vec4<f32>, vertexId: u32) -> FontFragmentInput {
    var tex_coord = fontGetTexCoord(char, vertexId);
    var p = position;

    if vertexId == 2 || vertexId == 4 || vertexId == 5 {
        p.y += u_font.height_normalized * p.w;
        tex_coord.y = 0.0;
    }

    if vertexId == 1 || vertexId == 2 || vertexId == 4 {
        p.x += u_font.width_normalized * p.w;
        tex_coord.x += f32(u_font.width);
    }
  return FontFragmentInput(p, tex_coord);
}
