#import camera

struct FontUniforms {
  size: f32,
  aspect: f32,
  chars_per_row: u32,
  n_rows: u32,
  x_shift: f32,
  y_shift: f32,
  size_scaling: f32,
  advance: f32,
  font_size: f32,
};

@group(0) @binding(2) var<uniform> u_font : FontUniforms;
@group(0) @binding(3) var u_font_texture : texture_2d<f32>;
@group(0) @binding(4) var u_font_sampler : sampler;

fn fontGetCharTexBox(char: u32) -> vec4<f32> {
    let size_in_texture = fontGetSizeInTexture();
    let i_row = (char-1)/ u_font.chars_per_row;
    let i_col = (char-1) % u_font.chars_per_row;
    let tex0 = vec2f(f32(i_col)*size_in_texture.x, 1.0-f32(i_row)*size_in_texture.y);

  return vec4f(
        tex0.x + u_font.x_shift,
        tex0.y - u_font.y_shift,
        tex0.x + size_in_texture.x - u_font.x_shift,
        tex0.y - size_in_texture.y + u_font.y_shift
    );
}

fn fontGetSizeInTexture() -> vec2<f32> {
    return vec2<f32>(1.0/f32(u_font.chars_per_row), 1.0/f32(u_font.n_rows));
}

fn fontGetSizeOnScreen() -> vec3<f32> {
    let size = f32(u_font.size * u_font.size_scaling);
    let h = 2.0 * size / f32(u_camera.height);
    let size_horizontal = 2.0 * size / f32(u_camera.width);
    let w = size_horizontal * u_font.aspect;
    let shift = 2.0 * u_font.advance * u_font.size / f32(u_camera.width);
    return vec3<f32>(w, h, shift);
}

struct FontFragmentInput {
    @builtin(position) fragPosition: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@fragment
fn fragmentFont(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let v = textureSample(
        u_font_texture,
        u_font_sampler,
        tex_coord,
    );

    var dist = max(min( v.x, v.y), min(max(v.x, v.y), v.z));
    if(dist == 0.0) {
      discard;
    }

    let width = max(1.0, 4*u_font.size / u_font.font_size);

    // make small fonts more bold for better readability
    var u_in_bias = 0.2 * smoothstep(0, 20, u_font.size) - 0.2;
    let u_out_bias = 0.0;

    let e = width * (dist - 0.5 - u_in_bias) + 0.5 + u_out_bias;
    let smoothing = 0.5;
    let alpha: f32 = smoothstep(0.5-smoothing, 0.5 + smoothing, e);

    if(alpha < 0.01) {
      discard;
    }

    return vec4(0., 0., 0., alpha);
}

fn fontCalc(char: u32, position: vec4<f32>, vertexId: u32) -> FontFragmentInput {
    if (char == 0) {
      return FontFragmentInput(vec4f(0.0, 0.0, 0.0, 0.0), vec2<f32>(0.0, 0.0));
    }

    let size_on_screen = fontGetSizeOnScreen();
    let bbox = fontGetCharTexBox(char);
    var tex_coord = bbox.xy;
    var p = position;

    if vertexId == 2 || vertexId == 4 || vertexId == 5 {
        p.y += size_on_screen.y * p.w;
    }
    else {
        tex_coord.y = bbox.w;
    }

    if vertexId == 1 || vertexId == 2 || vertexId == 4 {
        p.x += size_on_screen.x * p.w;
        tex_coord.x = bbox.z;
    }
  return FontFragmentInput(p, tex_coord);
}
