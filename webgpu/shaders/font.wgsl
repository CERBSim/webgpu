struct FontUniforms {
  width: u32,
  height: u32,

  canvas_width: u32,
  canvas_height: u32,
};

@group(0) @binding(2) var<uniform> u_font : FontUniforms;
@group(0) @binding(3) var u_font_texture : texture_2d<f32>;
