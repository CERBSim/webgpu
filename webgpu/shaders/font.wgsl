struct FontUniforms {
  width: u32,
  height: u32,

  padding0: u32,
  padding1: u32,
};

@group(0) @binding(2) var<uniform> u_font : FontUniforms;
@group(0) @binding(3) var u_font_texture : texture_2d<f32>;
