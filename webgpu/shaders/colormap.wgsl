@group(0) @binding(6) var u_colormap_texture : texture_1d<f32>;
@group(0) @binding(7) var u_colormap_sampler : sampler;
@group(0) @binding(5) var<uniform> u_cmap_uniforms : ColormapUniforms;

struct ColormapUniforms {
  min: f32,
  max: f32,
  padding0: f32,
  padding1: f32,
};


fn getColor(value: f32) -> vec4<f32> {
  let v = (value - u_cmap_uniforms.min) / (u_cmap_uniforms.max - u_cmap_uniforms.min);
    return textureSample(u_colormap_texture, u_colormap_sampler, v);
}

