@group(0) @binding(6) var u_colormap_texture : texture_1d<f32>;
@group(0) @binding(7) var u_colormap_sampler : sampler;
@group(0) @binding(5) var<uniform> u_cmap_uniforms : ColormapUniforms;

struct ColormapUniforms {
  min: f32,
  max: f32,
  position_x: f32,
  position_y: f32,
  discrete: u32,
  n_colors: u32,
  width: f32,
  height: f32,
};


fn getColor(value: f32) -> vec4<f32> {
  var v = (value - u_cmap_uniforms.min) / (u_cmap_uniforms.max - u_cmap_uniforms.min);
  if(u_cmap_uniforms.discrete == 1u) {
      v = f32(u32(f32(u_cmap_uniforms.n_colors) * v));
      v = v / f32(u_cmap_uniforms.n_colors);
    }
  return textureSample(u_colormap_texture, u_colormap_sampler, v);
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) val: f32,
};

@vertex
fn colormap_vertex(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId: u32) -> VertexOutput {
  var posx = trigId / 2u;
  if(vertId == 2u || (trigId % 2u == 1u && vertId == 1u))
    {
      posx = posx + 1u;
    }
  var posy = u_cmap_uniforms.position_y;
  if(vertId == 0u || (trigId % 2u == 1u && vertId == 1u))
    {
      posy = posy + u_cmap_uniforms.height;
    }
  let position = vec4<f32>(u_cmap_uniforms.position_x +
                           f32(posx) * u_cmap_uniforms.width / f32(u_cmap_uniforms.n_colors),
                           posy, 0.0, 1.0);
  return VertexOutput(position, f32(posx));
}

@fragment
fn colormap_fragment(vert: VertexOutput) -> @location(0) vec4<f32> {
  if (u_cmap_uniforms.height == 0.) {
    discard;
  }
  let min = u_cmap_uniforms.min;
  let max = u_cmap_uniforms.max;
  let v = min + (max-min) * vert.val / f32(u_cmap_uniforms.n_colors);
  let color = getColor(v);
  return color;
}
