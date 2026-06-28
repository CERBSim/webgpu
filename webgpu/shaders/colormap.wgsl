@group(0) @binding(6) var u_colormap_texture : texture_2d<f32>;
@group(0) @binding(7) var u_colormap_sampler : sampler;
@group(0) @binding(5) var<uniform> u_cmap_uniforms : ColormapUniforms;
@group(0) @binding(9) var<uniform> u_cbar_uniforms : ColorbarUniforms;

struct ColormapUniforms {
  min: f32,
  max: f32,
  discrete: u32,
  n_colors: u32,
};

struct ColorbarUniforms {
  position: vec2f,
  width: f32,
  height: f32,
  vertical: u32,
};



fn getColor(value: f32) -> vec4<f32> {
  var v = (value - u_cmap_uniforms.min) / (u_cmap_uniforms.max - u_cmap_uniforms.min);
  let N = u_cmap_uniforms.n_colors;
  let Nf = f32(N);

  v = clamp(v, 0.0, 1.0);

  var colorIndex: f32;
  if (u_cmap_uniforms.discrete == 1u) {
    colorIndex = min(floor(v * Nf), Nf - 1.0);
  } else {
    colorIndex = v * (Nf - 1.0);
  }

  var uv = vec2f(0.5, 0.5);
  if (N < 1024u) {
    uv.x = (colorIndex + 0.5) / Nf;
  } else {
    let texWidth = 1024.0;
    let x = colorIndex % texWidth;
    let y = floor(colorIndex / texWidth);
    let texHeight = ceil(Nf / texWidth);
    uv.x = (x + 0.5) / texWidth;
    uv.y = (y + 0.5) / texHeight;
  }

  return textureSample(u_colormap_texture, u_colormap_sampler, uv);
}
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) val: f32,
};

@vertex
fn colormap_vertex(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId: u32) -> VertexOutput {
  // index along the gradient (color) axis, 0 .. n_colors
  var grad = trigId / 2u;
  if(vertId == 2u || (trigId % 2u == 1u && vertId == 1u))
    {
      grad = grad + 1u;
    }
  // position across the bar thickness, 0 (near edge) or 1 (far edge)
  var cross = 0.0;
  if(vertId == 0u || (trigId % 2u == 1u && vertId == 1u))
    {
      cross = 1.0;
    }

  let g = f32(grad) / f32(u_cmap_uniforms.n_colors);
  var pos: vec2<f32>;
  if (u_cbar_uniforms.vertical == 1u) {
    // gradient runs bottom (min) to top (max) along height, thickness along width
    pos = vec2<f32>(u_cbar_uniforms.position.x + cross * u_cbar_uniforms.width,
                    u_cbar_uniforms.position.y + g * u_cbar_uniforms.height);
  } else {
    // gradient runs left (min) to right (max) along width, thickness along height
    pos = vec2<f32>(u_cbar_uniforms.position.x + g * u_cbar_uniforms.width,
                    u_cbar_uniforms.position.y + cross * u_cbar_uniforms.height);
  }
  return VertexOutput(vec4<f32>(pos, 0.0, 1.0), f32(grad));
}

@fragment
fn colormap_fragment(vert: VertexOutput) -> @location(0) vec4<f32> {
  if (u_cbar_uniforms.height == 0.) {
    discard;
  }
  let min = u_cmap_uniforms.min;
  let max = u_cmap_uniforms.max;
  let v = min + (max-min) * vert.val / f32(u_cmap_uniforms.n_colors);
  let color = getColor(v);
  return color;
}
