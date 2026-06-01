@group(0) @binding(50) var<uniform> u_bg_uniforms : BackgroundUniforms;

struct BackgroundUniforms {
  position: vec2f,
  width: f32,
  height: f32,
  bg_color: vec3f,
};

struct BgVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn background_vertex(@builtin(vertex_index) vertId: u32) -> BgVertexOutput {
  let pad_x = 0.055;
  let pad_top = 0.04;
  let pad_bottom = 0.1;

  let left = u_bg_uniforms.position.x - pad_x;
  let right = u_bg_uniforms.position.x + u_bg_uniforms.width + pad_x;
  let top = u_bg_uniforms.position.y + u_bg_uniforms.height + pad_top;
  let bottom = u_bg_uniforms.position.y - pad_bottom;

  var pos: vec2<f32>;
  var uv: vec2<f32>;

  switch (vertId) {
    case 0u: { pos = vec2f(left, bottom);  uv = vec2f(0.0, 0.0); }
    case 1u: { pos = vec2f(right, bottom); uv = vec2f(1.0, 0.0); }
    case 2u: { pos = vec2f(left, top);     uv = vec2f(0.0, 1.0); }
    case 3u: { pos = vec2f(right, bottom); uv = vec2f(1.0, 0.0); }
    case 4u: { pos = vec2f(right, top);    uv = vec2f(1.0, 1.0); }
    case 5u: { pos = vec2f(left, top);     uv = vec2f(0.0, 1.0); }
    default: { pos = vec2f(0.0); uv = vec2f(0.0); }
  }

  return BgVertexOutput(vec4f(pos, 0.0, 1.0), uv);
}

@fragment
fn background_fragment(vert: BgVertexOutput) -> @location(0) vec4<f32> {
  if (u_bg_uniforms.height == 0.) {
    discard;
  }

  // Compute quad aspect ratio for circular corners
  let pad_x = 0.055;
  let pad_top = 0.04;
  let pad_bottom = 0.1;
  let quad_w = u_bg_uniforms.width + 2.0 * pad_x;
  let quad_h = u_bg_uniforms.height + pad_top + pad_bottom;
  let aspect = quad_w / quad_h;

  // Rounded rectangle SDF in aspect-corrected space
  let p = (vert.uv - vec2f(0.5)) * vec2f(aspect, 1.0);
  let half_size = vec2f(0.5 * aspect, 0.5);
  let radius = 0.07;

  let q = abs(p) - half_size + vec2f(radius);
  let dist = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - radius;

  // Soft fade at the rounded edge
  let edge_softness = 0.045;
  let shape_alpha = 1.0 - smoothstep(-edge_softness, 0.005, dist);

  let alpha = 0.75 * shape_alpha;
  if (alpha < 0.001) {
    discard;
  }

  let color = u_bg_uniforms.bg_color;
  return vec4f(color, alpha);
}
