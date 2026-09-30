#import camera
#import light
#import clipping

struct DynTrigParams {
  wireframe: u32,
  line_width: f32,
  depth_bias: f32,
  wireframe_opacity: f32,
  wireframe_color: vec4f,
};

// per triangle: 9 f32 coordinates (bitcast), group id | hidden edges << 29,
// packed rgba8 colour (0 = group colour)
@group(0) @binding(95) var<storage> u_dyn_trigs : array<u32>;
@group(0) @binding(96) var<storage> u_dyn_colors : array<vec4f>;
@group(0) @binding(97) var<uniform> u_dyn_params : DynTrigParams;

const DYN_STRIDE: u32 = 11u;
const DYN_GROUP_MASK: u32 = 0x1fffffffu;

struct DynTrigOutput {
  @invariant @builtin(position) position: vec4f,
  @location(0) p: vec3f,
  @location(1) n: vec3f,
  @location(2) bary: vec3f,
  @location(3) @interpolate(flat) color: vec4f,
  @location(4) @interpolate(flat) trig: u32,
  @location(5) @interpolate(flat) group: u32,
  @location(6) @interpolate(flat) hidden: u32,
};

fn dynPoint(base: u32, i: u32) -> vec3f {
  let o = base + 3u * i;
  return vec3f(bitcast<f32>(u_dyn_trigs[o]),
               bitcast<f32>(u_dyn_trigs[o + 1u]),
               bitcast<f32>(u_dyn_trigs[o + 2u]));
}

fn dynPalette(g: u32) -> vec4f {
  let h = fract(f32(g) * 0.618034 + 0.1);
  let k = fract(vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0) + h) * 6.0 - 3.0;
  let rgb = clamp(abs(k) - 1.0, vec3f(0.0), vec3f(1.0));
  return vec4f(mix(vec3f(1.0), rgb, 0.55), 1.0);
}

fn dynColor(group: u32, packed: u32) -> vec4f {
  if (packed != 0u) {
    return unpack4x8unorm(packed);
  }
  if (group < arrayLength(&u_dyn_colors)) {
    let c = u_dyn_colors[group];
    if (c.a >= 0.0) {
      return c;
    }
  }
  return dynPalette(group);
}

@vertex
fn vertex_main(@builtin(vertex_index) vertexId: u32) -> DynTrigOutput {
  let trig = vertexId / 3u;
  let local = vertexId % 3u;
  let base = trig * DYN_STRIDE;
  let p0 = dynPoint(base, 0u);
  let p1 = dynPoint(base, 1u);
  let p2 = dynPoint(base, 2u);
  var p = p0;
  if (local == 1u) { p = p1; }
  if (local == 2u) { p = p2; }
  var bary = vec3f(0.0);
  bary[local] = 1.0;
  let packed = u_dyn_trigs[base + 9u];
  let group = packed & DYN_GROUP_MASK;
  let color = dynColor(group, u_dyn_trigs[base + 10u]);
  var position = cameraMapPoint(p);
  position.z -= u_dyn_params.depth_bias * position.w;
  return DynTrigOutput(position, p, cross(p1 - p0, p2 - p0), bary,
                       color, trig, group, packed >> 29u);
}

@fragment
fn fragment_main(input: DynTrigOutput) -> @location(0) vec4f {
  // edge v0v1 lies at bary.z == 0, v1v2 at bary.x == 0, v2v0 at bary.y == 0
  let h = input.hidden;
  let d = min(min(select(input.bary.z, 1.0, (h & 1u) != 0u),
                  select(input.bary.x, 1.0, (h & 2u) != 0u)),
              select(input.bary.y, 1.0, (h & 4u) != 0u));
  let w = fwidth(d) * u_dyn_params.line_width;
  checkClipping(input.p);
  var color = input.color;
  if (color.a < 0.01) {
    discard;
  }
#ifdef OPAQUE_PASS
  if (color.a < 1.0) { discard; }
#endif OPAQUE_PASS
#ifdef TRANSPARENT_PASS
  if (color.a >= 1.0) { discard; }
#endif TRANSPARENT_PASS
  var lit = lightCalcColor(input.p, input.n, color);
  if (u_dyn_params.wireframe != 0u) {
    let wc = u_dyn_params.wireframe_color;
    let edge = (1.0 - smoothstep(0.5 * w, w, d)) * wc.a * u_dyn_params.wireframe_opacity;
    lit = vec4f(mix(lit.rgb, wc.rgb * lit.a, edge), lit.a);
  }
  return lit;
}

#ifdef SELECT_PIPELINE
@fragment fn fragment_select(input: DynTrigOutput) -> @location(0) vec4<u32> {
  checkClipping(input.p);
  if (input.color.a < 0.01) {
    discard;
  }
  return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.position.z), input.trig, input.group);
}
#endif SELECT_PIPELINE
