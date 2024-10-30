struct EdgeP1 { p: array<f32, 6> };

struct TrigP1 { p: array<f32, 9>, index: i32 }; // 3 vertices with 3 coordinates each, don't use vec3 due to 16 byte alignment
struct TrigP2 { p: array<f32, 18>, index: i32 };

struct Uniforms {
  mat: mat4x4<f32>,
  clipping_plane: vec4<f32>,
  colormap: vec2<f32>,
  scaling: vec2<f32>,
  aspect: f32,
  eval_mode: u32,
  do_clipping: u32,
  font_width: u32,
  font_height: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
};

const VALUES_OFFSET: u32 = 2; // storing number of components and order of basis functions in first two entries

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var colormap : texture_1d<f32>;
@group(0) @binding(2) var colormap_sampler : sampler;

@group(0) @binding(4) var<storage> edges_p1 : array<EdgeP1>;
@group(0) @binding(5) var<storage> trigs_p1 : array<TrigP1>;
@group(0) @binding(6) var<storage> trig_function_values : array<f32>;
@group(0) @binding(7) var<storage> seg_function_values : array<f32>;
@group(0) @binding(8) var<storage> vertices : array<f32>;
@group(0) @binding(9) var<storage> index : array<u32>;

@group(0) @binding(10) var gBufferLam : texture_2d<f32>;
@group(0) @binding(11) var font : texture_2d<f32>;

struct VertexOutput1d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: f32,
  @location(2) @interpolate(flat) id: u32,
};

struct VertexOutput2d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec2<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

fn calcPosition(p: vec3<f32>) -> vec4<f32> {
    return uniforms.mat * vec4<f32>(p, 1.0);
}

fn checkClipping(p: vec3<f32>) {
    if uniforms.do_clipping != 0 {
        if dot(uniforms.clipping_plane, vec4<f32>(p, 1.0)) < 0 {
        discard;
        }
    }
}

fn getColor(value: f32) -> vec4<f32> {
    let v = (value - uniforms.colormap.x) / (uniforms.colormap.y - uniforms.colormap.x);
    return textureSample(colormap, colormap_sampler, v);
}

@vertex
fn mainVertexEdgeP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) edgeId: u32) -> VertexOutput1d {
    let edge = edges_p1[edgeId];
    var p: vec3<f32> = vec3<f32>(edge.p[3 * vertexId], edge.p[3 * vertexId + 1], edge.p[3 * vertexId + 2]);

    var lam: f32 = 0.0;
    if vertexId == 0 {
        lam = 1.0;
    }

    var position = calcPosition(p);
    return VertexOutput1d(position, p, lam, edgeId);
}

fn calcTrig(p: array<vec3<f32>, 3>, vertexId: u32, trigId: u32) -> VertexOutput2d {
    var lam: vec2<f32> = vec2<f32>(0.);
    if vertexId < 2 {
        lam[vertexId] = 1.0;
    }

    let position = calcPosition(p[vertexId]);
    let normal = cross(p[1] - p[0], p[2] - p[0]);

    return VertexOutput2d(position, p[vertexId], lam, trigId, normal);
}

@vertex
fn mainVertexTrigP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let trig = trigs_p1[trigId];
    var p = array<vec3<f32>, 3>(
        vec3<f32>(trig.p[0], trig.p[1], trig.p[2]),
        vec3<f32>(trig.p[3], trig.p[4], trig.p[5]),
        vec3<f32>(trig.p[6], trig.p[7], trig.p[8])
    );
    return calcTrig(p, vertexId, trigId);
}

@vertex
fn mainVertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let vid = array<u32, 3>(
        index[3 * trigId + 0],
        index[3 * trigId + 1],
        index[3 * trigId + 2]
    );
    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[3 * vid[0] ], vertices[3 * vid[0] + 1], vertices[3 * vid[0] + 2]),
        vec3<f32>(vertices[3 * vid[1] ], vertices[3 * vid[1] + 1], vertices[3 * vid[1] + 2]),
        vec3<f32>(vertices[3 * vid[2] ], vertices[3 * vid[2] + 1], vertices[3 * vid[2] + 2])
    );
    return calcTrig(p, vertexId, trigId);
}


@fragment
fn mainFragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let value = evalTrig(input.id, 0u, input.lam);
    return getColor(value);
}

@fragment
fn mainFragmentTrigMesh(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) @interpolate(flat) id: u32) -> @location(0) vec4<f32> {
    checkClipping(p);
    let value = id;
    return vec4<f32>(0., 1.0, 0.0, 1.0);
}

@fragment
fn mainFragmentEdge(@location(0) p : vec3< f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}

struct DeferredFragmentOutput {
  @builtin(position) fragPosition: vec4<f32>,

};

@fragment
fn mainFragmentDeferred(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let bufferSize = textureDimensions(gBufferLam);
    let coordUV = coord.xy / vec2f(bufferSize);

    let g_values = textureLoad(
        gBufferLam,
        vec2i(floor(coord.xy)),
        0
    );
    let lam = g_values.yz;
    if lam.x == -1.0 {discard;}
    let trigId = bitcast<u32>(g_values.x);

    let value = evalTrig(trigId, 0u, lam);
    return getColor(value);
}


@fragment
fn mainFragmentTrigToGBuffer(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) @interpolate(flat) id: u32) -> @location(0) vec4<f32> {
    checkClipping(p);
    let value = evalTrig(id, 0u, lam);
    return vec4<f32>(bitcast<f32>(id), lam, 0.0);
}

struct VertexOutputDeferred {
  @builtin(position) p: vec4<f32>,
};


@vertex
fn mainVertexDeferred(@builtin(vertex_index) vertexId: u32) -> VertexOutputDeferred {
    var position = vec4<f32>(-1., -1., 0., 1.);
    if vertexId == 1 || vertexId == 3 {
        position.x = 1.0;
    }
    if vertexId >= 2 {
        position.y = 1.0;
    }

    return VertexOutputDeferred(position);
}


struct FragmentTextInput {
    @builtin(position) fragPosition: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn mainVertexPointNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) pointId: u32) -> FragmentTextInput {
    var p = vec3<f32>(vertices[3 * pointId], vertices[3 * pointId + 1], vertices[3 * pointId + 2]);
    if uniforms.do_clipping != 0 {
        if dot(uniforms.clipping_plane, vec4<f32>(p, 1.0)) < 0 {
            return FragmentTextInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
        }
    }

    var position = calcPosition(p);
    let i_digit = vertexId / 6;
    let vi = vertexId % 6;

    var length = 1u;
    var n = 10u;
    while n <= pointId + 1 {
        length++;
        n *= 10u;
    }

    if i_digit >= length {
        return FragmentTextInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
    }

    var digit = pointId + 1;
    for (var i = 0u; i < i_digit; i++) {
        digit = digit / 10;
    }
    digit = digit % 10;

    let w: f32 = 2 * f32(uniforms.font_width) / 1000.;
    let h: f32 = 2 * f32(uniforms.font_height) / 800.;

    var tex_coord = vec2<f32>(
        f32((digit + 16) * uniforms.font_width),
        f32(uniforms.font_height)
    );

    if vi == 2 || vi == 4 || vi == 5 {
        position.y += h * position.w;
        tex_coord.y = 0.0;
    }

    position.x += f32(length - i_digit -1) * w * position.w;

    if vi == 1 || vi == 2 || vi == 4 {
        position.x += w * position.w;
        tex_coord.x += f32(uniforms.font_width);
    }

    return FragmentTextInput(position, tex_coord);
}

@fragment
fn mainFragmentText(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let alpha: f32 = textureLoad(
        font,
        vec2i(floor(tex_coord)),
        0
    ).x;
    if alpha < 0.01 {
      discard;
    }
    return vec4(0., 0., 0., alpha);
}
