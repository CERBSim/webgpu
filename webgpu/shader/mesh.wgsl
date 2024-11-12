// Mesh element types
struct Seg { p: array<u32, 2>, nr: u32, index: u32 };
struct Trig { p: array<u32, 3>, nr: u32, index: u32 };
struct Quad { p: array<u32, 4>, nr: u32, index: u32 };
struct Tet { p: array<u32, 4>, nr: u32, index: u32 };
struct Pyramid { p: array<u32, 5>, nr: u32, index: u32 };
struct Prism { p: array<u32, 6>, nr: u32, index: u32 };
struct Hex { p: array<u32, 8>, nr: u32, index: u32 };

// Inner edges, wireframe
struct Edge { p: array<u32, 2> };

@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;
@group(0) @binding(21) var<storage> u_edges : array<Edge>;
@group(0) @binding(22) var<storage> u_segs : array<Seg>;
@group(0) @binding(23) var<storage> u_trigs : array<Trig>;
@group(0) @binding(24) var<storage> u_quads : array<Quad>;
@group(0) @binding(25) var<storage> u_tets : array<Tet>;
@group(0) @binding(26) var<storage> u_pyramids : array<Pyramid>;
@group(0) @binding(27) var<storage> u_prisms : array<Prism>;
@group(0) @binding(28) var<storage> u_hexes : array<Hex>;

struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) p: vec3<f32>,
  @location(2) n: vec3<f32>,
  @location(3) @interpolate(flat) id: u32,
  @location(4) @interpolate(flat) index: u32,
};

fn calcMeshFace(color: vec4<f32>, p: array<vec3<f32>, 3>, vertId: u32, nr: u32, index: u32) -> MeshFragmentInput {
    let n = cross(p[1] - p[0], p[2] - p[0]);
    let point = p[vertId % 3];
    let position = calcPosition(point);
    return MeshFragmentInput(position, color, point, n, nr, index);
}

@fragment
fn fragmentMesh(input: MeshFragmentInput) -> @location(0) vec4<f32> {
    // checkClipping(input.p);
    let n4 = u_view.normal_mat * vec4(input.n, 1.0);
    let n = normalize(n4.xyz);
    let brightness = clamp(dot(n, normalize(vec3<f32>(-1., -3., -3.))), .0, 1.) * 0.7 + 0.3;
    let color = input.color.xyz * brightness;
    return vec4<f32>(color, input.color.w);
}

@vertex
fn vertexMeshTet(@builtin(vertex_index) vertId: u32, @builtin(instance_index) elId: u32) -> MeshFragmentInput {
    const N: u32 = 4;
    let faceId: u32 = vertId / 3;
    let el = u_tets[elId];
    var p: array<vec3<f32>, 4>;

    var center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < N; i++) {
        let n = 3 * el.p[i];
        p[i] = vec3<f32>(vertices[n], vertices[n + 1], vertices[n + 2]);
        center += p[i] / f32(N);
    }

    for (var i = 0u; i < 4u; i++) {
        p[i] = mix(center, p[i], u_mesh.shrink);
    }

    let pi = TET_FACES[faceId];
    let points = array<vec3<f32>, 3>(p[pi[0] ], p[pi[1] ], p[pi[2] ]);

    return calcMeshFace(vec4<f32>(1., 0., 0., 1.), points, vertId, el.nr, el.index);
}


@vertex
fn vertexMeshPyramid(@builtin(vertex_index) vertId: u32, @builtin(instance_index) elId: u32) -> MeshFragmentInput {
    const N: u32 = 5;
    let faceId = vertId / 3;
    let el = u_pyramids[elId];
    var p: array<vec3<f32>, 5>;

    var center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < N; i++) {
        p[i] = vec3<f32>(vertices[3 * el.p[i] ], vertices[3 * el.p[i] + 1], vertices[3 * el.p[i] + 2]);
        center += p[i] / f32(N);
    }

    for (var i = 0u; i < N; i++) {
        p[i] = mix(center, p[i], u_mesh.shrink);
    }

    let pi = PYRAMID_FACES[faceId];
    var points = array<vec3<f32>, 3>(p[pi[0] ], p[pi[1] ], p[pi[2] ]);

    return calcMeshFace(vec4<f32>(1., 0., 1., 1.), points, vertId, el.nr, el.index);
}


@vertex
fn vertexMeshPrism(@builtin(vertex_index) vertId: u32, @builtin(instance_index) elId: u32) -> MeshFragmentInput {
    const N: u32 = 6;
    let faceId = vertId / 3;
    let el = u_prisms[elId];
    var p: array<vec3<f32>, 6>;

    var center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < N; i++) {
        p[i] = vec3<f32>(vertices[3 * el.p[i] ], vertices[3 * el.p[i] + 1], vertices[3 * el.p[i] + 2]);
        center += p[i] / f32(N);
    }

    for (var i = 0u; i < N; i++) {
        p[i] = mix(center, p[i], u_mesh.shrink);
    }

    let pi = PRISM_FACES[faceId];
    var points = array<vec3<f32>, 3>(p[pi[0] ], p[pi[1] ], p[pi[2] ]);

    return calcMeshFace(vec4<f32>(0., 1., 1., 1.), points, vertId, el.nr, el.index);
}


@vertex
fn vertexMeshHex(@builtin(vertex_index) vertId: u32, @builtin(instance_index) elId: u32) -> MeshFragmentInput {
    const N: u32 = 8;
    let faceId = vertId / 3;
    let el = u_hexes[elId];
    var p: array<vec3<f32>, 8>;

    var center = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < N; i++) {
        p[i] = vec3<f32>(vertices[3 * el.p[i] ], vertices[3 * el.p[i] + 1], vertices[3 * el.p[i] + 2]);
        center += p[i] / f32(N);
    }

    for (var i = 0u; i < N; i++) {
        p[i] = mix(center, p[i], u_mesh.shrink);
    }

    let pi = HEX_FACES[faceId];
    var points = array<vec3<f32>, 3>(p[pi[0] ], p[pi[1] ], p[pi[2] ]);

    return calcMeshFace(vec4<f32>(1., 1., 0., 1.), points, vertId, el.nr, el.index);
}


const TET_FACES = array(
    vec3(0, 2, 1),
    vec3(0, 1, 3),
    vec3(1, 2, 3),
    vec3(2, 0, 3)
);

const PYRAMID_FACES = array(
    vec3(0, 2, 1),
    vec3(0, 3, 2),
    vec3(0, 1, 4),
    vec3(1, 2, 4),
    vec3(2, 3, 4),
    vec3(3, 0, 4)
);

const PRISM_FACES = array(
    vec3(0, 2, 1),
    vec3(3, 4, 5),
    vec3(0, 1, 4),
    vec3(0, 4, 3),
    vec3(1, 2, 5),
    vec3(1, 5, 4),
    vec3(2, 0, 3),
    vec3(2, 3, 5)
);

const HEX_FACES = array(
    vec3(0, 3, 1),
    vec3(3, 2, 1),
    vec3(4, 5, 6),
    vec3(4, 6, 7),
    vec3(0, 1, 5),
    vec3(0, 5, 4),
    vec3(1, 2, 6),
    vec3(1, 6, 5),
    vec3(2, 3, 7),
    vec3(2, 7, 6),
    vec3(3, 0, 4),
    vec3(3, 4, 7)
);
