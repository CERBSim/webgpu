struct MeshFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) p: vec3<f32>,
  @location(2) n: vec3<f32>,
  @location(3) @interpolate(flat) id: u32,
  @location(4) @interpolate(flat) index: u32,
};

// A triangle as part of a 3d element (thus, 3 barycentric coordinates)
struct SubTrig {
  lam: array<vec3f, 3>,
  id: u32,
}

struct ClipTetResult {
  n: u32,
  trigs: array<SubTrig, 2>,
}

// clip tet such that the clip triangle(s) have value 0 everywhere
fn clipTet(lam: array<vec3f, 4>, values: array<f32, 4>) -> ClipTetResult {
    let ei = 0u;
    var trigs = ClipTetResult(0, array<SubTrig, 2>(SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0), SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0)));
    var p_pos = array<u32, 4>(0u, 0u, 0u, 0u);
    var p_neg = array<u32, 4>(0u, 0u, 0u, 0u);

    var n_pos: u32 = 0u;
    var n_neg: u32 = 0u;

    for (var i = 0u; i < 4u; i++) {
        if values[i] > 0.0 {
            p_pos[n_pos] = i;
            n_pos++;
        } else {
            p_neg[n_neg] = i;
            n_neg++;
        }
    }

    if n_pos == 4u || n_neg == 4u {
        return trigs;
    }

    if n_pos == 3u {
        trigs.n = 1u;
        n_pos = 1u;
        n_neg = 3u;
        p_pos[3] = p_neg[0];
        p_neg[0] = p_pos[0];
        p_neg[1] = p_pos[1];
        p_neg[2] = p_pos[2];
        p_pos[0] = p_pos[3];
    }

    if n_pos == 1u {
        trigs.n = 1u;
        trigs.trigs[0].id = ei;
        for (var i = 0u; i < 3u; i++) {
            let t = values[p_pos[0] ] / (values[p_pos[0] ] - values[p_neg[i] ]);
            let lam_trig = mix(lam[p_pos[0] ], lam[p_neg[i] ], t);
            trigs.trigs[0].lam[i] = lam_trig;
        }
        return trigs;
    }

  // two points before, two points behind clipping plane
  // -> two triangles
    trigs.n = 2u;
    var pairs = array<vec2u,4>(
        vec2u(p_pos[1], p_neg[0]),
        vec2u(p_pos[0], p_neg[0]),
        vec2u(p_pos[0], p_neg[1]),
        vec2u(p_pos[1], p_neg[1])
    );
    var points: array<vec3f, 4> = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    for (var i = 0; i < 4; i++) {
        let p0 = pairs[i].x;
        let p1 = pairs[i].y;
        let t = values[p0 ] / (values[p0] - values[p1]);
        let lam_trig = mix(lam[p0], lam[p1], t);
        points[i] = lam_trig;
    }
    trigs.trigs[0].id = ei;
    trigs.trigs[0].lam = array(points[0], points[1], points[2]);
    trigs.trigs[1].id = ei;
    trigs.trigs[1].lam = array(points[0], points[2], points[3]);
    return trigs;
}

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
fn vertexClipTet(@builtin(vertex_index) vertId: u32, @builtin(instance_index) elId: u32) -> MeshFragmentInput {
    const N: u32 = 4;
    let faceId: u32 = vertId / 3;
    let el = u_tets[elId];
    var p: array<vec3f, 4>;

    for (var i = 0u; i < N; i++) {
        let n = 3 * el.p[i];
        p[i] = vec3<f32>(vertices[n], vertices[n + 1], vertices[n + 2]);
    }

    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));

    let s = u_mesh.shrink;
    let values = array(p[0].z-s, p[1].z-s, p[2].z-s, p[3].z-s);

    let trigs = clipTet(lam, values);

    var points = array<vec3<f32>, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    if(trigs.n > faceId) {
      let trig = trigs.trigs[faceId];
      for (var k = 0u; k < 3u; k++) {
       let lam = vec4f(trig.lam[k], 1.0 - trig.lam[k].x - trig.lam[k].y - trig.lam[k].z);
        for (var i = 0u; i < 3u; i++) {
          points[k][i] = lam.x * p[0][i] + lam.y * p[1][i] + lam.z * p[2][i] + lam.w * p[3][i];
        }
      }
    }

    return calcMeshFace(vec4<f32>(1., 0., 0., 1.), points, vertId, el.nr, el.index);
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
