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
    return u_view.model_view_projection * vec4<f32>(p, 1.0);
}

fn calcClipping(p: vec3<f32>) -> bool {
    var result: bool = true;
    if (u_clipping.mode & 0x01u) == 0x01u {
        if dot(u_clipping.plane, vec4<f32>(p, 1.0)) < 0 {
            result = false;
        }
    }
    if (u_clipping.mode & 0x02) == 0x02 {
        let d = distance(p, u_clipping.sphere.xyz);
        if d > u_clipping.sphere.w {
            result = false;
        }
    }
    return result;
}

fn checkClipping(p: vec3<f32>) {
    if calcClipping(p) == false {
    discard;
    }
}

fn getColor(value: f32) -> vec4<f32> {
    let v = (value - u_function.colormap.x) / (u_function.colormap.y - u_function.colormap.x);
    return textureSample(u_colormap_texture, u_colormap_sampler, v);
}

@vertex
fn vertexEdgeP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) edgeId: u32) -> VertexOutput1d {
    let edge = edges_p1[edgeId];
    var p: vec3<f32> = vec3<f32>(edge.p[3 * vertexId], edge.p[3 * vertexId + 1], edge.p[3 * vertexId + 2]);

    var lam: f32 = 0.0;
    if vertexId == 0 {
        lam = 1.0;
    }

    var position = calcPosition(p);
    return VertexOutput1d(position, p, lam, edgeId);
}

fn calcTrig(p: array<vec3<f32>, 3>, vertexId: u32, trigId: u32, faceSort: vec3u) -> VertexOutput2d {
    var lam: vec3<f32> = vec3<f32>(0.);
    lam[faceSort[vertexId] ] = 1.0;

    let position = calcPosition(p[vertexId]);
    let normal = cross(p[1] - p[0], p[2] - p[0]);

    return VertexOutput2d(position, p[vertexId], lam.xy, trigId, normal);
}

@vertex
fn vertexTrigP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let trig = trigs_p1[trigId];
    var p = array<vec3<f32>, 3>(
        vec3<f32>(trig.p[0], trig.p[1], trig.p[2]),
        vec3<f32>(trig.p[3], trig.p[4], trig.p[5]),
        vec3<f32>(trig.p[6], trig.p[7], trig.p[8])
    );
    return calcTrig(p, vertexId, trigId, vec3u(0,1,2));
}

@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    var vid = 3*vec3u(
        trigs[3 * trigId + 0],
        trigs[3 * trigId + 1],
        trigs[3 * trigId + 2]
    );

    var f = vec3u(0, 1, 2);

    if vid[f[0] ] > vid[f[1] ] {
        let t = f[0];
        f[0] = f[1];
        f[1] = t;
    }
    if vid[ f[1] ] > vid[f[2] ] {
        let t = f[1];
        f[1] = f[2];
        f[2] = t;
    }
    if vid[f[0] ] > vid[f[1] ] {
        let t = f[0];
        f[0] = f[1];
        f[1] = t;
    }

    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );
    return calcTrig(p, vertexId, trigId, f);
}


@fragment
fn fragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let value = evalTrig(input.id, 0u, input.lam);
    return getColor(value);
}

@fragment
fn fragmentTrigMesh(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) @interpolate(flat) id: u32) -> @location(0) vec4<f32> {
    checkClipping(p);
    let value = id;
    return vec4<f32>(0., 1.0, 0.0, 1.0);
}

@fragment
fn fragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}

struct DeferredFragmentOutput {
  @builtin(position) fragPosition: vec4<f32>,

};


@fragment
fn fragmentDeferred(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
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
fn fragmentTrigToGBuffer(@location(0) p: vec3<f32>, @location(1) lam: vec2<f32>, @location(2) @interpolate(flat) id: u32) -> @location(0) vec4<f32> {
    checkClipping(p);
    let value = evalTrig(id, 0u, lam);
    return vec4<f32>(bitcast<f32>(id), lam, 0.0);
}

struct VertexOutputDeferred {
  @builtin(position) p: vec4<f32>,
};


@vertex
fn vertexDeferred(@builtin(vertex_index) vertexId: u32) -> VertexOutputDeferred {
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
fn vertexPointNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) pointId: u32) -> FragmentTextInput {
    var p = vec3<f32>(vertices[3 * pointId], vertices[3 * pointId + 1], vertices[3 * pointId + 2]);
    if calcClipping(p) == false {
        return FragmentTextInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
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

    let w: f32 = 2 * f32(u_font.width) / 1000.;
    let h: f32 = 2 * f32(u_font.height) / 800.;

    var tex_coord = vec2<f32>(
        f32((digit + 16) * u_font.width),
        f32(u_font.height)
    );

    if vi == 2 || vi == 4 || vi == 5 {
        position.y += h * position.w;
        tex_coord.y = 0.0;
    }

    position.x += f32(length - i_digit -1) * w * position.w;

    if vi == 1 || vi == 2 || vi == 4 {
        position.x += w * position.w;
        tex_coord.x += f32(u_font.width);
    }

    return FragmentTextInput(position, tex_coord);
}

@fragment
fn fragmentText(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let alpha: f32 = textureLoad(
        u_font_texture,
        vec2i(floor(tex_coord)),
        0
    ).x;
    if alpha < 0.01 {
      discard;
    }
    return vec4(0., 0., 0., alpha);
}
