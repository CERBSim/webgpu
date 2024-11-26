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
    var result : bool = true;
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
fn vertexTrigP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let trig = trigs_p1[trigId];
    var p = array<vec3<f32>, 3>(
        vec3<f32>(trig.p[0], trig.p[1], trig.p[2]),
        vec3<f32>(trig.p[3], trig.p[4], trig.p[5]),
        vec3<f32>(trig.p[6], trig.p[7], trig.p[8])
    );
    return calcTrig(p, vertexId, trigId);
}

@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    let vid = array<u32, 3>(
        trigs[3 * trigId + 0],
        trigs[3 * trigId + 1],
        trigs[3 * trigId + 2]
    );
    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[3 * vid[0] ], vertices[3 * vid[0] + 1], vertices[3 * vid[0] + 2]),
        vec3<f32>(vertices[3 * vid[1] ], vertices[3 * vid[1] + 1], vertices[3 * vid[1] + 2]),
        vec3<f32>(vertices[3 * vid[2] ], vertices[3 * vid[2] + 1], vertices[3 * vid[2] + 2])
    );
    return calcTrig(p, vertexId, trigId);
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

fn random(seed: u32) -> u32 {
   var value : u32 = seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

fn getSeed(n: u32) -> u32 {
    // https://www.burtleburtle.net/bob/hash/integer.html
    var seed = (n ^ 61) ^ (n >> 16);
    seed *= seed * 9;
    seed = seed ^ (seed >> 4);
    seed = seed * 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}
fn vector_field(p_: vec2f) -> vec2f {
    // Normalize and shift coordinates to center around (500, 400)
    var p = p_ - vec2f(500.0, 400.0);
    p.x = p.x / 1000.0;
    p.y = p.y / 800.0;

    // Initialize the vector field
    var v = vec2f(0.0, 0.0);

    // Parameters for vortex centers
    let vortex_centers = array(
        vec2f(-0.3, -0.2),
        vec2f(0.3, 0.1),
        vec2f(-0.2, 0.3),
        vec2f(0.2, -0.4),
    );

    // Add vortex contributions
    for (var i = 0; i < 4; i++) {
        var d = p - vortex_centers[i]; // Displacement vector from vortex center
        var dist_sq = max(dot(d, d), 0.0001); // Squared distance
        var strength = 0.2; // Vortex strength (adjustable)
        var sign = 1.0;
        if(i==0) {sign = -1.0;}
        v += sign*vec2f(-d.y, d.x) * (strength / dist_sq);
    }

    // Add a global swirling effect
    var swirl_strength = 0.1;
    v += vec2f(-p.y, p.x) * swirl_strength;

    // Add a radial outward flow
    var radial_strength = 0.05;
    v += p * radial_strength;

    return v;
}


fn vector_field1(p_: vec2f) -> vec2f {
    var p = p_-vec2f(500.0, 400.0);
    p.x = p.x / 1000.0;
    p.y = p.y / 800.0;
    return vec2f(-p.y, p.x);
}

fn noise(x: u32, y: u32) -> f32 {
  let seed = getSeed(0xFFFFu * y + x);
  var rand = random(seed);

  return f32(rand) / f32(0xFFFFFFFFu);
}

fn dropletNoise(x: u32, y: u32) -> f32 {
  let rand : f32 = noise(x/5, y/5);

  let dx = x%5-2;
  let dy = y%5-2;
  let d = dx*dx + dy*dy;

  if (rand < 0.85) {
    return 0.0;
  }

  return 1.0 - 0.1 * f32(d);
}


fn lineIntegralConvolution(x: u32, y: u32, w: u32, h: u32) -> f32 {
    var sum : f32 = 0;
    var weight : f32 = 0;
    let KERNEL_LEN = 125u;
    let oriented : u32 = 0u;

    for (var dir : i32 = -1; dir <= 1; dir += 2) {
        var p = vec2f(f32(x)+.5, f32(y)+.5);

        for (var k : u32 = 0; k < KERNEL_LEN; k++) {
            var v : vec2f = vector_field(p);
            v = normalize(v);

            p += f32(dir) * v;

            if(p.x < 0.0 || p.x >= f32(w) || p.y < 0.0 || p.y >= f32(h)) {
              break;
            }

            let ix = u32(p.x);
            let iy = u32(p.y);

            var kernel_weight : f32 = f32(k) / f32(KERNEL_LEN); 
            if(oriented==0) {
              kernel_weight  = 1.0 - f32(k) / f32(KERNEL_LEN); 
              sum += kernel_weight * noise(ix, iy);
              weight += kernel_weight;
            }
            else {
              var t = 0.5*(1.0 + f32(dir) * f32(k) / f32(KERNEL_LEN));
              kernel_weight = 0.1+0.9*t*t*t*t;
              // kernel_weight = t*t;
              sum += 1.3*kernel_weight * dropletNoise(ix, iy);
              // weight = 0.5*f32(2*KERNEL_LEN+1);
              weight += kernel_weight;
            }
        }
    }

    return sum / weight;
}

@fragment
fn fragmentDeferred(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(coord.x);
    let y = u32(coord.y);
    let frand = 1.6*lineIntegralConvolution(x, y, 1000u, 800u);
    let value = 0.25*length(vector_field(vec2f(f32(x), f32(y))));
    // let color = getColor(value);
    let color = getCetL20(value);
    // return vec4<f32>((0.6*frand+0.4)*color.xyz, 1.0);
    return vec4<f32>((0.8*frand+0.2)*color.xyz, 1.0);
    // return vec4<f32>(frand*color.xyz, 1.0);
    // return vec4<f32>(color.xyz, 1.0);
}

@fragment
fn fragmentDeferred1(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
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


fn getCetL20(t_: f32) -> vec4<f32> {
  let colors = array(
    vec3f(0.189, 0.189, 0.189),
    vec3f(0.250, 0.222, 0.416),
    vec3f(0.262, 0.267, 0.619),
    vec3f(0.238, 0.335, 0.762),
    vec3f(0.176, 0.447, 0.743),
    vec3f(0.166, 0.561, 0.553),
    vec3f(0.367, 0.636, 0.380),
    vec3f(0.608, 0.682, 0.211),
    vec3f(0.847, 0.711, 0.071),
    vec3f(0.976, 0.766, 0.078),
    vec3f(0.999, 0.864, 0.067),
    vec3f(0.975, 0.975, 0.039)
  );
  var t = clamp(t_, 0.0, 1.0);
  if(t==1.0) {return vec4f(colors[11], 1.0);}
  t = t;

  t = 11*t;

  let i = u32(t);
  let frac = t - f32(i);
  let c0 = colors[i];
  let c1 = colors[i+1];
  return vec4<f32>(
    mix(c0, c1, frac),
    1.0
  );
}
