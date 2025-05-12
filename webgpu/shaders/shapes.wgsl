#import light
#import camera
#import colormap

@group(0) @binding(101) var<storage, read> shape_values: array<f32>;
@group(0) @binding(102) var<storage, read> shape_positions: array<f32>;

struct ShapeVertexIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
};

struct ShapeVertexOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) value: f32,
};

@vertex fn cylinder_vertex_main(
    vert: ShapeVertexIn,
    @builtin(instance_index) instance_index: u32,
) -> ShapeVertexOut {
    var out: ShapeVertexOut;
    let i0 = 2 * instance_index * 3;
    let pstart = vec3f(shape_positions[i0], shape_positions[i0 + 1], shape_positions[i0 + 2]);
    let pend = vec3f(shape_positions[i0 + 3], shape_positions[i0 + 4], shape_positions[i0 + 5]);
    let v = pend - pstart;
    let q = quaternion(v, vec3f(0., 0., 1.));
    var pref = vert.position;
    pref.z *= length(v);
    let p = pstart + rotate(pref, q);
    out.position = cameraMapPoint(p);
    out.normal = cameraMapNormal(rotate(vert.normal, q)).xyz;
    out.value = mix(shape_values[2 * instance_index], shape_values[2 * instance_index + 1], vert.position.z);
    return out;
}

@fragment fn shape_fragment_main(
    input: ShapeVertexOut,
) -> @location(0) vec4f {
    let color = getColor(input.value);
    return lightCalcColor(normalize(input.normal), color);
}


fn quaternion(vTo: vec3f, vFrom: vec3f) -> vec4f {
    const EPS: f32 = 1e-6;
    // assume that vectors are not normalized
    let n = length(vTo);
    var r = n + dot(vFrom, vTo);
    var tmp: vec3f;

    if r < EPS {
        r = 0.0;
        if abs(vFrom.x) > abs(vFrom.z) {
            tmp = vec3(-vFrom.y, vFrom.x, 0.0);
        } else {
            tmp = vec3(0, -vFrom.z, vFrom.y);
        }
    } else {
        tmp = cross(vFrom, vTo);
    }
    return normalize(vec4(tmp.x, tmp.y, tmp.z, r));
}

// apply a rotation-quaternion to the given vector
// (source: https://goo.gl/Cq3FU0)
fn rotate(v: vec3f, q: vec4f) -> vec3f {
    let t: vec3f = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
