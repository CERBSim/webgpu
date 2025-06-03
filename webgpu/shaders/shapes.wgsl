#import light
#import camera
#import colormap

struct ShapeVertexIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) instance_position: vec3f,
    @location(3) instance_direction: vec3f,
    @location(4) instance_color_bot: vec4f,
    @location(5) instance_color_top: vec4f,
    @location(6) z_range: vec2f,
};

struct ShapeVertexOut {
    @builtin(position) position: vec4f,
    @location(0) p: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) @interpolate(flat) instance: u32,
};

@group(0) @binding(10) var<uniform> u_shape: ShapeUniform;

struct ShapeUniform {
    scale: f32,
    scale_mode: u32,
    padding1: f32,
    padding2: f32,
};

@vertex fn shape_vertex_main(
    vert: ShapeVertexIn,
    @builtin(instance_index) instance_index: u32,
) -> ShapeVertexOut {
    var out: ShapeVertexOut;
    let i0 = 2 * instance_index * 3;
    let pstart = vert.instance_position;
    let pend = vert.instance_position + vert.instance_direction;
    let v = pend - pstart;
    let q = quaternion(v, vec3f(0., 0., 1.));
    var pref = vert.position;
    if(u_shape.scale_mode == 0u) {
        pref *= length(v);
    }
    else if(u_shape.scale_mode == 1u) {
        pref.z *= length(v);
    }
    let p = pstart + u_shape.scale * rotate(pref, q);
    out.p = p;
    out.position = cameraMapPoint(p);
    out.normal = normalize(rotate(vert.normal, q));
    let lam = (vert.position.z-vert.z_range.x) / (vert.z_range.y-vert.z_range.x);
    out.color = mix(vert.instance_color_bot, vert.instance_color_top, lam);
    out.instance = instance_index;
    return out;
}

@fragment fn shape_fragment_main_value(
    input: ShapeVertexOut,
) -> @location(0) vec4f {
    let color = getColor(input.color.x);
    return lightCalcColor(input.p, input.normal, color);
}

@fragment fn shape_fragment_main_color(
    input: ShapeVertexOut,
) -> @location(0) vec4f {
    return lightCalcColor(input.p, input.normal, input.color);
}

@fragment fn shape_fragment_main_select(
    input: ShapeVertexOut,
) -> @location(0) vec4<u32> {
    return vec4<u32>(@RENDER_OBJECT_ID@, input.instance, bitcast<u32>(input.color.x), 0);
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
