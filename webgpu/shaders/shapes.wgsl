#import light
#import camera
#import colormap
#import clipping

struct ShapeVertexIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) instance_position: vec3f,
    @location(3) instance_direction: vec3f,
    @location(4) instance_color_bot: vec4f,
    @location(5) instance_color_top: vec4f,
    @location(6) z_range: vec2f,
    @location(7) instance_direction_imag: vec3f,
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

struct ShapeComplexUniform {
    phase: f32,
    is_complex: u32,
    color_override: u32,
    padding: f32,
};
@group(0) @binding(11) var<uniform> u_shape_complex: ShapeComplexUniform;

@vertex fn shape_vertex_main(
    vert: ShapeVertexIn,
    @builtin(instance_index) instance_index: u32,
) -> ShapeVertexOut {
    var out: ShapeVertexOut;
    let i0 = 2 * instance_index * 3;
    let pstart = vert.instance_position;

    // Combine real and imag directions based on complex phase
    var v: vec3f;
    if u_shape_complex.is_complex == 1u {
        let c = cos(u_shape_complex.phase);
        let s = sin(u_shape_complex.phase);
        v = vert.instance_direction * c - vert.instance_direction_imag * s;
    } else {
        v = vert.instance_direction;
    }

    let q = quaternion(v, vec3f(0., 0., 1.));
    var pref = vert.position;
    if(u_shape.scale_mode == 0u) {
        pref *= length(v);
    }
    else if(u_shape.scale_mode == 1u) {
        pref.z *= length(v);
    }
    else if(u_shape.scale_mode == 2u) {
        // fixed size, fade out near zero
        pref *= clamp(length(v) * 1e4, 0.0, 1.0);
    }
    let p = pstart + u_shape.scale * rotate(pref, q);
    out.p = p;
    out.position = cameraMapPoint(p);
    out.normal = normalize(rotate(vert.normal, q));
    let lam = (vert.position.z-vert.z_range.x) / (vert.z_range.y-vert.z_range.x);

    // For complex mode with color_override, use current magnitude for color
    if u_shape_complex.is_complex == 1u && u_shape_complex.color_override == 1u {
        let val = length(v);
        out.color = mix(vec4f(val, val, val, val), vec4f(val, val, val, val), lam);
    } else {
        out.color = mix(vert.instance_color_bot, vert.instance_color_top, lam);
    }
    out.instance = instance_index;
    return out;
}

@fragment fn shape_fragment_main_value(
    input: ShapeVertexOut,
) -> @location(0) vec4f {
    checkClipping(input.p);
    let color = getColor(input.color.x);
    return lightCalcColor(input.p, input.normal, color);
}

@fragment fn shape_fragment_main_color(
    input: ShapeVertexOut,
) -> @location(0) vec4f {
    checkClipping(input.p);
    return lightCalcColor(input.p, input.normal, input.color);
}

@fragment fn shape_fragment_main_select(
    input: ShapeVertexOut,
) -> @location(0) vec4<u32> {
    checkClipping(input.p);
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
