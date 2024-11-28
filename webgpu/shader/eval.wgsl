const VALUES_OFFSET: u32 = 2; // storing number of components and order of basis functions in first two entries

fn evalSeg(id: u32, icomp: u32, lam: f32) -> f32 {
    let order: u32 = u32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    let ndof: u32 = order + 1;

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    var v: array<f32, 7>;
    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = seg_function_values[offset + i * stride];
    }

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = seg_function_values[offset + i * stride];
    }

    let b = vec2f(lam, 1.0 - lam);

    for (var n = order; n > 0; n--) {
        for (var i = 0u; i < n; i++) {
            v[i] = dot(b, vec2f(v[i], v[i + 1]));
        }
    }

    return v[0];
}

fn evalTrig(id: u32, icomp: u32, lam: vec2<f32>) -> f32 {
    var order: i32 = i32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    var v: array<f32, 28>;
    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = trig_function_values[offset + i * stride];
    }

    let dy = order + 1;
    let b = vec3f(lam.x, lam.y, 1.0 - lam.x - lam.y);

    for (var n = order; n > 0; n--) {
        var i0 = 0;
        for (var iy = 0; iy < n; iy++) {
            for (var ix = 0; ix < n - iy; ix++) {
                v[i0 + ix] = dot(b, vec3f(v[i0 + ix], v[i0 + ix + 1], v[i0 + ix + dy - iy]));
            }
            i0 += dy - iy;
        }
    }

    return v[0];
}
