fn evalSegP1Basis(x: f32) -> array<f32, 2> {
    let y = 1.0 - x;
    return array(x, y);
}

fn evalSegP1(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP1Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    return result;
}

fn evalSegP2Basis(x: f32) -> array<f32, 3> {
    let y = 1.0 - x;
    return array(x * x, 2.0 * x * y, y * y);
}

fn evalSegP2(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP2Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    result += basis[2] * seg_function_values[offset + 2 * stride];
    return result;
}

fn evalSegP3Basis(x: f32) -> array<f32, 4> {
    let y = 1.0 - x;
    return array(x * x * x, 3.0 * x * x * y, 3.0 * x * y * y, y * y * y);
}

fn evalSegP3(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP3Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    result += basis[2] * seg_function_values[offset + 2 * stride];
    result += basis[3] * seg_function_values[offset + 3 * stride];
    return result;
}

fn evalSegP4Basis(x: f32) -> array<f32, 5> {
    let y = 1.0 - x;
    return array(x * x * x * x, 4.0 * x * x * x * y, 6.0 * x * x * y * y, 4.0 * x * y * y * y, y * y * y * y);
}

fn evalSegP4(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP4Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    result += basis[2] * seg_function_values[offset + 2 * stride];
    result += basis[3] * seg_function_values[offset + 3 * stride];
    result += basis[4] * seg_function_values[offset + 4 * stride];
    return result;
}

fn evalSegP5Basis(x: f32) -> array<f32, 6> {
    let y = 1.0 - x;
    return array(x * x * x * x * x, 5.0 * x * x * x * x * y, 10.0 * x * x * x * y * y, 10.0 * x * x * y * y * y, 5.0 * x * y * y * y * y, y * y * y * y * y);
}

fn evalSegP5(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP5Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    result += basis[2] * seg_function_values[offset + 2 * stride];
    result += basis[3] * seg_function_values[offset + 3 * stride];
    result += basis[4] * seg_function_values[offset + 4 * stride];
    result += basis[5] * seg_function_values[offset + 5 * stride];
    return result;
}

fn evalSegP6Basis(x: f32) -> array<f32, 7> {
    let y = 1.0 - x;
    return array(x * x * x * x * x * x, 6.0 * x * x * x * x * x * y, 15.0 * x * x * x * x * y * y, 20.0 * x * x * x * y * y * y, 15.0 * x * x * y * y * y * y, 6.0 * x * y * y * y * y * y, y * y * y * y * y * y);
}

fn evalSegP6(offset: u32, stride: u32, lam: f32) -> f32 {
    let basis = evalSegP6Basis(lam);
    var result: f32 = basis[0] * seg_function_values[offset + 0 * stride];
    result += basis[1] * seg_function_values[offset + 1 * stride];
    result += basis[2] * seg_function_values[offset + 2 * stride];
    result += basis[3] * seg_function_values[offset + 3 * stride];
    result += basis[4] * seg_function_values[offset + 4 * stride];
    result += basis[5] * seg_function_values[offset + 5 * stride];
    result += basis[6] * seg_function_values[offset + 6 * stride];
    return result;
}


fn evalSeg(id: u32, icomp: u32, lam: f32) -> f32 {
    let order: u32 = u32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    let ndof: u32 = order + 1;

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    if order == 1 { return evalSegP1(offset, stride, lam); }
    if order == 2 { return evalSegP2(offset, stride, lam); }
    if order == 3 { return evalSegP3(offset, stride, lam); }
    if order == 4 { return evalSegP4(offset, stride, lam); }
    if order == 5 { return evalSegP5(offset, stride, lam); }
    if order == 6 { return evalSegP6(offset, stride, lam); }

    return 0.0;
}
fn evalTrigP1Basis(lam: vec2<f32>) -> array<f32, 3> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    return array(x, y, z);
}

fn evalTrigP1(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP1Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    return result;
}

fn evalTrigP2Basis(lam: vec2<f32>) -> array<f32, 6> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 2.0 * x;
    return array(x * x, x0 * y, y * y, x0 * z, 2.0 * y * z, z * z);
}

fn evalTrigP2(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP2Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    result += basis[3] * trig_function_values[offset + 3 * stride];
    result += basis[4] * trig_function_values[offset + 4 * stride];
    result += basis[5] * trig_function_values[offset + 5 * stride];
    return result;
}

fn evalTrigP3Basis(lam: vec2<f32>) -> array<f32, 10> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 3.0 * x * x;
    let x1 = 3.0 * y * y;
    let x2 = 3.0 * z * z;
    return array(x * x * x, x0 * y, x * x1, y * y * y, x0 * z, 6.0 * x * y * z, x1 * z, x * x2, x2 * y, z * z * z);
}

fn evalTrigP3(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP3Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    result += basis[3] * trig_function_values[offset + 3 * stride];
    result += basis[4] * trig_function_values[offset + 4 * stride];
    result += basis[5] * trig_function_values[offset + 5 * stride];
    result += basis[6] * trig_function_values[offset + 6 * stride];
    result += basis[7] * trig_function_values[offset + 7 * stride];
    result += basis[8] * trig_function_values[offset + 8 * stride];
    result += basis[9] * trig_function_values[offset + 9 * stride];
    return result;
}

fn evalTrigP4Basis(lam: vec2<f32>) -> array<f32, 15> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 4.0 * x * x * x;
    let x1 = y * y;
    let x2 = x * x;
    let x3 = 6.0 * x2;
    let x4 = 4.0 * y * y * y;
    let x5 = 12.0 * z;
    let x6 = z * z;
    let x7 = 4.0 * z * z * z;
    return array(x * x * x * x, x0 * y, x1 * x3, x * x4, y * y * y * y, x0 * z, x2 * x5 * y, x * x1 * x5, x4 * z, x3 * x6, 12.0 * x * x6 * y, 6.0 * x1 * x6, x * x7, x7 * y, z * z * z * z);
}

fn evalTrigP4(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP4Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    result += basis[3] * trig_function_values[offset + 3 * stride];
    result += basis[4] * trig_function_values[offset + 4 * stride];
    result += basis[5] * trig_function_values[offset + 5 * stride];
    result += basis[6] * trig_function_values[offset + 6 * stride];
    result += basis[7] * trig_function_values[offset + 7 * stride];
    result += basis[8] * trig_function_values[offset + 8 * stride];
    result += basis[9] * trig_function_values[offset + 9 * stride];
    result += basis[10] * trig_function_values[offset + 10 * stride];
    result += basis[11] * trig_function_values[offset + 11 * stride];
    result += basis[12] * trig_function_values[offset + 12 * stride];
    result += basis[13] * trig_function_values[offset + 13 * stride];
    result += basis[14] * trig_function_values[offset + 14 * stride];
    return result;
}

fn evalTrigP5Basis(lam: vec2<f32>) -> array<f32, 21> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 5.0 * x * x * x * x;
    let x1 = y * y;
    let x2 = x * x * x;
    let x3 = 10.0 * x2;
    let x4 = x * x;
    let x5 = y * y * y;
    let x6 = 10.0 * x5;
    let x7 = 5.0 * y * y * y * y;
    let x8 = 20.0 * z;
    let x9 = 30.0 * x4;
    let x10 = z * z;
    let x11 = z * z * z;
    let x12 = 10.0 * x11;
    let x13 = 5.0 * z * z * z * z;
    return array(x * x * x * x * x, x0 * y, x1 * x3, x4 * x6, x * x7, y * y * y * y * y, x0 * z, x2 * x8 * y, x1 * x9 * z, x * x5 * x8, x7 * z, x10 * x3, x10 * x9 * y, 30.0 * x * x1 * x10, x10 * x6, x12 * x4, 20.0 * x * x11 * y, x1 * x12, x * x13, x13 * y, z * z * z * z * z);
}

fn evalTrigP5(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP5Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    result += basis[3] * trig_function_values[offset + 3 * stride];
    result += basis[4] * trig_function_values[offset + 4 * stride];
    result += basis[5] * trig_function_values[offset + 5 * stride];
    result += basis[6] * trig_function_values[offset + 6 * stride];
    result += basis[7] * trig_function_values[offset + 7 * stride];
    result += basis[8] * trig_function_values[offset + 8 * stride];
    result += basis[9] * trig_function_values[offset + 9 * stride];
    result += basis[10] * trig_function_values[offset + 10 * stride];
    result += basis[11] * trig_function_values[offset + 11 * stride];
    result += basis[12] * trig_function_values[offset + 12 * stride];
    result += basis[13] * trig_function_values[offset + 13 * stride];
    result += basis[14] * trig_function_values[offset + 14 * stride];
    result += basis[15] * trig_function_values[offset + 15 * stride];
    result += basis[16] * trig_function_values[offset + 16 * stride];
    result += basis[17] * trig_function_values[offset + 17 * stride];
    result += basis[18] * trig_function_values[offset + 18 * stride];
    result += basis[19] * trig_function_values[offset + 19 * stride];
    result += basis[20] * trig_function_values[offset + 20 * stride];
    return result;
}

fn evalTrigP6Basis(lam: vec2<f32>) -> array<f32, 28> {
    let x = lam.x;
    let y = lam.y;
    let z = 1.0 - x - y;
    let x0 = 6.0 * x * x * x * x * x;
    let x1 = y * y;
    let x2 = x * x * x * x;
    let x3 = 15.0 * x2;
    let x4 = y * y * y;
    let x5 = x * x * x;
    let x6 = 20.0 * x5;
    let x7 = x * x;
    let x8 = y * y * y * y;
    let x9 = 15.0 * x8;
    let x10 = 6.0 * y * y * y * y * y;
    let x11 = 30.0 * z;
    let x12 = 60.0 * z;
    let x13 = z * z;
    let x14 = 60.0 * x13;
    let x15 = z * z * z;
    let x16 = 60.0 * x15;
    let x17 = z * z * z * z;
    let x18 = 15.0 * x17;
    let x19 = 6.0 * z * z * z * z * z;
    return array(x * x * x * x * x * x, x0 * y, x1 * x3, x4 * x6, x7 * x9, x * x10, y * y * y * y * y * y, x0 * z, x11 * x2 * y, x1 * x12 * x5, x12 * x4 * x7, x * x11 * x8, x10 * z, x13 * x3, x14 * x5 * y, 90.0 * x1 * x13 * x7, x * x14 * x4, x13 * x9, x15 * x6, x16 * x7 * y, x * x1 * x16, 20.0 * x15 * x4, x18 * x7, 30.0 * x * x17 * y, x1 * x18, x * x19, x19 * y, z * z * z * z * z * z);
}

fn evalTrigP6(offset: u32, stride: u32, lam: vec2<f32>) -> f32 {
    let basis = evalTrigP6Basis(lam);
    var result: f32 = basis[0] * trig_function_values[offset + 0 * stride];
    result += basis[1] * trig_function_values[offset + 1 * stride];
    result += basis[2] * trig_function_values[offset + 2 * stride];
    result += basis[3] * trig_function_values[offset + 3 * stride];
    result += basis[4] * trig_function_values[offset + 4 * stride];
    result += basis[5] * trig_function_values[offset + 5 * stride];
    result += basis[6] * trig_function_values[offset + 6 * stride];
    result += basis[7] * trig_function_values[offset + 7 * stride];
    result += basis[8] * trig_function_values[offset + 8 * stride];
    result += basis[9] * trig_function_values[offset + 9 * stride];
    result += basis[10] * trig_function_values[offset + 10 * stride];
    result += basis[11] * trig_function_values[offset + 11 * stride];
    result += basis[12] * trig_function_values[offset + 12 * stride];
    result += basis[13] * trig_function_values[offset + 13 * stride];
    result += basis[14] * trig_function_values[offset + 14 * stride];
    result += basis[15] * trig_function_values[offset + 15 * stride];
    result += basis[16] * trig_function_values[offset + 16 * stride];
    result += basis[17] * trig_function_values[offset + 17 * stride];
    result += basis[18] * trig_function_values[offset + 18 * stride];
    result += basis[19] * trig_function_values[offset + 19 * stride];
    result += basis[20] * trig_function_values[offset + 20 * stride];
    result += basis[21] * trig_function_values[offset + 21 * stride];
    result += basis[22] * trig_function_values[offset + 22 * stride];
    result += basis[23] * trig_function_values[offset + 23 * stride];
    result += basis[24] * trig_function_values[offset + 24 * stride];
    result += basis[25] * trig_function_values[offset + 25 * stride];
    result += basis[26] * trig_function_values[offset + 26 * stride];
    result += basis[27] * trig_function_values[offset + 27 * stride];
    return result;
}


fn evalTrig(id: u32, icomp: u32, lam: vec2<f32>) -> f32 {
    let order: u32 = u32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    let ndof: u32 = (order + 1) * (order + 2) / 2;

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    if order == 1 { return evalTrigP1(offset, stride, lam); }
    if order == 2 { return evalTrigP2(offset, stride, lam); }
    if order == 3 { return evalTrigP3(offset, stride, lam); }
    if order == 4 { return evalTrigP4(offset, stride, lam); }
    if order == 5 { return evalTrigP5(offset, stride, lam); }
    if order == 6 { return evalTrigP6(offset, stride, lam); }

    return 0.0;
}
