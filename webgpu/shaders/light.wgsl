#import camera

fn lightCalcBrightness(n: vec3f) -> f32 {
    let n4 = cameraMapNormal(n);
    let l = normalize(cameraMapNormal(vec3f(1., 3., 3.)).xyz);
    return clamp(dot(normalize(n4.xyz), l), .0, 1.) * 0.7 + 0.3;
}

fn lightCalcColor(n: vec3f, color: vec4f) -> vec4f {
    let brightness = lightCalcBrightness(n);
    return vec4f(color.xyz * brightness, color.w);
}
