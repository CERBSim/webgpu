fn random(seed: u32) -> u32 {
   // Xorshift32, see https://en.wikipedia.org/wiki/Xorshift
    var value: u32 = seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

fn randomFloat(seed: u32) -> f32 {
    return f32(random(seed)) / f32(0xFFFFFFFFu);
}

fn getSeed(j: u32, k: u32) -> u32 {
    // create hash value as seed for random number generator
    // https://www.burtleburtle.net/bob/hash/integer.html
    let n = (k << 16) + j;
    var seed = (n ^ 61) ^ (n >> 16);
    seed *= seed * 9;
    seed = seed ^ (seed >> 4);
    seed = seed * 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

fn noise(x: u32, y: u32) -> f32 {
    let seed = getSeed(x, y);
    return randomFloat(seed);
}

fn dropletNoise(x: u32, y: u32) -> f32 {
  // large droplets for oriented line integral convolution
    let n = u_line_integral_convolution.thickness;
    let rand: f32 = noise(x / n, y / n);

    if rand < 0.85 {
        return 0.0;
    }

    let dx = f32(x % n) - f32(n) / 2.;
    let dy = f32(y % n) - f32(n) / 2.;
    let d = 4 * (dx * dx + dy * dy) / f32(n * n);

    return clamp(1.0 - d, 0.0, 1.0);
}


fn lineIntegralConvolution(x: u32, y: u32) -> f32 {
    let w = u_line_integral_convolution.width;
    let h = u_line_integral_convolution.height;
    let kernel_length = u_line_integral_convolution.kernel_length;
    let oriented = u_line_integral_convolution.oriented;
    var sum: f32 = 0;
    var weight: f32 = 0;

    for (var dir: i32 = -1; dir <= 1; dir += 2) {
        var p = vec2f(f32(x) + .5, f32(y) + .5);

        for (var k: u32 = 0; k < kernel_length; k++) {
            var v = textureLoad(u_line_integral_convolution_input, vec2i(p), 0).xy;
            v = normalize(v);

            p += f32(dir) * v;

            if p.x < 0.0 || p.x >= f32(w) || p.y < 0.0 || p.y >= f32(h) {
              break;
            }

            let ix = u32(p.x);
            let iy = u32(p.y);

            var kernel_weight: f32 = f32(k) / f32(kernel_length);
            if oriented == 0 {
                kernel_weight = 1.0 - f32(k) / f32(kernel_length);
                sum += kernel_weight * noise(ix, iy);
                weight += kernel_weight;
            } else {
                var t = 0.5 * (1.0 + f32(dir) * f32(k) / f32(kernel_length));
                kernel_weight = 0.1 + 0.9 * t * t * t * t;
                sum += kernel_weight * dropletNoise(ix, iy);
                weight += kernel_weight;
            }
        }
    }

    return sum / weight;
}

@compute  @workgroup_size(16, 16, 1)
fn computeLineIntegralConvolution(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u_line_integral_convolution.width;
    let h = u_line_integral_convolution.height;

    if gid.x >= w || gid.y >= h {
        return;
    }

    let value = lineIntegralConvolution(gid.x, gid.y);
    textureStore(u_line_integral_convolution_output, gid.xy, vec4f(value, .0, .0, .0));
}
