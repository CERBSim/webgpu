struct TrigP1 { p: array<f32, 9>, index: i32};
@group(0) @binding(9) var<storage, read_write> trigs_p1 : array<TrigP1>;
@group(0) @binding(10) var<storage, read_write> trig_function_values : array<f32>;
@group(0) @binding(12) var<storage, read_write> vertex_buffer : array<f32>;
@group(0) @binding(13) var<storage, read_write> index_buffer : array<u32>;

@compute  @workgroup_size(16, 16, 1)
fn create_mesh(@builtin(num_workgroups) n_groups: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n: u32 = n_groups.x * 16;
    let h: f32 = 1.0 / (f32(n) + 1.);

    if lid.x == 0 && lid.y == 0 && wid.x == 0 {
        trig_function_values[0] = 1.0;
        trig_function_values[1] = 1.0;
    }

    let ix: u32 = wid.x * 16u + lid.x;
    let x: f32 = h * f32(ix);
    for (var iy: u32 = wid.y * 16u + lid.y; iy < n + 1; iy += 16u) {
        let y: f32 = h * f32(iy);
        for (var k: u32 = 0u; k < 2u; k++) {
            if iy < n {
                let i = 2 * (ix + iy * n) + k;
                let i1 = ix + iy * (n + 1);
                var px: array<f32, 3>;
                var py: array<f32, 3>;
                if k == 0 {
                    px = array<f32, 3>(x, x + h, x);
                    py = array<f32, 3>(y, y, y + h);
                } else {
                    px = array<f32, 3>(x + h, x + h, x);
                    py = array<f32, 3>(y, y + h, y + h);
                }
                trigs_p1[i].index = 1;
                for (var pi: u32 = 0u; pi < 3u; pi++) {
                    trigs_p1[i].p[3 * pi + 0] = px[pi];
                    trigs_p1[i].p[3 * pi + 1] = py[pi];
                    trigs_p1[i].p[3 * pi + 2] = 0.0;
                    trig_function_values[2 + 3 * i + pi] = px[pi];
                }

                if k == 0 {
                    index_buffer[3 * i] = i1;
                    index_buffer[3 * i + 1] = i1 + 1;
                    index_buffer[3 * i + 2] = i1 + n + 1;
                } else {
                    index_buffer[3 * i] = i1 + 1;
                    index_buffer[3 * i + 1] = i1 + n + 1 + 1;
                    index_buffer[3 * i + 2] = i1 + n + 1;
                }
            }
        }

        let iv = 3 * (ix + iy * (n + 1));
        vertex_buffer[iv] = x;
        vertex_buffer[iv + 1] = y;
        vertex_buffer[iv + 2] = 0.0;

        if ix + 1 == n {
            vertex_buffer[iv + 3] = x + h;
            vertex_buffer[iv + 4] = y;
            vertex_buffer[iv + 5] = 0.;
        }
    }
}
