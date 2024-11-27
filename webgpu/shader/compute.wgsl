@compute  @workgroup_size(16, 16, 1)
fn create_mesh(@builtin(num_workgroups) n_groups: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n: u32 = n_groups.x * 16;
    let h: f32 = 1.0 / (f32(n) + 1.);

    if lid.x == 0 && lid.y == 0 && wid.x == 0 {
        create_mesh_trig_function_values[0] = 1.0;
        create_mesh_trig_function_values[1] = 1.0;
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
                create_mesh_trigs_p1[i].index = 1;
                for (var pi: u32 = 0u; pi < 3u; pi++) {
                    create_mesh_trigs_p1[i].p[3 * pi + 0] = px[pi];
                    create_mesh_trigs_p1[i].p[3 * pi + 1] = py[pi];
                    create_mesh_trigs_p1[i].p[3 * pi + 2] = 0.0;
                    create_mesh_trig_function_values[2 + 3 * i + pi] = px[pi];
                }

                if k == 0 {
                    create_mesh_index_buffer[3 * i] = i1;
                    create_mesh_index_buffer[3 * i + 1] = i1 + 1;
                    create_mesh_index_buffer[3 * i + 2] = i1 + n + 1;
                } else {
                    create_mesh_index_buffer[3 * i] = i1 + 1;
                    create_mesh_index_buffer[3 * i + 1] = i1 + n + 1 + 1;
                    create_mesh_index_buffer[3 * i + 2] = i1 + n + 1;
                }
            }
        }

        let iv = 3 * (ix + iy * (n + 1));
        create_mesh_vertex_buffer[iv] = x;
        create_mesh_vertex_buffer[iv + 1] = y;
        create_mesh_vertex_buffer[iv + 2] = 0.0;

        if ix + 1 == n {
            create_mesh_vertex_buffer[iv + 3] = x + h;
            create_mesh_vertex_buffer[iv + 4] = y;
            create_mesh_vertex_buffer[iv + 5] = 0.;
        }
    }
}
