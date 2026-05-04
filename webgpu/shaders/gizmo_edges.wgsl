#import camera

@group(0) @binding(90) var<storage> u_edges: array<f32>;

struct GizmoUniforms {
    corner: vec2f,
    scale: f32,
    thickness: f32,
};
@group(0) @binding(93) var<uniform> u_gizmo: GizmoUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) edgeId: u32) -> VertexOutput {
    var out: VertexOutput;

    let p1_3d = vec3f(u_edges[edgeId * 6u], u_edges[edgeId * 6u + 1u], u_edges[edgeId * 6u + 2u]);
    let p2_3d = vec3f(u_edges[edgeId * 6u + 3u], u_edges[edgeId * 6u + 4u], u_edges[edgeId * 6u + 5u]);

    let r1 = (u_camera.rot_mat * vec4f(p1_3d, 0.0)).xyz;
    let r2 = (u_camera.rot_mat * vec4f(p2_3d, 0.0)).xyz;

    var off1 = r1.xy * u_gizmo.scale;
    var off2 = r2.xy * u_gizmo.scale;
    off1.x /= u_camera.aspect;
    off2.x /= u_camera.aspect;
    let sp1 = u_gizmo.corner + off1;
    let sp2 = u_gizmo.corner + off2;

    let v = normalize(sp2 - sp1);
    var normal = vec2f(-v.y, v.x) * u_gizmo.thickness;

    var pos: vec2f;
    var z: f32;
    if (vertId == 0u) {
        pos = sp1 - normal;
        z = r1.z;
    } else if (vertId == 1u) {
        pos = sp1 + normal;
        z = r1.z;
    } else if (vertId == 2u) {
        pos = sp2 - normal;
        z = r2.z;
    } else {
        pos = sp2 + normal;
        z = r2.z;
    }

    let depth = 0.01 - z * 0.015;
    out.position = vec4f(pos, depth, 1.0);
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(0.15, 0.15, 0.15, 1.0);
}
