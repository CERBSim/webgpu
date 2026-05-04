#import camera

@group(0) @binding(90) var<storage> u_positions: array<f32>;
@group(0) @binding(91) var<storage> u_normals: array<f32>;
@group(0) @binding(92) var<storage> u_colors: array<f32>;

struct GizmoUniforms {
    corner: vec2f,
    scale: f32,
    padding: f32,
};
@group(0) @binding(93) var<uniform> u_gizmo: GizmoUniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) @interpolate(flat) face_index: u32,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    let pos = vec3f(u_positions[vid * 3u], u_positions[vid * 3u + 1u], u_positions[vid * 3u + 2u]);
    let normal = vec3f(u_normals[vid * 3u], u_normals[vid * 3u + 1u], u_normals[vid * 3u + 2u]);
    let color = vec4f(u_colors[vid * 4u], u_colors[vid * 4u + 1u], u_colors[vid * 4u + 2u], u_colors[vid * 4u + 3u]);

    let rotated = (u_camera.rot_mat * vec4f(pos, 0.0)).xyz;
    let rot_normal = normalize((u_camera.rot_mat * vec4f(normal, 0.0)).xyz);

    // Simple directional lighting
    let light_dir = normalize(vec3f(0.4, 0.7, 1.0));
    let ambient = 0.35;
    let diffuse = max(dot(rot_normal, light_dir), 0.0) * 0.55;
    let back_diffuse = max(dot(-rot_normal, light_dir), 0.0) * 0.2;
    let brightness = ambient + diffuse + back_diffuse;

    // Aspect-correct positioning: only scale the gizmo part, not the corner
    var gizmo_offset = rotated.xy * u_gizmo.scale;
    gizmo_offset.x /= u_camera.aspect;
    let ndc = u_gizmo.corner + gizmo_offset;

    // Depth: near 0 so gizmo is always in front, but spread for self-occlusion
    let depth = 0.02 - rotated.z * 0.015;

    out.position = vec4f(ndc, depth, 1.0);
    out.color = vec4f(color.rgb * brightness, color.a);
    out.face_index = vid / 6u;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(input.color.rgb * input.color.a, input.color.a);
}

#ifdef SELECT_PIPELINE
@fragment
fn fragment_select(input: VertexOutput) -> @location(0) vec4<u32> {
    return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.position.z), 0u, input.face_index);
}
#endif SELECT_PIPELINE
