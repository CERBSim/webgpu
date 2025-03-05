
@group(0) @binding(90) var<storage> u_vertices : array<f32>;
@group(0) @binding(91) var<storage> u_normals : array<f32>;

struct TriangleFragmentInput {
  @builtin(position) position: vec4<f32>,
    @location(0) p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) @interpolate(flat) vertId: u32,
    @location(3) @interpolate(flat) trigId: u32,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32, @builtin(instance_index) trigId: u32) -> TriangleFragmentInput {
  let point = vec3<f32>(u_vertices[trigId * 9 + vertId * 3],
                        u_vertices[trigId * 9 + vertId * 3 + 1],
                        u_vertices[trigId * 9 + vertId * 3 + 2]);
  let normal = -vec3<f32>(u_normals[trigId * 9 + vertId * 3],
                         u_normals[trigId * 9 + vertId * 3 + 1],
                         u_normals[trigId * 9 + vertId * 3 + 2]);

  let position = cameraMapPoint(point);
  return TriangleFragmentInput(position,
                               point,
                               normal,
                               vertId,
                               trigId,
                               );
}

@fragment
fn fragment_main(input: TriangleFragmentInput) -> @location(0) vec4<f32> {
  return lightCalcColor(input.n, getColor(input.vertId, input.trigId));
}
