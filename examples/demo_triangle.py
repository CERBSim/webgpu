
from webgpu.render_object import RenderObject

shader_code = lambda: """
// Data structure which is the output of the vertex shader and the input of the fragment shader
struct FragmentInput {
    @builtin(position) p: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Vertex shader, returns a FragmentInput object
@vertex
fn vertex_main(
  @builtin(vertex_index) vertex_index : u32
) -> FragmentInput {

  var pos = array<vec3f, 3>(
    vec3f(0., 0., 0.),
    vec3f(1., 1., 0.),
    vec3f(0., 1., 1.)
  );

  var color = array<vec4f, 3>(
    vec4f(1., 0., 0., 1.),
    vec4f(0., 1., 0., 1.),
    vec4f(0., 0., 1., 1.)
  );
  return FragmentInput( cameraMapPoint(pos[vertex_index]), color[vertex_index] );
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  return input.color;
}
"""

class Triangle(RenderObject):
    def __init__(self, gpu):
        super().__init__(gpu)
        self.n_vertices = 3

    def get_shader_code(self):
        return shader_code() + self.gpu.camera.get_shader_code()

    def get_bindings(self):
        return self.gpu.camera.get_bindings()
