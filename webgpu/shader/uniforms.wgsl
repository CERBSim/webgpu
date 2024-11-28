// Uniforms are global variables that are constant for all invocations of a shader.
// They are used to store configuration data (no mesh etc.)
// Each uniform must have a unique binding group and number
// They are used to pass data from the CPU to the GPU (variable names are only relevant within the shader code)

// general uniforms
@group(0) @binding(0) var<uniform> u_view : ViewUniforms;
@group(0) @binding(1) var<uniform> u_clipping : ClippingUniforms;
@group(0) @binding(2) var<uniform> u_font : FontUniforms;
@group(0) @binding(3) var u_font_texture : texture_2d<f32>;
@group(0) @binding(5) var<uniform> u_function : FunctionUniforms;
@group(0) @binding(6) var u_colormap_texture : texture_1d<f32>;
@group(0) @binding(7) var u_colormap_sampler : sampler;
@group(0) @binding(14) var gBufferLam : texture_2d<f32>;

// legacy uniforms
@group(0) @binding(8) var<storage> edges_p1 : array<EdgeP1>;
@group(0) @binding(9) var<storage> trigs_p1 : array<TrigP1>;
@group(0) @binding(10) var<storage> trig_function_values : array<f32>;
@group(0) @binding(11) var<storage> seg_function_values : array<f32>;
@group(0) @binding(12) var<storage> vertices : array<f32>;
@group(0) @binding(13) var<storage> trigs : array<u32>;

// mesh uniforms
@group(0) @binding(20) var<uniform> u_mesh : MeshUniforms;
@group(0) @binding(21) var<storage> u_edges : array<Edge>;
@group(0) @binding(22) var<storage> u_segs : array<Seg>;
@group(0) @binding(23) var<storage> u_trigs : array<Trig>;
@group(0) @binding(24) var<storage> u_quads : array<Quad>;
@group(0) @binding(25) var<storage> u_tets : array<Tet>;
@group(0) @binding(26) var<storage> u_pyramids : array<Pyramid>;
@group(0) @binding(27) var<storage> u_prisms : array<Prism>;
@group(0) @binding(28) var<storage> u_hexes : array<Hex>;

// deformation uniforms
@group(0) @binding(30) var<storage> u_segs_deformation : array<f32>;
@group(0) @binding(31) var<storage> u_trigs_deformation : array<f32>;
@group(0) @binding(32) var<storage> u_quads_deformation : array<f32>;
@group(0) @binding(33) var<storage> u_tets_deformation : array<f32>;
@group(0) @binding(34) var<storage> u_pyramids_deformation : array<f32>;
@group(0) @binding(35) var<storage> u_prisms_deformation : array<f32>;
@group(0) @binding(36) var<storage> u_hexes_deformation : array<f32>;

// function uniforms
@group(0) @binding(30) var<storage> u_segs_function : array<f32>;
@group(0) @binding(31) var<storage> u_trigs_function : array<f32>;
@group(0) @binding(32) var<storage> u_quads_function : array<f32>;
@group(0) @binding(33) var<storage> u_tets_function : array<f32>;
@group(0) @binding(34) var<storage> u_pyramids_function : array<f32>;
@group(0) @binding(35) var<storage> u_prisms_function : array<f32>;
@group(0) @binding(36) var<storage> u_hexes_function : array<f32>;

// Line integral convolution
@group(0) @binding(40) var<uniform> u_line_integral_convolution : LineIntegralConvolutionUniforms;
@group(0) @binding(41) var u_line_integral_convolution_input: texture_2d<f32>;
@group(0) @binding(42) var u_line_integral_convolution_output: texture_storage_2d<r32float, write>;

// Create mesh
@group(0) @binding(50) var<storage, read_write> create_mesh_trigs_p1 : array<TrigP1>;
@group(0) @binding(51) var<storage, read_write> create_mesh_trig_function_values : array<f32>;
@group(0) @binding(52) var<storage, read_write> create_mesh_vertex_buffer : array<f32>;
@group(0) @binding(53) var<storage, read_write> create_mesh_index_buffer : array<u32>;

// general uniform structures
struct ViewUniforms {
  model_view: mat4x4<f32>,
  model_view_projection: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  aspect: f32,

  padding0: u32,
  padding1: u32,
  padding2: u32,
};

struct ClippingUniforms {
  plane: vec4<f32>,
  sphere: vec4<f32>,
  mode: u32, // 0: disabled, 1: plane, 2: sphere, 3: both

  padding0: u32,
  padding1: u32,
  padding2: u32,
};

struct FunctionUniforms {
  colormap: vec2<f32>,

  padding0: f32,
  padding1: f32,
};

struct FontUniforms {
  width: u32,
  height: u32,

  padding0: u32,
  padding1: u32,
};

struct MeshUniforms {
  subdivision: u32,
  shrink: f32,

  padding0: f32,
  padding1: f32,
};

struct LineIntegralConvolutionUniforms {
  width: u32,         // canvas width
  height: u32,        // canvas height
  kernel_length: u32,
  oriented: u32,      // 0: not oriented, 1: oriented
  thickness: u32,     // thickness of the lines (only used for oriented)

  padding0: u32,
  padding1: u32,
  padding2: u32,
};
// Mesh element structures

struct Edge { p: array<u32, 2> }; // Inner edge, for wireframe
struct Seg { p: array<u32, 2>, nr: u32, index: u32 };
struct Trig { p: array<u32, 3>, nr: u32, index: u32 };
struct Quad { p: array<u32, 4>, nr: u32, index: u32 };
struct Tet { p: array<u32, 4>, nr: u32, index: u32 };
struct Pyramid { p: array<u32, 5>, nr: u32, index: u32 };
struct Prism { p: array<u32, 6>, nr: u32, index: u32 };
struct Hex { p: array<u32, 8>, nr: u32, index: u32 };


// legacy mesh structs
struct EdgeP1 { p: array<f32, 6> };
struct TrigP1 { p: array<f32, 9>, index: i32 }; // 3 vertices with 3 coordinates each, don't use vec3 due to 16 byte alignment
struct TrigP2 { p: array<f32, 18>, index: i32 };

