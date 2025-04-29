
@group(0) @binding(81) var<storage, read> vec_points: array<f32>;
@group(0) @binding(82) var<storage, read> vec_vectors: array<f32>;
@group(0) @binding(83) var<uniform> vec_options: VectorOptions;

struct VectorOptions {
 length: f32,
 scale_with_vector_length: u32,
 padding2: f32,
 padding3: f32
};


fn Cross(a: vec3f, b: vec3f) -> vec3f {

    return vec3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    );
}


struct VectorFragmentInput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) color_val: f32,
  @location(1) n: vec3<f32>
};

fn get_point(index: u32) -> vec3f {
    let i = index * 3u;
    return vec3f(vec_points[i], vec_points[i + 1u], vec_points[i + 2u]);
}

fn get_vector(index: u32) -> vec3f {
    let i = index * 3u;
    return vec3f(vec_vectors[i], vec_vectors[i + 1u], vec_vectors[i + 2u]);
}

// Draw vector cone with triangle strip

const cone_points =
  array<vec3f, 4>(vec3f(-sqrt(0.5), -sqrt(0.5), 0.),
    vec3f(sqrt(0.5), -sqrt(0.5), 0.),
    vec3f(0., 1., 0.),
    vec3f(0., 0., 5.));

const cone_strip = array<u32, 10>(0u, 1u, 2u, 2u, 3u, 1u, 3u, 0u, 3u, 2u);
const cone_strip_normals = array<u32, 10>(0u, 0u, 0u, 1u, 4u, 2u, 5u, 3u, 6u, 1u);

@vertex
fn vertex_main(@builtin(vertex_index) index: u32,
    @builtin(instance_index) instance: u32) -> VectorFragmentInput {
    let point = get_point(instance);
    let vector = get_vector(instance);
    // if length(vector) < 0.0001 {
    //     return VectorFragmentInput(vec4<f32>(0., 0., 0., 0.),
    //         0., vec3<f32>(0., 0., 0.));
    // }
    let cp = cone_points[cone_strip[index]];
    let v = normalize(vector);
    let z_axis = vec3<f32>(0., 0., 1.);
    var rotation_matrix = mat3x3<f32>(1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.);
    var rotation_axis = -Cross(z_axis, v);
    let axis_norm = length(rotation_axis);
    if axis_norm > 0.0001 {
        rotation_axis = rotation_axis / axis_norm;
    
    // Compute the angle between the z-axis and the target vector
        let angle = acos(clamp(dot(z_axis, v), -1.0, 1.0));
    
    // Compute Rodrigues' rotation formula components
        let K = mat3x3<f32>(0.0, -rotation_axis.z, rotation_axis.y,
            rotation_axis.z, 0.0, -rotation_axis.x,
            -rotation_axis.y, rotation_axis.x, 0.0);

        // Final rotation matrix
        rotation_matrix += sin(angle) * K + (1.0 - cos(angle)) * (K * K);
    } else {
        if dot(z_axis, v) < 0. {
            rotation_matrix = mat3x3<f32>(-1., 0., 0.,
                0., -1., 0.,
                0., 0., -1.);
        }
    }
    var rotated = rotation_matrix * cp;
    rotated = rotated * 0.2 * vec_options.length;
    if(vec_options.scale_with_vector_length != 0u) {
      rotated = rotated * length(vector);
    }
    let position = point + rotated;
    let view_position = cameraMapPoint(position);
    let cone_normals = array<vec3f, 7>(-normalize(Cross(cone_points[1] - cone_points[0],
        cone_points[2] - cone_points[0])),
        cone_points[2],
        cone_points[1],
        cone_points[0],
        -normalize(Cross(cone_points[2] - cone_points[3],
        cone_points[1] - cone_points[3])),
        -normalize(Cross(cone_points[1] - cone_points[3],
        cone_points[0] - cone_points[3])),
        -normalize(Cross(cone_points[0] - cone_points[3],
        cone_points[2] - cone_points[3])));

    let normal = cone_normals[cone_strip_normals[index]];
    return VectorFragmentInput(view_position, length(vector),
        rotation_matrix * normal);
}

@fragment
fn fragment_main(input: VectorFragmentInput) -> @location(0) vec4<f32> {
    if length(input.n) == 0. {
    discard;
    }
    return lightCalcColor(-input.n, getColor(input.color_val));
}
