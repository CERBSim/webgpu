struct CameraUniforms {
  model_view: mat4x4<f32>,
  model_view_projection: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  aspect: f32,

  padding0: u32,
  padding1: u32,
  padding2: u32,
};

@group(0) @binding(0) var<uniform> u_camera : CameraUniforms;

fn cameraMapPoint(p: vec3f) -> vec4f {
    return u_camera.model_view_projection * vec4<f32>(p, 1.0);
}

fn cameraMapNormal(n: vec3f) -> vec4f {
    return u_camera.normal_mat * vec4(n, 1.0);
}

