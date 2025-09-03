struct CameraUniforms {
  model_view: mat4x4<f32>,
  model_view_projection: mat4x4<f32>,
  rot_mat: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  aspect: f32,
  width: u32,
  height: u32,

  padding: u32,
};

@group(0) @binding(0) var<uniform> u_camera : CameraUniforms;

fn cameraMapPoint(p: vec3f) -> vec4f {
    return u_camera.model_view_projection * vec4<f32>(p, 1.0);
}

fn cameraMapNormal(n: vec3f) -> vec4f {
    return u_camera.normal_mat * vec4(n, 1.0);
}

#ifdef SELECT_PIPELINE
@fragment fn fragment_select_no_clipping(
    @builtin(position) p: vec4f,
) -> @location(0) vec4<u32> {
    return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(p.z), 0, 0);
}
#endif SELECT_PIPELINE
