#import camera

struct LightUniforms {
  key: vec4f,
  fill: vec4f,

  ambient: f32,       // 0..1
  diffuse: f32,       // overall diffuse strength
  specular: f32,      // overall spec strength (keep small for colormaps)
  shininess: f32,     // e.g. 8..32

  wrap: f32,          // 0..1 (0=Lambert, ~0.4-0.7 good for vis)
  minDiffuse: f32,    // 0..1 (brightness floor for diffuse term)

  rimStrength: f32,   // 0..0.3 (optional silhouette cue)
  rimPower: f32,      // 1..4 (2 is a good start)
};

@group(0) @binding(8) var<uniform> u_light : LightUniforms;

// Wrapped diffuse: keeps “dark side” from going black
fn wrappedDiffuse(n: vec3f, L: vec3f, wrap: f32) -> f32 {
  let d = dot(n, L); // [-1..1]
  return clamp((d + wrap) / (1.0 + wrap), 0.0, 1.0);
}

fn lightCalcBrightnessScientific(p: vec3f, normal: vec3f) -> vec2f {
  // View-space normal & view direction
  let n = normalize(cameraMapNormal(normal).xyz);

  let posVS = (u_camera.model_view * vec4f(p, 1.0)).xyz;
  let v = normalize(-posVS);

  let Lk  = normalize(u_light.key.xyz);
  let Lf = normalize(u_light.fill.xyz);

  // Wrapped diffuse + brightness floor
  let dk = wrappedDiffuse(n, Lk, u_light.wrap) * u_light.key.w;
  let df = wrappedDiffuse(n, Lf, u_light.wrap) * u_light.fill.w;

  var diffuse = (dk + df) * u_light.diffuse;
  diffuse = max(diffuse, u_light.minDiffuse); // keep colormap readable

  // Very gentle spec (Blinn-Phong), only when facing light
  var spec = 0.0;
  if (u_light.shininess > 0.0 && u_light.specular > 0.0) {
    // Gate spec by a non-wrapped facing term to avoid highlights on “back side”
    let facingK = max(dot(n, Lk), 0.0);
    let facingF = max(dot(n, Lf), 0.0);

    if (facingK > 0.0) {
      let Hk = normalize(Lk + v);
      spec += pow(max(dot(n, Hk), 0.0), u_light.shininess) * u_light.key.w;
    }
    if (facingF > 0.0) {
      let Hf = normalize(Lf + v);
      spec += pow(max(dot(n, Hf), 0.0), u_light.shininess) * u_light.fill.w;
    }

    // Keep it subtle for scalar colormaps
    spec *= u_light.specular;
  }

  // Optional rim for silhouette cue (adds light, doesn’t darken)
  if (u_light.rimStrength > 0.0) {
    let rim = pow(1.0 - max(dot(n, v), 0.0), u_light.rimPower);
    diffuse += rim * u_light.rimStrength;
  }

  return vec2f(u_light.ambient + diffuse, spec);
}

fn lightCalcColor(p: vec3f, n: vec3f, color: vec4f) -> vec4f {
  let b = lightCalcBrightnessScientific(p, n);

  // Add spec as white (or you can tint it slightly if you prefer)
  let lit = color.rgb * b.x + b.y * vec3f(1.0);

  return vec4f(lit, color.a);
}
