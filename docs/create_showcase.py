"""Generate the landing page showcase visualization.

Run once to produce docs/_static/showcase.html:

    python docs/create_showcase.py

Requires: ngsolve, ngsolve_webgpu, webgpu (with WEBGPU_EXPORTING support)
"""

import base64
import os

os.environ["WEBGPU_EXPORTING"] = "1"

from netgen.occ import *
from ngsolve import *
from ngsolve_webgpu import *
from webgpu.colormap import Colorbar
from webgpu.jupyter import Draw
from webgpu.engine import engine_js

# NGSolve logo geometry (OCC version)
sphere_outer = Sphere((50, 50, 50), 80)
sphere_inner = Sphere((50, 50, 50), 50)
cyl_x = Cylinder((50, 0, 0), X, r=40, h=200, mantle="dirichlet").Move((-100, 0, 0))
cyl_y = Cylinder((0, 50, 100), Y, r=40, h=200).Move((100, -100, 0))
cyl_z = Cylinder((0, 100, 50), Z, r=40, h=200).Move((0, 0, -100))

shape = sphere_outer - cyl_x - cyl_y - cyl_z - sphere_inner
shape = shape.Rotate(Axis((0,0,0), Y), -45)
shape = shape.Rotate(Axis((0,0,0), Z), 20)
shape = shape.Rotate(Axis((0,0,0), X), -5)

mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=25))
mesh.Curve(5)

fes = H1(mesh, order=3, dirichlet="dirichlet")
u, v = fes.TnT()

a = BilinearForm(grad(u) * grad(v) * dx).Assemble()
f = LinearForm(v * dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Render
mesh_data = MeshData(mesh)
function_data = FunctionData(mesh_data, gfu, order=3)

colormap = Colormap()
cfr = CFRenderer(function_data, colormap=colormap)

scene = Draw(cfr)
blob_b64 = base64.b64encode(scene.export()).decode()

canvas_id = "showcase_canvas"
html = f"""\
<style>
#showcase_root {{ --webgpu-canvas-bg: #ffffff; }}
@media (prefers-color-scheme: dark) {{
  #showcase_root {{ --webgpu-canvas-bg: #adadad; }}
}}
</style>
<div id="showcase_root" style="width:min(800px,100%); max-width:100%; margin:0 auto 1em auto; position:relative;">
<canvas id="{canvas_id}" width="800" height="400" style="background-color:var(--webgpu-canvas-bg,#ffffff); width:100%; max-width:100%; height:auto; aspect-ratio:800 / 400; touch-action:none; border-radius:8px; display:block;"></canvas>
<script>
{engine_js}
RenderEngine.create("{canvas_id}", "{blob_b64}");
</script>
</div>
"""

out = os.path.join(os.path.dirname(__file__), "_static", "showcase.html")
with open(out, "w") as f:
    f.write(html)
print(f"Written to {out}")
