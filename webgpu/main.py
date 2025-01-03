"""Main file for the webgpu example, creates a small 2d mesh and renders it using WebGPU"""

import urllib.parse

import js
import ngsolve as ngs
from netgen.occ import unit_cube
from netgen.geom2d import unit_square
from pyodide.ffi import create_proxy

from .gpu import WebGPU, init_webgpu
from .mesh import *
from .lic import LineIntegralConvolutionRenderObject

from .webgpu_api import BackendType, RenderPassEncoder, ColorTargetState

# s = ColorTargetState()
# print("BackendType", BackendType)
# print("ColorTargetState", s)


def f(encoder: RenderPassEncoder):
    return


gpu: WebGPU = None
mesh_object: RenderObject = None
elements_object = None
point_number_object = None

cf = None
render_function = None


async def main():
    global gpu, mesh_object, cf, render_function

    gpu = await init_webgpu(js.document.getElementById("canvas"))
    # print("DEVICE", dir(gpu.native_device))

    point_number_object = None

    if 1:
        from ngsolve.meshes import MakeStructured3DMesh

        # create new ngsolve mesh and evaluate arbitrary function on it
        # mesh = ngs.Mesh(unit_cube.GenerateMesh(maxh=0.2))
        # mesh = MakeStructured3DMesh(True, 10, prism=True)

        import netgen.occ as occ
        from netgen.meshing import IdentificationType

        idtype = IdentificationType.CLOSESURFACES
        inner = occ.Box((0, 0, 0), (1, 1, 1))
        trafo = occ.gp_Trsf().Scale(inner.center, 1.1)
        outer = trafo(inner)

        # inner.Identify(outer, "", idtype, trafo)
        shape = occ.Glue([outer - inner, inner])

        geo = occ.OCCGeometry(shape)
        mesh = geo.GenerateMesh(maxh=0.3)

        # mesh = unit_square.GenerateMesh(maxh=0.3)
        mesh = ngs.Mesh(mesh)

        order = 3
        cf = cf or ngs.sin(10 * ngs.x) * ngs.sin(10 * ngs.y)
        # cf = ngs.x
        data = MeshData(mesh, cf, order)
        gpu.u_function.min = -1
        gpu.u_function.max = 1
    else:
        # use compute shader to create a unit_square mesh
        # but has always P1 and 'x' hard-coded as function
        query = urllib.parse.parse_qs(js.location.search[1:])
        N = 10
        N = int(query.get("n", [N])[0])
        data = create_testing_square_mesh(gpu, N)
        gpu.u_function.min = 0
        gpu.u_function.max = 1

    # lic = LineIntegralConvolutionRenderObject(gpu, 1000, 800)
    # print("LIC", lic)

    # mesh_object = MeshRenderObject(gpu, data)
    # mesh_object = MeshRenderObjectIndexed(gpu, data) # function values are wrong, due to ngsolve vertex numbering order
    # mesh_object = MeshRenderObjectDeferred(
    #     gpu, data
    # )  # function values are wrong, due to ngsolve vertex numbering order
    point_number_object = PointNumbersRenderObject(gpu, data, font_size=16)
    elements_object = Mesh3dElementsRenderObject(gpu, data)

    t_last = 0
    fps = 0
    frame_counter = 0
    params = pyodide.ffi.to_js({"shrink": 0.3})

    def render(time):
        # this is the render function, it's called for every frame
        if not isinstance(time, float):
            time = 0

        nonlocal t_last, fps, frame_counter
        print("params", params.shrink)
        dt = time - t_last
        t_last = time
        frame_counter += 1
        print(f"frame time {dt:.2f} ms")

        gpu.u_mesh.shrink = params.shrink

        # copy camera position etc. to GPU
        gpu.update_uniforms()

        command_encoder = gpu.device.createCommandEncoder()

        if mesh_object is not None:
            mesh_object.render(command_encoder)

        if elements_object is not None:
            elements_object.render(command_encoder)

        if point_number_object is not None:
            point_number_object.render(command_encoder)

        gpu.device.queue.submit([command_encoder.finish()])
        if frame_counter < 20:
            js.requestAnimationFrame(render_function)

    render_function = create_proxy(render)
    gpu.input_handler._update_uniforms()
    gpu.input_handler.render_function = render_function

    render_function.request_id = js.requestAnimationFrame(render_function)

    try:
        gui = js.gui

        gui.reset(recursive=True)

        folder = js.window.folder
        folder.reset()
        folder.add(params, "shrink", 0.1, 1.0)
        gui.onChange(render_function)
    except Exception as e:
        print(e)


def cleanup():
    print("cleanup")
    global gpu, mesh_object
    if "gpu" in globals():
        del gpu
    if "mesh_object" in globals():
        del mesh_object


async def user_function(data):
    code, expr = data
    import base64
    import marshal
    import types

    code = base64.b64decode(code.encode("utf-8"))
    code = marshal.loads(code)
    func = types.FunctionType(code, globals(), "user_function")
    func(expr)


async def reload(*args, **kwargs):
    print("reload")
    cleanup()
    reload_package("webgpu")
    from webgpu.main import main

    await main()
