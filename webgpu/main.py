"""Main file for the webgpu example, creates a small 2d mesh and renders it using WebGPU"""

import urllib.parse

import js
import ngsolve as ngs
from netgen.occ import unit_square
from pyodide.ffi import create_once_callable, create_proxy

from .gpu import init_webgpu
from .mesh import *

gpu = None
mesh_object = None

cf = None
render_function = None


async def main():
    global gpu, mesh_object, cf, render_function

    gpu = await init_webgpu(js.document.getElementById("canvas"))

    if 1:
        # create new ngsolve mesh and evaluate arbitrary function on it
        mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.5))
        order = 6
        region = mesh.Region(ngs.VOL)
        cf = cf or ngs.sin(10 * ngs.x) * ngs.sin(10 * ngs.y)
        n_trigs, buffers = create_mesh_buffers(gpu.device, region)
        buffers = buffers | create_function_value_buffers(gpu.device, cf, region, order)
        mesh_object = MeshRenderObject(gpu, buffers, n_trigs)

    else:
        # create testing mesh, this one also supports indexed or deferred rendering
        # but has always P1 and 'x' hard-coded as function
        query = urllib.parse.parse_qs(js.location.search[1:])
        N = 100
        # N = int(5000/2**.5)
        # N = int(2000 / 2**0.5)
        # N = int(50/2**.5)
        # N = 1
        N = int(query.get("n", [N])[0])
        # print("creating ", N * N, "triangles")
        n_trigs, buffers = create_testing_square_mesh(gpu, N)

        mesh_object = MeshRenderObject(gpu, buffers, n_trigs)
        # mesh_object = MeshRenderObjectIndexed(gpu, buffers, n_trigs)
        # mesh_object = MeshRenderObjectDeferred(gpu, buffers, n_trigs)

    # move mesh to center and scale it
    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    # translate to center
    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8

    t_last = 0
    fps = 0
    frame_counter = 0

    def render(time):
        nonlocal t_last, fps, frame_counter
        dt = time - t_last
        t_last = time
        frame_counter += 1
        # if dt < 20:
        #     print('returning')
        #     return
        print(f"frame time {dt:.2f} ms")
        # if dt > 1e-3:
        #     frame_counter += 1
        #     fps = 0.9 * fps + 0.1 * 1000 / dt
        #     if frame_counter % 30 == 0:
        #         print(f"fps {fps:.2f}")

        # this is the render function, it's called for every frame

        # copy camera position etc. to GPU
        gpu.uniforms.update_buffer()

        command_encoder = gpu.device.createCommandEncoder()

        mesh_object.render(command_encoder)

        gpu.device.queue.submit([command_encoder.finish()])
        if frame_counter < 20:
            js.requestAnimationFrame(render_function)
            # gpu.device.queue.onSubmittedWorkDone().then(
            #     create_once_callable(
            #         lambda _: js.requestAnimationFrame(render_function)
            #     )
            # )

    render_function = create_proxy(render)
    gpu.input_handler._update_uniforms()
    gpu.input_handler.render_function = render_function

    render_function.request_id = js.requestAnimationFrame(render_function)


def cleanup():
    print("cleanup")
    global gpu, mesh_object
    if "gpu" in globals():
        del gpu
    if "mesh_object" in globals():
        del mesh_object


def reload_package(package_name):
    """Reload python package and all submodules (searches in modules for references to other submodules)"""
    import importlib
    import os
    import types

    package = importlib.import_module(package_name)
    assert hasattr(package, "__package__")
    file_name = package.__file__
    package_dir = os.path.dirname(file_name) + os.sep
    reloaded_modules = {file_name: package}

    def reload_recursive(module):
        module = importlib.reload(module)

        for var in vars(module).values():
            if isinstance(var, types.ModuleType):
                file_name = getattr(var, "__file__", None)
                if file_name is not None and file_name.startswith(package_dir):
                    if file_name not in reloaded_modules:
                        reloaded_modules[file_name] = reload_recursive(var)

        return module

    reload_recursive(package)
    return reloaded_modules


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
