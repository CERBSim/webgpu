import json
import struct

MAGIC = b"WGPU"
VERSION = 1


def serialize_scene(scene) -> bytes:
    """Pack ExportScene into a binary blob."""
    # 1. Collect all binary chunks (buffer data + texture data)
    binary_chunks = []  # list of (id, data_bytes)

    for buf_id, buf in scene.buffers.items():
        binary_chunks.append((buf_id, buf.data))

    for tex_id, tex in scene.textures.items():
        binary_chunks.append((tex_id, tex.data))

    # 2. Compute offsets (each chunk 16-byte aligned)
    offsets = {}
    current_offset = 0
    padded_chunks = []
    for chunk_id, data in binary_chunks:
        offsets[chunk_id] = current_offset
        padded_size = (len(data) + 15) & ~15  # round up to 16
        padded_chunks.append(data + b"\x00" * (padded_size - len(data)))
        current_offset += padded_size

    # 3. Build JSON metadata
    metadata = {
        "buffers": {},
        "textures": {},
        "compute_passes": [],
        "render_passes": [],
        "interactions": [],
        "camera": scene.camera,
        "light": _serialize_light(scene.light),
    }

    for buf_id, buf in scene.buffers.items():
        metadata["buffers"][buf_id] = {
            "usage": buf.usage,
            "size": buf.size,
            "offset": offsets[buf_id],
            "byte_length": len(buf.data),
        }

    for tex_id, tex in scene.textures.items():
        metadata["textures"][tex_id] = {
            "width": tex.width,
            "height": tex.height,
            "format": tex.format,
            "sampler": tex.sampler,
            "offset": offsets[tex_id],
            "byte_length": len(tex.data),
        }

    for cp in scene.compute_passes:
        metadata["compute_passes"].append(
            {
                "id": cp.id,
                "shader": cp.shader,
                "bindings": {str(k): v for k, v in cp.bindings.items()},
                "workgroups": cp.workgroups
                if isinstance(cp.workgroups, list)
                else [cp.workgroups, 1, 1],
                "triggers": cp.triggers,
                "reset_buffers": cp.reset_buffers,
                "indirect_setup": cp.indirect_setup,
                "count_then_fill": cp.count_then_fill,
            }
        )

    for rp in scene.render_passes:
        metadata["render_passes"].append(
            {
                "id": rp.id,
                "shader": rp.shader,
                "bindings": {str(k): v for k, v in rp.bindings.items()},
                "vertex_count": rp.vertex_count,
                "instance_count": rp.instance_count,
                "draw_indirect": rp.draw_indirect,
                "topology": rp.topology,
                "depth_write": rp.depth_write,
                "depth_bias": rp.depth_bias,
                "pass_type": rp.pass_type,
                "vertex_entry_point": rp.vertex_entry_point,
                "fragment_entry_point": rp.fragment_entry_point,
                "vertex_buffers": rp.vertex_buffers,
                "index_buffer_id": rp.index_buffer_id,
                "index_format": rp.index_format,
            }
        )

    for inter in scene.interactions:
        metadata["interactions"].append(
            {
                "type": inter.type,
                "buffer_id": inter.buffer_id,
                "config": inter.config,
            }
        )

    json_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    # 4. Pad JSON to 16-byte alignment
    json_padded_size = (len(json_bytes) + 15) & ~15
    json_padded = json_bytes + b"\x00" * (json_padded_size - len(json_bytes))

    # 5. Assemble final blob
    header = struct.pack("<4sII", MAGIC, VERSION, len(json_bytes))
    binary_data = b"".join(padded_chunks)

    return header + json_padded + binary_data


def deserialize_scene(blob: bytes):
    """Unpack a binary blob back to ExportScene (for testing/validation)."""
    from .format import (
        ExportBuffer,
        ExportComputePass,
        Interaction,
        ExportRenderPass,
        ExportScene,
        ExportTexture,
    )

    magic = blob[:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic}")

    version, json_length = struct.unpack_from("<II", blob, 4)
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    json_start = 12
    json_bytes = blob[json_start : json_start + json_length]
    metadata = json.loads(json_bytes.decode("utf-8"))

    # Binary data starts after padded JSON
    json_padded_size = (json_length + 15) & ~15
    binary_start = 12 + json_padded_size
    binary_data = blob[binary_start:]

    # Reconstruct buffers
    buffers = {}
    for buf_id, info in metadata["buffers"].items():
        data = binary_data[info["offset"] : info["offset"] + info["byte_length"]]
        buffers[buf_id] = ExportBuffer(
            id=buf_id,
            data=data,
            usage=info["usage"],
            size=info["size"],
        )

    # Reconstruct textures
    textures = {}
    for tex_id, info in metadata["textures"].items():
        data = binary_data[info["offset"] : info["offset"] + info["byte_length"]]
        textures[tex_id] = ExportTexture(
            id=tex_id,
            data=data,
            width=info["width"],
            height=info["height"],
            format=info["format"],
            sampler=info["sampler"],
        )

    # Reconstruct passes
    compute_passes = [
        ExportComputePass(
            id=cp["id"],
            shader=cp["shader"],
            bindings={int(k): v for k, v in cp["bindings"].items()},
            workgroups=cp["workgroups"],
            triggers=cp["triggers"],
            reset_buffers=cp["reset_buffers"],
            indirect_setup=cp.get("indirect_setup"),
            count_then_fill=cp.get("count_then_fill"),
        )
        for cp in metadata["compute_passes"]
    ]

    render_passes = [
        ExportRenderPass(
            id=rp["id"],
            shader=rp["shader"],
            bindings={int(k): v for k, v in rp["bindings"].items()},
            vertex_count=rp["vertex_count"],
            instance_count=rp["instance_count"],
            draw_indirect=rp.get("draw_indirect"),
            topology=rp.get("topology", "triangle-list"),
            depth_write=rp.get("depth_write", True),
            depth_bias=rp.get("depth_bias", 0),
            pass_type=rp.get("pass_type", "opaque"),
            vertex_entry_point=rp.get("vertex_entry_point", "vertex_main"),
            fragment_entry_point=rp.get("fragment_entry_point", "fragment_main"),
            vertex_buffers=rp.get("vertex_buffers", []),
            index_buffer_id=rp.get("index_buffer_id"),
            index_format=rp.get("index_format", "uint32"),
        )
        for rp in metadata["render_passes"]
    ]

    interactions = [
        Interaction(
            type=i["type"],
            buffer_id=i["buffer_id"],
            config=i.get("config", {}),
        )
        for i in metadata["interactions"]
    ]

    return ExportScene(
        buffers=buffers,
        textures=textures,
        compute_passes=compute_passes,
        render_passes=render_passes,
        interactions=interactions,
        camera=metadata.get("camera", {}),
        light=metadata.get("light", {}),
    )


def _serialize_light(light_dict):
    """Convert light dict for JSON (handle bytes)."""
    if not light_dict:
        return {}
    result = {}
    for k, v in light_dict.items():
        if isinstance(v, (bytes, bytearray)):
            import base64

            result[k] = base64.b64encode(v).decode("ascii")
        else:
            result[k] = v
    return result
