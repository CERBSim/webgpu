// Parse binary scene blob produced by Python serialize_scene()
//
// Layout:
//   [4 bytes] magic "WGPU"
//   [4 bytes] version (uint32 LE, = 1)
//   [4 bytes] JSON length N (uint32 LE)
//   [N bytes] JSON metadata (UTF-8)
//   [padding to 16-byte alignment from JSON start]
//   [remaining] concatenated binary data (each chunk 16-byte aligned)

function parseSceneBlob(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  // Header
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  );
  if (magic !== 'WGPU') throw new Error('Invalid scene blob magic');
  const version = view.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported version: ${version}`);
  const jsonLength = view.getUint32(8, true);

  // JSON metadata
  const jsonBytes = new Uint8Array(arrayBuffer, 12, jsonLength);
  const metadata = JSON.parse(new TextDecoder().decode(jsonBytes));

  // Binary data starts after 16-byte-aligned JSON
  const jsonPaddedSize = (jsonLength + 15) & ~15;
  const binaryStart = 12 + jsonPaddedSize;
  const binaryData = new Uint8Array(arrayBuffer, binaryStart);

  // Attach binary slices
  for (const buf of Object.values(metadata.buffers)) {
    buf.data = binaryData.slice(buf.offset, buf.offset + buf.byte_length);
  }
  for (const tex of Object.values(metadata.textures)) {
    tex.data = binaryData.slice(tex.offset, tex.offset + tex.byte_length);
  }

  return metadata;
}

export { parseSceneBlob };
