#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.1",
#   "zarr==3.2.1",
# ]
# ///

"""Generate Zarr v3 fixtures using zarr-python as the reference implementation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec, Crc32cCodec, GzipCodec, ZstdCodec


OUTPUT_DIR = Path(__file__).parent.parent / "images" / "zarr"
STRING_VALUES = np.asarray(["A", "BC"], dtype="U2")


def create_string_array(
    path: Path,
    *,
    serializer: object,
    compressors: list[object] | None = None,
    chunk_key_encoding: dict[str, object] | None = None,
    shape: tuple[int, ...] = (2,),
    chunks: tuple[int, ...] = (2,),
    write_values: bool = True,
    values: np.ndarray = STRING_VALUES,
) -> zarr.Array:
    array = zarr.create_array(
        store=path,
        shape=shape,
        chunks=chunks,
        dtype=values.dtype,
        zarr_format=3,
        chunk_key_encoding=chunk_key_encoding
        or {"name": "default", "configuration": {"separator": "/"}},
        serializer=serializer,
        compressors=compressors or [],
        fill_value="",
    )
    if write_values:
        array[:] = values
    return array


def chunk_path(array_path: Path) -> Path:
    metadata = zarr.open_array(array_path, mode="r").metadata
    if metadata.chunk_key_encoding.name == "v2":
        return array_path / "0"
    separator = metadata.chunk_key_encoding.separator
    return array_path / ("c.0" if separator == "." else "c/0")


def generate_string_fixtures() -> None:
    output_dir = OUTPUT_DIR / "string"
    output_dir.mkdir(parents=True)

    create_string_array(
        output_dir / "zstd_overhang",
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        chunks=(4,),
    )
    create_string_array(
        output_dir / "v2_key",
        serializer=BytesCodec(endian="little"),
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "."}},
    )

    gzip_path = output_dir / "gzip"
    create_string_array(
        gzip_path,
        serializer=BytesCodec(endian="little"),
        compressors=[GzipCodec(level=1)],
    )
    gzip_chunk = bytearray(chunk_path(gzip_path).read_bytes())
    if gzip_chunk[:2] != bytes((0x1F, 0x8B)):
        raise RuntimeError("zarr-python did not produce a gzip stream")
    gzip_chunk[4:8] = bytes(
        4
    )  # Normalize the gzip MTIME header for reproducible fixtures.
    chunk_path(gzip_path).write_bytes(gzip_chunk)

    create_string_array(
        output_dir / "blosc",
        serializer=BytesCodec(endian="little"),
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle="noshuffle")],
    )
    create_string_array(
        output_dir / "big_endian",
        serializer=BytesCodec(endian="big"),
        values=np.asarray(["Ω", "🙂"], dtype="U2"),
    )
    create_string_array(
        output_dir / "dot_key",
        serializer=BytesCodec(endian="little"),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
    )
    create_string_array(
        output_dir / "missing_chunk",
        serializer=BytesCodec(endian="little"),
        write_values=False,
    )
    create_string_array(
        output_dir / "crc_before_after_zstd",
        serializer=BytesCodec(endian="little"),
        compressors=[Crc32cCodec(), ZstdCodec(level=1), Crc32cCodec()],
    )

    crc_mismatch = output_dir / "crc_mismatch"
    create_string_array(
        crc_mismatch,
        serializer=BytesCodec(endian="little"),
        compressors=[Crc32cCodec()],
    )
    crc_bytes = bytearray(chunk_path(crc_mismatch).read_bytes())
    crc_bytes[-1] ^= 0xFF
    chunk_path(crc_mismatch).write_bytes(crc_bytes)

    truncated = output_dir / "truncated"
    create_string_array(truncated, serializer=BytesCodec(endian="little"))
    truncated_bytes = chunk_path(truncated).read_bytes()
    chunk_path(truncated).write_bytes(truncated_bytes[:-4])

    invalid_unicode = output_dir / "invalid_unicode"
    create_string_array(invalid_unicode, serializer=BytesCodec(endian="little"))
    invalid_bytes = bytearray(chunk_path(invalid_unicode).read_bytes())
    invalid_bytes[:4] = bytes((0x00, 0xD8, 0x00, 0x00))  # UTF-32LE surrogate U+D800.
    chunk_path(invalid_unicode).write_bytes(invalid_bytes)


def create_numeric_array(
    path: Path,
    values: np.ndarray,
    *,
    dimension_names: tuple[str, ...],
    chunks: tuple[int, ...] | None = None,
    attributes: dict[str, Any] | None = None,
) -> zarr.Array:
    array = zarr.create_array(
        store=path,
        shape=values.shape,
        chunks=chunks or values.shape,
        dtype=values.dtype,
        zarr_format=3,
        dimension_names=dimension_names,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        fill_value=0,
        attributes=attributes,
    )
    array[:] = values
    return array


def generate_xradio_fixture() -> None:
    path = OUTPUT_DIR / "xradio" / "minimal"
    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "coordinate_system_info": {
                "projection": "SIN",
                "reference_direction": {
                    "data": [1.0, 0.5],
                    "attrs": {"frame": "fk5", "equinox": "J2000"},
                },
                "native_pole_direction": {"data": [0.0, 1.5707963267948966]},
                "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        }
    )

    sky_shape = (1, 3, 2, 4, 5)
    create_numeric_array(
        path / "SKY",
        np.zeros(sky_shape, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={
            "units": "Jy/beam",
            "type": "Intensity",
            "object_name": "Zarr test source",
            "observer": "CARTA",
            "obsdate": {"data": 59000.0, "attrs": {"format": "MJD", "scale": "UTC"}},
            "telescope": {
                "name": "Test scope",
                "direction": {"data": [0.0, 0.0]},
                "distance": {"data": [6371000.0]},
            },
            "user": {"origin": "zarr-python", "exposure": 12.5},
            "beam_fit_params": "BEAM",
        },
    )
    create_numeric_array(
        path / "l",
        np.asarray([-0.001, 0.0, 0.001, 0.002], dtype=np.float64),
        dimension_names=("l",),
    )
    create_numeric_array(
        path / "m",
        np.asarray([-0.002, -0.001, 0.0, 0.001, 0.002], dtype=np.float64),
        dimension_names=("m",),
    )
    create_numeric_array(
        path / "frequency",
        np.asarray([1.4e9, 1.401e9, 1.402e9], dtype=np.float64),
        dimension_names=("frequency",),
        attributes={
            "reference_frequency": {"attrs": {"units": "Hz", "observer": "lsrk"}},
            "rest_frequency": {"data": 1.420405751e9},
        },
    )
    polarization = create_string_array(
        path / "polarization",
        serializer=BytesCodec(endian="little"),
        values=np.asarray(["I", "Q"], dtype="U1"),
    )
    polarization.attrs.update({"dimension_names": ["polarization"]})

    parameter_labels = create_string_array(
        path / "beam_params_label",
        serializer=BytesCodec(endian="little"),
        shape=(3,),
        chunks=(3,),
        values=np.asarray(["minor", "pa", "major"], dtype="U5"),
    )
    parameter_labels.attrs.update({"dimension_names": ["beam_params_label"]})

    beam_values = np.zeros((1, 3, 2, 3), dtype=np.float64)
    for channel in range(3):
        for stokes in range(2):
            major = 2.0e-5 + (channel * 1.0e-6) + (stokes * 1.0e-7)
            beam_values[0, channel, stokes] = (major / 2.0, 0.1 + channel * 0.01, major)
    create_numeric_array(
        path / "BEAM",
        beam_values,
        dimension_names=("time", "frequency", "polarization", "beam_params_label"),
        chunks=(1, 1, 1, 3),
        attributes={"units": "rad"},
    )


def main() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    generate_string_fixtures()
    generate_xradio_fixture()


if __name__ == "__main__":
    main()
