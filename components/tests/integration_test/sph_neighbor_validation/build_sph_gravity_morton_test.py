#!/usr/bin/env python3
"""Link the gravity-order regression against the exact production solver objects."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess


SOURCE = pathlib.Path(__file__).with_name("test_sph_gravity_morton.cu")
MAIN = "CMakeFiles/granit_multigpu.dir/main_multigpu.cu.o"
DLINK = "CMakeFiles/granit_multigpu.dir/cmake_device_link.o"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def flag_line(path, name):
    for line in path.read_text().splitlines():
        if line.startswith(name + " = "):
            return shlex.split(line.split(" = ", 1)[1])
    raise ValueError(f"{name} missing in {path}")


def cuda_compiler(build):
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_CUDA_COMPILER:"):
            return line.split("=", 1)[1]
    raise ValueError(f"CMAKE_CUDA_COMPILER missing in {build / 'CMakeCache.txt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    build, out = args.build.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prod = build / "CMakeFiles/granit_multigpu.dir"
    obj = out / "test_sph_gravity_morton.o"
    flags = prod / "flags.make"
    compiler = cuda_compiler(build)
    compile_command = [compiler, "-dc", *flag_line(flags, "CUDA_DEFINES"),
                       *flag_line(flags, "CUDA_INCLUDES"),
                       *flag_line(flags, "CUDA_FLAGS"), "-o", str(obj), str(SOURCE)]
    subprocess.run(compile_command, cwd=build, check=True)

    original = shlex.split((prod / "deviceObjects1.rsp").read_text())
    if original.count(MAIN) != 1:
        raise ValueError("Expected exactly one production main object")
    objects = [obj if item == MAIN else build / item for item in original]
    if not all(path.is_file() for path in objects):
        raise FileNotFoundError("Production build objects are incomplete")
    response = out / "deviceObjects1.gravity_test.rsp"
    response.write_text(" ".join(shlex.quote(str(path)) for path in objects) + "\n")

    dlink_obj = out / "gravity_test_device_link.o"
    dlink = shlex.split((prod / "dlink.txt").read_text())
    dlink[dlink.index("CMakeFiles/granit_multigpu.dir/deviceObjects1.rsp")] = str(response)
    dlink[dlink.index(DLINK)] = str(dlink_obj)
    subprocess.run(dlink, cwd=build, check=True)

    binary = out / "test_sph_gravity_morton"
    link = shlex.split((prod / "link.txt").read_text())
    link = [str(obj) if item == MAIN else str(dlink_obj) if item == DLINK
            else str(binary) if item == "bin/granit_multigpu" else item for item in link]
    subprocess.run(link, cwd=build, check=True)

    gravity_obj = build / "CMakeFiles/granit_multigpu.dir/src/core/acceleration/self_gravity.cu.o"
    receipt = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "production_gravity_object": str(gravity_obj),
        "production_gravity_object_sha256": sha256(gravity_obj),
        "candidate_build": str(build),
        "candidate_solver_sha256": sha256(build / "bin/granit_multigpu"),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "compile_command": compile_command,
        "device_link_command": dlink,
        "host_link_command": link,
        "note": "Only the solver main object is replaced; gravity and every other solver object come from the candidate build.",
    }
    (out / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
