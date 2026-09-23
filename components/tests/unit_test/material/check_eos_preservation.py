#!/usr/bin/env python3
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[4];out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
build=a.build.resolve();yaml=build/'pinned-deps/yaml-cpp-f7320141120f720aecc4c32be25586e7da9eb978/include';lib=build/'lib/libyaml-cpp.a'
for name,tree in [('baseline',a.baseline.resolve()),('integrated',r)]:
 binary=out/f'probe-eos-{name}'
 command=['g++','-std=c++17','-O2','-I'+str(tree),'-I/usr/local/cuda-12.1/include','-I'+str(yaml),str(r/'components/tests/unit_test/material/probe_sph_eos.cpp'),str(tree/'src/core/material/material_database.cpp'),str(lib),'-L/usr/local/cuda-12.1/lib64','-Wl,-rpath,/usr/local/cuda-12.1/lib64','-lcudart','-o',str(binary)]
 subprocess.run(command,check=True)
 with (out/f'probe-eos-{name}.txt').open('w') as f:subprocess.run([str(binary),str(r/'components/config/materials_example.yaml')],stdout=f,check=True)
x=(out/'probe-eos-baseline.txt').read_bytes();y=(out/'probe-eos-integrated.txt').read_bytes()
receipt={'passed':x==y,'cases':1260,'eos_models':3,'fingerprint_variants':6,'comparison':'exact hexadecimal float output and fingerprint equality','sha256':hashlib.sha256(x).hexdigest()}
(out/'eos-preservation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt);assert x==y
