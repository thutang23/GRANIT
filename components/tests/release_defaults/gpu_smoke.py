#!/usr/bin/env python3
"""Opt-in release smoke/force checks, including the initial RESPA snapshot.

Requires NumPy/PyYAML and two GPUs; writes inputs, logs and verdicts in a new
output directory. The 0.2% force gate is retained even when a case fails.
No timing or long-term conservation claim is made by this short fixture.
"""
from pathlib import Path
import argparse, hashlib, json, os, struct, subprocess
import numpy as np
import yaml

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--gpus', default='0,1', help='Two available physical GPU IDs')
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
build=args.build_dir.resolve()
work=args.output_dir.resolve()
work.mkdir(parents=True, exist_ok=False)
gpus=args.gpus.split(',')
if len(gpus)!=2: raise ValueError('--gpus requires two GPU IDs')
rng=np.random.default_rng(20260923)
side=8
pos=np.indices((side,)*3).reshape(3,-1).T.astype(float)
pos=(pos-3.5)*.5+rng.uniform(-.07,.07,pos.shape)
N=len(pos)
def write(physics,pos,fields,name):
 n=len(pos); offset=0; descriptors=[];data=[]
 for fid,dt,dtype,values in fields:
  values=np.asarray(values,dtype=dtype)
  descriptors.append(struct.pack('<HBBIQ',fid,dt,1,0,offset));blob=values.tobytes();data.append(blob);offset+=len(blob)
 header=struct.pack('<8sHHQqddHBBIIQQ',b'GRANIT\0\0',3,72,n,0,0.,0.,len(fields),physics,1,0,0,0,0)
 assert len(header)==72
 (work/name).write_bytes(header+b''.join(descriptors)+pos.T.astype('<f8').tobytes()+np.zeros((3,n),'<f8').tobytes()+np.arange(n,dtype='<i4').tobytes()+b''.join(data))
write(1,pos,[(6,2,'<i4',np.zeros(N)),(0x11,0,'<f4',np.full(N,.1))],'dem.bin')
for name,depth,mg in [('dem_auto',None,False),('dem_d8',8,False),('dem_d9',9,False),('dem_multi_auto',None,True)]:
 c={'material_database':str(root/'components/config/materials_example.yaml'),
 'simulation':{'physics':'dem','time':{'stop':8e-5,'output_step':2e-5,'dt':1e-5}},
 'dem':{'damping_model':'restitution'},'domain':{'box_min':[-3.,-3.,-3.],'box_max':[3.,3.,3.]},
 'accelerations':{'self_gravity':{'enabled':True}},
 'io':{'input':{'particle_file':str(work/'dem.bin')},'output':{'directory':str(work/name),'initial_state':True,'fields':['gravity_force']},'checkpoint':{'enabled':False}},
 'gpu':{'device_id':0}}
 if depth:c['accelerations']['self_gravity']['unified_fmm']={'tree_depth':depth}
 if mg:c['multigpu']={'enabled':True,'max_gpus':2}
 (work/(name+'.yaml')).write_text(yaml.safe_dump(c))
# SPH no-gravity startup with the public production precision defaults.
pos_sph=(np.indices((5,)*3).reshape(3,-1).T.astype(float)-2)*.01
n=len(pos_sph)
write(2,pos_sph,[(6,2,'<i4',np.zeros(n)),(0x12,0,'<f4',np.full(n,.0027)),(1,0,'<f4',np.full(n,2700))],'sph.bin')
c={'material_database':str(root/'components/config/materials_example.yaml'),'simulation':{'physics':'sph','time':{'stop':2e-7,'output_step':1e-7}},'sph_parameters':{'smoothing_length':.012,'dt_max':1e-7},'domain':{'box_min':[-.1]*3,'box_max':[.1]*3},'io':{'input':{'particle_file':str(work/'sph.bin')},'output':{'directory':str(work/'sph_default'),'initial_state':True},'checkpoint':{'enabled':False}}}
(work/'sph_default.yaml').write_text(yaml.safe_dump(c))

# Sequential runs keep the resource footprint bounded to the chosen two GPUs.
for name,solver,multi in [('dem_auto','granit_dem',False),('dem_d8','granit_dem',False),
                          ('dem_d9','granit_dem',False),('dem_multi_auto','granit_multigpu',True),
                          ('sph_default','granit_sph',False)]:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus) if multi else gpus[0])
    with (work/f'{name}.log').open('w') as log:
        subprocess.run([str(build/'bin'/solver),str(work/f'{name}.yaml')], env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
receipt={'gpus':args.gpus,'build_dir':str(build),
         'binary_sha256':{p.name:sha256(p) for p in (build/'bin').glob('granit_*') if p.is_file()},
         'g_table_sha256':sha256(build/'share/granit/fmm/g_table_v1.bin'),
         'input_sha256':{p.name:sha256(p) for p in work.glob('*.bin')}}
(work/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')

def read(path):
 b=path.read_bytes();h=struct.unpack_from('<8sHHQqddHBBIIQQ',b)
 magic,version,hlen,n,step,time,dt,nf,physics,kind,stride,contacts,matfp,clumpfp=h
 assert magic==b'GRANIT\0\0' and version==3 and hlen==72 and kind==2
 assert physics in [1,2] and stride==contacts==0 and matfp!=0
 pos0=72+16*nf; t2=pos0+52*n
 pos=np.frombuffer(b,'<f8',3*n,pos0).reshape(3,n).T.copy()
 vel=np.frombuffer(b,'<f8',3*n,pos0+24*n).reshape(3,n).T.copy()
 ids=np.frombuffer(b,'<i4',n,pos0+48*n).copy();assert len(set(ids))==n
 fields={};end=t2
 for i in range(nf):
  fid,dtype,count,res,offset=struct.unpack_from('<HBBIQ',b,72+16*i)
  assert res==0 and fid not in fields
  typ=['<f4','<f8','<i4','<i2','u1','?','<i8'][dtype]
  field=np.frombuffer(b,typ,n*count,t2+offset).reshape(count,n).T.copy()
  assert np.isfinite(field).all();fields[fid]=field
  end=max(end,t2+offset+field.nbytes)
 assert len(b)==end and np.isfinite(pos).all() and np.isfinite(vel).all()
 return n,step,time,dt,pos,vel,ids,fields

results={}
for name in ['dem_auto','dem_d8','dem_d9','dem_multi_auto','sph_default']:
 files=sorted((work/name).glob('output_step_*.bin'))
 assert len(files)>=2,(name,files)
 rows=[read(p) for p in files]
 assert rows[-1][2]>= (2e-7 if name=='sph_default' else 8e-5)-1e-12
 result={'frames':len(rows),'final_step':rows[-1][1],'final_time':rows[-1][2]}
 if name.startswith('dem'):
  n,step,t,dt,pos,vel,ids,f=rows[0]
  side=Path(str(files[0])+'.gravity_force');bb=side.read_bytes()
  magic,version,ng,sg,tg,dtg=struct.unpack_from('<8sIiqdd',bb)
  assert ng==n and sg==step and len(bb)==40+24*n
  force=np.frombuffer(bb,'<f8',3*n,40).reshape(3,n).T.copy()
  order=np.argsort(ids);pos=pos[order];mass=f[0x12][order,0].astype(float)
  eps=float(f[0x11][:,0].min())*.01
  diff=pos[None,:,:]-pos[:,None,:]
  rr=np.sum(diff**2,axis=2)+eps**2;np.fill_diagonal(rr,np.inf)
  direct=6.67430e-11*mass[:,None]*np.sum(diff*(mass[None,:]/rr**1.5)[:,:,None],axis=1)
  rms=float(np.linalg.norm(force-direct)/np.linalg.norm(direct))
  net=float(np.linalg.norm(force.sum(axis=0))/np.linalg.norm(force,axis=1).sum())
  com=(mass[:,None]*pos).sum(axis=0)/mass.sum();r=pos-com
  torque=np.cross(r,force)
  net_torque=float(np.linalg.norm(torque.sum(axis=0))/np.sum(np.linalg.norm(r,axis=1)*np.linalg.norm(force,axis=1)))
  result['force_rms_gate_0.002'] = 'pass' if rms <= 2e-3 else 'fail'
  result['net_force_gate_1e-6'] = 'pass' if net <= 1e-6 else 'fail'
  result['net_torque_gate_1e-6'] = 'pass' if net_torque <= 1e-6 else 'fail'
  assert np.isfinite(force).all()
  result.update(force_relative_rms=rms,net_force_fraction=net,net_torque_fraction=net_torque)
 results[name]=result
(work/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

if any(value == 'fail' for result in results.values() for value in result.values()):
    raise SystemExit(1)
