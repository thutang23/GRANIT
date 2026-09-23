#!/usr/bin/env python3
"""Bounded SPH RESPA/event/restart/auto-depth checks on one and two GPUs.

Uses the existing SPH replay field tolerances and 1% RMS/5% maximum gravity
error gates. Writes all inputs, commands, hashes, logs and measured verdicts.
"""
from pathlib import Path
import argparse, csv, dataclasses, importlib.util, json, os, shutil, subprocess, sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/plugins/scripts'))
from cudadem_pipeline.common.io import read_particle_binary, write_particle_binary
spec=importlib.util.spec_from_file_location('replay',ROOT/'components/tests/integration_test/sph_neighbor_validation/replay_sph_backends.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)

def ordered(data):
    order=np.argsort(data.global_id)
    for field in dataclasses.fields(data):
        value=getattr(data,field.name)
        if isinstance(value,np.ndarray) and len(value)==data.num_particles:
            setattr(data,field.name,value[order])
    return data

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--gpus',default='0,1')
    ap.add_argument('--force-only',action='store_true')
    ap.add_argument('--timeout',type=float,default=180,help='Per-run seconds; allow extra time for a cold PTX JIT')
    ap.add_argument('--ranks',default='0,1,2',help='0 for single GPU runner; positive for multi-GPU runner')
    args=ap.parse_args();ranks_to_test=[int(x) for x in args.ranks.split(',')];work=args.output_dir.resolve();work.mkdir(parents=True,exist_ok=False)
    build=args.build_dir.resolve();source=work/'input.bin';replay.fixture(source,6)
    material=ROOT/'components/config/materials_example.yaml'
    report={'binaries':{name:replay.digest(build/'bin'/name) for name in ['granit_sph','granit_multigpu']},
            'gpus':args.gpus,'commands':[],'checks':{},'passed':False}
    def save(): (work/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def run(name,ranks=0,gravity=False,k=2,adaptive=False,events=False,resume=None,zero=False,method='unified_fmm',far=False,empty=False,defaults=False,reuse_output=None):
        folder=work/name;folder.mkdir()
        c=replay.config(folder,source,material,'cuda',1,12,4,True)
        c['neighbor'].update(verlet_buffer_enabled=False,update_frequency=1)
        c['accelerations']['self_gravity'].update(enabled=gravity,method=method,gravity_update_rate=k)
        c['accelerations']['self_gravity']['direct']={'gravitational_constant':1.0}
        fmm=c['accelerations']['self_gravity']['unified_fmm']
        fmm.update(tree_depth=5,gravitational_constant=1.0,enable_global_projection=True)
        if adaptive:
            fmm['tree_depth']='auto'
            fmm['adaptive_depth']={'enabled':True,'min_depth':2,'max_depth':10,
                'trial_interval_evaluations':2,'trial_warmup_evaluations':0,'trial_samples':1,'confirmation_trials':1}
        if defaults:
            fmm.pop('tree_depth',None);fmm.pop('adaptive_depth',None)
        c['multigpu']={'enabled':bool(ranks),'max_gpus':max(1,ranks),'morton_ownership':True,
            'halo_width_factor':1.5,'buffer_capacity_factor':3.0,'rebalance_min_interval':1000}
        c['io']['checkpoint']['interval']=3
        c['io']['output']['scalar_interval']=1
        if reuse_output:c['io']['output']['directory']=str(reuse_output)
        if events:
            c['sph_parameters'].update(dt_min=1e-10,dt_max=1e-5,cfl_number=.1,track_history=True,
                stages=[{'mode':'RELAX','end_time':4.7e-7},{'mode':'SHOCK','end_time':1.09e-6}])
            c['simulation']['time'].update(stop=1.09e-6,output_step=3.1e-7)
        if far:
            far_path=work/'far.bin'
            if not far_path.exists():
                d=read_particle_binary(source);d.lifecycle_state=None;d.positions*=12
                write_particle_binary(far_path,d,physics='sph')
            c['io']['input']['particle_file']=str(far_path)
        if empty:
            empty_path=work/'coincident.bin'
            if not empty_path.exists():
                d=read_particle_binary(source);d.lifecycle_state=None;d.positions[:]=0;d.velocities[:]=0
                write_particle_binary(empty_path,d,physics='sph')
            c['io']['input']['particle_file']=str(empty_path)
        if resume:
            c['io']['checkpoint'].update(force_fresh=False,resume_from=str(resume))
            if zero:c['simulation']['time']['stop']=read_particle_binary(resume).current_time
        elif zero:c['simulation']['time']['stop']=0.0
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
        binary=build/'bin'/('granit_multigpu' if ranks else 'granit_sph')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus if ranks>1 else args.gpus.split(',')[0])
        command=[str(binary),str(path)]
        with (folder/'run.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        report['commands'].append({'argv':command,'returncode':result.returncode,'config_sha256':replay.digest(path)})
        save()
        if result.returncode:raise RuntimeError(f'{name}: see {folder / "run.log"}')
        files=sorted((folder/'checkpoints').glob('checkpoint_step_*.bin'))
        assert files,name
        data=ordered(read_particle_binary(files[-1]));assert data.force.dtype==np.float64
        assert data.gravity_acceleration.dtype==np.float64
        start_step=read_particle_binary(resume).current_step if resume else 0
        expected_first=min(data.current_step,start_step+((3+k-1)//k)*k)
        assert read_particle_binary(files[0]).current_step==expected_first,(name,expected_first,files)
        for f in files:
            d=read_particle_binary(f)
            assert d.current_step%k==0,(name,f,d.current_step)
            assert f.with_suffix('.bin.respa_sync').exists()
            if adaptive:assert f.with_suffix('.bin.adaptive_fmm_depth').exists()
        receipts=folder/'output/sph_macros.csv'
        if receipts.exists():
            rows=list(csv.DictReader(receipts.open()))
            for i,row in enumerate(rows):
                assert int(row['inner_count'])==k
                assert float(row['inner_dt']) <= float(row['cfl_dt'])*(1+1e-14)
                if i:assert int(row['start_step'])==int(rows[i-1]['start_step'])+k
                if 'cfl_evaluations' in row:assert int(row['cfl_evaluations'])==i+1
            if events:
                endpoints=[float(row['end_time']) for row in rows]
                for t in [3.1e-7,4.7e-7,6.2e-7,9.3e-7,1.09e-6]:
                    if not resume or t > read_particle_binary(resume).current_time + 1e-20:
                        assert any(abs(t-x)<1e-20 for x in endpoints),(name,t,endpoints)
                assert len(set(row['inner_dt'] for row in rows))>1
            if gravity and not events and not resume:
                assert [int(row['gravity_evaluations']) for row in rows]==list(range(1,len(rows)+1)),(name,rows)
        report['checks'][name]={'pass':True,'final_time':data.current_time,'final_step':data.current_step,'checkpoints':len(files)}
        save();print('PASS',name,flush=True)
        return files,data
    def compare(name,left,right):
        result=replay.compare_pair(left,right)
        # Retain existing field gates; slow acceleration is checked additionally.
        delta=np.linalg.norm(left.gravity_acceleration-right.gravity_acceleration)
        ref=np.linalg.norm(left.gravity_acceleration)
        result['gravity_relative_l2']=float(delta/max(ref,1e-30))
        result['pass'] = bool(result['pass'] and delta <= 2e-5*ref+1e-12)
        for field in ('energy_cold','energy_thermal_peak','pressure_peak','fracture_area','fragment_length','damping_energy'):
            a,b=getattr(left,field),getattr(right,field)
            if a is not None:
                result['pass'] = bool(result['pass'] and np.all(np.abs(a-b) <= max(1e-12,replay.ROUNDOFF*np.abs(a).max())+replay.RTOL*np.abs(a)))
        if name.endswith('-roundtrip'):
            for field in dataclasses.fields(left):
                a,b=getattr(left,field.name),getattr(right,field.name)
                if isinstance(a,np.ndarray):result['pass']=bool(result['pass'] and np.array_equal(a,b))
        report['checks'][name]=result;save()
        if not result['pass']:raise AssertionError(name)
        print('PASS',name,flush=True)
    def reject(name,ranks,source_config,kind,checkpoint=None,diagnostics=None):
        folder=work/name;folder.mkdir()
        c=yaml.safe_load(source_config.read_text())
        c['io']['output']['directory']=str(folder/'output')
        c['io']['checkpoint']['path']=str(folder/'checkpoints')
        if checkpoint:
            c['io']['checkpoint'].update(force_fresh=False,resume_from=str(checkpoint))
        if kind=='cadence':
            c['accelerations']['self_gravity']['gravity_update_rate']=1
        elif kind=='physical':
            c['sph_parameters']['hsml_min']=1e-5
        elif kind=='corrupt':
            copied=folder/checkpoint.name
            for path in checkpoint.parent.glob(checkpoint.name+'*'):
                shutil.copy2(path,folder/path.name)
            c['io']['checkpoint']['resume_from']=str(copied)
            sidecar=Path(str(copied)+'.adaptive_fmm_depth')
            sidecar.write_bytes(sidecar.read_bytes()[:16])
        elif kind=='csv_gap':
            output=folder/'output';output.mkdir()
            for suffix in ('','.start'):
                shutil.copy2(diagnostics/('adaptive_fmm_depth.csv'+suffix),output)
            csv_path=output/'adaptive_fmm_depth.csv'
            lines=csv_path.read_text().splitlines()
            assert len(lines)>2
            csv_path.write_text('\n'.join([lines[0]]+lines[2:])+'\n')
        path=folder/'config.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False))
        command=[str(build/'bin'/('granit_multigpu' if ranks else 'granit_sph')),str(path)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus if ranks>1 else args.gpus.split(',')[0])
        with (folder/'run.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        output=(folder/'run.log').read_text()
        expected={'cadence':('RESPA',),'physical':('h_max/hsml_min','smoothing lengths'),
                  'corrupt':('checkpoint',),'csv_gap':('sequence gap','CSV watermark is unavailable')}[kind]
        passed=result.returncode!=0 and any(s.lower() in output.lower() for s in expected)
        report['commands'].append({'argv':command,'returncode':result.returncode,'expected_rejection':True,'config_sha256':replay.digest(path)})
        report['checks'][name]={'pass':passed,'expected_rejection':True};save()
        assert passed,(name,output[-2000:])
        print('PASS',name,flush=True)
    try:
        if not args.force_only:
            for ranks in ranks_to_test:
                run(f'hydro-k1-r{ranks}',ranks,k=1)
                event_files,event_data=run(f'events-k2-r{ranks}',ranks,events=True)
                _,event_resume=run(f'events-resume-r{ranks}',ranks,events=True,resume=event_files[0])
                compare(f'events-continuation-r{ranks}',event_data,event_resume)
                for adaptive in (False,True):
                    name=f'gravity-k2-r{ranks}-auto{int(adaptive)}'
                    files,data=run(name,ranks,gravity=True,adaptive=adaptive)
                    split=files[0]
                    continued_files,continued=run(name+'-resume',ranks,gravity=True,adaptive=adaptive,resume=split)
                    compare(name+'-continuation',data,continued)
                    if adaptive and ranks==0:
                        _,again=run(name+'-resume-again',ranks,gravity=True,adaptive=True,
                            resume=continued_files[0],reuse_output=work/(name+'-resume')/'output')
                        compare(name+'-continuation-again',data,again)
                        reject('reject-csv-segment-gap',ranks,work/name/'config.yaml','csv_gap',
                            continued_files[0],work/(name+'-resume')/'output')
                    _,roundtrip=run(name+'-zero',ranks,gravity=True,adaptive=adaptive,resume=files[-1],zero=True)
                    compare(name+'-roundtrip',data,roundtrip)
                    if adaptive:
                        config_path=work/name/'config.yaml'
                        reject(f'reject-cadence-r{ranks}',ranks,config_path,'cadence',files[0])
                        reject(f'reject-sidecar-r{ranks}',ranks,config_path,'corrupt',files[0])
                        reject(f'reject-physical-r{ranks}',ranks,config_path,'physical')
            if 2 in ranks_to_test:run('empty-owner-k2',2,gravity=True,adaptive=True,empty=True)
            for ranks in ranks_to_test:run(f'default-depth-r{ranks}',ranks,gravity=True,adaptive=True,defaults=True)
        for far in (False,True):
            _,direct=run(f'direct-zero-far{far}',gravity=True,zero=True,method='direct',far=far)
            for ranks in ranks_to_test:
                _,candidate=run(f'fmm-zero-r{ranks}-far{far}',ranks,gravity=True,zero=True,adaptive=True,far=far,defaults=True)
                delta=candidate.gravity_acceleration-direct.gravity_acceleration
                relative=float(np.linalg.norm(delta)/np.linalg.norm(direct.gravity_acceleration))
                maximum=float(np.linalg.norm(delta,axis=1).max()/(np.linalg.norm(direct.gravity_acceleration)/np.sqrt(direct.num_particles)))
                force=candidate.mass[:,None]*candidate.gravity_acceleration
                net=float(np.linalg.norm(force.sum(axis=0))/np.linalg.norm(force,axis=1).sum())
                radius=candidate.positions-np.average(candidate.positions,axis=0,weights=candidate.mass)
                torque=float(np.linalg.norm(np.cross(radius,force).sum(axis=0))/np.sum(np.linalg.norm(radius,axis=1)*np.linalg.norm(force,axis=1)))
                check={'relative_rms':relative,'maximum_over_reference_rms':maximum,'net_force_fraction':net,
                       'net_torque_fraction':torque,'pass':relative<=.01 and maximum<=.05 and net<=1e-6 and torque<=1e-6}
                report['checks'][f'force-r{ranks}-far{far}']=check;save()
                assert check['pass'],check
        report['passed']=True;save()
    finally:save()
if __name__=='__main__':main()
