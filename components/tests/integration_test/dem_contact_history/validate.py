#!/usr/bin/env python3
"""Bounded fresh-only trajectory, history churn, reciprocal and restart audit."""
import argparse
import json
import math
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'dem_multigpu_clumps'))
import validate as base
import extension_validation as ext


def rows():
    # A low-stiffness isolated sphere passes 21 separated spheres. The offset
    # supplies tangential velocity; no prescribed motion or external force.
    yield (-.10, 0., 0.), (1., 0., 0.), (0., 0., 1.), .03, 1
    for j in range(21):
        yield (j*.12, .035, 0.), (0., 0., 0.), (0., 0., 0.), .01, j+2


def overflow_rows():
    import math
    yield (0.,0.,0.), (0.,)*3, (0.,)*3, .04, 1
    for j in range(17):
        angle=2*math.pi*j/17
        yield (.041*math.cos(angle),.041*math.sin(angle),0.), (0.,)*3, (0.,)*3, .002, j+2


def prepare(work, ranks, resume=None, overflow=False):
    work.mkdir(parents=True, exist_ok=False)
    base.write_input(work/'input.bin', overflow_rows() if overflow else rows())
    base.dump(work/'materials.json', {'version': 1, 'materials': [{'id': 0, 'name': 'churn', 'dem': {'density': 1000., 'youngs_modulus': .1, 'poisson_ratio': .25}}], 'dem_interactions': [{'materials': [0, 0], 'restitution': 1., 'friction': {'static': .4, 'rolling': .03, 'twisting': .02}, 'shape_parameter': 1., 'cohesion': {'model': 'none'}}]})
    cfg=base.make_config(work,'contact-churn',18 if overflow else 22,ranks,gravity=False,steps=1400)
    cfg['simulation']['time'].update(dt=.002,stop=math.nextafter(sum(.002 for _ in range(1400)),-math.inf),output_step=.02)
    cfg['io']['checkpoint']={'enabled':True,'interval':10,'path':str(work/'checkpoints'),'force_fresh':True}
    if resume: cfg['io']['checkpoint'].update(resume_from=str(resume),force_fresh=False,rebootstrap_forces=False)
    base.dump(work/'config.json',cfg)


def momentum(record):
    linear=[0.,0.,0.];angular=[0.,0.,0.]
    for gid,row in record["rows"].items():
        mass=record["fields"][0x12][gid][0];inertia=record["fields"][0x13][gid][0]
        spin=record["fields"][0x10][gid]
        orbital=base.cross(row[:3],row[3:6])
        for axis in range(3):
            linear[axis]+=mass*row[3+axis]
            angular[axis]+=mass*orbital[axis]+inertia*spin[axis]
    return linear,angular


def audit(work):
    partners=set(); active=0; nonzero=0; maximum=0.; records=[]
    for path in sorted((work/'checkpoints').glob('checkpoint_step_*.bin')):
        r=ext.read_record(path); records.append(r)
        for (a,b),h in r['histories'].items():
            reverse=r['histories'].get((b,a)); assert reverse is not None, (path,a,b,'missing reciprocal')
            maximum=max(maximum,max(abs(x+y) for x,y in zip(h[2:],reverse[2:])))
            if a==0: partners.add(b)
            active+=1;nonzero+=any(x!=0 for x in h[2:])
    assert len(partners)>16, ('fixture did not exercise >16 distinct partners',len(partners))
    assert maximum==0 and nonzero>0
    initial=momentum(records[0]);drift=[0.,0.]
    for record in records:
        current=momentum(record)
        for kind in range(2):drift[kind]=max(drift[kind],base.norm([a-b for a,b in zip(current[kind],initial[kind])]))
    assert max(drift)<1e-8, ('linear/angular momentum drift',drift)
    return {'pass':True,'probe_distinct_partners':sorted(partners),'active_directed_records_checked':active,'nonzero_records':nonzero,'max_reciprocal_spring_residual':maximum,'max_linear_momentum_drift':drift[0],'max_angular_momentum_drift':drift[1],'momentum_absolute_gate':1e-8},records


def audit_overflow(work):
    artifacts=list((work/'output').glob('**/*'))+list((work/'checkpoints').glob('**/*'))
    files=[str(p.relative_to(work)) for p in artifacts if p.is_file()]
    assert not files, ('overflow emitted output or checkpoint',files)
    return {'pass':True,'output_and_checkpoint_files':files}


def audit_transport(work):
    text=(work/'run.log').read_text()
    migrations=re.findall(r'\[clumps-migration\] rank=(\d+) owned_particles=\d+ previous_members=(\d+) owned_members=(\d+)',text)
    changed=[{'rank':int(rank),'previous_members':int(before),'owned_members':int(after)}
             for rank,before,after in migrations if before!=after]
    remote=re.findall(r'\[clumps-contact\] rank=(\d+) step=\d+ remote_interclump_overlap_pairs=(\d+)',text)
    remote_ranks=sorted({int(rank) for rank,count in remote if int(count)>0})
    assert len({row['rank'] for row in changed})>=2, ('no observed two-rank member migration',changed)
    assert len(remote_ranks)>=2, ('no observed two-rank remote contacts',remote_ranks)
    return {'pass':True,'changed_member_counts':changed,'remote_contact_ranks':remote_ranks}


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--work',type=Path,required=True);p.add_argument('--ranks',type=int,nargs='+',default=[1,2]);p.add_argument('--timeout',type=float,default=120);a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False);report={'pass':False,'checks':{},'launches':[]};base.dump(a.work/'report.json',report)
    try:
        for ranks in a.ranks:
            rejected=a.work/f'overflow-gpu{ranks}';prepare(rejected,ranks,overflow=True)
            launch=base.launch(a.binary,rejected,a.timeout);report['launches'].append(launch)
            log=(rejected/'run.log').read_text()
            assert launch['returncode']==1 and not launch['timeout'] and 'contact-history' in log.lower() and 'capacity exceeded' in log.lower(), (launch,log[-2000:])
            report['checks'][f'overflow-gpu{ranks}']={**audit_overflow(rejected),'exit_code':1}
            full=a.work/f'churn-gpu{ranks}';prepare(full,ranks)
            launch=base.launch(a.binary,full,a.timeout);report['launches'].append(launch);assert launch['returncode']==0 and not launch['timeout']
            report['checks'][f'churn-gpu{ranks}'],records=audit(full)
            # Split while a physical contact carries nonzero spring memory.
            candidates=[r for r in records if 400 <= r['step'] <= 1000 and
                        any(any(v != 0 for v in h[2:]) for h in r['histories'].values())]
            assert candidates, 'no nonzero contact history available for restart split'
            split=min(candidates,key=lambda r:abs(r['step']-700))
            checkpoint=Path(split['path'])
            resumed=a.work/f'churn-restart-gpu{ranks}';prepare(resumed,ranks,checkpoint)
            launch=base.launch(a.binary,resumed,a.timeout);report['launches'].append(launch);assert launch['returncode']==0 and not launch['timeout']
            check=ext.compare_records(records[-1],ext.read_record(ext.latest_checkpoint(resumed)));assert check['pass']
            check['split_step']=split['step'];check['split_active_histories']=len(split['histories'])
            report['checks'][f'churn-restart-gpu{ranks}']=check
        # Existing clump fixture supplies exact rigid-body ownership transport,
        # nonzero tangential memory and midpoint restart without long churn.
        campaign=ext.Campaign(a.binary,a.work/'clump-ownership-restart',a.ranks)
        for ranks in a.ranks:
            campaign.restart('restart-contact',ranks)
            if ranks>1:
                report['checks'][f'observed-clump-transport-gpu{ranks}']=audit_transport(campaign.work/f'restart-contact-gpu{ranks}-full')
        campaign.report['pass']=all(c['pass'] for c in campaign.report['checks'].values())
        report['checks']['clump_ownership_restart']=campaign.report
        if a.baseline:
            # Migration packs only active histories even in the broken solver,
            # so it can postpone exhaustion. The negative control must retain
            # one-GPU ownership throughout all encounters.
            for ranks in [1]:
                negative=a.work/f'negative-baseline-churn-gpu{ranks}';prepare(negative,ranks)
                launch=base.launch(a.baseline,negative,a.timeout);report['launches'].append(launch)
                assert launch['returncode']==0 and not launch['timeout'], launch
                try:
                    audit(negative)
                except AssertionError as failure:
                    reason=str(failure)
                    assert 'missing reciprocal' in reason or '>16' in reason, ('unexpected baseline failure',reason)
                    report['checks'][f'negative-baseline-churn-gpu{ranks}']={'pass':True,'expected_history_failure':reason}
                else:
                    raise AssertionError('frozen baseline unexpectedly passed churn history audit')
            for ranks in a.ranks:
                current=a.work/f'matched-current-gpu{ranks}';old=a.work/f'matched-baseline-gpu{ranks}'
                for directory,binary in ((current,a.binary),(old,a.baseline)):
                    ext.prepare(directory,'restart-contact',ranks,40);launch=base.launch(binary,directory,a.timeout);report['launches'].append(launch);assert launch['returncode']==0 and not launch['timeout']
                check=ext.compare_records(ext.read_record(ext.latest_checkpoint(old)),ext.read_record(ext.latest_checkpoint(current)));assert check['pass'];report['checks'][f'matched-fresh-gpu{ranks}']=check
        report['pass']=True
    finally: base.dump(a.work/'report.json',report)
    return 0
if __name__=='__main__':raise SystemExit(main())
