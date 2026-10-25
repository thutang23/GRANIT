#!/usr/bin/env python3
"""Independent Newtonian pair reference for GRANIT's fixed D3 stencil.
No CUDA, numpy, production G matrices or production basis functions are used.
The generated stencil header is READ ONLY to recover the exact accepted cells.
A pair accepted at cells (ct,cs) is represented by the bivariate Taylor series
  1/|cs-ct + lambda*(source-cs) - mu*(target-ct)|
through degree four in lambda AND degree four in mu. Differentiate in the
physical target displacement, set lambda=mu=1, multiply by G*mi*mj.
This is the Cartesian full-rank form of p4 source-to-p4 local translation.
It tests the mathematical operator, not its stored G-table coefficients.
"""
import argparse, collections, json, math, pathlib, re, struct
from decimal import Decimal, localcontext

REPO=pathlib.Path(__file__).resolve().parents[4]
def read_tables():
    text=(REPO/'src/core/acceleration/body_fitted_fmm/grouped_stencil_tables.cuh').read_text()
    def section(name):
        start=text.index('HOST_GROUPED_D3_'+name)
        return text[start:text.index('\n};',start)]
    cross=re.findall(r'\{\{(-?\d+), (-?\d+), (-?\d+)\}, (0x[0-9a-f]+)\}',section('CROSS'))
    assert len(cross)==8*16
    crosses=[{tuple(map(int,e[:3])):int(e[3],16) for e in cross[q*16:(q+1)*16]} for q in range(8)]
    blocks=[{} for _ in range(8)]
    for name,n in [('ALL_FAR',6),('MIXED',11)]:
        entries=re.findall(r'\{\{(-?\d+), (-?\d+), (-?\d+)\}, (0x[0-9a-f]+), (0x[0-9a-f]+), \{',section(name))
        assert len(entries)==8*n
        for q in range(8):
            for e in entries[q*n:(q+1)*n]:
                blocks[q][tuple(map(int,e[:3]))]=(int(e[3],16),int(e[4],16))
    return crosses,blocks
CROSS,BLOCKS=read_tables()
def cell(pos, level, root_min, root_size):
    h=root_size/(1<<level)
    return tuple(max(0,min((1<<level)-1,math.floor((x-mn)/h))) for x,mn in zip(pos,root_min))
def center(c,level,root_min,root_size):
    h=root_size/(1<<level)
    return tuple(mn+(v+.5)*h for mn,v in zip(root_min,c))
def qchild(c):return ((c[0]&1)<<2)|((c[1]&1)<<1)|(c[2]&1)
def route(t,s,depth,root_min,root_size):
    hits=[]
    for level in range(2,depth+1):
        tc=cell(t,level,root_min,root_size);sc=cell(s,level,root_min,root_size)
        tp=tuple(x>>1 for x in tc);sp=tuple(x>>1 for x in sc)
        delta=tuple(b-a for a,b in zip(tp,sp));q=qchild(tc)
        if delta in CROSS[q]:
            hits.append(('cross',level,level-1,tc,sp,CROSS[q][delta]))
        if delta in BLOCKS[q]:
            far,near=BLOCKS[q][delta];bit=1<<qchild(sc)
            if far&bit:hits.append(('same',level,level,tc,sc,None))
            elif level==depth and near&bit:hits.append(('direct',level,level,tc,sc,None))
    assert len(hits)==1, ('pair partition error',t,s,hits)
    kind,tl,sl,tc,sc,op=hits[0]
    return dict(kind=kind,target_level=tl,source_level=sl,target_cell=tc,source_cell=sc,
                target_center=center(tc,tl,root_min,root_size),source_center=center(sc,sl,root_min,root_size))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def norm(x):return math.sqrt(dot(x,x))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def scale(a,s):return tuple(x*s for x in a)
def pair_taylor(R,s,t,order=4,source_order=None,target_order=None):
    """Coefficient pairs are (potential value, d/dtarget xyz), Python FP64."""
    source_order=order if source_order is None else source_order
    target_order=order if target_order is None else target_order
    maximum_degree=source_order+target_order
    D=dot(R,R)
    zero=(0.,0.,0.)
    q={(1,0):(2*dot(R,s)/D,zero),
       (0,1):(-2*dot(R,t)/D,scale(R,-2/D)),
       (2,0):(dot(s,s)/D,zero),
       (0,2):(dot(t,t)/D,scale(t,2/D)),
       (1,1):(-2*dot(s,t)/D,scale(s,-2/D))}
    power={(0,0):(1.,zero)};gradient=zero;binomial=1.
    for n in range(maximum_degree+1):
        gradient=add(gradient,scale(tuple(sum(v[1][k] for v in power.values()) for k in range(3)),binomial))
        if n==maximum_degree:break
        product={}
        for (a,b),(v,g) in power.items():
            for (c,d),(w,h) in q.items():
                if a+c>source_order or b+d>target_order:continue
                key=(a+c,b+d)
                pv,pg=product.get(key,(0.,zero))
                product[key]=(pv+v*w,add(pg,add(scale(g,w),scale(h,v))))
        power=product
        binomial*=(-.5-n)/(n+1)
    return scale(gradient,1/math.sqrt(D))
def direct(t,s,eps):
    r=sub(s,t);d=dot(r,r)+eps*eps
    return scale(r,1/(d*math.sqrt(d)))
def generated_fixture(depth=12,members=64):
    # Decimal generation avoids Python-double fmod drift from the C++ long-double fixture.
    with localcontext() as c:
        c.prec=50;D=Decimal
        rows=[dict(position=[-16.,-16.,-16.],mass=.0625,clump_id=0),dict(position=[16.,16.,16.],mass=.0625,clump_id=0)]
        constants=list(map(D,['.7548776662466927','.5698402909980532','.438579021']))
        for i in range(48):rows.append(dict(position=[float(((i+1)*v%1)*24-12) for v in constants],mass=.5+(i%13)*.0625,clump_id=0))
        h=math.ldexp(32.+2*(32.*.01),-depth);smallscale=.006*min(1.,h/1.02)
        offsets=[[-.78,-.19,.11],[.43,-.31,-.23],[.08,.57,.37],[.34,.12,-.46]]
        for off,mass in zip(offsets,[.625,1.375,.875,1.125]):rows.append(dict(position=[float(D(str(x))+D(str(y))*D(str(smallscale))) for x,y in zip([-.35,.22,.11],off)],mass=mass,clump_id=1))
        ctr=[-3+(v%1)*6 for v in map(D,['.6180339887498948','.414213562373095','.732050807568877'])]
        extent=max(2.2,h*.75);bits=[0,5,6,3,7,2,1,4]
        for i in range(members):
            u=[D(1 if bits[i]&(1<<k) else -1) for k in range(3)] if i<8 else [2*((i+1)*v%1)-1 for v in constants]
            rows.append(dict(position=[float(x+y*D(str(extent))) for x,y in zip(ctr,u)],mass=.5+((i*7)%17)*.0625,clump_id=2))
        return dict(root_min=[-16.32]*3,root_size=32.64,depth=depth,epsilon=float(1e-3*1e-3),rows=rows)
def analyze(data,target_index):
    rows=data['rows'];a=rows[target_index];t=a['position'];tid=a.get('clump_id',0)
    approx=(0.,0.,0.);exact=(0.,0.,0.);terms=[];counts=collections.Counter()
    node_counts=collections.Counter((level,cell(row['position'],level,data['root_min'],data['root_size'])) for level in range(1,data['depth']+1) for row in rows)
    for j,b in enumerate(rows):
        if j==target_index or (tid>0 and tid==b.get('clump_id',0)):continue
        s=b['position'];r=route(t,s,data['depth'],data['root_min'],data['root_size'])
        if r['kind']!='direct':
            population_work=node_counts[(r['target_level'],r['target_cell'])]*node_counts[(r['source_level'],r['source_cell'])]
            threshold=data.get('direct_threshold_leaf' if r['target_level']==data['depth'] else 'direct_threshold_internal',0)
            if threshold and population_work<=threshold:r['kind']='direct_cutoff'
        counts[(r['kind'],r['target_level'],r['source_level'])]+=1
        f=direct(t,s,data.get('epsilon',0.));mass=a['mass']*b['mass']*data.get('G',1.)
        if r['kind'].startswith('direct'):p=f
        else:p=pair_taylor(sub(r['source_center'],r['target_center']),sub(s,r['source_center']),sub(t,r['target_center']))
        p=scale(p,mass);f=scale(f,mass);error=sub(p,f);approx=add(approx,p);exact=add(exact,f)
        terms.append(dict(source=j,source_clump=b.get('clump_id',0),**r,force=p,direct_force=f,error=error,error_norm=norm(error),target_offset_ratio=norm(sub(t,r['target_center']))/norm(sub(r['source_center'],r['target_center'])) if not r['kind'].startswith('direct') else None,source_offset_ratio=norm(sub(s,r['source_center']))/norm(sub(r['source_center'],r['target_center'])) if not r['kind'].startswith('direct') else None))
    output=dict(target=target_index,position=t,clump_id=tid,analytic_force=approx,direct_force=exact,error=sub(approx,exact),relative_error=norm(sub(approx,exact))/max(norm(exact),1e-28),direct_force_sum_norm=sum(norm(e['direct_force']) for e in terms),route_counts=[dict(kind=k[0],target_level=k[1],source_level=k[2],pairs=v) for k,v in counts.items()],largest_pair_errors=sorted(terms,key=lambda e:e['error_norm'],reverse=True)[:12])
    if 'actual_force' in a:output.update(actual_force=a['actual_force'],analytic_minus_actual_relative=norm(sub(approx,a['actual_force']))/max(norm(exact),1e-28))
    if 'reference_force' in a:output.update(reference_force=a['reference_force'],direct_minus_dump_relative=norm(sub(exact,a['reference_force']))/max(norm(exact),1e-28))
    return output

def load_dump(path):
    contents=pathlib.Path(path).read_text()
    try:data=json.loads(contents)
    except json.JSONDecodeError:
        records=[json.loads(line) for line in contents.splitlines() if line.strip()]
        data=next(row for row in records if row.get('method')=='bffmm' and 'force_diagnostics' in row)
    if 'rows' in data:return data
    if data.get('aligned_inner',{}).get('used') and (data.get('direct_threshold_leaf',0) or data.get('direct_threshold_internal',0)):
        raise ValueError('Aligned nonzero-cutoff replay requires component target occupancies; use the forced-0/0 aligned diagnostic or an ordinary all-small tree.')
    diag=data['force_diagnostics'];root=diag.get('root',data.get('aligned_inner',{}))
    f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
    eps=f32(f32(data['member_radius'])*f32(data['softening_factor'])) if 'member_radius' in data and 'softening_factor' in data else data.get('epsilon',1e-6)
    result=dict(rows=diag['members'],root_min=root.get('root_min',root.get('domain_min')),root_size=root['root_size'],depth=root.get('depth',root.get('evaluation_depth')),epsilon=eps,G=data.get('oracle_gravitational_constant',1.),direct_threshold_leaf=data.get('direct_threshold_leaf',0),direct_threshold_internal=data.get('direct_threshold_internal',0))
    # Dump index is canonical; require a complete live-prefix row sequence.
    assert all(row.get('particle_index',i)==i for i,row in enumerate(result['rows']))
    for row in result['rows']:
        if row.get('actual_force') is None:row.pop('actual_force',None)
        if row.get('reference_force') is None:row.pop('reference_force',None)
    return result

def selftest():
    # Source offset zero: direct gradient of the quartic local potential.
    R=(2.,1.,0.);t=(.49,.245,.013)
    a=dot(R,t);b=dot(t,t);D=dot(R,R)
    f=add(scale(R,D**-1.5),scale(sub(scale(R,3*a),scale(t,D)),D**-2.5))
    f=add(f,scale(sub(scale(R,15*a*a-3*D*b),scale(t,6*D*a)),.5*D**-3.5))
    f=add(f,scale(add(scale(R,140*a*a*a-60*D*a*b),scale(t,-60*D*a*a+12*D*D*b)),.125*D**-4.5))
    assert norm(sub(f,pair_taylor(R,(0.,0.,0.),t)))<1e-14
    # Centered target and source: exact unsoftened point force.
    assert norm(sub(pair_taylor(R,(0.,0.,0.),(0.,0.,0.)),direct((0.,0.,0.),R,0.)))<1e-15

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input');ap.add_argument('--target',type=int,default=76);ap.add_argument('--depth',type=int,default=12);ap.add_argument('--output');args=ap.parse_args()
    selftest();data=load_dump(args.input) if args.input else generated_fixture(args.depth)
    result=analyze(data,args.target);text=json.dumps(result,indent=2)
    if args.output:pathlib.Path(args.output).write_text(text+'\n')
    print(text)

# Optional import-only high-precision scalar-potential validation. This follows
# the same defining series but uses no dual-number/analytic gradient code.
def pair_taylor_decimal_gradient(R,s,t,step='1e-20'):
    with localcontext() as ctx:
        ctx.prec=70
        D=Decimal
        R=tuple(D.from_float(float(x)) for x in R);s=tuple(D.from_float(float(x)) for x in s);t=tuple(D.from_float(float(x)) for x in t)
        h=D(step);radius2=dot(R,R)
        def potential(target):
            q={(1,0):2*dot(R,s)/radius2,(0,1):-2*dot(R,target)/radius2,(2,0):dot(s,s)/radius2,(0,2):dot(target,target)/radius2,(1,1):-2*dot(s,target)/radius2}
            power={(0,0):D(1)};total=D(0);binomial=D(1)
            for n in range(9):
                total+=binomial*sum(power.values())
                if n==8:break
                product={}
                for (a,b),v in power.items():
                    for (c,d),w in q.items():
                        if a+c<=4 and b+d<=4:
                            key=(a+c,b+d);product[key]=product.get(key,D(0))+v*w
                power=product;binomial*=(-D('.5')-n)/(n+1)
            return total/radius2.sqrt()
        gradient=[]
        for axis in range(3):
            plus=list(t);minus=list(t);plus[axis]+=h;minus[axis]-=h
            gradient.append(float((potential(plus)-potential(minus))/(2*h)))
        return tuple(gradient)
