#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -J i2a_verify
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/verify_fix_%j.out
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
python3 - <<'PY'
import pickle, numpy as np, glob
init=pickle.load(open(glob.glob('/scratch/users/terran/i2a_runs/crafter_done_ach3/ckpt/2026*/')[-1]+'agent.pkl','rb')); init=init.get('params',init)
d=sorted(glob.glob('/scratch/users/terran/i2a_runs/crafter_done_pilot/ckpt/2026*/'))[-1]; new=pickle.load(open(d+'agent.pkl','rb')); new=new.get('params',new)
ks=[k for k in new if k.startswith('stop_head/')]
ch={k.split('/',1)[1]: float(np.abs(np.asarray(new[k]).astype(np.float64)-np.asarray(init[k]).astype(np.float64)).mean()) for k in ks}
print('mean |delta| per stop_head param after the pilot:', {k: round(v,6) for k,v in ch.items()})
ok = ch['linear/bias']>0 and ch['mlp/linear0/kernel']>0; print('STOP HEAD TRAINS:', ok); open('/scratch/users/terran/i2a_stop/fix_ok','w').write('1' if ok else '0')
PY
if [ "$(cat /scratch/users/terran/i2a_stop/fix_ok)" = "1" ]; then rm -rf /scratch/users/terran/i2a_runs/crafter_done_ach4; J=$(LOGDIR=/scratch/users/terran/i2a_runs/crafter_done_ach4 sbatch --parsable --export=ALL train_done.sbatch); echo "300k with optimizer fix: $J"; fi
echo "== VERIFY DONE =="
