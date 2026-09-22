"""Correct metadata of untrained derived controls; never modify source arrays."""
import json
from pathlib import Path
import numpy as np
from derive_cache import sha

root=Path(__file__).resolve().parent
for path in (root/'controls').glob('goal_*/cache_stride8/manifest.json'):
    m=json.loads(path.read_text())
    if 'parent_diagnostics' in m:continue
    m['parent_diagnostics']=m.pop('diagnostics',{})
    m['diagnostics']={}
    for split in ('train','val','test'):
        data_path=path.parent/m['files'][split]
        assert sha(data_path)==m['file_sha256'][split]
        with np.load(data_path,allow_pickle=False) as f:
            rms=f['midpoint10_rmse']
            assert len(rms)==m['counts'][split]
            m['diagnostics'][split]={'paired_expert_midpoint10_executed_rmse_mean':float(rms.mean()),
                'paired_expert_midpoint10_executed_rmse_p95':float(np.quantile(rms,.95))}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(path)
    print(f'Corrected thinned-control diagnostics: {path}')
