"""Write a diagnostics config (smoke/overfit) next to config.ext.json.

Diagnostics go to a separate, timestamped folder so they never mix with
scientific runs and never collide with earlier smoke directories.
"""
import json
import sys
from pathlib import Path

src, dst, stamp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
settings = json.loads(src.read_text(encoding='utf-8'))
settings['output_dir'] = settings['output_dir'].rstrip('/\\') + '_diagnostics'
settings['cohort'] = f'diagnostics_{stamp}'
dst.write_text(json.dumps(settings, indent=2), encoding='utf-8')
print('diagnostics ->', Path(settings['output_dir']) / settings['cohort'])
