"""Portable CPU reconstruction of the exported ESM representation contract."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class HR:
    def __init__(self, name='frozen', *, fold, root=ROOT):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        entry = manifest['entries'][name]
        if fold not in (0, 1) or (entry['fold'] is not None and entry['fold'] != fold):
            raise ValueError('Use the matching adapter fold (0 or 1)')
        path = self.root / entry['cache']
        if sha256(path) != manifest['files'][entry['cache']]['sha256']:
            raise ValueError('Cache checksum mismatch')
        self.p = torch.load(path, map_location='cpu', weights_only=True)
        self.roles = json.loads((self.root / f'fold{fold}_roles.json').read_text())
        self.metadata = {m['var_id']: m for m in json.loads((self.root / 'metadata.json').read_text())}
        self.lookup = {v: i for i, v in enumerate(self.p['var_id'])}
        if set(self.lookup) != set(self.roles):
            raise ValueError('Cohort mismatch')
        self.layers = self.p['layers']
        self.slot_kind_vocab = self.p['slot_kind_vocab']
        self.provenance = self.p['provenance']

    def ids(self, role):
        if role not in ('train', 'val', 'test'):
            raise ValueError('Unknown role')
        return sorted(v for v, r in self.roles.items() if r == role)

    def batch(self, ids):
        ids = list(ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Provide nonempty unique variant IDs')
        idx = torch.tensor([self.lookup[v] for v in ids], dtype=torch.long)
        p = self.p
        out = {k: p[k].index_select(0, idx) for k in
               ('wt_pos', 'mut_pos', 'wt_present', 'mut_present', 'delta_valid', 'token_valid', 'slot_kind')}
        pos = (out['wt_pos'] - 1).clamp(min=0)
        wt = p['wt_full'][:, pos, :].permute(1, 0, 2, 3).contiguous()
        wt = wt.masked_fill(~out['wt_present'][:, None, :, None], 0)
        mut = p['mut_windows'].index_select(0, p['row_to_mut_window'].index_select(0, idx))
        delta = (mut.float() - wt.float()).masked_fill(~out['delta_valid'][:, None, :, None], 0)
        out.update(H_WT=wt, H_MUT=mut, delta_H=delta, var_id=ids,
                   split=[self.roles[v] for v in ids], layers=self.layers,
                   slot_kind_vocab=self.slot_kind_vocab,
                   edit_metadata=[dict(self.metadata[v], split=self.roles[v]) for v in ids])
        return out

if __name__ == '__main__':
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        if sha256(ROOT / name) != record['sha256']:
            raise SystemExit('Checksum mismatch: ' + name)
    print('All manifest checksums verified.')
