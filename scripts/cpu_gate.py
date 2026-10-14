"""Quick CPU checks on tiny tensors: imports, shapes and known values.
Exits non-zero on the first failure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.manual_seed(0)
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src import metrics_v2 as m2
from src.seed import set_seed
from src.model import InstrumentedGRIT
from src.graphgps.model import InstrumentedGPS
from src.diagnostics_v2 import run_diagnostics_v2, aggregate_v2, compute_stability

set_seed(0)
FAILS = []


def check(name, cond):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        FAILS.append(name)


# ---------------- metrics_v2 known-value tests ----------------
H, N = 2, 6
uni = torch.full((H, N, N), 1.0 / N)
s = m2.compute_sink_metrics(uni)
check('uniform max_sink_score = 1/N', abs(s['max_sink_score'] - 1.0 / N) < 1e-6)
check('uniform norm_sink_score = 1', abs(s['norm_sink_score'] - 1.0) < 1e-5)
check('uniform sink_rate(0.3) = 0', s['sink_rate_eps0.3'] == 0.0)
sp = m2.compute_attention_spectra(uni)
check('uniform SLEM(A) ~ 0 (rank-1)', sp['slem_A'] < 1e-4)
check('uniform SLEM(lazy) ~ 0.5', abs(sp['slem_lazy'] - 0.5) < 1e-3)

sink = torch.zeros(H, N, N)
sink[:, :, 0] = 1.0
s2 = m2.compute_sink_metrics(sink)
check('point-sink score = 1 at node 0', s2['max_sink_score'] == 1.0 and s2['max_sink_node'] == 0)
check('point-sink rate(0.8) = 1', s2['sink_rate_eps0.8'] == 1.0)
check('point-sink headrate(0.8) = 1', s2['headrate_eps0.8'] == 1.0)

Hf = torch.tensor([[0., 0.], [1., 0.], [0., 0.]])
ei = torch.tensor([[0, 1], [1, 0]])
d = m2.compute_dirichlet_v2(Hf, ei)
check('dirichlet raw = 2 on unit edge (both directions)', abs(d['dirichlet_raw'] - 2.0) < 1e-6)
check('dirichlet per-edge = 1', abs(d['dirichlet_per_edge'] - 1.0) < 1e-6)
check('dirichlet rayleigh = 2', abs(d['dirichlet_rayleigh'] - 2.0) < 1e-5)

ns = m2.compute_norm_stats_v2(torch.tensor([[3., 4.], [0.3, 0.4], [0.3, 0.4]]))
check('norm ratio unsq', abs(ns['norm_ratio_unsq'] - 5.0 / 2.0) < 1e-4)
check('norm ratio squared = unsq^2', abs(ns['max_to_mean_ratio'] - (5.0 / 2.0) ** 2) < 1e-3)

# logit growth: logits |l_ij| proportional to norm_j ** 2 -> slope ~ 2
norms = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
lg = (norms ** 2).repeat(H, N, 1)
g = m2.compute_logit_growth(norms, torch.ones(N) / N, logits=lg)
check('logit slope recovers exponent 2', abs(g['logit_norm_slope'] - 2.0) < 1e-3)


# ---------------- synthetic graph builders ----------------
def mk_data(nn_, K, categorical=True, rwse_dim=8):
    ei_ = torch.stack([torch.arange(nn_ - 1), torch.arange(1, nn_)])
    ei_ = torch.cat([ei_, ei_.flip(0)], 1)
    if categorical:
        x = torch.randint(0, 28, (nn_,))
        ea = torch.randint(0, 4, (ei_.size(1),))
    else:
        x = torch.randn(nn_, 9)
        ea = torch.randn(ei_.size(1), 3)
    dd = Data(x=x, edge_index=ei_, edge_attr=ea, y=torch.randn(1))
    dd.rrwp_node = torch.randn(nn_, K)
    dd.rrwp_edge = torch.randn(nn_ * nn_, K)
    dd.random_walk_pe = torch.randn(nn_, rwse_dim)
    return dd


def mk_batch(K=21):
    return Batch.from_data_list([mk_data(5 + i, K) for i in range(3)])


INFO = {'task': 'regression', 'metric': 'mae', 'num_classes': 1, 'num_node_types': 28,
        'num_edge_types': 4, 'input_type': 'categorical'}
MODEL = dict(hidden_dim=32, num_heads=4, attn_dropout=0.0, dropout=0.0, pool='sum')


# ---------------- GRIT: all attn modes forward + stashes ----------------
for mode in ('grit', 'dot_bias', 'dot'):
    cfg = {'model': dict(MODEL, attn_mode=mode),
           'architecture': {'num_layers': 2, 'task': 'regression'},
           'pe': {'type': 'rrwp', 'dim': 21}}
    m = InstrumentedGRIT(cfg, INFO)
    m.eval()
    with torch.no_grad():
        out = m(mk_batch(), collect_diagnostics=True)
    ok = (out.shape == (3,) and len(m.attn_weights) == 2
          and m.attn_weights[0] is not None
          and m.attn_logits[0] is not None
          and m.value_norms[0] is not None)
    check(f'GRIT[{mode}] forward + stashes', ok)
    aw = m.attn_weights[0]
    rows = aw[0, :, :5, :5].sum(-1)
    check(f'GRIT[{mode}] attention row-stochastic', torch.allclose(rows, torch.ones_like(rows), atol=1e-4))

# grit mode must keep the parameter names of the shipped checkpoints
m = InstrumentedGRIT({'model': dict(MODEL), 'architecture': {'num_layers': 1, 'task': 'regression'},
                      'pe': {'type': 'rrwp', 'dim': 21}}, INFO)
ks = set(m.state_dict().keys())
need = ['layers.0.mha.W_E.weight', 'layers.0.mha.W_VeRow.weight', 'layers.0.mha.W_A.weight',
        'layers.0.mha.W_Eo.weight', 'layers.0.mha.W_Q.weight', 'layers.0.mha.W_O.weight',
        'pe_node_enc.weight', 'pe_edge_enc.weight', 'layers.0.bn_edge.weight',
        'layers.0.deg_scaler_1', 'node_emb.weight', 'edge_emb.weight']
check('GRIT state-dict keys intact', all(k in ks for k in need))

# entropy forcing has grad
cfg = {'model': dict(MODEL), 'architecture': {'num_layers': 2, 'task': 'regression'},
       'pe': {'type': 'rrwp', 'dim': 21}}
m = InstrumentedGRIT(cfg, INFO)
for l in m.layers:
    l.mha.entropy_track = True
m.train()
_ = m(mk_batch())
ent = m.pop_attn_entropy()
check('entropy term exists + grad', ent is not None and ent.requires_grad)

# sink-bias path (vnode designated column)
cfg = {'model': dict(MODEL, sink_bias=True), 'architecture': {'num_layers': 1, 'task': 'regression'},
       'pe': {'type': 'rrwp', 'dim': 21}, 'vnode': {'enabled': True}}
m = InstrumentedGRIT(cfg, INFO)
with torch.no_grad():
    out = m(mk_batch(), collect_diagnostics=False)
check('sink-bias forward runs', out.shape == (3,) and m.sink_bias_param is not None)

# ---------------- GPS: mpnn variants + offline logits ----------------
for mpnn in ('gine', 'none', 'gatedgcn'):
    cfg = {'model': dict(MODEL), 'architecture': {'num_layers': 2, 'mpnn': mpnn, 'task': 'regression'},
           'pe': {'type': 'rwse', 'dim': 8}, 'vnode': {'enabled': False}}
    g = InstrumentedGPS(cfg, INFO)
    g.eval()
    g._register_attn_hooks()
    with torch.no_grad():
        out = g(mk_batch(), collect_diagnostics=True)
    ok = out.shape == (3,) and g.attn_weights[0] is not None
    check(f'GPS[{mpnn}] forward + attn capture', ok)
    x0 = g.layer_data[0]['h'][:5]
    logits, vnorm = g.compute_logits_values(0, x0)
    check(f'GPS[{mpnn}] offline logits shape', logits.shape == (4, 5, 5) and vnorm.shape == (4, 5))
    g._remove_attn_hooks()

# ---------------- diagnostics_v2 end-to-end on tiny GRIT ----------------
cfg = {'model': dict(MODEL), 'architecture': {'num_layers': 2, 'task': 'regression'},
       'pe': {'type': 'rrwp', 'dim': 21}}
m = InstrumentedGRIT(cfg, INFO)
loader = DataLoader([mk_data(5 + i, 21) for i in range(6)], batch_size=3, shuffle=False)
res = run_diagnostics_v2(m, loader, torch.device('cpu'), max_graphs=None, spectra=True)
agg, layers = aggregate_v2(res)
stab = compute_stability(res)
ok = (len(res[0]) == 6 and 'dirichlet_per_edge' in agg and 'slem_lazy' in agg
      and 'logit_norm_slope' in agg and 'sink_value_ratio' in agg
      and 'sink_identity_consistency_mean' in stab)
check('diagnostics_v2 end-to-end keys', ok)
# per-graph edge slicing: graphs have different sizes -> per-edge dirichlet must be finite/nonzero
vals = [r['dirichlet_per_edge'] for r in res[1]]
check('dirichlet finite per graph', all(v == v and v >= 0 for v in vals))

print('=' * 50)
if FAILS:
    print(f'CPU gate failed: {len(FAILS)} failures: {FAILS}')
    sys.exit(1)
print('CPU gate passed')
