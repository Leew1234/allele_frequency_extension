"""Score every trained configuration on the held-out test chromosome, inside
the functional blocks and inside the neutral remainder, next to the oracle."""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from score import category_ratios, load_class_probabilities  # noqa: E402

CONFIGS = [('none', 'whole'), ('none', 'neutral'), ('common', 'whole'),
           ('common', 'neutral'), ('all', 'whole'), ('all', 'neutral')]


def fmt(r, key):
    ci = r.get('ci95', {}).get(key)
    return f"{r[key]:.2f}" + (f" ({ci[0]:.2f}-{ci[1]:.2f})" if ci else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True)
    p.add_argument('--test', default='chrE')
    args = p.parse_args()
    w = Path(args.work)
    y = np.load(w / f'arr/{args.test}_frequency_labels.npy')
    func = np.load(w / f'masks/{args.test}_functional.npy')
    rows = [('oracle (true neutral rates)', np.load(w / 'oracle.npz')[args.test])]
    rows += [(f'{scope} training, {cal} calibration',
              load_class_probabilities(w / f'pred/{scope}_{cal}_{args.test}.npy')) for scope, cal in CONFIGS]
    print(f'Test chromosome {args.test}; 95% block-bootstrap intervals (20 kb blocks)\n')
    print(f"{'configuration':36s} {'functional oe_any':>20s} {'functional oe_common_share':>27s} {'neutral oe_any':>15s}")
    for name, pred in rows:
        f = category_ratios(y, pred, func, block_size=20_000, n_boot=300)
        n = category_ratios(y, pred, ~func, block_size=20_000, n_boot=0)
        print(f"{name:36s} {fmt(f, 'oe_any'):>20s} {fmt(f, 'oe_common_share'):>27s} {n['oe_any']:>15.2f}")


if __name__ == '__main__':
    main()
