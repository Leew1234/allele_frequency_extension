"""Check results against the pre-registered criteria (criteria.json).

  python evaluate_plan.py --criteria criteria.json --results WORK/results \\
      --report WORK/results/criteria_report.md

Each criterion compares a value read from a result file with a constant or
with another value, for every combination in its "over" lists. It passes only
if every combination passes, except dimensions listed in "require", where at
least that many values must pass for each combination of the other
dimensions. A missing file or value counts as MISSING (never as a pass).
A decision holds only if all of its criteria pass.

Freezing: criteria.sha256 (next to criteria.json) records the file's SHA-256
when the plan was frozen. The report states whether the criteria file still
matches it; a mismatch means the criteria were edited after freezing and must
be reported as a deviation. Re-freeze only before any real-data result
exists: `python evaluate_plan.py --freeze`.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

OPS = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b, '>': lambda a, b: a > b,
       '>=': lambda a, b: a >= b, 'between': lambda a, b: b[0] <= a <= b[1]}


def _fill(x, combo):
    if isinstance(x, str):
        for k, v in combo.items():
            x = x.replace('{' + k + '}', str(v))
    return x


def read_value(ref, combo, results_dir, cache):
    """A number from a file path + key path, or a literal number/list."""
    if not isinstance(ref, dict):
        return ref
    path = Path(results_dir) / _fill(ref['file'], combo)
    if path not in cache:
        cache[path] = json.loads(path.read_text()) if path.exists() else None
    node = cache[path]
    if node is None:
        raise LookupError(f'missing file {path.name}')
    for key in ref['path']:
        key = _fill(key, combo)
        try:
            node = node[int(key)] if isinstance(node, list) else node[key]
        except (KeyError, IndexError, ValueError, TypeError):
            raise LookupError(f'missing {"/".join(str(_fill(k, combo)) for k in ref["path"])} in {path.name}')
    if node is None:
        raise LookupError(f'null value in {path.name}')
    return node


def check(criterion, results_dir, cache=None):
    cache = {} if cache is None else cache
    over = criterion.get('over', {})
    dims = list(over)
    require = criterion.get('require', {})
    rows = []
    for values in itertools.product(*(over[d] for d in dims)) if dims else [()]:
        combo = dict(zip(dims, values))
        try:
            left = read_value(criterion['left'], combo, results_dir, cache)
            right = read_value(criterion['right'], combo, results_dir, cache)
            ok = bool(OPS[criterion['op']](left, right))
            rows.append({'combo': combo, 'left': left, 'right': right, 'status': 'PASS' if ok else 'FAIL'})
        except LookupError as err:
            rows.append({'combo': combo, 'left': None, 'right': None, 'status': 'MISSING', 'why': str(err)})
    if any(r['status'] == 'MISSING' for r in rows):
        status = 'MISSING'
    elif not require:
        status = 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL'
    else:
        status = 'PASS'
        rest = [d for d in dims if d not in require]
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r['combo'][d] for d in rest), []).append(r)
        for dim, k in require.items():
            for g in groups.values():
                if sum(r['status'] == 'PASS' for r in g) < k:
                    status = 'FAIL'
    return status, rows


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.4g}'
    if isinstance(v, list):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    return str(v)


def evaluate(criteria_doc, results_dir):
    cache = {}
    out = {c['id']: dict(zip(('status', 'rows'), check(c, results_dir, cache))) | {'criterion': c}
           for c in criteria_doc['criteria']}
    decisions = {}
    for name, ids in criteria_doc.get('decisions', {}).items():
        sts = [out[i]['status'] for i in ids]
        decisions[name] = ('YES' if all(s == 'PASS' for s in sts)
                           else 'INCOMPLETE' if any(s == 'MISSING' for s in sts) else 'NO')
    return out, decisions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_status(criteria_path):
    """(current hash, frozen hash or None, matches)"""
    cur = sha256(criteria_path)
    ref = Path(criteria_path).with_suffix('.sha256')
    frozen = ref.read_text().split()[0] if ref.exists() else None
    return cur, frozen, frozen == cur


def report(criteria_doc, out, decisions, freeze=None):
    lines = [f"# Pre-registered criteria check (criteria v{criteria_doc.get('version', '?')}, "
             f"frozen {criteria_doc.get('frozen_on', '?')})", '']
    if freeze is not None:
        cur, frozen, ok = freeze
        lines += [f"Criteria file SHA-256 `{cur}`: "
                  + ('**matches the frozen copy**.' if ok else
                     f"**DOES NOT match the frozen copy** (`{frozen}`); report this as a deviation."
                     if frozen else '**no frozen hash found** (criteria.sha256 missing).'), '']
    lines += ['## Decisions', '', '| Decision | Result | Criteria |', '| --- | --- | --- |']
    for name, res in decisions.items():
        ids = criteria_doc['decisions'][name]
        parts = ', '.join(i + ' ' + out[i]['status'] for i in ids)
        lines.append(f'| {name} | **{res}** | {parts} |')
    lines += ['', '## Criteria', '', '| ID | Group | Claim | Result | Values |', '| --- | --- | --- | --- | --- |']
    for cid, o in out.items():
        c = o['criterion']
        vals = '; '.join(
            (' '.join(f'{k}={v}' for k, v in r['combo'].items()) + ': ' if r['combo'] else '')
            + (f"{_fmt(r['left'])} {c['op']} {_fmt(r['right'])} {r['status']}" if r['status'] != 'MISSING'
               else f"MISSING ({r['why']})") for r in o['rows'])
        lines.append(f"| {cid} | {c['group']} | {c['claim']} | **{o['status']}** | {vals} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--criteria', default=str(Path(__file__).with_name('criteria.json')))
    p.add_argument('--results', help='results directory written by run_plan.sh')
    p.add_argument('--freeze', action='store_true',
                   help='write criteria.sha256 for the current criteria file (only before real results exist)')
    p.add_argument('--report', help='write a Markdown report here')
    p.add_argument('--json', help='write machine-readable results here')
    args = p.parse_args()
    if args.freeze:
        h = sha256(args.criteria)
        Path(args.criteria).with_suffix('.sha256').write_text(f'{h}  {Path(args.criteria).name}\n')
        print(f'Frozen {args.criteria}: {h}')
        return
    if not args.results:
        p.error('--results is required')
    doc = json.loads(Path(args.criteria).read_text())
    out, decisions = evaluate(doc, args.results)
    fz = freeze_status(args.criteria)
    text = report(doc, out, decisions, fz)
    if args.report:
        Path(args.report).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'criteria_sha256': fz[0], 'matches_frozen': fz[2], 'decisions': decisions,
             'criteria': {k: {'status': v['status'], 'rows': v['rows']} for k, v in out.items()}},
            indent=2, default=str))
    print(text)


if __name__ == '__main__':
    main()
