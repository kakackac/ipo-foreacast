"""Compare overlapping official calendar observations without inferring withdrawal."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dart_subscription_calendar import replay


def compare(previous, current):
    previous, current = Path(previous), Path(current)
    reports = [json.loads((p / 'report.json').read_text()) for p in (previous, current)]
    windows = [{(r['year'], r['month']) for r in report['requests']} for report in reports]
    overlap = windows[0] & windows[1]
    if not overlap:
        raise ValueError('No common calendar months')
    groups = []
    for folder in (previous, current):
        grouped = {}
        for event in replay(folder, {'companies': []})['events']:
            if tuple(map(int, event['subscription_date'].split('-')[:2])) not in overlap:
                continue
            grouped.setdefault(event['corp_code'], set()).add(
                (event['rcept_no'], event['event_type'], event['subscription_date']))
        groups.append(grouped)
    before, after = groups
    rows = []
    for corp in sorted(before.keys() | after.keys()):
        old, new = before.get(corp, set()), after.get(corp, set())
        status = ('unchanged' if old == new else 'newly_observed' if not old
                  else 'no_longer_observed_requires_review' if not new else 'receipt_or_schedule_changed')
        rows.append({'corp_code': corp, 'status': status, 'previous': sorted(old), 'current': sorted(new)})
    return {'previous_snapshot': str(previous), 'current_snapshot': str(current),
            'common_months': sorted(overlap), 'rows': rows,
            'withdrawal_inferred': False, 'model_eligible': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.previous, args.current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    from collections import Counter
    print(json.dumps(dict(Counter(row['status'] for row in result['rows']))))
