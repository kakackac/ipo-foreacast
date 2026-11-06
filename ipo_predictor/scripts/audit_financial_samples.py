"""Compare cached official sections with manually reviewed annual table cells."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pipelines.disclosure_financials import parse_disclosure_financials
from data.pipelines.financial_history import atomic_json, summarize_asof


# Amounts below were read from the original summary tables, not parser output.
SAMPLES = [
    ('엔에스쇼핑', '00406392', '20150312000834', '2015-03-26 21:00', 2013,
     [347135000000, 69534000000, 336642000000, 109236000000, 227406000000, 54619000000]),
    ('파멥신', '00972293', '20181108000408', '2018-11-20 21:00', 2017,
     [208000000, -3835000000, 7526000000, 5246000000, 2281000000, -6483000000]),
    ('바이오플러스', '00636656', '20210910000471', '2021-09-26 21:00', 2020,
     [24366600859, 12327486055, 37935325802, 9415034892, 28520290910, 9703010829]),
    ('코셈', '01493793', '20240206000494', '2024-02-22 21:00', 2022,
     [12525000000, 1780000000, 14078000000, 3363000000, 10715000000, 1716000000]),
    ('스카이랩스', '01586923', '20260825000414', '2026-09-03 21:00', 2025,
     [7935865376, -14725265236, 30180942922, 4728384269, 25452558653, -62995166524]),
    ('싸이맥스', '00659161', '20150602000317', '2015-06-16 21:00', 2014,
     [54714000000, 5836000000, 39268000000, 12976000000, 26292000000, 5993000000]),
    ('SG', '00963976', '20180122000271', '2018-01-25 21:00', 2016,
     [93642364279, 11464092002, 83188236557, 58375494119, 24812742438, 9004249362]),
    ('휴젤', '00888347', '20151214000155', '2015-12-23 21:00', 2014,
     [40350000000, 15488000000, 97250000000, 54576000000, 42674000000, 13150000000]),
    ('서진시스템', '00838005', '20170315000227', '2017-03-26 21:00', 2016,
     [165871000000, 24391000000, 227323000000, 154819000000, 72504000000, 19627000000]),
]


def main():
    cache = Path(__file__).resolve().parents[1] / 'data/raw/dart_ipo_financial_sections'
    failed = False
    results = []
    for issuer, corp, receipt, cutoff, year, expected in SAMPLES:
        path = cache / f'{receipt}.json'
        if not path.exists():
            print(json.dumps({'issuer': issuer, 'receipt': receipt, 'status': 'source_cache_missing'}, ensure_ascii=False))
            failed = True
            continue
        payload = json.loads(path.read_text())
        digest = hashlib.sha256(payload['html'].encode()).hexdigest()
        frame = parse_disclosure_financials(payload['html'], corp, receipt,
                                            payload.get('official_publication_date') or f'{receipt[:4]}-{receipt[4:6]}-{receipt[6:8]}', payload['source_url'])
        result = summarize_asof(frame, corp, cutoff)
        fields = ['revenue', 'operating_income', 'total_assets', 'total_liabilities', 'equity', 'net_income']
        mismatches = {key: {'expected': value, 'actual': result.get(key)}
                      for key, value in zip(fields, expected) if result.get(key) != value}
        if result.get('financial_as_of_year') != year:
            mismatches['financial_as_of_year'] = {'expected': year, 'actual': result.get('financial_as_of_year')}
        if digest != payload.get('sha256') or payload.get('issuer') != issuer:
            mismatches['source_integrity'] = False
        failed |= bool(mismatches)
        record = {'issuer': issuer, 'receipt': receipt, 'source_url': payload['source_url'],
                          'sha256': digest, 'status': 'mismatch' if mismatches else 'sample_passed',
                          'mismatches': mismatches}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    atomic_json(cache.parent.parent / 'processed/financial_sample_audit.json',
                {'sample_count': len(SAMPLES), 'passed': len(results) == len(SAMPLES) and not failed,
                 'samples': results, 'training_executed': False})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
