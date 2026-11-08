"""Compare cached official sections with manually reviewed annual table cells."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.pipelines.disclosure_financials import parse_disclosure_financials
from data.pipelines.financial_history import atomic_json, summarize_asof
from data.pipelines.disclosure_connections import viewer_identity


# Amounts below were read from original issuer statements, not parser output.
SAMPLES = [
    ('디앤씨미디어', '01186404', '20170721000127', '2017-07-31 21:00', 2016,
     [18893266114, 3693827393, 9632000000, 2864000000, 6769000000, 3011991038]),
    ('우진아이엔에스', '00143262', '20180904000072', '2018-09-13 21:00', 2017,
     [140446264438, 22811552380, 97690983885, 20686984883, 77003999002, 16845827865]),
    ('클리노믹스', '01201970', '20201120000295', '2020-12-03 21:00', 2019,
     [4101115062, -4492065268, 26935607774, 21885905205, 5049702569, -5760155545]),
    ('한켐', '01016309', '20241002000206', '2024-10-21 21:00', 2023,
     [26940818816, 5021083704, 30916487835, 10212029630, 20704458205, 4160972900]),
    ('명인제약', '00179489', '20250917000213', '2025-09-30 21:00', 2024,
     [269434788789, 92754381974, 581691991637, 49871470841, 531820520796, 68670651454]),
    ('달바글로벌', '01649204', '20250508000654', '2025-05-21 21:00', 2024,
     [309063000000, 59845000000, 136108000000, 34039000000, 102069000000, 15409000000]),
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

# Reviewed original annual cells: scope, year, revenue, operating income, assets.
# A None income pair explicitly checks that interim-only income stays absent.
BASIS_SAMPLES = [
    ('전진건설로봇', '00372129', '20240807000292',
     '48f65bca5496a43df10f469a9da642138d5a4a79108ed1fca8d115db44fe1f1b',
     [('CFS', 2023, 158370290546, 32904822481, 158316498662),
      ('OFS', 2023, 158079710420, 32920653826, 157583434174)], {2023}),
    ('두올', '00119672', '20160715000252',
     'c6f66ec26bed267290919d2cf759e36eb6c46862277eb37621757577f66224a9',
     [('CFS', 2015, 366288000000, 26056000000, 230283000000),
      ('CFS', 2014, 316832000000, 18397000000, 208984000000),
      ('OFS', 2015, 197562000000, 2553000000, 184920000000)], {2014, 2015}),
    ('앤디포스', '00942131', '20160929000463',
     'daa4ed23055912bc57747606e7feef7a1e3c7331296676c7e8ba150de2bf2d92',
     [('OFS', 2015, 57457561947, 16343803656, 34925025091)], {2015}),
    ('LG에너지솔루션', '01515323', '20220114000408',
     'dbdcd980a3eeeb9100fa42d6d631371cd97acb759c1179e2203a527165ce0913',
     [('CFS', 2020, None, None, 19941795000000)], {2020}),
    ('디티에스', '00534729', '20261012000323',
     'e83b38f7b396005341134dc3d4d31fdeb4fc77b1c20e9020484c011ee5d75e22',
     [('OFS', 2025, None, None, 167871407282)], {2025}),
    ('에치디프로', '00820389', '20150930000232',
     'e4d81fded831a7617feb383ab9fe0b47ddc466653fc17c1c906020b77aef4d37',
     [('OFS', 2014, 66063000000, 3738000000, 32094000000),
      ('OFS', 2013, 71254000000, 6976000000, 28041000000)], {2013, 2014}),
    ('오션브릿지', '01113754', '20161118000262',
     '24d107732915a37192ea9aae02d218ad23ff066cdd9dc91e2624c048096ed841',
     [('OFS', 2015, 25389000000, 3762000000, 13446000000)], {2015}),
    ('차백신연구소', '01141599', '20211008000552',
     'c1dc4c546c6a746c686223a449e0f5c070516cdf610475f1c2da10ff8f8d237a',
     [('OFS', 2020, 77796488, -4234422385, 25747443759),
      ('OFS', 2019, None, None, 9556199032)], {2019, 2020}),
    ('LG씨엔에스', '00139834', '20250117000578',
     'b864b37c67e790abe0e440d0a959ee4a7b87424e61a3a92123890e4327478657',
     [('CFS', 2023, 5605300168000, 464048383000, 4040679600000),
      ('OFS', 2023, 5112688400000, 373164112000, 3689410848000)], {2023}),
    ('미래에셋생명', '00112332', '20150626000234',
     '10b651330dd54a4fa8dddb91b0b1a2fb7e2b2fe90a04da3720baf79a5ba0ca6f',
     [('CFS', 2014, 3928349000000, 151667000000, 24681134000000),
      ('OFS', 2014, 3692466000000, 84574000000, 24080045000000)], {2014}),
    ('명인제약', '00179489', '20250917000213',
     'eae5a6093be90a853c9b6939fa1c5ca7bf17feaacc7015337f593dd99989c5e2',
     [('CFS', 2024, 269434788789, 92754381974, 581691991637),
      ('OFS', 2024, 269585959989, 90057625868, 572923047279)], {2024}),
    ('AJ네트웍스', '00365387', '20150807000011',
     '74403a517a9d1059ff282239e93b8683d39dea0b9f4d51ea0a8522348ec0f2b8',
     [('CFS', 2014, 1011150088928, 76977721440, 1437917070538),
      ('OFS', 2014, 194184554953, 26285715749, 484674125513)], {2013, 2014}),
    ('오에스피', '01491607', '20220930000631',
     'bb05f8f4ff8f6c071614cc0121e09894ea86021b6e79c28c99f5766b92b5da9f',
     [('OFS', 2021, 15694537506, 2803559295, 18495736461),
      ('OFS', 2020, 15566711787, 3934799221, 15780501749)], {2019, 2020, 2021}),
    ('웹케시', '00323868', '20190114000238',
     '3254ef6ee67ebf7ef461de08b2b89910ed9d11c278469c13914a773accceff1f',
     [('CFS', 2017, 77432986691, 3804075535, 35716017994),
      ('OFS', 2017, 77086240876, 4007904731, 36425027314)], {2016, 2017}),
    ('에이스토리', '00975290', '20190708000136',
     '65f7ea23554390dbbc06b38a55a2a1b223f488fce0e3632e8319299ee6b14b74',
     [('OFS', 2018, 46446059000, 1185323000, 19456504000)], {2016, 2017, 2018}),
    ('쿠콘', '01311055', '20210416000329',
     '03dacc698c5a8a6b2452ceee78902dd25b6334fbf8986c8063ebffbceff19e49',
     [('CFS', 2020, 51374145182, 11238744762, 141997802981),
      ('OFS', 2020, 49373922266, 11001158992, 141307999675)], {2019, 2020}),
    ('일진하이솔루스', '00972503', '20210823000386',
     '115d3e115b62bf714943cbb9374acf2484f1c247f890bc46cf68f23351ed2ec2',
     [('OFS', 2020, 113524940199, 15074663005, 81301533377)], {2018, 2019, 2020}),
    ('원방테크', '00304076', '20200911000117',
     '0bdb3d46d82708775ec3afa5f04f9beadcd8f7a201eca20ffbd3464c0810a6c8',
     [('CFS', 2018, 160275380000, 19116447000, 153392507000),
      ('OFS', 2017, 158171848000, 18939847000, 112783514000)], {2017, 2018, 2019}),
    ('에스바이오메딕스', '01258020', '20230420000520',
     'cee482e7b21c978b4684ede270d2e5077cb4c59bb2727fa27aafb40eded6349c',
     [('OFS', 2022, 361312000, -6536886000, 8855320000)], {2019, 2020, 2021, 2022}),
    ('삼영에스앤씨', '00405506', '20210510000418',
     '864cc88b76dc376c51f9b57692476b7261f65e3c2389f52fe95f586bc69e2bfd',
     [('OFS', 2020, 13557858486, 182225063, 18148583909)], {2019, 2020}),
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
                                            payload['official_publication_date'], payload['source_url'])
        result = summarize_asof(frame, corp, cutoff)
        fields = ['revenue', 'operating_income', 'total_assets', 'total_liabilities', 'equity', 'net_income']
        mismatches = {key: {'expected': value, 'actual': result.get(key)}
                      for key, value in zip(fields, expected) if result.get(key) != value}
        if result.get('financial_as_of_year') != year:
            mismatches['financial_as_of_year'] = {'expected': year, 'actual': result.get('financial_as_of_year')}
        index=json.loads((cache.parent/'dart_connection_sources'/
                          (hashlib.sha256(payload['index_url'].encode()).hexdigest()+'.json')).read_text())
        viewer_identity(index['text'],receipt,corp,payload['official_publication_date'])
        if (digest != payload.get('sha256') or payload.get('official_corp_code') != corp
                or hashlib.sha256(index['text'].encode()).hexdigest()!=index['sha256']
                or index['sha256']!=payload['index_sha256']):
            mismatches['source_integrity'] = False
        failed |= bool(mismatches)
        record = {'issuer': issuer, 'receipt': receipt, 'source_url': payload['source_url'],
                          'sha256': digest, 'status': 'mismatch' if mismatches else 'sample_passed',
                          'mismatches': mismatches}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    basis_results=[]
    for issuer,corp,receipt,expected_hash,expected,years in BASIS_SAMPLES:
        payload=json.loads((cache/(receipt+'.json')).read_text())
        body_hash=hashlib.sha256(payload['html'].encode()).hexdigest()
        index=json.loads((cache.parent/'dart_connection_sources'/
                          (hashlib.sha256(payload['index_url'].encode()).hexdigest()+'.json')).read_text())
        assert body_hash==payload['sha256']==expected_hash
        assert hashlib.sha256(index['text'].encode()).hexdigest()==index['sha256']==payload['index_sha256']
        viewer_identity(index['text'],receipt,corp,payload['official_publication_date'])
        assert payload['official_corp_code']==corp
        frame=parse_disclosure_financials(payload['html'],corp,receipt,
                                          payload['official_publication_date'],payload['source_url'])
        mismatches=[]
        if frame.empty or set(frame.year)!=years:mismatches.append('annual_years')
        cells=[]
        for scope,year,*values in expected:
            for name,value in zip(['revenue','operating_income','total_assets'],values):
                selected=frame[frame.fs_div.eq(scope)&frame.year.eq(year)&frame.account_name_en.eq(name)]
                actual=set(selected.amount)
                if actual!=({value} if value is not None else set()):
                    mismatches.append([scope,year,name,value,sorted(actual)])
                for row in selected.to_dict('records'):
                    cells.append({key:row[key] for key in ['year','fs_div','account_name_en','amount',
                        'raw_amount','unit_multiplier','table_id','row_id','column_id','period_name',
                        'accounting_standard_evidence','extraction_rule_id']})
        record={'issuer_at_ipo':issuer,'corp_code':corp,'receipt':receipt,'source_url':payload['source_url'],
                'sha256':body_hash,'index_sha256':index['sha256'],
                'published_at':payload['official_publication_date'],'checked_cells_and_absences':len(expected)*3,
                'status':'mismatch' if mismatches else 'sample_passed','mismatches':mismatches,'cells':cells}
        basis_results.append(record);failed|=bool(mismatches)
        print(json.dumps({k:v for k,v in record.items() if k!='cells'},ensure_ascii=False))
    atomic_json(cache.parent.parent / 'processed/financial_sample_audit.json',
                {'sample_count': len(SAMPLES)+len(BASIS_SAMPLES),
                 'passed': len(results) == len(SAMPLES) and not failed,
                 'samples': results,'basis_samples':basis_results,
                 'parser_sha256':hashlib.sha256(Path(sys.modules[parse_disclosure_financials.__module__].__file__).read_bytes()).hexdigest(),
                 'training_executed': False})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
