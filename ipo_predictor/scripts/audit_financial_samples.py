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
    ('뷰노', '01344202', '20210215001088', '2021-02-25 21:00', 2019,
     [162097981, -6021756963, 7452661169, 2033877398, 5418783771, -5912548138]),
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
     [309062616157, 59844712688, 136108165193, 34039092723, 102069072470, 15408977423]),
    ('엔에스쇼핑', '00406392', '20150312000834', '2015-03-26 21:00', 2013,
     [347135000000, 69534000000, 336642000000, 109236000000, 227406000000, 54619000000]),
    ('파멥신', '00972293', '20181108000408', '2018-11-20 21:00', 2017,
     [208000000, -3835000000, 7526344656, 5245707983, 2280636673, -6483000000]),
    ('바이오플러스', '00636656', '20210910000471', '2021-09-26 21:00', 2020,
     [24366600859, 12327486055, 37935325802, 9415034892, 28520290910, 9703010829]),
    ('코셈', '01493793', '20240206000494', '2024-02-22 21:00', 2022,
     [12525000000, 1780000000, 14077570000, 3362721000, 10714849000, 1716000000]),
    ('스카이랩스', '01586923', '20260825000414', '2026-09-03 21:00', 2025,
     [7935865376, -14725265236, 30180942922, 4728384269, 25452558653, -62995166524]),
    ('싸이맥스', '00659161', '20150602000317', '2015-06-16 21:00', 2014,
     [54714000000, 5836000000, 39268761071, 12976491271, 26292269800, 5993000000]),
    ('SG', '00963976', '20180122000271', '2018-01-25 21:00', 2016,
     [93642364279, 11464092002, 83188236557, 58375494119, 24812742438, 9004249362]),
    ('휴젤', '00888347', '20151214000155', '2015-12-23 21:00', 2014,
     [40350000000, 15488000000, 97249646946, 54576067221, 42673579725, 13150000000]),
    ('서진시스템', '00838005', '20170315000227', '2017-03-26 21:00', 2016,
     [165870701012, 24390510140, 227322922042, 154818667715, 72504254327, 19626660876]),
]

# Reviewed original annual cells: scope, year, revenue, operating income, assets.
# A None income pair explicitly checks that interim-only income stays absent.
BASIS_SAMPLES = [
    ('덱스터', '01021949', '20151211000128',
     '6fe3b18f68db166923d9759df19e193f385b57ca5e6b5d853304501e8483b534',
     [('CFS', 2014, 18665493000, 5049200000, 18436977410),
      ('OFS', 2014, 18665493000, 4760906000, 18458736002)], {2013, 2014}),
    ('바이오노트', '00622787', '20221212000271',
     '797660b3e926c7a8d5f5d1750d4580e9969cbaa86cd142a0e1befd4fa3c0844e',
     [('CFS', 2021, None, None, 1488867273227), ('OFS', 2021, None, None, 993760029494)], {2021}),
    ('한솔씨앤피', '00493431', '20160118000101',
     '3dc2409f3c8d2b44bd86ae46e228c0c7c6b3411c8aac108130fc03d708446953',
     [('CFS', 2014, None, None, 44391639696), ('OFS', 2014, None, None, 34737775683)], {2014}),
    ('카카오게임즈', '01137383', '20200831000067',
     '9b549c182d0c323db622573ef02e5a8dca48e4e693ef18499922b998a5e80b17',
     [('CFS', 2019, None, None, 591751778679), ('OFS', 2019, None, None, 520693596752)], {2019}),
    ('카카오페이', '01244601', '20211022000418',
     '8940aca954121a900a44f3509d2d1956b6f99bebada7dc616b04af657fc9cb87',
     [('CFS', 2020, None, None, 1207982675732), ('OFS', 2020, None, None, 761403131316)], {2020}),
    ('엘앤씨바이오', '01181515', '20181022000079',
     '7802d3f6288da9f415bfebd3696958e985bb9141ad9b2cf306f586934634fed0',
     [('CFS', 2017, None, None, 20490472647), ('OFS', 2017, None, None, 20490472647)], {2017}),
    ('비아이매트릭스', '01462739', '20231027000529',
     'b33cd51459325ddaf68e8542a0ef886c66144670d6cc137558b038036f6a3dca',
     [('OFS', 2022, None, None, 25527720751)], {2022}),
    ('야스', '00875307', '20170919000194',
     '9215b2415c672c58bd5ae807db2861bf6a931bf820e9728d0307b44805d08d00',
     [('OFS', 2016, 82982000000, 16056000000, 81082801895)], {2015, 2016}),
    ('이지트로닉스', '01140837', '20220118000224',
     '3ffbe413ab2c6832cd0bf0f0a6c42fb22218b7cbe63e5435a12febce74b21be3',
     [('OFS', 2020, 15348581080, 2829540431, 34289823450)], {2020}),
    ('쿼드메디슨', '01671285', '20251201000478',
     '14d4d8327350e675bbdc524b5175e87aa629bf0b7582e0c823a51a3ad3e3a13b',
     [('CFS', 2024, 9303610913, -4449177465, 32098000000),
      ('OFS', 2024, 9303610913, -3282107523, 31895000000)], {2024}),
    ('셀리드', '01235296', '20190201000199',
     '2e433e0636a8a5bf1399f1638c8710cb6b4edb384a93b6f21e51ebdf03d80ac6',
     [('OFS', 2017, None, -1516714459, 4963151186)], {2017}),
    ('엔에스', '01020843', '20151124000081',
     'e6b99671046ff72ffed41973ae837909379158521ebd931d8d91a4cea19dc98a',
     [('OFS', 2014, 20779567395, 2332022741, 22660581489)], {2013, 2014}),
    ('에코프로비엠', '01160363', '20190220000799',
     '961cd71b0ee13fccd0f2a59c4d34be986c4e3c6a91ca85fc6561b2d22eee5595',
     [('OFS', 2017, None, None, 296290921990)], {2017}),
    ('나우로보틱스', '01594791', '20250422000541',
     'b486dec3c11c81ca9bcbc46d381d025541b88c690b5824851bbf3deaf2387dff',
     [('OFS', 2024, 12059591829, -2856450125, 16316925751)], {2021, 2022, 2023, 2024}),
    ('루미르', '01636565', '20241008000450',
     'a8d578aab6fbdd87cccbcb70b849c1ede9ea97167febc3bdd4e6bef5337f8dbf',
     [('OFS', 2023, 12126540000, -3023273000, 46679313086)], {2023}),
    ('피피아이', '00403766', '20191213000083',
     'a963431acb85848428a50673824cf0a40ca52be500dcb8a31d567d943b672394',
     [('OFS', 2016, 15523077931, 1104260564, None)], {2016, 2017, 2018}),
    ('제주맥주', '01230255', '20210512000262',
     '6f2db4f80c52f663a9d3dfb9bea22ab2bfaa83d3625203bb0ad78708e9219607',
     [('OFS', 2020, 21525592162, -4347327422, 62682620070),
      ('OFS', 2019, 7210611126, -9218206523, 43085862224)], {2019, 2020}),
    ('카카오뱅크', '01133217', '20210722000374',
     'eb112ad666e2ca2caf4565efdd387146ee8b89c17db0712b3abbed027421bdf7',
     [('OFS', 2020, 804157323123, 122551246633, 26649990466590),
      ('OFS', 2018, 375581509526, -21208717701, 12126722813103)], {2018, 2019, 2020}),
    ('핀텔', '01524507', '20221007000279',
     '5baa7fd5db1bd3d8454a64022fe17c95e347e42d235b36045f972b7aebbc8069',
     [('OFS', 2020, 4886319291, -837626, 2445934794),
      ('OFS', 2019, 2596328239, 93064237, 2910102543)], {2019, 2020, 2021}),
    ('제주항공', '00555874', '20151026000276',
     '24f3f78cb6a9c04a82e28f49c9ff387ea139644b501c0df1e2d3de80952c1fbf',
     [('OFS', 2014, 510636286681, 29519225853, 263681131155),
      ('OFS', 2013, 434146480411, 17026017114, 180901506051)], {2013, 2014}),
    ('포시에스', '00939942', '20150129000089',
     '20b088ef862d75c2bb0b29d4b855ee0069d214aa3bd55bda610d1b347fe5b1ee',
     [('OFS', 2014, 12846942899, 4111424306, 24724607959)], {2013, 2014}),
    ('아이엔지생명', '00176792', '20170424000341',
     'd19317902d31fb71e985b9d024f8de96eb70db66a09f98e572a6ebde4e32e3fd',
     [('OFS', 2016, 4265748689342, 321884373980, 30668730146645),
      ('OFS', 2013, 3853774000000, 272284000000, 23339170000000, '2013-03-31')], {2013, 2014, 2015, 2016}),
    ('에이텀', '01444380', '20231117000118',
     '843e1cdee1385c8049e5f3a2ae57d05938901ad823a878c0c5b09c0860339c0c',
     [('CFS', 2023, 47971262133, -5922867618, 38055185236)], {2021, 2022, 2023}),
    ('SV인베스트먼트', '00599151', '20180625000357',
     '4edb9ce1f4ad545ada7bb7f6014116faf2d6fb9285a47900ac6faeb226ef7dc6',
     [('CFS', 2018, 17961441215, 7561825679, 33886719627)], {2017, 2018}),
    ('이지케어텍', '00490090', '20190308000278',
     'b60637df20c7810795d24312584fb87dfa8536593772a3943e65d88d0798bc05',
     [('CFS', 2018, 55374000000, 4492000000, 39215213400)], {2017, 2018}),
    ('코리아에셋투자증권', '00304915', '20191107000232',
     '2e4c5e31fb9c2d2bfd2ffec1846edead3950390d04f75b768847aa2f3df13958',
     [('CFS', 2019, 166372000000, 8240000000, 78637934769)], {2019}),
    ('마크로밀엠브레인', '00877174', '20200617000383',
     '9a2975527798060bb92daf4174f6aa80100fa169882f7492beda309ee7b9abe9',
     [('CFS', 2019, 39074815984, 2576070308, 17347766482)], {2017, 2018, 2019}),
    ('위더스제약', '00765851', '20200623000326',
     'f522e97a69e6aca462544ed59c56283b8bf35e6f7afcb4ded0ba0e8556551c91',
     [('OFS', 2019, 51732082000, 10934657000, 76282983000)], {2019}),
    ('전진건설로봇', '00372129', '20240807000292',
     '48f65bca5496a43df10f469a9da642138d5a4a79108ed1fca8d115db44fe1f1b',
     [('CFS', 2023, 158370290546, 32904822481, 158316498662),
      ('OFS', 2023, 158079710420, 32920653826, 157583434174)], {2023}),
    ('두올', '00119672', '20160715000252',
     'c6f66ec26bed267290919d2cf759e36eb6c46862277eb37621757577f66224a9',
     [('CFS', 2015, 366288000000, 26056000000, 230283413058),
      ('CFS', 2014, 316832000000, 18397000000, 208983800118),
      ('OFS', 2015, 197562000000, 2553000000, 184919511060)], {2014, 2015}),
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
     [('OFS', 2014, 66063000000, 3738000000, 32094356307),
      ('OFS', 2013, 71254000000, 6976000000, 28041636862)], {2013, 2014}),
    ('오션브릿지', '01113754', '20161118000262',
     '24d107732915a37192ea9aae02d218ad23ff066cdd9dc91e2624c048096ed841',
     [('OFS', 2015, 25389000000, 3762000000, 13445887745)], {2015}),
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
     [('CFS', 2014, 3928349000000, 151667000000, 24681134224854),
      ('OFS', 2014, 3692466000000, 84574000000, 24080044889046)], {2014}),
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
     [('OFS', 2022, 361311518, -6536885880, 8855319849)], {2019, 2020, 2021, 2022}),
    ('삼영에스앤씨', '00405506', '20210510000418',
     '864cc88b76dc376c51f9b57692476b7261f65e3c2389f52fe95f586bc69e2bfd',
     [('OFS', 2020, 13557858486, 182225063, 18148583909)], {2019, 2020}),
]


def main():
    parser_path=Path(sys.modules[parse_disclosure_financials.__module__].__file__)
    parser_sha=hashlib.sha256(parser_path.read_bytes()).hexdigest()
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
        if receipt in {'20180625000357','20190308000278','20191107000232','20200617000383','20200623000326','20231117000118'}:
            from data.pipelines.disclosure_financials import annual_date_range
            if frame.empty or any(annual_date_range(str(a.period_start)+'~'+str(a.period_end)) is None
                                 or str(a.period_end)[:4]!=str(a.year) for a in frame.itertuples()):
                mismatches.append('explicit_full_fiscal_period')
        cells=[]
        for scope,year,*values in expected:
            fiscal_end=values.pop() if len(values)==4 else None
            for name,value in zip(['revenue','operating_income','total_assets'],values):
                selected=frame[frame.fs_div.eq(scope)&frame.year.eq(year)&frame.account_name_en.eq(name)]
                if fiscal_end:selected=selected[selected.period_end.eq(fiscal_end)]
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
    if (parser_sha!=hashlib.sha256(parser_path.read_bytes()).hexdigest()
            or script_sha!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise RuntimeError('financial_sample_audit_code_changed_during_run')
    atomic_json(cache.parent.parent / 'processed/financial_sample_audit.json',
                {'sample_count': len(SAMPLES)+len(BASIS_SAMPLES),
                 'passed': len(results) == len(SAMPLES) and not failed,
                 'samples': results,'basis_samples':basis_results,
                 'parser_sha256':parser_sha,'audit_script_sha256':script_sha,
                 'training_executed': False})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
