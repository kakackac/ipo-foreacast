"""Read float and post-offering totals from the same dated ownership table."""
import json
import re


def ownership_float(tables):
    compact = lambda s: re.sub(r"\s+", "", s)
    def shares(cell):
        match = re.fullmatch(r"([0-9][0-9,]*)(?:주)?", compact(cell))
        return int(match[1].replace(',', '')) if match else None
    def percent(cell):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)%", compact(cell))
        return float(match[1])/100 if match else None
    def display_quantum(cell):
        printed=compact(cell).removesuffix('%')
        decimals=len(printed.split('.')[1]) if '.' in printed else 0
        return 10**(-decimals)/100
    def compatible(numerator, denominator, cell):
        value=percent(cell);quantum=display_quantum(cell);computed=numerator/denominator
        return value is not None and (value-quantum/2-1e-12<=computed<=value+quantum/2+1e-12 or
                                     value-1e-12<=computed<value+quantum-1e-12)
    candidates = []
    paragraph_bindings = {}
    paragraph_conflicts = []
    ownership_totals = []
    for ti, table in enumerate(tables):
        if not table:
            continue
        for ri, row in enumerate(table):
            if len(row)!=1:continue
            text=compact(row[0])
            totals={int(value.replace(',','')) for value in re.findall(
                r'(?:당사의|(?<![가-힣])회사의)(?:상장예정주식수|상장후총발행(?:예정)?주식수|상장후발행주식총수)([0-9][0-9,]*)주',text)}
            if len(totals)!=1:continue
            total=totals.pop()
            patterns=[
                r'(?:당사의|(?<![가-힣])회사의)(?:상장예정주식수|상장후총발행(?:예정)?주식수|상장후발행주식총수)[0-9][0-9,]*주중(?:약)?(?P<pct>[0-9]+(?:\.[0-9]+)?)%에해당하는(?P<n>[0-9][0-9,]*)주(?:는|가)?상장직후유통가능(?:물량|주식)',
                r'상장예정주식수의(?P<pct>[0-9]+(?:\.[0-9]+)?)%에해당하는(?P<n>[0-9][0-9,]*)주(?:는|가)?상장직후유통가능(?:물량|주식)',
                r'(?P<n>[0-9][0-9,]*)주(?:\((?:공모후(?:주식수)?기준)?(?P<pct>[0-9]+(?:\.[0-9]+)?)%\))?(?:는|가|이)?상장직후유통가능(?:물량|주식)',
            ]
            seen_spans=[]
            for pattern in patterns:
                for match in re.finditer(pattern,text):
                    if any(start<=match.start()<end for start,end in seen_spans):continue
                    seen_spans.append(match.span())
                    numerator=int(match['n'].replace(',',''));ratio=float(match['pct'])/100 if match['pct'] is not None else None
                    binding={'text':row[0],'tradable_clause':match[0],'numerator':numerator,'denominator':total,
                             'reported_fraction':ratio,'reported_percentage_text':match['pct'],
                             'rule':'same_row_explicit_issuer_initial_float_v1'}
                    if (total<=0 or numerator>total or ratio is not None and
                            (not 0<=ratio<=1 or not compatible(numerator,total,match['pct']+'%'))):
                        paragraph_conflicts.append(binding);continue
                    candidates.append((numerator,total,ratio,ti,ri,ri,0,0,table))
                    paragraph_bindings[(ti,ri)]=binding
        headers = table[:2]
        post = lambda c: any(c < len(h) and re.search(r'公募後|공모후|상장후', compact(h[c])) for h in headers)
        for ri, row in enumerate(table):
            label = compact(' '.join(row[:3]))
            total_label = bool(re.search(r'상장후총발행주식수|상장예정주식(?:총)?수', label))
            is_total = total_label or bool(re.fullmatch(r'합계|총계', compact(row[0])))
            if not is_total:
                continue
            # Layout A: total row has separate post-offering and tradable columns.
            for c in range(len(row)):
                float_header = any(c < len(h) and re.fullmatch(r'유통가능(?:물량|주식수)?', compact(h[c])) for h in headers)
                if not float_header or shares(row[c]) is None:
                    continue
                schema=compact(' '.join(' '.join(h) for h in headers))
                if ('주주명' in schema and re.search(r'주식의?종류|주식종류',schema)
                        and re.search(r'매각제한|보호예수',schema) and c+1<len(row) and percent(row[c+1]) is not None):
                    ownership_totals.append({'shares':shares(row[c]),'percentage':row[c+1],
                        'table_index':ti,'row':ri,'column':c,'table':table})
                denominators = [j for j in range(len(row)) if post(j) and shares(row[j]) is not None
                                and not any(j < len(h) and '유통가능' in compact(h[j]) for h in headers)]
                if len(denominators) == 1:
                    dc = denominators[0]
                    ratio = percent(row[c+1]) if c+1 < len(row) else None
                    candidates.append((shares(row[c]), shares(row[dc]), ratio, ti, ri, ri, c, dc, table))
            # Layout B: float subtotal and total share one post-offering column.
            for fr, frow in enumerate(table):
                flabel = compact(' '.join(frow[:3]))
                if not (re.search(r'^유통가능주식수(?:\(\d+\))?', flabel)
                        or any(compact(cell)=='유통가능물량소계' for cell in frow[:3])
                        or (compact(frow[0])=='유통가능물량' and len(frow)>1 and compact(frow[1])=='소계')):
                    continue
                columns = [c for c in range(min(len(row), len(frow)))
                           if shares(row[c]) is not None and shares(frow[c]) is not None
                           and (post(c) or total_label)]
                if len(columns) != 1:
                    continue
                c = columns[0]
                ratio = percent(frow[c+1]) if c+1 < len(frow) else None
                candidates.append((shares(frow[c]), shares(row[c]), ratio, ti, fr, ri, c, c, table))
    valid = []
    def reported_cell(item):
        binding=paragraph_bindings.get((item[3],item[4]))
        return binding['reported_percentage_text']+'%' if binding else item[8][item[4]][item[6]+1]
    for item in candidates:
        numerator, denominator, ratio, *_ = item
        if not denominator or not 0 <= numerator <= denominator:
            continue
        if ratio is not None and (not 0 <= ratio <= 1 or not compatible(numerator,denominator,reported_cell(item))):
            continue
        valid.append(item)
    pairs = {(x[0], x[1]) for x in valid}
    ratios = {x[2] for x in valid if x[2] is not None}
    # A standalone ownership subtotal cannot supply a missing denominator. It
    # can, however, contradict an issuer-bound initial-float paragraph.
    paragraph_pairs={(x[0],x[1]) for x in valid if (x[3],x[4]) in paragraph_bindings}
    for subtotal in ownership_totals:
        for n,d in paragraph_pairs:
            if subtotal['shares']!=n or not compatible(n,d,subtotal['percentage']):
                paragraph_conflicts.append({'rule':'issuer_initial_paragraph_ownership_total_conflict_v1',
                    'paragraph_numerator':n,'paragraph_denominator':d,'ownership_total':subtotal})
    if len(pairs) > 1 or paragraph_conflicts:
        return {'public_float_shares': None, 'public_float_ratio_disclosed': None,
                'public_float_parse_method': 'conflicting_float_ratios_review_required',
                'public_float_parse_evidence': json.dumps({'quantity_pairs':sorted(pairs),
                    'reported_ratios':sorted(ratios),'paragraph_conflicts':paragraph_conflicts},ensure_ascii=False),
                'public_float_basis_status': 'same_table_candidates_conflict'}
    if not valid:
        return {}
    n, d, ratio, ti, fr, tr, fc, dc, table = min(valid,
        key=lambda item:display_quantum(reported_cell(item)) if item[2] is not None else float('inf'))
    evidence = json.dumps({'rule': 'same_table_post_offering_scope_v1', 'table_index': ti,
        'float_row': fr, 'total_row': tr, 'float_column': fc, 'total_column': dc,
        'basis': 'post_offering_ordinary_or_explicit_listed_shares', 'table': table,
        'compatible_reported_ratios':sorted(ratios),
        'paragraph_binding':paragraph_bindings.get((ti,fr))}, ensure_ascii=False)
    return {'public_float_shares': n, 'public_float_ratio_disclosed': ratio,
            'public_float_parse_method': ('disclosed_public_float_ratio_direct_context' if ratio is not None
                                          else 'disclosed_public_float_shares_direct_context'),
            'public_float_parse_evidence': evidence, 'total_post_listing_shares': d,
            'total_post_listing_shares_parse_method': 'same_table_post_offering_scope_v1',
            'total_post_listing_shares_parse_evidence': evidence,
            'total_post_listing_shares_parser_validation_status': 'structurally_verified',
            'public_float_basis_status': 'same_table_post_offering_scope_verified'}
