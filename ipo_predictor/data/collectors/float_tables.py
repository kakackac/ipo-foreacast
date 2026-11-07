"""Read float and post-offering totals from the same dated ownership table."""
import json
import re


def ownership_float(tables):
    compact = lambda s: re.sub(r"\s+", "", s)
    def shares(cell):
        match = re.fullmatch(r"([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:주)?", compact(cell))
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
    schema_bindings = {}
    table_conflicts = []
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
        quantity_columns = {c for h in headers for c,cell in enumerate(h)
                            if re.fullmatch(r'주식수(?:\(주\))?', compact(cell))}
        has_pre_column = any(re.search(r'공모전|상장전',compact(cell)) for h in headers for cell in h)
        def post(c):
            if any(c < len(h) and re.search(r'公募後|공모후|상장후', compact(h[c])) for h in headers):
                return True
            # A sole share-quantity column can be dated by its adjacent explicit
            # post-offering percentage header, never by a pre/post mixed grid.
            return (quantity_columns == {c} and not has_pre_column and any(
                c+1 < len(h) and re.fullmatch(r'공모후(?:지분율|비율)(?:\(%\))?',compact(h[c+1]))
                for h in headers))
        def float_label(cell):
            return bool(re.fullmatch(r'(?:상장직후|상장일|공모후|상장후)?유통가능(?:물량|주식수|주식)?'
                                     r'(?:(?:\(주\))|(?:주\))|(?:\(주?\d+\)))?', compact(cell)))
        def total_percentage(row,c):
            if c+1>=len(row):
                return False
            if percent(row[c+1])==1:
                return True
            return (bool(re.fullmatch(r'100(?:\.0+)?',compact(row[c+1]))) and any(
                c+1<len(h) and re.fullmatch(r'(?:공모후)?(?:지분율|비율)(?:\(%\))?',compact(h[c+1]))
                for h in headers))
        def share_class_unverified(c):
            kind_columns = {j for h in headers for j, cell in enumerate(h)
                            if re.fullmatch(r'주식의?종류|증권의종류|종류', compact(cell))}
            return any(c < len(r) and shares(r[c]) not in (None, 0) and any(
                (j in kind_columns and re.search(r'우선주|종류주|RCPS|전환사채|신주인수권|스톡옵션', compact(cell), re.I))
                or re.fullmatch(r'(?:전환상환|상환전환|전환|상환)?우선주(?:식)?|종류주|RCPS|전환사채', compact(cell), re.I)
                for j, cell in enumerate(r[:c])) for r in table[1:])
        def ordinary_scope(row,c):
            return any(compact(cell)=='보통주' for cell in row[:c])
        def subtotal_label(row):
            prefix = [compact(cell) for cell in row[:3]]
            reference = r'(?:\((?:주)?\d+\))*'
            if any(re.fullmatch(r'(?:상장직후|상장일|공모후|상장후)?유통가능(?:물량|주식수|주식)?(?:소계|합계)'+reference, cell)
                   for cell in prefix):
                return True
            if not prefix or not float_label(prefix[0]):
                return False
            # A top-level subtotal may repeat the group or ordinary-share class,
            # but an investor subgroup must never become the entire float.
            labels = []
            for cell in row[1:]:
                value = compact(cell)
                if shares(cell) is not None or percent(cell) is not None:
                    break
                labels.append(value)
            return (any(re.fullmatch(r'소계'+reference, cell) for cell in labels) and all(
                cell in ('', '보통주') or float_label(cell) or re.fullmatch(r'소계'+reference, cell)
                for cell in labels))
        for ri, row in enumerate(table):
            label = compact(' '.join(row[:3]))
            total_label = bool(re.search(r'상장후총발행주식(?:수|합계)|상장예정주식(?:총)?수', label))
            is_total = total_label or bool(re.fullmatch(r'합계|총계|총합계', compact(row[0])))
            if not is_total:
                continue
            # Layout A: total row has separate post-offering and tradable columns.
            for c in range(len(row)):
                float_header = any(c < len(h) and float_label(h[c]) for h in headers)
                float_header = float_header and not any(c<len(h) and re.search(r'지분율|비율',compact(h[c])) for h in headers)
                inline = re.fullmatch(r'([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:주)?\(([0-9]+(?:\.[0-9]+)?%)\)', compact(row[c]))
                numerator = shares(row[c]) if inline is None else shares(inline[1])
                if not float_header or numerator is None:
                    continue
                schema=compact(' '.join(' '.join(h) for h in headers))
                if ('주주명' in schema and re.search(r'주식의?종류|주식종류',schema)
                        and re.search(r'매각제한|보호예수',schema) and c+1<len(row) and percent(row[c+1]) is not None):
                    ownership_totals.append({'shares':shares(row[c]),'percentage':row[c+1],
                        'table_index':ti,'row':ri,'column':c,'table':table})
                denominators = [j for j in range(len(row)) if post(j) and shares(row[j]) is not None
                                and not any(j < len(h) and (re.search(r'유통가능|유통불가능|매도금지|매각제한|의무보유', compact(h[j]))
                                    or re.fullmatch(r'(?:공모후|상장후)?(?:지분율|비율)(?:\(%\))?',compact(h[j]))) for h in headers)]
                if len(denominators) == 1:
                    dc = denominators[0]
                    ordinary=ordinary_scope(row,dc)
                    if (share_class_unverified(dc) and not ordinary or
                            not total_label and not ordinary and not total_percentage(row,dc)):
                        continue
                    printed = inline[2] if inline else row[c+1] if c+1 < len(row) and percent(row[c+1]) is not None else None
                    diluted_percentage = ((total_label or ordinary) and dc+1<len(row) and
                        percent(row[dc+1]) is not None and percent(row[dc+1])!=1)
                    within_column_percentage = (printed is not None and percent(printed)==1 and numerator<shares(row[dc]) and any(
                        j+1<len(row) and shares(row[j]) is not None and numerator+shares(row[j])==shares(row[dc]) and
                        percent(row[j+1])==1 and any(j<len(h) and re.fullmatch(
                            r'유통불가능(?:물량|주식수)?|매각제한물량|매도금지물량',compact(h[j])) for h in headers)
                        and not any(j<len(h) and re.search(r'지분율|비율',compact(h[j])) for h in headers)
                        for j in range(len(row))))
                    ignored_printed = printed if diluted_percentage or within_column_percentage else None
                    if diluted_percentage or within_column_percentage:
                        printed = None
                    ratio = percent(printed) if printed else None
                    candidates.append((numerator, shares(row[dc]), ratio, ti, ri, ri, c, dc, table))
                    schema_bindings[(ti,ri,c)] = {'rule':'explicit_post_offering_float_column_v2',
                        'reported_percentage_text':printed,
                        'share_class_scope':'ordinary' if ordinary else 'explicit_post_total',
                        'ignored_diluted_basis_percentage':ignored_printed,
                        'percentage_scope':'within_float_and_locked_columns' if within_column_percentage else 'issuer_or_diluted_total',
                        'denominator_percentage_text':row[dc+1] if diluted_percentage else None}
            # Layout B: float subtotal and total share one post-offering column.
            for fr, frow in enumerate(table):
                flabel = compact(' '.join(frow[:3]))
                reference = r'(?:\((?:주)?\d+\))?'
                direct_label = bool(re.fullmatch(r'유통가능주식수(?:\(\d+\))?', compact(frow[0])))
                leading_labels=[]
                for cell in frow:
                    if shares(cell) is not None:
                        break
                    leading_labels.append(compact(cell))
                direct_label = direct_label and bool(leading_labels) and all(
                    cell==compact(frow[0]) for cell in leading_labels)
                legacy_label = (direct_label
                        or any(re.fullmatch(r'(?:상장직후)?유통가능물량소계'+reference, compact(cell)) for cell in frow[:3])
                        or (re.fullmatch(r'(?:상장직후)?유통가능물량',compact(frow[0])) and len(frow)>1
                            and re.fullmatch(r'소계'+reference, compact(frow[1]))))
                if not (legacy_label or subtotal_label(frow)):
                    continue
                columns = [c for c in range(min(len(row), len(frow)))
                           if shares(row[c]) is not None and shares(frow[c]) is not None
                           and (post(c) or total_label)]
                if len(columns) != 1:
                    continue
                c = columns[0]
                ordinary=ordinary_scope(row,c) and ordinary_scope(frow,c)
                if share_class_unverified(c) and not ordinary:
                    continue
                ratio = percent(frow[c+1]) if c+1 < len(frow) else None
                diluted_percentage=ordinary and c+1<len(row) and percent(row[c+1]) is not None and percent(row[c+1])!=1
                ignored_printed=frow[c+1] if diluted_percentage and ratio is not None else None
                if diluted_percentage:
                    ratio=None
                candidates.append((shares(frow[c]), shares(row[c]), ratio, ti, fr, ri, c, c, table))
                if not legacy_label or ordinary:
                    schema_bindings[(ti,fr,c)] = {'rule':'top_level_initial_float_subtotal_v2',
                        'reported_percentage_text':frow[c+1] if ratio is not None else None,
                        'share_class_scope':'ordinary' if ordinary else 'explicit_post_total',
                        'ignored_diluted_basis_percentage':ignored_printed,
                        'denominator_percentage_text':row[c+1] if diluted_percentage else None}
    valid = []
    def reported_cell(item):
        binding=paragraph_bindings.get((item[3],item[4]))
        schema=schema_bindings.get((item[3],item[4],item[6]))
        return binding['reported_percentage_text']+'%' if binding else schema['reported_percentage_text'] if schema else item[8][item[4]][item[6]+1]
    for item in candidates:
        numerator, denominator, ratio, *_ = item
        if not denominator or not 0 <= numerator <= denominator:
            continue
        if ratio is not None and (not 0 <= ratio <= 1 or not compatible(numerator,denominator,reported_cell(item))):
            if (item[3],item[4],item[6]) in schema_bindings:
                table_conflicts.append({'rule':'explicit_float_table_quantity_percentage_conflict_v2',
                    'table_index':item[3], 'float_row':item[4], 'total_row':item[5],
                    'float_column':item[6], 'total_column':item[7],
                    'numerator':numerator, 'denominator':denominator, 'reported_percentage':reported_cell(item),
                    'table':item[8]})
            continue
        valid.append(item)
    pairs = {(x[0], x[1]) for x in valid}
    ratios = {x[2] for x in valid if x[2] is not None}
    rejected_headers = []
    for conflict in table_conflicts:
        table=conflict['table'];row=table[conflict['total_row']];n=conflict['numerator']
        header_rows=table[:3]
        # A malformed upper header can mark the old outstanding-share column
        # as post-offering. Ignore it only when an independent issuer-bound
        # candidate and the same grid's float+locked quantities prove another
        # explicit 100% quantity column. Otherwise the conflict stays held.
        alternatives=[j for j in range(len(row)) if j!=conflict['total_column'] and
            shares(row[j]) is not None and (n,shares(row[j])) in pairs and j+1<len(row) and percent(row[j+1])==1 and
            any(j<len(h) and re.fullmatch(r'보유주식|주식수(?:\(주\))?',compact(h[j])) for h in header_rows)]
        locked=[j for j in range(len(row)) if shares(row[j]) is not None and any(
            j<len(h) and re.fullmatch(r'매각제한물량|매도금지물량|유통불가능물량',compact(h[j])) for h in header_rows)
            and not any(j<len(h) and re.search(r'지분율|비율',compact(h[j])) for h in header_rows)]
        supported=[j for j in alternatives if any(n+shares(row[c])==shares(row[j]) for c in locked)]
        if len(supported)==1:
            rejected_headers.append({**conflict,'rejection':'independently_verified_issuer_total_and_locked_float_grid_resolve_bad_header',
                                     'supported_total_column':supported[0]})
        else:
            paragraph_conflicts.append(conflict)
    # A standalone ownership subtotal cannot supply a missing denominator. It
    # can, however, contradict an issuer-bound initial-float paragraph.
    paragraph_pairs={(x[0],x[1]) for x in valid if (x[3],x[4]) in paragraph_bindings}
    for ti, table in enumerate(tables):
        adjusted_columns = {c for h in table[:2] for c, cell in enumerate(h) if re.fullmatch(
            r'자발적매각제한확약후유통가능(?:주식수|물량)',compact(cell))}
        base_columns = {c for h in table[:2] for c, cell in enumerate(h) if re.fullmatch(
            r'유통가능(?:주식수|물량)',compact(cell))}
        for ri, row in enumerate(table):
            if not row or compact(row[0])!='상장일유통가능':
                continue
            for bc in base_columns:
                for ac in adjusted_columns:
                    if max(bc,ac)+1>=len(row) or shares(row[bc]) is None or shares(row[ac]) is None:
                        continue
                    for n,d in pairs:
                        if (n==shares(row[bc]) and n!=shares(row[ac]) and
                                percent(row[bc+1]) is not None and compatible(n,d,row[bc+1]) and
                                percent(row[ac+1]) is not None and compatible(shares(row[ac]),d,row[ac+1])):
                            paragraph_conflicts.append({'rule':'explicit_initial_float_basis_adjustment_requires_reconciliation_v2',
                                'table_index':ti,'row':ri,'base_column':bc,'adjusted_column':ac,
                                'base_shares':n,'adjusted_shares':shares(row[ac]),'denominator':d,'table':table})
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
        'schema_binding':schema_bindings.get((ti,fr,fc)),
        'rejected_header_candidates':rejected_headers,
        'paragraph_binding':paragraph_bindings.get((ti,fr))}, ensure_ascii=False)
    return {'public_float_shares': n, 'public_float_ratio_disclosed': ratio,
            'public_float_parse_method': ('disclosed_public_float_ratio_direct_context' if ratio is not None
                                          else 'disclosed_public_float_shares_direct_context'),
            'public_float_parse_evidence': evidence, 'total_post_listing_shares': d,
            'total_post_listing_shares_parse_method': 'same_table_post_offering_scope_v1',
            'total_post_listing_shares_parse_evidence': evidence,
            'total_post_listing_shares_parser_validation_status': 'structurally_verified',
            'public_float_basis_status': 'same_table_post_offering_scope_verified'}
