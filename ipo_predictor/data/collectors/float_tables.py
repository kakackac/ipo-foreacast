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
    ratio_only = []
    scope_rejections = []
    unverified_grids = []
    for ti, table in enumerate(tables):
        if not table:
            continue
        for ri, row in enumerate(table):
            if len(row)!=1:continue
            text=compact(row[0])
            total_label=(r'(?:(?:당사의|(?<![가-힣])회사의)'
                r'(?:상장예정주식수|상장후총발행(?:예정)?주식수|상장후발행주식총수)'
                r'(?:\(보통주기준\))?(?:인)?|공모주식(?:등)?을포함한상장예정주식수)')
            other_scope=(r'비교회사|타회사|종속회사|관계회사|경쟁회사|피투자회사|다른회사|다른기업|타사|경쟁사|'
                         r'비교기업|종속기업|관계기업|피투자기업|행사가정|전환가정|전량행사시|전환시')
            total_matches=[m for m in re.finditer(total_label+r'(?P<total>[0-9][0-9,]*)주',text)
                if shares(m['total']) is not None and not re.search(other_scope,text[max(0,m.start()-40):m.start()])]
            totals={int(m['total'].replace(',','')) for m in total_matches}
            if len(totals)!=1:continue
            total=totals.pop()
            total_clause=r'(?:'+'|'.join(re.escape(m[0]) for m in total_matches)+r')'
            initial_percentage=(r'(?P<pct>[0-9]+(?:\.[0-9]+)?)%'
                r'(?P<alternate_basis_note>\(우선주(?:전환후|포함(?:기준)?)[0-9]+(?:\.[0-9]+)?%\))?')
            patterns=[
                total_clause+r'중(?:약)?'+initial_percentage+r'에해당하는(?P<n>[0-9][0-9,]*)주(?:는|가)?상장직후유통가능(?:물량|주식)',
                r'상장예정주식수의'+initial_percentage+r'에해당하는(?P<n>[0-9][0-9,]*)주(?:는|가)?상장직후유통가능(?:물량|주식)',
                r'(?P<n>[0-9][0-9,]*)주(?:\((?:공모후(?:주식수)?기준|상장주식수대비)?(?P<pct>[0-9]+(?:\.[0-9]+)?)%\))?(?:는|가|이)?상장직후유통가능(?:물량|주식)',
            ]
            seen_spans=[]
            for pattern in patterns:
                for match in re.finditer(pattern,text):
                    if any(start<=match.start()<end for start,end in seen_spans):continue
                    if re.search(other_scope,text[max(0,match.start()-40):match.start()]):continue
                    seen_spans.append(match.span())
                    numerator=shares(match['n']);percentage_text=match['pct'];percentage_raw=percentage_text
                    alternate_note=match.groupdict().get('alternate_basis_note')
                    if numerator is None:
                        paragraph_conflicts.append({'rule':'malformed_initial_float_share_token_v2',
                            'text':row[0],'raw_quantity':match['n'],'tradable_clause':match[0]})
                        continue
                    clause=match[0]
                    if percentage_text is None:
                        linked=re.search(r'(?P<pct>[0-9][0-9.,]*)%(?P<note>(?:\([^()]*\))*)에해당하는$',text[:match.start()])
                        if linked:
                            percentage_raw=linked['pct'];percentage_text=percentage_raw;alternate_note=linked['note'] or None
                            if re.fullmatch(r'(?:[0-9]{1,2}|100),[0-9]{2}',percentage_text):
                                percentage_text=percentage_text.replace(',','.')
                            if (not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',percentage_text) or alternate_note
                                    and not re.fullmatch(r'\(우선주(?:전환후|포함(?:기준)?)[0-9]+(?:\.[0-9]+)?%\)',alternate_note)):
                                paragraph_conflicts.append({'rule':'unbound_initial_float_percentage_link_v2',
                                    'text':row[0],'percentage_clause':linked[0],'tradable_clause':clause})
                                continue
                            clause=linked[0]+clause
                    ratio=float(percentage_text)/100 if percentage_text is not None else None
                    binding={'text':row[0],'tradable_clause':clause,'numerator':numerator,'denominator':total,
                             'reported_fraction':ratio,'reported_percentage_text':percentage_text,
                             'reported_percentage_raw_text':percentage_raw,
                             'percentage_normalization':('decimal_comma_two_places_quantity_checked'
                                 if percentage_raw!=percentage_text else None),
                             'alternate_basis_note':alternate_note,
                             'issuer_total_clauses':[m[0] for m in total_matches],
                             'rule':'same_row_explicit_issuer_initial_float_v2'}
                    if (total<=0 or numerator>total or ratio is not None and
                            (not 0<=ratio<=1 or not compatible(numerator,total,percentage_text+'%'))):
                        paragraph_conflicts.append(binding);continue
                    candidates.append((numerator,total,ratio,ti,ri,ri,0,0,table))
                    paragraph_bindings[(ti,ri)]=binding
        headers = table[:2]
        quantity_columns = {c for h in headers for c,cell in enumerate(h)
                            if re.fullmatch(r'주식수(?:\(주\))?', compact(cell))}
        has_pre_column = any(re.search(r'공모전|상장전',compact(cell)) for h in headers for cell in h)
        conditional_header = any(re.search(r'희석|행사가정|행사후|전환후|확약후|가정시|전량행사',compact(cell)) for h in headers for cell in h)
        def public_offering_holders(c):
            return any(c<len(r) and shares(r[c]) not in (None,0) and any(
                re.fullmatch(r'공모주주(?:\(.*\))?|공모주식수|공모주식(?:\(.*\))?|기관투자자및일반(?:청약자|투자자)',compact(cell))
                for cell in r[:c]) for r in table[1:])
        def post(c):
            if any(c<len(h) and re.search(r'희석|행사전|행사후|전환전|전환후|행사가정',compact(h[c])) for h in headers):
                return False
            if any(c < len(h) and re.search(r'公募後|공모후|상장후', compact(h[c])) for h in headers):
                return True
            # A sole share-quantity column can be dated by its adjacent explicit
            # post-offering percentage header, never by a pre/post mixed grid.
            return (quantity_columns == {c} and not has_pre_column and any(
                c+1 < len(h) and re.fullmatch(r'공모후(?:지분율|비율)(?:\(%\))?',compact(h[c+1]))
                for h in headers)) or inferred_partition(c)
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
                (j in kind_columns and re.search(r'우선주|종류주|RCPS|CPS|전환사채|신주인수권|스톡옵션|주식매수선택권', compact(cell), re.I))
                or re.fullmatch(r'(?:전환상환|상환전환|전환|상환)?우선주(?:식)?|종류주|RCPS|CPS|전환사채|주식매수선택권', compact(cell), re.I)
                for j, cell in enumerate(r[:c])) for r in table[1:])
        def ordinary_scope(row,c):
            return any(compact(cell)=='보통주' or re.fullmatch(r'상장예정주식(?:총)?수\(보통주\)',compact(cell)) for cell in row[:c])
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
        def inferred_partition(c):
            if (has_pre_column or conditional_header or any('상장신청' in compact(cell) for h in headers for cell in h)
                    or not public_offering_holders(c)):
                return False
            if not any(c<len(h) and re.fullmatch(r'(?:총|소유)?주식수(?:\(주\))?',compact(h[c])) for h in headers):
                return False
            if any(c<len(h) and re.search(r'유통|매각|매도|의무보유|지분율|비율',compact(h[c])) for h in headers):
                return False
            totals=[r for r in table if r and re.fullmatch(r'합계|총계|총합계',compact(r[0])) and c<len(r)
                    and shares(r[c]) is not None and total_percentage(r,c)]
            floats=[r for r in table if r and subtotal_label(r) and c<len(r) and shares(r[c]) is not None]
            blocked=[]
            for r in table:
                if not r or not re.fullmatch(r'유통(?:불가능|제한)(?:주식수|물량|주식)?',compact(r[0])) or c>=len(r):continue
                labels=[compact(x) for x in r[1:c]]
                if (shares(r[c]) is not None and '소계' in labels and all(x in ('소계','보통주','') for x in labels)):
                    blocked.append(r)
            if len(totals)==len(floats)==len(blocked)==1:
                n,b,d=shares(floats[0][c]),shares(blocked[0][c]),shares(totals[0][c])
                if d and n+b==d and c+1<len(floats[0]) and c+1<len(blocked[0]):
                    return compatible(n,d,floats[0][c+1]) and compatible(b,d,blocked[0][c+1])
            # Separate float/locked columns also prove a post-offering total,
            # but only with a positive public-offering holder and exact sum.
            for r in totals:
                fs=[j for j in range(len(r)) if shares(r[j]) is not None and any(
                    j<len(h) and float_label(h[j]) for h in headers) and not any(
                    j<len(h) and re.search(r'지분율|비율',compact(h[j])) for h in headers)]
                bs=[j for j in range(len(r)) if shares(r[j]) is not None and any(j<len(h) and re.fullmatch(
                    r'유통불가능(?:물량|주식수)?|매도금지물량|매각제한물량|의무보호예수주식수',compact(h[j])) for h in headers)
                    and not any(j<len(h) and re.search(r'지분율|비율',compact(h[j])) for h in headers)]
                explicit_locked_total=[j for j in bs if any(j<len(h) and re.fullmatch(
                    r'총보호예수물량|총매각제한물량',compact(h[j])) for h in headers)]
                if len(explicit_locked_total)==1:bs=explicit_locked_total
                if len(fs)==len(bs)==1 and shares(r[fs[0]])+shares(r[bs[0]])==shares(r[c]):return True
            return False

        # An explicitly labelled listing-day schedule discloses its percentage
        # directly. Future 100% rows must not manufacture today's denominator.
        schedule_q={c for h in headers for c,cell in enumerate(h) if re.fullmatch(r'(?:유통가능)?주식수(?:\(주\))?',compact(cell))}
        schedule_p={c for h in headers for c,cell in enumerate(h) if re.fullmatch(r'유통가능(?:주식수)?비율(?:\(%\))?',compact(cell))}
        if (len(schedule_q)<=1 and len(schedule_p)==1 and not any(re.search(
                r'공모전|상장신청|행사|전환|희석|가정|자발적|확약후|주주명|기업명',compact(cell)) for h in headers for cell in h)):
            qc=next(iter(schedule_q),None);pc=next(iter(schedule_p))
            initial=[(ri,r) for ri,r in enumerate(table) if r and re.fullmatch(r'상장(?:일|직후)유통가능',compact(r[0]))]
            if len(initial)==1 and (qc is None or pc==qc+1):
                ri,r=initial[0]
                if (pc<len(r) and (qc is None or shares(r[qc]) is not None)
                        and percent(r[pc]) is not None and 0<=percent(r[pc])<=1):
                    ratio_only.append({'rule':'explicit_listing_day_schedule_ratio_v1','table_index':ti,'row':ri,
                        'quantity_column':qc,'ratio_column':pc,'shares':None if qc is None else shares(r[qc]),'ratio':percent(r[pc]),
                        'percentage_text':r[pc],'table':table,'denominator_inferred':False})
        if any('유통가능' in compact(cell) for h in headers for cell in h):
            unverified_grids.append({'table_index':ti,'table':table})
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
                if not float_header:
                    continue
                schema=compact(' '.join(' '.join(h) for h in headers))
                if ('주주명' in schema and re.search(r'주식의?종류|주식종류',schema)
                        and numerator is not None and re.search(r'매각제한|보호예수',schema)
                        and c+1<len(row) and percent(row[c+1]) is not None):
                    ownership_totals.append({'shares':shares(row[c]),'percentage':row[c+1],
                        'table_index':ti,'row':ri,'column':c,'table':table})
                denominators = [j for j in range(len(row)) if (post(j) or total_label and j in quantity_columns) and shares(row[j]) is not None
                                and not any(j < len(h) and (re.search(r'유통가능|유통불가능|매도금지|매각제한|의무보유', compact(h[j]))
                                    or re.fullmatch(r'(?:공모후|상장후)?(?:지분율|비율)(?:\(%\))?',compact(h[j]))) for h in headers)]
                if not denominators and numerator is not None and not has_pre_column and not conditional_header:
                    locked=[j for j in range(len(row)) if j+1<len(row) and shares(row[j]) is not None and any(
                        j<len(h) and re.fullmatch(r'유통불가능(?:물량|주식수)?|매도금지물량|매각제한물량',compact(h[j])) for h in headers)
                        and not any(j<len(h) and re.search(r'지분율|비율',compact(h[j])) for h in headers)]
                    if (len(locked)==1 and c+1<len(row) and not share_class_unverified(c)
                            and not share_class_unverified(locked[0]) and percent(row[c+1]) is not None
                            and percent(row[locked[0]+1]) is not None):
                        lc=locked[0];partition_total=numerator+shares(row[lc])
                        if (partition_total and compatible(numerator,partition_total,row[c+1])
                                and compatible(shares(row[lc]),partition_total,row[lc+1])
                                and abs(percent(row[c+1])+percent(row[lc+1])-1)<1e-12):
                            ratio_only.append({'rule':'explicit_float_locked_total_ratio_v1','table_index':ti,'row':ri,
                                'quantity_column':c,'ratio_column':c+1,'shares':numerator,'ratio':percent(row[c+1]),
                                'percentage_text':row[c+1],'table':table,'locked_column':lc,'denominator_inferred':False})
                if len(denominators) == 1:
                    dc = denominators[0]
                    ordinary=ordinary_scope(row,dc)
                    if (share_class_unverified(dc) and not ordinary or
                            not total_label and not ordinary and not total_percentage(row,dc)):
                        scope_rejections.append({'table_index':ti,'row':ri,'reason':'post_total_share_class_or_scope_unverified','table':table})
                        continue
                    printed = inline[2] if inline else row[c+1] if c+1 < len(row) and percent(row[c+1]) is not None else None
                    if numerator is None:
                        if printed and 0<=percent(printed)<=1 and total_percentage(row,dc) and not conditional_header:
                            ratio_only.append({'rule':'explicit_post_issuer_total_ratio_v1','table_index':ti,'row':ri,
                                'quantity_column':c,'ratio_column':c+1,'shares':None,'ratio':percent(printed),
                                'percentage_text':printed,'table':table,'post_total_column':dc,
                                'post_total_shares':shares(row[dc]),'unparsed_quantity_text':row[c],'denominator_inferred':False})
                        continue
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
                    scope_rejections.append({'table_index':ti,'row':ri,'reason':'post_total_share_class_or_scope_unverified','table':table})
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
    # A direct aggregate percentage must agree with a separately bound quantity
    # pair at its printed precision; neither path silently overrides the other.
    for direct in ratio_only:
        for n,d in pairs:
            if (direct['shares'] is not None and direct['shares']!=n or
                    not compatible(n,d,direct['percentage_text'])):
                paragraph_conflicts.append({'rule':'direct_initial_ratio_quantity_conflict_v1',
                    'numerator':n,'denominator':d,'direct_candidate':direct})
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
        ratios_only={(r['shares'],r['ratio']) for r in ratio_only}
        if len(ratios_only)==1 and not scope_rejections:
            chosen=ratio_only[0]
            return {'public_float_shares':chosen['shares'],'public_float_ratio_disclosed':chosen['ratio'],
                    'public_float_parse_method':'disclosed_public_float_ratio_direct_context',
                    'public_float_parse_evidence':json.dumps(chosen,ensure_ascii=False),
                    'public_float_basis_status':'explicit_initial_issuer_ratio_verified'}
        if ratio_only or scope_rejections or unverified_grids:
            return {'public_float_shares':None,'public_float_ratio_disclosed':None,
                    'public_float_parse_method':'conflicting_float_ratios_review_required',
                    'public_float_parse_evidence':json.dumps({'rule':'initial_float_scope_requires_review_v1',
                        'ratio_candidates':ratio_only,'scope_rejections':scope_rejections,
                        'unverified_float_grids':unverified_grids},ensure_ascii=False),
                    'public_float_basis_status':'initial_float_scope_unverified'}
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
