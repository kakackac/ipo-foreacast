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
    candidates = []
    for ti, table in enumerate(tables):
        if not table:
            continue
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
    for item in candidates:
        numerator, denominator, ratio, *_ = item
        if not denominator or not 0 <= numerator <= denominator:
            continue
        if ratio is not None and (not 0 <= ratio <= 1 or abs(numerator/denominator-ratio) > .000051):
            continue
        valid.append(item)
    pairs = {(x[0], x[1], x[2]) for x in valid}
    if len(pairs) > 1:
        return {'public_float_shares': None, 'public_float_ratio_disclosed': None,
                'public_float_parse_method': 'conflicting_float_ratios_review_required',
                'public_float_parse_evidence': json.dumps(sorted(pairs, key=str)),
                'public_float_basis_status': 'same_table_candidates_conflict'}
    if not valid:
        return {}
    n, d, ratio, ti, fr, tr, fc, dc, table = valid[0]
    evidence = json.dumps({'rule': 'same_table_post_offering_scope_v1', 'table_index': ti,
        'float_row': fr, 'total_row': tr, 'float_column': fc, 'total_column': dc,
        'basis': 'post_offering_ordinary_or_explicit_listed_shares', 'table': table}, ensure_ascii=False)
    return {'public_float_shares': n, 'public_float_ratio_disclosed': ratio,
            'public_float_parse_method': ('disclosed_public_float_ratio_direct_context' if ratio is not None
                                          else 'disclosed_public_float_shares_direct_context'),
            'public_float_parse_evidence': evidence, 'total_post_listing_shares': d,
            'total_post_listing_shares_parse_method': 'same_table_post_offering_scope_v1',
            'total_post_listing_shares_parse_evidence': evidence,
            'total_post_listing_shares_parser_validation_status': 'structurally_verified',
            'public_float_basis_status': 'same_table_post_offering_scope_verified'}
