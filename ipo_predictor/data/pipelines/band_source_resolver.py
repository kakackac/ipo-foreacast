"""Keep a currency band paired and bind amendments to their own official price."""
import ast
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from requests import RequestException


BAND_FIELDS = ('price_band_low', 'price_band_high', 'price_band_parse_method',
               'price_band_parse_evidence', 'price_band_validation_status')


def currency_band(document):
    if (document.get('price_band_validation_status') != 'direct_currency_band'
            or document.get('price_band_parse_method') != 'direct_currency_range_v1'):
        return None
    try:
        pair = tuple(float(document[field]) for field in BAND_FIELDS[:2])
        evidence = json.loads(document['price_band_parse_evidence'])
        if not isinstance(evidence,list) or not evidence or not all(
                isinstance(text,str) and re.search(r'원|KRW',text,re.IGNORECASE) for text in evidence):
            return None
        observed = {tuple(float(n.replace(',', '')) for n in re.findall(r'\d[\d,]*', text))
                    for text in evidence}
    except (KeyError, TypeError, ValueError):
        return None
    return pair if all(math.isfinite(v) and v > 0 for v in pair) and pair[1] >= pair[0] and observed == {pair} else None


def confirmed_price(document, price):
    try:
        valid = not isinstance(price,bool) and math.isfinite(float(price)) and price > 0
    except (TypeError,ValueError):
        valid = False
    return (valid and document.get('offering_price_finality') == 'confirmed_price_language'
            and document.get('offering_price_review_status') == 'verified_currency_unit'
            and document.get('offering_price') == price)


def verified_after_view(view):
    if not isinstance(view,dict):return False
    if view.get('rule')=='explicit_marker_before_after_sections_v1':return True
    if view.get('rule')!='explicit_overview_empty_before_additions_v1':return False
    notes=view.get('empty_before_notes');pairs=view.get('pair_count');original=view.get('original_paired_sections')
    if (not isinstance(notes,dict) or not notes or type(pairs) is not int or type(original) is not int
            or original<0 or pairs!=original+len(notes)
            or view.get('strict_section_rejection')!='correction_after_without_matching_before'):
        return False
    if not all(isinstance(view.get(k),str) and re.fullmatch(r'[a-f0-9]{64}',view[k])
               for k in ['original_sha256','selected_sha256','empty_before_expansion_sha256']):return False
    for note,proof in notes.items():
        if not re.fullmatch(r'[1-9][0-9]*',str(note)) or not isinstance(proof,dict):return False
        cells=proof.get('cells',[])
        if (len(cells)!=3 or re.sub(r'\s+','',cells[1])!='-'
                or re.sub(r'\s+','',cells[2])!='주'+str(note)+')'
                or any(type(proof.get(k)) is not int or proof[k]<0 for k in ['table','row'])):return False
    return True


def superseded_terms_evidence(day, chosen):
    active=[item for item in day if '증권신고서' in item[0].get('report_nm','')
        and '지분증권' in item[0].get('report_nm','')
        and item[0]['rcept_no'] in chosen[2].get('related_receipts',[])
        and chosen[0]['rcept_no'] in item[2].get('related_receipts',[])
        and item[2]['corp_code']==chosen[2]['corp_code']]
    if len(active)!=1 or '투자설명서' not in chosen[0].get('report_nm',''):return {}
    latest=active[0];result={}
    for m,body,proof in sorted(day,key=lambda item:item[0]['rcept_no']):
        if (proof.get('content_scope','full')!='full' and '발행조건확정' in m.get('report_nm','')
                and m['rcept_no']<min(latest[0]['rcept_no'],chosen[0]['rcept_no'])
                and m['rcept_no'] not in chosen[2].get('related_receipts',[])
                and proof['corp_code']==chosen[2]['corp_code']
                and {latest[0]['rcept_no'],chosen[0]['rcept_no']}.issubset(proof.get('related_receipts',[]))):
            result[m['rcept_no']]={'receipt':m['rcept_no'],'replacement_receipt':latest[0]['rcept_no'],
                'complete_receipt':chosen[0]['rcept_no'],'published_at':m['rcept_dt'],
                'rule':'same_day_explicit_active_related_terms_pair_v1',
                'prior_related_receipts':proof['related_receipts'],
                'complete_related_receipts':chosen[2]['related_receipts']}
    return result


def select_notice_band(complete, notices, price, *, complete_receipt=None):
    """A verified after-notice can fill an absent band, never override a conflict."""
    checks = []
    pair = currency_band(complete)
    for item in sorted(notices, key=lambda x: x['receipt'], reverse=True):
        document = item['document']
        check = {'receipt':item['receipt'], 'linked':item['linked'],
                 'explicit_after_view':bool(item.get('view')), 'band':currency_band(document)}
        checks.append(check)
        if not (item['linked'] and verified_after_view(item.get('view'))
                and confirmed_price(complete, price) and confirmed_price(document, price)):
            check['status'] = 'notice_link_view_or_confirmed_price_unverified'
            continue
        if document.get('price_band_validation_status') == 'conflicting_bands_review_required':
            if pair is not None and complete_receipt and complete_receipt > item['receipt']:
                check['status'] = 'later_complete_band_retained_notice_parser_ambiguous'
                return complete, None, checks
            check['status'] = 'notice_band_conflict'
        elif check['band'] is None:
            check['status'] = 'notice_has_no_verified_currency_band'
            continue
        elif pair == check['band']:
            check['status'] = 'same_band_complete_document_retained'
            return complete, None, checks
        elif (pair is None and complete.get('price_band_validation_status') != 'conflicting_bands_review_required'
              and all(complete.get(f) is None for f in BAND_FIELDS[:2])):
            check['status'] = 'verified_after_notice_fills_absent_complete_band'
            return {**complete, **{f:document.get(f) for f in BAND_FIELDS}}, item, checks
        else:
            check['status'] = 'same_day_official_band_conflict'
        held = {**complete, **dict.fromkeys(BAND_FIELDS)}
        held['price_band_validation_status'] = 'conflicting_bands_review_required'
        return held, None, checks
    return complete, None, checks


def historical_indicative_band(chosen, family, load, analyse, cutoff):
    """Retain the last pre-bookbuilding range only through a checked filing lineage."""
    from data.pipelines.stage_snapshots import available, local, demand_schedule, amendment_view, added_note_after_view

    meta,body,origin=chosen
    evidence={'rule':'pre_bookbuilding_band_checked_lineage_v1','checks':[]}
    donor=None
    try:
        if available(meta)>=local(cutoff):raise ValueError('historical_band_final_document_not_available_at_cutoff')
        start,end,schedule=demand_schedule(body)
        pre_cutoff=local(start).normalize()-pd.Timedelta(hours=24 if 'scoped_windows' in schedule else 3)
        ordered=sorted({m['rcept_no']:m for m in family if available(m)<local(cutoff)}.values(),
                       key=lambda m:(m['rcept_dt'],m['rcept_no']))
        terms=[m for m in ordered if re.search(r'증권신고서.*지분증권|투자설명서',m['report_nm'])]
        eligible=[m for m in terms if available(m)<pre_cutoff]
        if not eligible:raise ValueError('historical_band_pre_period_source_missing')
        candidate=eligible[-1]
        if any(re.search(r'철회|취소',m['report_nm']) and m['rcept_dt']>=candidate['rcept_dt'] for m in ordered):
            raise ValueError('historical_band_withdrawal_crosses_lineage')
        earlier_body,earlier_origin=load(candidate)
        if (earlier_origin.get('content_scope','full')!='full'
                or earlier_origin['corp_code']!=origin['corp_code']
                or meta['rcept_no'] not in earlier_origin.get('related_receipts',[])):
            raise ValueError('historical_band_pre_period_identity_or_scope_unverified')
        announced_start,announced_end,announced_schedule=demand_schedule(earlier_body)
        if (announced_start,announced_end)!=(start,end):
            raise ValueError('historical_band_demand_round_conflict')
        parsed=analyse(candidate,earlier_body,origin['corp_code'])['offering']
        pair=currency_band(parsed)
        if pair is None:raise ValueError('historical_band_latest_pre_period_range_unverified')
        evidence.update(candidate_receipt=candidate['rcept_no'],candidate_origin=earlier_origin,
            demand_start=str(start.date()),demand_end=str(end.date()),
            pre_cutoff=pre_cutoff.isoformat(),announced_schedule=announced_schedule)
        # Check the later full documents and every non-superseded after-view.
        # A scalar final price is not an indicative range and cannot replace it.
        days={}
        for m in terms:
            if (m['rcept_dt'],m['rcept_no'])<(candidate['rcept_dt'],candidate['rcept_no']):continue
            b,p=load(m);days.setdefault(m['rcept_dt'],[]).append((m,b,p))
        for day in days.values():
            selected=next((item for item in day if item[0]['rcept_no']==meta['rcept_no']),None)
            if selected is None:
                full=[item for item in day if item[2].get('content_scope','full')=='full'
                      and '투자설명서' in item[0].get('report_nm','')]
                selected=max(full,key=lambda item:item[0]['rcept_no']) if full else None
            superseded=superseded_terms_evidence(day,selected) if selected else {}
            for m,b,p in day:
                check={'receipt':m['rcept_no'],'origin':p};evidence['checks'].append(check)
                if p['corp_code']!=origin['corp_code']:raise ValueError('historical_band_company_conflict')
                if m['rcept_no']!=meta['rcept_no'] and meta['rcept_no'] not in p.get('related_receipts',[]):
                    raise ValueError('historical_band_lineage_not_linked')
                if m['rcept_no'] in superseded:
                    check.update(status='explicitly_superseded_terms',supersession=superseded[m['rcept_no']]);continue
                view=None
                if p.get('content_scope','full')!='full':
                    try:b,view=amendment_view(b)
                    except ValueError as exc:
                        if str(exc)!='correction_after_without_matching_before':raise
                        b,view=added_note_after_view(b);view['strict_section_rejection']=str(exc)
                    if not verified_after_view(view):raise ValueError('historical_band_after_view_unverified')
                current=analyse(m,b,p['corp_code'])['offering'];current_pair=currency_band(current)
                check.update(band=current_pair,after_view=view)
                if (current.get('price_band_validation_status')=='conflicting_bands_review_required'
                        or any(current.get(f) is not None for f in BAND_FIELDS[:2]) and current_pair is None):
                    raise ValueError('historical_band_later_range_ambiguous')
                if current_pair is not None and current_pair!=pair:
                    raise ValueError('historical_band_later_range_conflict')
                if p.get('content_scope','full')=='full':
                    later_start,later_end,_=demand_schedule(b)
                    if (later_start,later_end)!=(start,end):raise ValueError('historical_band_demand_round_conflict')
                check['status']='same_indicative_range' if current_pair else 'no_replacement_range_in_checked_document'
        donor={'metadata':candidate,'origin':earlier_origin,'document':parsed,'view':None,
               'history':evidence}
        evidence['status']='verified_historical_indicative_band'
    except ValueError as exc:
        evidence.update(status='historical_band_review_required',reason=str(exc))
        if getattr(exc,'url',None):evidence['failed_url']=exc.url
    except RequestException:
        evidence.update(status='historical_band_review_required',reason='source_access_failed')
    return donor,evidence


def resolve_offering_band(raw, metadata, proof, price, listing_date, dart, *,
                          parsed_documents=None, primary_body=None):
    from data.pipelines.lockup_reconciliation import documents, original_body
    from data.pipelines.stage_snapshots import amendment_view, added_note_after_view, document_analysis

    record = {'rcept_no':metadata['rcept_no'], 'corp_code':proof['corp_code'],
              'offering_price':price}
    cutoff = pd.Timestamp(listing_date).normalize()-pd.Timedelta(hours=3)
    parsed = []
    admitted = documents(Path(raw), record, proof) if parsed_documents is None else parsed_documents
    for meta, origin, body in admitted:
        if pd.Timestamp(meta['rcept_dt'])+pd.Timedelta(days=1) >= cutoff:
            raise RuntimeError('band_source_not_available_before_cutoff')
        document = (document_analysis(dart,body,meta['rcept_no'],str(proof['corp_code']),raw)['offering']
                    if parsed_documents is None else body)
        parsed.append((meta,origin,document))
    parsed.sort(key=lambda x:(x[0]['rcept_dt'],x[0]['rcept_no']),reverse=True)
    if not parsed:
        raise ValueError('band_complete_document_missing')
    base = parsed[0]
    for candidate in parsed:
        if candidate[2].get('price_band_validation_status') == 'conflicting_bands_review_required':
            base = candidate
            break
        if currency_band(candidate[2]) is not None and confirmed_price(candidate[2],price):
            base = candidate
            break
    meta, origin, document = base
    notices = []
    extra = proof.get('supplementary', {})
    same_day = (proof.get('content_scope') == 'partial_or_terms_notice'
        and not proof.get('correction_selection')
        and proof.get('supplementary_status') == 'official_related_prospectus_verified'
        and proof.get('corp_code') == extra.get('corp_code')
        and meta['rcept_dt'] == metadata['rcept_dt'])
    view_failure = None
    if same_day:
        try:
            original=original_body(Path(raw),proof) if primary_body is None else primary_body
            try:body,view=amendment_view(original)
            except ValueError as exc:
                if str(exc)!='correction_after_without_matching_before':raise
                body,view=added_note_after_view(original)
                view['strict_section_rejection']=str(exc)
            notice = (document_analysis(dart,body,metadata['rcept_no'],str(proof['corp_code']),raw)['offering']
                      if parsed_documents is None else dart._parse_offering_html(body,metadata['rcept_no']))
            notices.append({'receipt':metadata['rcept_no'], 'document':notice, 'origin':proof,
                            'metadata':metadata, 'view':view, 'linked':True})
        except ValueError as exc:
            view_failure = str(exc)
    selected, donor, checks = select_notice_band(document,notices,price,complete_receipt=meta['rcept_no'])
    history=None
    if (currency_band(selected) is None and confirmed_price(selected,price)
            and selected.get('price_band_validation_status')!='conflicting_bands_review_required'
            and all(selected.get(f) is None for f in BAND_FIELDS[:2])):
        from data.pipelines.stage_snapshots import read_document
        family={m['rcept_no']:m for bundle in (proof,extra) for m in bundle.get('family',[])}
        def load(m):return read_document(dart,raw,m,str(proof['corp_code']),True)
        analyse=lambda m,b,c:document_analysis(dart,b,m['rcept_no'],c,raw)
        try:
            b,p=load(meta)
            donor,history=historical_indicative_band((meta,b,p),family.values(),load,analyse,cutoff)
        except (ValueError,RequestException) as exc:
            history={'rule':'pre_bookbuilding_band_checked_lineage_v1','checks':[],
                'status':'historical_band_review_required',
                'reason':str(exc) if isinstance(exc,ValueError) else 'source_access_failed'}
            if getattr(exc,'url',None):history['failed_url']=exc.url
        if donor:
            selected={**selected,**{f:donor['document'].get(f) for f in BAND_FIELDS}}
        support=[]
        for source in [history.get('candidate_origin'),*[c.get('origin') for c in history['checks']]]:
            if source:
                support.extend([{'url':source['source_url'],'sha256':source['index_sha256']},*source['sections']])
        proof['band_support_sources']=list({s['url']:s for s in support}.values())
    result = dict.fromkeys((*BAND_FIELDS,'price_band_rcept_no','price_band_rcept_dt',
                           'price_band_source_url','price_band_source_sha256'))
    valid = currency_band(selected) is not None and confirmed_price(selected,price)
    result['price_band_resolution_status'] = ('verified_currency_band_source_selected' if valid else
        'latest_band_conflict_requires_review' if selected.get('price_band_validation_status') == 'conflicting_bands_review_required'
        else 'no_verified_currency_band_in_admitted_documents')
    if valid:
        if donor: meta,origin = donor['metadata'],donor['origin']
        result.update({f:selected.get(f) for f in BAND_FIELDS})
        result.update(price_band_rcept_no=meta['rcept_no'],price_band_rcept_dt=meta['rcept_dt'],
            price_band_source_url=origin['source_url'],
            price_band_source_sha256='|'.join(sorted(s['sha256'] for s in origin['sections'])))
    else:
        result['price_band_validation_status'] = result['price_band_resolution_status']
    result['price_band_resolution_evidence'] = json.dumps({'checks':checks,
        'notice_view_failure':view_failure,'selected_after_view':donor['view'] if donor else None,
        'complete_receipt':base[0]['rcept_no'],'complete_band':currency_band(document),
        'confirmed_price':price,'historical_band':history},ensure_ascii=False)
    return result


def refresh_cached_band(raw_dir, processed_dir, dart=None):
    """Migrate only bands from pinned originals; preserve every other raw field."""
    from data.collectors.dart_collector import DARTCollector
    from data.pipelines.data_delivery import checked_artifact, digest
    from data.pipelines.financial_history import atomic_json
    from data.pipelines.historical_reconciliation import connection_contract,feature_tasks,load_inputs
    from data.pipelines.historical_event_recovery import capture_sources
    from data.pipelines.reconciled_dataset import pin_connections

    raw,processed = Path(raw_dir).resolve(),Path(processed_dir).resolve()
    if not (processed/'current_historical_source_connections.json').exists(): return None
    parent,report = pin_connections(processed)
    previous = report.get('replay_contract',report['inputs']['contract'])
    contract = connection_contract()
    if previous == contract: return None
    owned = {'band','adapter','record','resolver'}
    if {k:v for k,v in previous.items() if k not in owned} != {k:v for k,v in contract.items() if k not in owned}:
        return None
    baseline = 'e9d1595'
    if 'band' not in previous:
        for key,name in [('adapter','historical_reconciliation.py'),('record','historical_event_recovery.py'),
                         ('resolver','event_disclosures.py')]:
            original = subprocess.check_output(['git','show',baseline+':ipo_predictor/data/pipelines/'+name],
                                               cwd=Path(__file__).parents[3])
            if hashlib.sha256(original).hexdigest() != previous[key]: return None
    else:
        if previous.get('record') != contract['record']: return None
        if previous.get('resolver') != contract['resolver']:
            from data.pipelines.lockup_reconciliation import parser_snapshot
            original=parser_snapshot(raw,previous['resolver'])
            if original is None:return None
            old_tree=ast.parse(original)
            tree=ast.parse(Path(__file__).with_name('event_disclosures.py').read_text())
            target=ast.parse("urls.update(source['url'] for source in proof.get('band_support_sources',[]))").body[0]
            removed=0
            for function in tree.body:
                if isinstance(function,ast.FunctionDef) and function.name=='connect_event':
                    before=len(function.body)
                    function.body=[n for n in function.body if ast.dump(n,include_attributes=False)!=ast.dump(target,include_attributes=False)]
                    removed+=before-len(function.body)
            if removed!=1 or ast.dump(old_tree,include_attributes=False)!=ast.dump(tree,include_attributes=False):return None
        if previous['adapter'] != contract['adapter']:
            from data.pipelines.lockup_reconciliation import parser_snapshot
            source = parser_snapshot(raw,previous['adapter'])
            def outside_contract(text):
                tree=ast.parse(text)
                tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='connection_contract')]
                return ast.dump(tree,include_attributes=False)
            if source is None or outside_contract(source) != outside_contract(
                    Path(__file__).with_name('historical_reconciliation.py').read_text()): return None
    inputs = load_inputs(raw,processed)
    if inputs['input_sha256'] != report['inputs']['inputs']: return None
    rows = pd.read_parquet(parent/'connection_results.parquet')
    general = set(inputs['events'].loc[inputs['events'].event_class.eq('general_ipo'),'event_id'])
    if set(rows.event_id) != general or rows.event_id.duplicated().any() or not rows.status.eq('official_sources_connected').all():
        return None
    identity = {'parent_manifest':str((parent/'manifest.json').relative_to(processed)),
        'parent_sha256':digest(parent/'manifest.json'),'contract':contract,'repair':digest(__file__),
        'scope':'band_fields_only_original_identity_prices_and_other_features_unchanged',
        'initial_contract_baseline':baseline if 'band' not in previous else None}
    version = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    checkpoints = raw/'band_field_repairs'/version
    dart = dart or DARTCollector(api_key='unused')
    audit = [];archive={}
    for index,row in rows.iterrows():
        old,proof = json.loads(row.record),json.loads(row.proof)
        capture_sources(raw,proof.get('sources',[]),{})
        path = checkpoints/(hashlib.sha256(row.event_id.encode()).hexdigest()+'.json')
        if path.exists():
            saved = json.loads(path.read_text());payload = saved['payload']
            if saved['event_id'] != row.event_id or saved['sha256'] != hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():
                raise RuntimeError('band_field_checkpoint_integrity_failed')
        else:
            metadata = next(m for m in proof['family'] if m['rcept_no'] == old['rcept_no'])
            resolved = resolve_offering_band(raw,metadata,proof,old['offering_price'],old['listing_date'],dart)
            sources={s['url']:s for s in proof.get('sources',[])}
            for source in proof.get('band_support_sources',[]):
                cache=raw/'dart_connection_sources'/(hashlib.sha256(source['url'].encode()).hexdigest()+'.json')
                saved=json.loads(cache.read_text())
                if saved['url']!=source['url'] or hashlib.sha256(saved['text'].encode()).hexdigest()!=source['sha256']:
                    raise RuntimeError('band_support_source_integrity_failed')
                sources[source['url']]={'url':source['url'],'sha256':source['sha256'],'collected_at':saved.get('collected_at')}
            proof['sources']=sorted(sources.values(),key=lambda s:s['url'])
            updated = {k:v for k,v in old.items() if not k.startswith('price_band_')}
            updated.update(resolved)
            proof['resolved_band'] = resolved
            for field in BAND_FIELDS[:2]:
                proof.setdefault('field_sources',{}).pop(field,None)
                if resolved['price_band_rcept_no'] is not None:
                    proof['field_sources'][field] = {'receipt':resolved['price_band_rcept_no'],
                        'source_url':resolved['price_band_source_url'],'available_at':(
                            pd.Timestamp(resolved['price_band_rcept_dt']).tz_localize('Asia/Seoul')+pd.Timedelta(days=1)).isoformat()}
            payload = {'record':updated,'proof':proof}
            atomic_json(path,{'event_id':row.event_id,'payload':payload,
                'sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        updated = payload['record']
        old_urls={s['url'] for s in json.loads(row.proof).get('sources',[])}
        capture_sources(raw,[s for s in payload['proof'].get('sources',[]) if s['url'] not in old_urls],archive)
        if {k:v for k,v in old.items() if not k.startswith('price_band_')} != {
                k:v for k,v in updated.items() if not k.startswith('price_band_')}:
            raise RuntimeError('band_field_unrelated_value_changed')
        rows.at[index,'record'] = json.dumps(updated,ensure_ascii=False)
        rows.at[index,'proof'] = json.dumps(payload['proof'],ensure_ascii=False)
        audit.append({'event_id':row.event_id,'corp_name':row.corp_name,'before_low':old.get('price_band_low'),
            'before_high':old.get('price_band_high'),'after_low':updated.get('price_band_low'),
            'after_high':updated.get('price_band_high'),'resolution_status':updated['price_band_resolution_status'],
            'receipt':updated.get('price_band_rcept_no'),'unrelated_fields_unchanged':True,
            'evidence':updated.get('price_band_resolution_evidence')})
        if len(audit)%100 == 0: print(json.dumps({'band_reprocessed':len(audit),'population':len(rows)}),flush=True)
    if contract != connection_contract() or identity['repair'] != digest(__file__) or inputs['input_sha256'] != load_inputs(raw,processed)['input_sha256']:
        raise RuntimeError('band_field_inputs_changed')
    destination = processed/'historical_reconciliations'/version
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix='.band.',dir=destination.parent))
        rows.to_parquet(staging/'connection_results.parquet',index=False)
        pd.read_parquet(parent/'events.parquet').to_parquet(staging/'events.parquet',index=False)
        feature_tasks(rows.to_dict('records')).to_parquet(staging/'feature_tasks.parquet',index=False)
        pd.DataFrame(audit).to_parquet(staging/'band_reprocessing.parquet',index=False)
        atomic_json(staging/'source_documents.json',archive)
        amended = {**report,'version':version,'amendment':identity,'replay_contract':contract,
            'source_archive_policy':'parent_manifest_archive_plus_amendment_originals',
            'band_field_reprocessing':{'processed':len(rows),'unrelated_fields_unchanged':True,
                'status_counts':pd.DataFrame(audit).resolution_status.value_counts().to_dict()},
            'output_sha256':{f.name:digest(f) for f in staging.iterdir()}}
        atomic_json(staging/'manifest.json',amended);staging.rename(destination)
    atomic_json(processed/'current_historical_source_connections.json',{
        'manifest':str((destination/'manifest.json').relative_to(processed)),'sha256':digest(destination/'manifest.json')})
    return {**checked_artifact(destination),'path':str(destination)}
