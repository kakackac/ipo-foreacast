import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.pipelines import stage_snapshots as stages
from data.pipelines.data_delivery import digest
from data.pipelines.financial_history import atomic_json
from features.model_profiles import FEATURE_SOURCE_APPROVAL, get_model_profile


class StageModelInputTests(unittest.TestCase):
    def fixture(self):
        rows=[];observations=[]
        for stage in ('pre_demand','post_demand'):
            row={'event_id':'event','corp_code':'00000001','corp_name':'Issuer',
                'event_class':'general_ipo','market':'KOSDAQ','listing_date':'2024-04-01',
                'prediction_stage':stage,'status':'historical_document_selected',
                'prediction_at':'2024-02-01T21:00:00+09:00' if stage=='pre_demand' else '2024-02-20T21:00:00+09:00',
                'prediction_time_source':'historical_stage_documents_v1',
                'demand_start_at':'2024-02-02T00:00:00+09:00',
                'demand_start_time_source':'official_schedule',
                'demand_result_available_at':'2024-02-20T00:00:00+09:00',
                'demand_result_time_is_effective':True,'demand_result_time_source':'official_result',
                'offering_price':None if stage=='pre_demand' else 10000,
                'target_offering_price':10000,'offering_price_review_status':'verified_currency_unit',
                'price_target_validation_status':'official_price_verified',
                'open_return_pct':10.0,'close_return_pct':5.0}
            for field in stages.stage_collection_fields():
                value=1.0
                if field in {'major_shareholder_lockup_months','risk_factor_count','recent_ipo_avg_return_sector'}:
                    value=None
                if stage=='pre_demand' and field in {'institutional_demand_ratio','lockup_commitment_ratio',
                        'offering_price_band_position','band_exceeded'}:value=None
                row[field]=value
                contract=FEATURE_SOURCE_APPROVAL.get(field)
                status=next(iter(contract[1])) if contract and contract[1] else 'verified_stage_asof_input_v1'
                observations.append(stages.stage_observation(row,field,value,{
                    'source_url':'https://dart.fss.or.kr/official','available_at':'2024-01-20T00:00:00+09:00',
                    'validation_status':status},{}))
            rows.append(row)
        return pd.DataFrame(rows),pd.DataFrame(observations)

    def test_profiles_include_contemporaneous_band_financials_and_post_inherits_pre(self):
        pre=get_model_profile('pre_demand');post=get_model_profile('post_demand')
        self.assertTrue({'price_band_low','price_band_high','revenue_growth_3y','operating_margin','debt_ratio'}
                        .issubset(pre.feature_names))
        self.assertTrue(set(pre.feature_names).issubset(post.feature_names))
        self.assertEqual(len(pre.feature_names),len(set(pre.feature_names)))
        self.assertNotIn('offering_price_band_position',pre.feature_names)
        self.assertNotIn('institutional_demand_ratio',pre.feature_names)
        self.assertNotIn('target_offering_price',pre.feature_names)
        self.assertNotIn('retail_subscription_ratio',post.feature_names)

    def test_materialization_connects_features_without_copying_post_results_to_pre(self):
        frame,audit=self.fixture()
        result=stages.build_stage_model_inputs(frame,audit)
        pre=result['pre_demand'];post=result['post_demand']
        self.assertNotIn('institutional_demand_ratio',pre)
        self.assertNotIn('offering_price',pre)
        self.assertIn('target_offering_price',pre)
        for stage,table in result.items():
            self.assertTrue(table.stage_model_candidate.all())
            self.assertTrue(set(get_model_profile(stage).feature_names).issubset(table))
            self.assertEqual(table.loc[0,'revenue_growth_3y'],1.0)
        self.assertEqual(pre.loc[0,'open_return_pct'],post.loc[0,'open_return_pct'])

    def test_optional_financial_missing_stays_null_and_does_not_block(self):
        frame,audit=self.fixture();frame['revenue_growth_3y']=None
        audit.loc[audit.feature_name.eq('revenue_growth_3y'),['value','is_missing','validation_status']]=[None,True,'unverified_or_missing']
        result=stages.build_stage_model_inputs(frame,audit)
        for table in result.values():
            self.assertTrue(table.revenue_growth_3y.isna().all())
            self.assertTrue(table.revenue_growth_3y__missing.all())
            self.assertTrue(table.stage_model_candidate.all())

    def test_unapproved_financial_or_raw_band_is_not_a_valid_model_input(self):
        for field in ('debt_ratio','price_band_low'):
            frame,audit=self.fixture()
            audit.loc[audit.feature_name.eq(field),'validation_status']='legacy_unverified'
            result=stages.build_stage_model_inputs(frame,audit)
            self.assertTrue(all(not table.stage_model_candidate.any() for table in result.values()))

    def test_financial_available_after_cutoff_blocks_only_that_stage(self):
        frame,audit=self.fixture()
        audit.loc[audit.feature_name.eq('operating_margin'),'available_at']='2024-02-10T00:00:00+09:00'
        result=stages.build_stage_model_inputs(frame,audit)
        self.assertFalse(result['pre_demand'].stage_model_candidate.any())
        self.assertTrue(result['post_demand'].stage_model_candidate.all())

    def test_incomplete_duplicate_or_changed_observation_lattice_cannot_publish_inputs(self):
        for kind in ('missing','duplicate','changed'):
            frame,audit=self.fixture()
            if kind=='missing':audit=audit.iloc[1:]
            elif kind=='duplicate':audit=pd.concat([audit,audit.iloc[:1]],ignore_index=True)
            else:audit.loc[0,'value']=999
            with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'stage_input_'):
                stages.build_stage_model_inputs(frame,audit)

    def test_loader_rejects_legacy_stage_file_or_changed_dependency_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            processed=Path(directory);root=processed/'stage_snapshots'/'fixture';root.mkdir(parents=True)
            pd.DataFrame({'legacy':[1]}).to_parquet(root/'stages.parquet')
            report={'version':'fixture','output_sha256':{'stages.parquet':digest(root/'stages.parquet')},
                'training_executed':False,'deployment_authorized':False,'inputs':{}}
            atomic_json(root/'manifest.json',report)
            atomic_json(processed/'current_stage_snapshots.json',{
                'manifest':str((root/'manifest.json').relative_to(processed)),'sha256':digest(root/'manifest.json')})
            with self.assertRaisesRegex(RuntimeError,'stage_input_'):
                stages.load_stage_model_inputs(processed,'pre_demand')

    def test_pre_result_contamination_is_rejected_not_silently_dropped(self):
        frame,audit=self.fixture();frame.loc[0,'institutional_demand_ratio']=100
        mask=audit.stage.eq('pre_demand') & audit.feature_name.eq('institutional_demand_ratio')
        audit.loc[mask,['value','is_missing']]=[100,False]
        with self.assertRaisesRegex(RuntimeError,'stage_input_pre_result_leakage'):
            stages.build_stage_model_inputs(frame,audit)

    def test_unapproved_market_observation_blocks_even_with_legacy_time_flag(self):
        frame,audit=self.fixture()
        audit.loc[audit.feature_name.eq('kospi_momentum_5d'),'validation_status']='legacy_unverified'
        result=stages.build_stage_model_inputs(frame,audit)
        self.assertTrue(all(not table.stage_source_valid.any() for table in result.values()))
        self.assertTrue(all(not table.stage_model_candidate.any() for table in result.values()))

    def test_pre_band_approval_does_not_require_or_approve_a_confirmed_offering_price(self):
        selected={'html':'official','metadata':{'rcept_no':'20240101000001'},'analysis':{
            'offering':{'price_band_low':9000,'price_band_high':11000,
                        'price_band_validation_status':'direct_currency_band'},
            'demand':{},'underwriter':{'underwriter_validation_status':'not_verified'}}}
        values,_=stages.values_from_document(None,selected,'00000001','pre_demand')
        self.assertEqual(values['price_band_low'],9000)
        self.assertNotIn('offering_price',values)
        self.assertNotIn('offering_price_band_position',values)
        self.assertIn(selected['field_validation']['price_band_low'],FEATURE_SOURCE_APPROVAL['price_band_low'][1])

    def test_pipeline_never_falls_back_to_existing_legacy_stage_files(self):
        from pipeline import step_load_or_build_data,TrainingReadinessError,require_training_ready
        with tempfile.TemporaryDirectory() as directory:
            processed=Path(directory);legacy=processed/'model_stage_datasets';legacy.mkdir()
            pd.DataFrame({'unsafe':[1]}).to_parquet(legacy/'pre_demand.parquet')
            with patch('config.PROC_DIR',processed),self.assertRaises(TrainingReadinessError):
                step_load_or_build_data(prediction_stage='pre_demand')
            with patch('config.PROC_DIR',processed),self.assertRaises(TrainingReadinessError):
                require_training_ready(pd.DataFrame({'unsafe':[1]}),'core','pre_demand')

    def test_feature_selection_includes_financials_and_never_selects_target_prices(self):
        from pipeline import step_feature_selection
        frame,audit=self.fixture();table=stages.build_stage_model_inputs(frame,audit)['pre_demand']
        selected=step_feature_selection(table,prediction_stage='pre_demand')
        self.assertIn('operating_margin',selected);self.assertIn('price_band_low',selected)
        self.assertNotIn('target_offering_price',selected);self.assertNotIn('offering_price',selected)
        self.assertNotIn('institutional_demand_ratio',selected)

    def test_input_schema_is_stable_across_process_hash_seeds(self):
        script=('import json; from tests.test_stage_model_inputs import StageModelInputTests; '
            'from data.pipelines.stage_snapshots import build_stage_model_inputs; '
            'f,a=StageModelInputTests().fixture(); '
            'print(json.dumps(build_stage_model_inputs(f,a)["pre_demand"].columns.tolist()))')
        schemas=[subprocess.check_output([sys.executable,'-c',script],
            env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]),'PYTHONHASHSEED':seed},text=True)
            for seed in ('1','2')]
        self.assertEqual(schemas[0],schemas[1])

    def test_verified_data_candidates_do_not_authorize_training(self):
        from pipeline import require_training_ready,TrainingReadinessError
        frame,audit=self.fixture();table=stages.build_stage_model_inputs(frame,audit)['pre_demand']
        table.attrs['stage_snapshot_reference']={'manifest':'fixed','sha256':'fixed'}
        with patch('data.pipelines.stage_snapshots.load_stage_model_inputs',return_value=(table,audit,{})), \
                self.assertRaisesRegex(TrainingReadinessError,'학습이 승인되지'):
            require_training_ready(table,prediction_stage='pre_demand')

    def test_loader_pins_inputs_and_stage_filtered_observations_to_one_verified_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            processed=Path(directory).resolve();root=processed/'stage_snapshots'/'fixture';root.mkdir(parents=True)
            finance=processed/'finance';finance.mkdir();pd.DataFrame({'fixture':[1]}).to_parquet(finance/'accounts.parquet')
            frame,audit=self.fixture();inputs=stages.build_stage_model_inputs(frame,audit)
            for stage,table in inputs.items():table.to_parquet(root/(stage+'_inputs.parquet'),index=False)
            audit.to_parquet(root/'observations.parquet',index=False)
            frame.to_parquet(root/'stages.parquet',index=False)
            report={'version':'fixture','scope':'all_connected_general_ipo','events':1,'input_policy':stages.INPUT_POLICY,
                'training_executed':False,'deployment_authorized':False,
                'input_profiles':{s:{'features':list(get_model_profile(s).feature_names),
                    'critical_features':list(get_model_profile(s).critical_features)} for s in inputs},
                'inputs':{'builder':digest(stages.__file__),'dataset':'dataset','sources':'sources',
                    'accounts':digest(finance/'accounts.parquet'),
                    'parser':digest(Path(stages.__file__).parents[1]/'collectors/dart_collector.py'),
                    'financial_document_structure':stages.financial_document_structure_contract(),
                    'dependency_sha256':{key:digest(file) for key,file in stages.stage_dependency_paths().items()}},
                'output_sha256':{f.name:digest(f) for f in root.iterdir()}}
            atomic_json(root/'manifest.json',report)
            ref={'manifest':str((root/'manifest.json').relative_to(processed)),'sha256':digest(root/'manifest.json')}
            atomic_json(processed/'current_stage_snapshots.json',ref)
            parent={'run':{'input_sha256':{'dataset':'dataset','sources':'sources'}},'paths':{'financial':finance},
                    'reports':{'sources':{'population':1}}}
            with patch('data.pipelines.reconciliation_publication.committed_reconciliation',return_value=parent):
                table,evidence,_=stages.load_stage_model_inputs(processed,'pre_demand')
                self.assertEqual(table.attrs['stage_snapshot_reference'],ref)
                self.assertTrue(evidence.stage.eq('pre_demand').all())
                self.assertNotIn('institutional_demand_ratio',evidence.feature_name.tolist())
                changed={**report['inputs'],'dependency_sha256':{**report['inputs']['dependency_sha256'],'float_tables':'old'}}
                atomic_json(root/'manifest.json',{**report,'inputs':changed})
                atomic_json(processed/'current_stage_snapshots.json',{**ref,'sha256':digest(root/'manifest.json')})
                with self.assertRaisesRegex(RuntimeError,'stage_input_parent_or_dependency_changed'):
                    stages.load_stage_model_inputs(processed,'pre_demand')
                atomic_json(root/'manifest.json',{**report,'inputs':{**report['inputs'],'builder':'old'}})
                atomic_json(processed/'current_stage_snapshots.json',{**ref,'sha256':digest(root/'manifest.json')})
                with self.assertRaisesRegex(RuntimeError,'stage_input_contract_missing_or_changed'):
                    stages.load_stage_model_inputs(processed,'pre_demand')


if __name__=='__main__':unittest.main()
