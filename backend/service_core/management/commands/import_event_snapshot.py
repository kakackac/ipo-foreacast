import json
import sys
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from service_core.models import Offering


class Command(BaseCommand):
    help = 'Import newly price-linked IPO events, without releasing predictions.'

    def add_arguments(self, parser):
        parser.add_argument('--include-disclosures',action='store_true')

    def handle(self, *args, **options):
        sys.path.insert(0, str(Path(settings.BASE_DIR).parent/'ipo_predictor'))
        from data.pipelines.event_delivery import current_event_delivery
        root, manifest = current_event_delivery(settings.IPO_PROCESSED_DIR)
        rows = pd.read_parquet(root/'added_events.parquet')
        rows = rows[rows.event_class.isin(['general_ipo','spac_ipo'])
                    & rows.price_resolution_status.isin(['official_price_verified','pending_collection'])]
        if rows.event_id.duplicated().any(): raise ValueError('event_import_duplicate_identity')
        if current_event_delivery(settings.IPO_PROCESSED_DIR)[0] != root:
            raise ValueError('event_version_changed_during_import')
        disclosures, disclosure_version = pd.DataFrame(), None
        if options.get('include_disclosures'):
            from data.pipelines.event_disclosures import current_disclosures
            disclosure_root, evidence = current_disclosures(settings.IPO_PROCESSED_DIR)
            if evidence['event_version'] != manifest['version']:
                raise ValueError('disclosures_bound_to_different_event_version')
            disclosures = pd.read_parquet(disclosure_root/'events.parquet')
            if disclosures.event_id.duplicated().any() or not set(disclosures.event_id).issubset(set(rows.event_id)):
                raise ValueError('disclosure_import_population_invalid')
            disclosures = disclosures[disclosures.status.eq('official_identity_family_and_price_verified')]
            disclosure_version = evidence['version']
            if current_disclosures(settings.IPO_PROCESSED_DIR) != (disclosure_root,evidence):
                raise ValueError('disclosure_version_changed_during_import')
        created = updated = skipped = 0
        disclosure_updated = 0
        with transaction.atomic():
            for row in rows.itertuples(index=False):
                fields = dict(company_name=row.corp_name,ticker=row.ticker,market=row.market,
                    listing_date=pd.Timestamp(row.listing_date).date(),offering_type=row.event_class,
                    data_version=manifest['version'],prediction_status=('awaiting_price_collection'
                        if row.price_resolution_status=='pending_collection' else 'awaiting_disclosure_verification'))
                existing = Offering.objects.filter(event_id=row.event_id).first()
                if existing:
                    # Do not downgrade a research or reviewed release from another pipeline.
                    if (existing.market != row.market or existing.listing_date != pd.Timestamp(row.listing_date).date()
                            or existing.ticker and existing.ticker != row.ticker):
                        raise ValueError('existing_event_trading_identity_conflict')
                    if (existing.prediction_status in {'awaiting_price_collection','awaiting_disclosure_verification'}
                            and any(getattr(existing,k)!=v for k,v in fields.items())):
                        for key,value in fields.items():setattr(existing,key,value)
                        existing.save(update_fields=[*fields,'updated_at'])
                        updated += 1
                    else: skipped += 1
                    continue
                Offering.objects.create(event_id=row.event_id, **fields)
                created += 1
            for row in disclosures.itertuples(index=False):
                existing = Offering.objects.get(event_id=row.event_id)
                if existing.ticker != row.ticker or existing.listing_date != pd.Timestamp(row.listing_date).date():
                    raise ValueError('disclosure_import_trading_identity_conflict')
                if (existing.prediction_status in {'awaiting_disclosure_verification','awaiting_feature_verification'}
                        and (existing.data_version != disclosure_version or existing.prediction_status != 'awaiting_feature_verification')):
                    existing.prediction_status = 'awaiting_feature_verification'
                    existing.data_version = disclosure_version
                    existing.save(update_fields=['prediction_status','data_version','updated_at'])
                    disclosure_updated += 1
        self.stdout.write(json.dumps({'created':created,'updated':updated,'skipped_existing':skipped,'eligible_events':len(rows),
                                     'data_version':manifest['version'],'disclosure_updated':disclosure_updated,
                                     'disclosure_version':disclosure_version,'predictions_released':False}))
