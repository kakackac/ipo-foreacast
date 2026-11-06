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
        created = updated = skipped = 0
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
        self.stdout.write(json.dumps({'created':created,'updated':updated,'skipped_existing':skipped,'eligible_events':len(rows),
                                     'data_version':manifest['version'],'predictions_released':False}))
