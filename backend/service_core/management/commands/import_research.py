import json
import sys
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from service_core.models import Notification, Offering, Watchlist


class Command(BaseCommand):
    help = 'Import the pinned research cohort as historical events, never as released predictions.'

    def handle(self, *args, **options):
        sys.path.insert(0, str(Path(settings.BASE_DIR).parent / 'ipo_predictor'))
        from data.pipelines.data_delivery import checked_artifact, current_financial_repair
        processed = Path(settings.IPO_PROCESSED_DIR)
        pointer_bytes = (processed / 'current_research_data.json').read_bytes()
        repair = current_financial_repair(processed)
        if (processed / 'current_research_data.json').read_bytes() != pointer_bytes:
            raise ValueError('research_version_changed_during_import')
        pointer = json.loads(pointer_bytes)
        snapshot = json.loads((processed / pointer['manifest']).read_text())
        research = pd.read_parquet(processed / snapshot['research_dataset'] / 'dataset.parquet')
        raw = pd.read_parquet(repair / 'features_all.parquet')
        imported = research[['event_id']].merge(raw, on='event_id', validate='one_to_one')
        checked_artifact(repair)
        checked_artifact(processed / snapshot['research_dataset'])
        if (processed / 'current_research_data.json').read_bytes() != pointer_bytes:
            raise ValueError('research_version_changed_during_import')
        if len(imported) != snapshot['rows']:
            raise ValueError('research_import_event_count_mismatch')
        version = snapshot['version']
        created = updated = 0
        with transaction.atomic():
            for row in imported.itertuples(index=False):
                if row.event_class != 'general_ipo' or row.market not in {'KOSPI', 'KOSDAQ'}:
                    raise ValueError('research_import_scope_mismatch')
                fields = {'company_name': row.corp_name, 'ticker': str(getattr(row, 'ticker', '') or ''),
                          'market': row.market, 'listing_date': pd.Timestamp(row.listing_date).date(),
                          'offering_type': 'general_ipo', 'data_version': version, 'prediction_status': 'not_released'}
                previous = Offering.objects.filter(event_id=row.event_id).first()
                if previous and all(getattr(previous, k) == v for k, v in fields.items()):
                    continue
                offering, is_new = Offering.objects.update_or_create(event_id=row.event_id, defaults=fields)
                created += int(is_new)
                updated += int(not is_new)
                if not is_new:
                    for item in Watchlist.objects.filter(offering=offering, user__notification_enabled=True):
                        Notification.objects.get_or_create(user=item.user, deduplication_key=f'event:{offering.pk}:{version}',
                            defaults={'message': f'{offering.company_name} data updated'})
        self.stdout.write(json.dumps({'created': created, 'updated': updated, 'data_version': version,
                                      'predictions_released': False}))
