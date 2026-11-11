"""Train or inspect research-only opening/closing return models, never release them."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import MODEL_DIR, PROC_DIR
from models.training.listing_eve import TARGETS, load_model, predict, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--research-only', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--challenger', action='store_true', help='Use a learned challenger, not the selected baseline policy')
    args = parser.parse_args()
    parent = MODEL_DIR / 'listing_eve_candidates'
    if args.action == 'train':
        result = train(PROC_DIR, parent, research_only=args.research_only)
        report = result['report']
        print(json.dumps({'run_id': result['run_id'], 'path': result['path'], 'reused': result['reused'],
            'selected_models': report['selected_models'], 'nested_selection_metrics': report['nested_selection_metrics'],
            'retrospective_development_signal': report['retrospective_development_signal'],
            'deployment_authorized': False}, ensure_ascii=False))
    else:
        if not args.run_id or not re.fullmatch('[a-f0-9]{20}', args.run_id) or args.input is None:
            parser.error('predict requires --run-id and --input')
        records = json.loads(args.input.read_text())
        if not isinstance(records, list) or not 1 <= len(records) <= 1000:
            raise ValueError('input_requires_1_to_1000_numeric_feature_records')
        frame = pd.DataFrame(records)
        outputs = {target: predict(load_model(parent / args.run_id, target, challenger=args.challenger), frame).to_dict(orient='records') for target in TARGETS}
        print(json.dumps({'predictions': outputs, 'input_source_verified': False, 'research_only': True,
                          'deployment_authorized': False}, allow_nan=False))


if __name__ == '__main__':
    main()
