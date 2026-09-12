"""Audit completed local study outputs without generating or rating any answer."""
import json
from collections import Counter
from pathlib import Path

from persona_memory_selector.data import read_csv, dump
from persona_memory_selector.experiment import verify, response_status
from persona_memory_selector.ranking import memory_block
from selector_study_checks import verify_blind_cases

root = Path(__file__).resolve().parents[1]
frozen = verify(root)
rows = {r['row_id']: r for r in read_csv(root / 'data/processed/selector/rows.csv')}
report = {}

for split, group_count, methods in (
    ('val', 10, ('A', 'B', 'C', 'D')),
    ('test', 100, ('A', 'B', 'C', 'D')),
    ('test-diagnostics', 20, ('C_seed311', 'D_seed311', 'D_shuffled')),
):
    partition = split.split('-')[0]
    folder = root / f'artifacts/selector/responses/{split}'
    paths = sorted(folder.glob('*.json'))
    assert {p.stem for p in paths} == {f'{partition}-{i:03d}' for i in range(group_count)}, 'Incomplete case set'
    groups = set()
    strata = Counter()
    valid = invalid = largest_pack = 0
    for path in paths:
        document = json.loads(path.read_text(encoding='utf-8'))
        case = document['case']
        assert case['case_id'] == path.stem and case['group_id'] not in groups
        groups.add(case['group_id'])
        strata[(case['model'], case['kind'])] += 1
        if case['kind'] == 'persona':
            query = rows[case['row_id']]
            assert (query['split'], query['role'], query['group_id']) == (partition, 'query', case['group_id'])
            assert query['model_context'] == case['query'] and query['label'] == case['trait']
        candidates = document['candidate_ids']
        assert len(candidates) == len(set(candidates)) == 50
        for identifier in candidates:
            source = rows[identifier]
            assert source['split'] == partition and source['role'] == 'memory'
            assert source['group_id'] != case['group_id'], 'Query scenario leaked into memory bank'
        assert set(document['responses']) == set(methods)
        for method, response in document['responses'].items():
            selected = response['selected']
            assert len(selected) <= 5
            assert len({m['source_ref'] for m in selected}) == len(selected)
            if method == 'A':
                assert not selected
            for memory in selected:
                assert memory['memory_id'] in candidates
                source = rows[memory['memory_id']]
                assert memory['text'] == source['context'] + '\nTARGET: ' + source['response']
                assert memory['source_ref'] == source['source_ref']
                assert memory['memory_type'] == 'behavioral_exemplar'
            used = len(memory_block(selected).encode('utf-8'))
            assert used == response['memory_budget_upper_bound'] and used <= 2000
            largest_pack = max(largest_pack, used)
            expected_digest = frozen['models'][case['model']]['digest']
            assert response['model_digest'] == expected_digest
            cache = root / f"artifacts/selector/generation_cache/{response['cache_key']}.json"
            cached = json.loads(cache.read_text(encoding='utf-8'))
            status = response_status(cached)
            assert cached['message']['content'] == response['text']
            assert all(response[k] == value for k, value in status.items())
            valid += status['valid_completion']
            invalid += not status['valid_completion']
    if split == 'test':
        assert strata == {('primary', 'persona'): 60, ('secondary', 'persona'): 20, ('primary', 'control'): 20}
    report[split] = dict(groups=len(paths), answers=valid + invalid,
                         valid_completions=valid, invalid_completions=invalid,
                         maximum_memory_bytes=largest_pack,
                         original_sources_and_group_isolation=True)

for split, count in (('val', 10), ('test', 100)):
    blinded = json.loads((root / f'deliverables/selector/review/{split}-blinded.json').read_text(encoding='utf-8'))
    key = read_csv(root / f'artifacts/selector/review/{split}-key.csv')
    originals = {f'{split}-{i:03d}': json.loads((root / f'artifacts/selector/responses/{split}/{split}-{i:03d}.json').read_text(encoding='utf-8')) for i in range(count)}
    assert verify_blind_cases(blinded, key, originals)

report['blind_export'] = 'Exact answer text; method keys absent from blinded JSON'
report['human_ratings'] = 'Not created or judged by this audit'
dump(root / 'reports/selector/study_acceptance.json', report)
print(json.dumps(report, indent=2))
