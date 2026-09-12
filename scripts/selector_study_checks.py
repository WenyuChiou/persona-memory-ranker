"""Pure checks for identity and metadata in a blinded answer export."""


def verify_blind_cases(blinded, key_rows, originals):
    ids = [case['case_id'] for case in blinded]
    assert len(ids) == len(set(ids)) and set(ids) == set(originals), 'Blinded case identity mismatch'
    key_ids = [(r['case_id'], r['answer_id']) for r in key_rows]
    assert len(key_ids) == len(set(key_ids)) == 4 * len(originals), 'Duplicate or missing private key'
    key = {identity: row['method'] for identity, row in zip(key_ids, key_rows)}
    tags = {f'Answer {i}' for i in range(1, 5)}
    assert set(key) == {(identifier, tag) for identifier in originals for tag in tags}, 'Private key identity mismatch'
    for case in blinded:
        assert set(case) == {'case_id', 'query', 'trait', 'kind', 'answers'}, 'Blinded fields include unexpected metadata'
        original = originals[case['case_id']]
        assert all(case[k] == original['case'][k] for k in ('case_id', 'query', 'trait', 'kind')), 'Blinded query/persona metadata changed'
        answers = case['answers']
        assert len(answers) == 4 and {a['answer_id'] for a in answers} == tags, 'Exactly four unique answers required'
        methods = []
        for answer in answers:
            assert set(answer) == {'answer_id', 'text'}, 'Answer contains method or retrieval metadata'
            method = key[(case['case_id'], answer['answer_id'])]
            methods.append(method)
            assert answer['text'] == original['responses'][method]['text'], 'Blinded answer text changed'
        assert set(methods) == {'A', 'B', 'C', 'D'}, 'Private key must map all four methods once'
    return True
