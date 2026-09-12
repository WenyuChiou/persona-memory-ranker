import copy
import pytest
from scripts.selector_study_checks import verify_blind_cases


def fixture():
    originals = {}
    blinded = []
    key = []
    for i in range(2):
        case = dict(case_id=f'test-{i:03d}', query=f'Question {i}', trait='openness_high', kind='persona')
        responses = {m: {'text': f'Output {i} {m}'} for m in 'ABCD'}
        originals[case['case_id']] = dict(case=case, responses=responses)
        answers = []
        for j, method in enumerate('CADB', 1):
            tag = f'Answer {j}'
            answers.append(dict(answer_id=tag, text=responses[method]['text']))
            key.append(dict(case_id=case['case_id'], answer_id=tag, method=method))
        blinded.append(dict(case, answers=answers))
    return blinded, key, originals


def test_exact_blind_export_and_mutated_identity():
    blinded, key, originals = fixture()
    assert verify_blind_cases(blinded, key, originals)
    blinded[1] = copy.deepcopy(blinded[0])
    with pytest.raises(AssertionError, match='identity'):
        verify_blind_cases(blinded, key, originals)


@pytest.mark.parametrize('field', ['query', 'trait', 'kind'])
def test_blinded_metadata_mutation(field):
    blinded, key, originals = fixture()
    blinded[0][field] = 'incorrect metadata'
    with pytest.raises(AssertionError, match='metadata changed'):
        verify_blind_cases(blinded, key, originals)


def test_duplicate_answer_and_private_key():
    blinded, key, originals = fixture()
    blinded[0]['answers'].append(copy.deepcopy(blinded[0]['answers'][0]))
    with pytest.raises(AssertionError, match='four unique'):
        verify_blind_cases(blinded, key, originals)
    blinded, key, originals = fixture()
    key.append(copy.deepcopy(key[0]))
    with pytest.raises(AssertionError, match='Duplicate'):
        verify_blind_cases(blinded, key, originals)
