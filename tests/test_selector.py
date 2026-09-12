import numpy as np
import pytest

from persona_memory_selector.data import clean_rows
from persona_memory_selector.ranking import rank_candidates, pack, memory_block


def raw(i, text, head=None, level='high'):
    return dict(original_index=str(i), trait='openness', level=level,
                train_input=text, train_output='I would consider alternatives.',
                head=head or f'event {i}', personx='Alice', persony='Bob')


def test_transitive_duplicate_groups_never_split():
    rows, excluded, report = clean_rows([raw(1, 'Alice asks Bob'), raw(2, 'Alice asks Bob'),
                                       raw(2, 'Another question'), raw(3, 'Another question')])
    assert len({r['group_id'] for r in rows}) == 1
    assert len({r['split'] for r in rows}) == 1
    assert all('Alice' not in r['model_text'] for r in rows)


def test_missing_fields_are_not_imputed():
    x = raw(1, '')
    rows, excluded, report = clean_rows([x])
    assert not rows and len(excluded) == 1


def candidate(i, similarity=.7, probability=.8, kind='behavioral_exemplar'):
    return dict(memory_id=str(i), text='An example', source_ref=f'source:{i}',
                memory_type=kind, similarity=similarity, trait_score=probability,
                cluster_id=i % 2)


def test_ranking_empty_and_duplicate_source():
    assert rank_candidates([], 'D') == []
    a, b = candidate(1), candidate(2)
    b['source_ref'] = a['source_ref']
    with pytest.raises(ValueError, match='Duplicate'):
        rank_candidates([a, b], 'D')


def test_pack_counts_complete_rendered_memory_not_just_body():
    rows = rank_candidates([candidate(i) for i in range(8)], 'C')
    result = pack(rows, count=len, budget=300, limit=5)
    assert len(memory_block(result)) <= 300
    assert len(result) <= 5
    assert 'not your own experiences' in memory_block(result)


def test_owned_episode_requires_explicit_owner():
    with pytest.raises(ValueError, match='owner'):
        rank_candidates([candidate(1, kind='owned_episode')], 'B')


def test_graph_is_finite_and_uses_only_same_candidates():
    rows = [candidate(i, probability=.1 + .1*i) for i in range(8)]
    ranked = rank_candidates(rows, 'D')
    assert {x['memory_id'] for x in rows} == {x['memory_id'] for x in ranked}
    assert all(np.isfinite(x['rank_score']) for x in ranked)


def test_softmax_export_class_order_and_constants(tmp_path):
    from persona_memory_selector.features import Scorer
    from persona_memory_selector.data import dump
    p=tmp_path/'model.json'
    dump(p,{'schema_version':'selector-logistic-v1','class_order':['low','high'],
         'feature_source':'context','input_feature_order':['e000','e001'],
         'preprocessing':{'center':[1,5],'scale':[2,1],'selected_indices':[0,1]},
         'model':{'intercept':[0,0],'coefficients':[[0,0],[1,0]]}})
    probs=Scorer(p).predict([[3,5]])
    assert probs[0,1]==pytest.approx(1/(1+np.exp(-1)))
    with pytest.raises(ValueError): Scorer(p).predict([[float('nan'),5]])


def test_frozen_artifact_changes_rejected(tmp_path):
    from persona_memory_selector.experiment import verify,hashes
    from persona_memory_selector.data import dump
    (tmp_path/'input').write_text('before')
    dump(tmp_path/'artifacts/selector/frozen.json',{'hashes':hashes(tmp_path,['input']),'artifact_hashes':{}})
    verify(tmp_path)
    (tmp_path/'input').write_text('after')
    with pytest.raises(ValueError,match='changed'): verify(tmp_path)


def test_pending_ratings_never_produce_effects(tmp_path):
    from persona_memory_selector.data import write_csv
    from persona_memory_selector.review import summarize
    write_csv(tmp_path/'artifacts/selector/review/test-key.csv',[{'case_id':'test-000','answer_id':'Answer 1','method':'A'}])
    for i in (1,2):
        write_csv(tmp_path/f'deliverables/selector/review/test-reviewer_{i}.csv',
          [dict(reviewer=f'reviewer_{i}',case_id='test-000',answer_id='Answer 1',behavior='',voice='',relevance='',fabricated_history='',factual_error='',invalid='')])
    r=summarize(tmp_path)
    assert r['status']=='pending' and r['completed_ratings']==0 and not r['comparisons']


def test_truncation_is_retained_as_invalid_not_claimed_complete():
    from persona_memory_selector.experiment import response_status
    status=response_status({'done':True,'done_reason':'length','message':{'content':'cut off'}})
    assert status=={'valid_completion':False,'invalid_reason':'output_limit'}
    with pytest.raises(ValueError): response_status({'done':True,'message':{'content':'missing reason'}})


def test_feature_identity_changes_with_encoder_spec(monkeypatch):
    from persona_memory_selector import features
    before=features.feature_signature(['same text'])
    monkeypatch.setitem(features.FEATURE_SPEC,'window_overlap',1)
    assert features.feature_signature(['same text'])!=before


def test_partial_blind_export_does_not_create_templates(tmp_path,monkeypatch):
    from persona_memory_selector import experiment
    monkeypatch.setattr(experiment,'verify',lambda root:{})
    with pytest.raises(ValueError,match='partial'): experiment.blind_export(tmp_path,'test')
    assert not (tmp_path/'deliverables/selector/review').exists()


def test_reviewer_identity_cannot_be_changed_in_csv(tmp_path):
    from persona_memory_selector.data import write_csv
    from persona_memory_selector.review import summarize
    write_csv(tmp_path/'artifacts/selector/review/test-key.csv',[dict(case_id='test-000',answer_id='Answer 1',method='A')])
    write_csv(tmp_path/'deliverables/selector/review/test-reviewer_1.csv',
              [dict(reviewer='reviewer_2',case_id='test-000',answer_id='Answer 1')])
    with pytest.raises(ValueError,match='identity'): summarize(tmp_path)


def test_domain_rules_keep_ambiguous_cases_separate():
    from persona_memory_selector.experiment import situation_domain
    assert situation_domain('The client changed the deadline.')=='work'
    assert situation_domain('My friend and colleague called.')=='mixed'
    assert situation_domain('An unfamiliar situation.')=='other'


@pytest.mark.parametrize('budget',[2001,10000,-1,1.5,True])
def test_interface_rejects_budget_outside_contract_before_loading_model(tmp_path,budget):
    from persona_memory_selector.interface import retrieve
    with pytest.raises(ValueError,match='0..2000'):
        retrieve(tmp_path,{'query':'A situation','budget':budget})


def test_blind_export_does_not_reassign_existing_ratings_to_changed_answers(tmp_path,monkeypatch):
    from persona_memory_selector import experiment
    from persona_memory_selector.data import dump
    import json
    monkeypatch.setattr(experiment,'verify',lambda root:{})
    for i in range(10):
        dump(tmp_path/f'artifacts/selector/responses/val/val-{i:03d}.json',
             {'case':dict(case_id=f'val-{i:03d}',query='Question',trait='openness_high',kind='persona'),
              'responses':{m:{'text':m+' original response'} for m in experiment.ARMS}})
    experiment.blind_export(tmp_path,'val')
    blinded=tmp_path/'deliverables/selector/review/val-blinded.json';before=blinded.read_bytes()
    p=tmp_path/'artifacts/selector/responses/val/val-000.json';case=json.loads(p.read_text())
    case['responses']['A']['text']='Changed answer';dump(p,case)
    with pytest.raises(ValueError,match='content changed'): experiment.blind_export(tmp_path,'val')
    assert blinded.read_bytes()==before


@pytest.mark.parametrize('field',['model_context','model_response','model_text','label','context','response','persony'])
def test_source_acceptance_checks_every_feature_column_and_target(field):
    from persona_memory_selector.data import check_source_record
    source=raw(1,'Alice asks Bob')
    rows,_,_=clean_rows([source]);row=rows[0]
    check_source_record(row,source,0)
    row[field]='changed'
    with pytest.raises(ValueError,match=field): check_source_record(row,source,0)
