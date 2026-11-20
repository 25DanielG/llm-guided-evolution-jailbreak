"""Mocked client and server contracts, not real model inference verification."""
import asyncio
from unittest.mock import Mock

import pytest
import llm_utils
import server

pytestmark = pytest.mark.mock_inference


@pytest.mark.parametrize('status', [200, 503])
def test_legacy_generate_client_contract(status, monkeypatch):
    monkeypatch.setattr(llm_utils, 'get_llm_server_hostname', lambda: 'mock-compute')
    reply = Mock(status_code=status)
    reply.json.return_value = {'generated_text': 'fixture answer', 'response_time_sec': 0.01}
    post = Mock(return_value=reply)
    monkeypatch.setattr(llm_utils.requests, 'post', post)
    result = llm_utils.submit_mixtral_local('fixture prompt', max_new_tokens=123, temperature=0.2, top_p=0.7)
    assert result == ('fixture answer' if status == 200 else None)
    assert post.call_args.args[0] == f'http://mock-compute:{llm_utils.PORT}/generate'
    assert post.call_args.kwargs['json'] == {'prompt': 'fixture prompt', 'max_new_tokens': 123,
                                            'temperature': 0.2, 'top_p': 0.7}


@pytest.mark.parametrize('fails', [False, True])
def test_generate_endpoint_contract(fails, monkeypatch):
    recorded = []
    class FakeModel:
        async def start_batch_processor(self):
            recorded.append('startup')
        async def generate(self, request):
            recorded.append(request)
            if fails:
                raise RuntimeError('fixture inference failure')
            return {'generated_text': 'fixture answer', 'response_time_sec': 0.01, 'batch_size': 1}
    monkeypatch.setattr(server, 'LLMModel', FakeModel)
    asyncio.run(server.preload_model())
    request = server.LLMRequest(prompt='fixture prompt', max_new_tokens=123, temperature=0.2, top_p=0.7)
    if fails:
        with pytest.raises(server.HTTPException) as err:
            asyncio.run(server.generate_text(request))
        assert err.value.status_code == 500
        assert 'fixture inference failure' in err.value.detail
    else:
        result = asyncio.run(server.generate_text(request))
        assert result == {'generated_text': 'fixture answer', 'response_time_sec': 0.01, 'batch_size': 1}
    assert recorded == ['startup', {'prompt': 'fixture prompt', 'max_new_tokens': 123, 'temperature': 0.2, 'top_p': 0.7}]
