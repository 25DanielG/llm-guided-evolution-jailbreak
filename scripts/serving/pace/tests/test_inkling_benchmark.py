import json
import unittest

from inkling_benchmark import metrics_delta, parse_metrics, stream_statistics


def events(*data):
    return [b'data: ' + json.dumps(event).encode() for event in data] + [b'data: [DONE]']


class InklingBenchmarkTests(unittest.TestCase):
    def test_counts_raw_tokens_when_reasoning_and_final_chunks_are_split(self):
        stream = events(
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'reasoning': 'a'}, 'token_ids': [10, 11]}]},
            {'choices': [{'delta': {'content': 'b'}, 'token_ids': [12, 13, 14]}]},
            {'choices': [], 'usage': {'completion_tokens': 5, 'prompt_tokens': 9,
                                     'completion_tokens_details': {'reasoning_tokens': 2}}},
        )
        stamps = iter((10.125, 10.375, 10.5))
        result = stream_statistics(stream, 10, lambda: next(stamps))
        self.assertEqual(result['ttft_s'], 0.125)
        self.assertEqual(result['decode_tok_s'], 12)
        self.assertEqual(result['e2e_tok_s'], 10)
        self.assertEqual(result['reasoning_tokens'], 2)
        self.assertEqual(result['other_completion_tokens'], 3)

    def test_rejects_missing_or_inconsistent_usage(self):
        stream = events({'choices': [{'token_ids': [1]}]},
                        {'choices': [{'token_ids': [2]}]},
                        {'usage': {'completion_tokens': 4}})
        times = iter((1, 2, 3))
        with self.assertRaisesRegex(ValueError, 'Incomplete SSE token accounting'):
            stream_statistics(stream, 0, lambda: next(times))

    def test_histogram_deltas_exclude_the_warmup(self):
        before = parse_metrics('vllm:request_prefill_time_seconds_sum{model="inkling"} 0.1\n'
                               'vllm:request_prefill_time_seconds_count 1.0')
        after = parse_metrics('vllm:request_prefill_time_seconds_sum{model="inkling"} 0.4\n'
                              'vllm:request_prefill_time_seconds_count 4.0')
        stat = metrics_delta(before, after, 3)['request_prefill_time_seconds']
        self.assertEqual(stat['count'], 3)
        self.assertAlmostEqual(stat['average'], 0.1)
        self.assertTrue(stat['complete'])


if __name__ == '__main__':
    unittest.main()
