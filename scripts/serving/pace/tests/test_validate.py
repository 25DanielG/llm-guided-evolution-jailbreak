import json
import unittest
from unittest.mock import MagicMock, patch

from validate import qualification_suite, streaming_benchmark


def reply(finish, content=None, **message):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content, **message}}]}, 0.5


def events(*chunks):
    usage = {'choices': [], 'usage': {'completion_tokens': 5, 'prompt_tokens': 9}}
    return [b'data: ' + json.dumps(event).encode() for event in (*chunks, usage)] + [b'data: [DONE]']


class ValidationTests(unittest.TestCase):
    calls = [{'function': {'name': 'record_color', 'arguments': '{"color":"red"}'}}]

    def suite(self, *replies):
        with patch('validate.completion', side_effect=replies), \
                patch('validate.streaming_benchmark', return_value={'completion_tokens': 256}):
            return qualification_suite('http://127.0.0.1', 'muse', video=True)

    def test_suite_accepts_completed_answers_and_expected_tool_finish(self):
        results, benchmark = self.suite(
            reply('stop', '391'),
            reply('stop', '391', reasoning='17 times 23'),
            reply('tool_calls', tool_calls=self.calls),
            reply('stop', 'Red'),
            reply('stop', 'Red'),
        )
        self.assertTrue(all(check['passed'] for check in results.values()))
        self.assertTrue(benchmark['passed'])

    def test_matching_text_cannot_hide_truncation_or_an_unexpected_tool_call(self):
        results, _ = self.suite(
            reply('length', '391'),
            reply('stop', '391'),
            reply('stop', tool_calls=self.calls),
            reply('stop', 'Red', tool_calls=self.calls),
            reply('length', 'Red'),
        )
        self.assertEqual({name: result['passed'] for name, result in results.items()}, {
            'text': False, 'reasoning': False, 'tools': False, 'image': False, 'video': False,
        })

    def benchmark(self, stream, stamps):
        response = MagicMock()
        response.__enter__.return_value = stream
        with patch('validate.api_open', return_value=response) as opener, \
                patch('validate.time.monotonic', side_effect=stamps):
            result = streaming_benchmark('http://127.0.0.1', {'model': 'muse'})
        request = json.loads(opener.call_args.args[0].data)
        self.assertTrue(request['return_token_ids'])
        self.assertTrue(request['stream_options']['include_usage'])
        return result

    def test_streaming_benchmark_excludes_every_token_from_the_first_chunk(self):
        stream = events(
            {'choices': [{'token_ids': [1, 2], 'delta': {'reasoning': 'two tokens'}}]},
            {'choices': [{'token_ids': [3, 4, 5], 'delta': {'content': 'three more'}}]},
        )
        measured = self.benchmark(stream, (10, 10.125, 10.375, 10.5))
        self.assertEqual(measured['first_token_seconds'], 0.125)
        self.assertEqual(measured['generation_tokens_per_second'], 12)
        self.assertEqual(measured['end_to_end_tokens_per_second'], 10)

    def test_streaming_benchmark_reports_only_end_to_end_if_token_ids_do_not_match_usage(self):
        for choices, clock in [
            ([{'delta': {'content': 'reply'}}], (10, 10.5)),
            ([{'token_ids': [1, 2], 'delta': {'content': 'reply'}}], (10, 10.125, 10.5)),
        ]:
            with self.subTest(choices=choices):
                measured = self.benchmark(events({'choices': choices}), clock)
                self.assertEqual(measured['completion_tokens'], 5)
                self.assertEqual(measured['end_to_end_tokens_per_second'], 10)
                self.assertIsNone(measured['first_token_seconds'])
                self.assertIsNone(measured['generation_tokens_per_second'])
