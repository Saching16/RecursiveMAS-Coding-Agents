"""Unit tests for `_extract_tool_calls` / `_format_tools`, independent of any
model weights. This is the piece that can be verified without a live smoke
test: given the exact text a Qwen-family model would emit for its
`<tool_call>` format, does the parser produce the right LangChain `ToolCall`
dicts and leave the right visible content behind.

Run with: python3 -m unittest discover -s integrations -v
"""

from __future__ import annotations

import unittest

from deepagents_latent.tool_calling import extract_tool_calls, format_tools


class ExtractToolCallsTests(unittest.TestCase):
    def test_plain_text_no_tool_call(self):
        content, calls = extract_tool_calls("The answer is 42.")
        self.assertEqual(content, "The answer is 42.")
        self.assertEqual(calls, [])

    def test_single_tool_call_no_surrounding_text(self):
        text = '<tool_call>\n{"name": "search", "arguments": {"query": "latent MAS"}}\n</tool_call>'
        content, calls = extract_tool_calls(text)
        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search")
        self.assertEqual(calls[0]["args"], {"query": "latent MAS"})
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_tool_call_with_leading_reasoning_text(self):
        text = (
            "I should look this up first.\n"
            '<tool_call>\n{"name": "search", "arguments": {"query": "kv cache"}}\n</tool_call>'
        )
        content, calls = extract_tool_calls(text)
        self.assertEqual(content, "I should look this up first.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.py"}}\n</tool_call>'
        )
        content, calls = extract_tool_calls(text)
        self.assertEqual(content, "")
        self.assertEqual([c["args"]["path"] for c in calls], ["a.py", "b.py"])
        # ids must be unique so LangGraph can correlate each ToolMessage
        # back to the call that produced it.
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_tool_call_with_no_arguments_key_defaults_to_empty_dict(self):
        text = '<tool_call>\n{"name": "list_files"}\n</tool_call>'
        content, calls = extract_tool_calls(text)
        self.assertEqual(calls[0]["args"], {})

    def test_malformed_json_is_dropped_not_raised(self):
        text = '<tool_call>\n{"name": "search", "arguments": {broken json\n</tool_call>'
        content, calls = extract_tool_calls(text)
        self.assertEqual(calls, [])
        # the malformed block is still stripped out of visible content
        self.assertEqual(content, "")

    def test_tool_call_missing_name_is_dropped(self):
        text = '<tool_call>\n{"arguments": {"query": "x"}}\n</tool_call>'
        content, calls = extract_tool_calls(text)
        self.assertEqual(calls, [])

    def test_tool_call_arguments_not_a_dict_defaults_to_empty(self):
        text = '<tool_call>\n{"name": "noop", "arguments": "not-a-dict"}\n</tool_call>'
        content, calls = extract_tool_calls(text)
        self.assertEqual(calls[0]["args"], {})

    def test_mixed_valid_and_malformed_calls_keeps_the_valid_one(self):
        text = (
            '<tool_call>\n{"name": "ok_call", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\nnot json at all\n</tool_call>'
        )
        content, calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "ok_call")


class FormatToolsTests(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(format_tools(None))

    def test_empty_list_returns_none(self):
        self.assertIsNone(format_tools([]))

    def test_plain_function_is_converted_to_openai_tool_schema(self):
        def get_weather(city: str) -> str:
            """Get the current weather for a city.

            Args:
                city: Name of the city.
            """
            return "sunny"

        formatted = format_tools([get_weather])
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["type"], "function")
        self.assertEqual(formatted[0]["function"]["name"], "get_weather")
        self.assertIn("city", formatted[0]["function"]["parameters"]["properties"])

    def test_unconvertible_entry_is_dropped_not_raised(self):
        def good_tool(x: int) -> int:
            """Doubles a number.

            Args:
                x: The number to double.
            """
            return x * 2

        formatted = format_tools([good_tool, object()])
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["function"]["name"], "good_tool")


if __name__ == "__main__":
    unittest.main()
