import json

from SDYJ_Agents.llm.base import BaseLLM, parse_json_object
from SDYJ_Agents.utils.tracing import InstrumentedLLM


class FencedJSONLLM(BaseLLM):
    def __init__(self):
        super().__init__(api_key="fake", model="fake-json")
        self.last_prompt = ""
        self.last_usage = {"prompt_tokens": 3, "completion_tokens": 2}

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        return '```json\n{"ok": true, "items": [1, 2]}\n```'

    def stream_generate(self, prompt: str, **kwargs):
        yield self.generate(prompt, **kwargs)


class NativeJSONLLM(FencedJSONLLM):
    def generate_json(self, prompt: str, schema=None, **kwargs):
        self.last_prompt = prompt
        return {"native": True, "schema_seen": bool(schema)}


def test_parse_json_object_accepts_fenced_and_prefixed_json():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here is the object: {"b": 2}') == {"b": 2}


def test_base_generate_json_fallback_appends_schema_and_parses_response():
    llm = FencedJSONLLM()
    payload = llm.generate_json("Return data", schema={"type": "object"})

    assert payload == {"ok": True, "items": [1, 2]}
    assert "Return a single valid JSON object" in llm.last_prompt


def test_instrumented_generate_json_records_one_llm_call():
    trace = {
        "provider": "fake",
        "llm_calls": [],
        "events": [],
        "replay_cache": {"llm_calls": []},
        "errors": [],
    }
    llm = InstrumentedLLM(NativeJSONLLM(), trace)

    payload = llm.generate_json("Return native", schema={"type": "object"}, temperature=0)

    assert payload == {"native": True, "schema_seen": True}
    assert len(trace["llm_calls"]) == 1
    assert trace["llm_calls"][0]["response_format"] == "json_object"
    assert json.loads(trace["replay_cache"]["llm_calls"][0]["response"]) == payload
