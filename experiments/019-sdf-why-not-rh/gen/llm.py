"""Thin async client for the LiteLLM proxy with an on-disk cache, so every stage of the pipeline
is resumable: an identical request (model, messages, sampling parameters, tag) returns the cached
reply without a network call. Usage is appended to a ledger so the spend can be reconstructed.

Reads OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_CUSTOM_HEADERS from the repo's .env (or the
environment). Anthropic models get the system prompt marked for prompt caching.
"""
import asyncio
import collections
import hashlib
import json
import os
import random
import time

from openai import AsyncOpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXP))
CACHE = os.path.join(HERE, "cache")
LEDGER = os.path.join(HERE, "out", "usage_ledger.jsonl")


def load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def custom_headers():
    raw = os.environ.get("OPENAI_CUSTOM_HEADERS", "")
    out = {}
    for part in raw.replace(";", ",").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


class LLM:
    # Per-provider concurrency: the OpenRouter models take many parallel requests, the Anthropic
    # key behind the proxy is shared and rate-limited.
    CONCURRENCY = {"openrouter": 64, "anthropic": 12, "local": 8}

    def __init__(self, concurrency=None, timeout=600):
        load_env()
        # An OpenSSL context rather than the SDK's default TLS backend: inside Claude's sandbox
        # the default fails macOS trust evaluation on the HTTPS proxy (OSStatus -26276) while the
        # system OpenSSL store verifies it fine. Full verification either way.
        import ssl
        from openai import DefaultAsyncHttpxClient
        self.client = AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"],
            default_headers=custom_headers() or None, timeout=timeout, max_retries=0,
            http_client=DefaultAsyncHttpxClient(verify=ssl.create_default_context()),
        )
        self._sems = {}
        self._override = concurrency
        self.usage = collections.defaultdict(collections.Counter)  # (model, tag) -> counts
        self.n_cached = 0
        self.n_calls = 0
        os.makedirs(CACHE, exist_ok=True)
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)

    def sem_for(self, model):
        provider = model.split("/", 1)[0]
        if provider not in self._sems:
            n = self._override or self.CONCURRENCY.get(provider, 8)
            self._sems[provider] = asyncio.Semaphore(n)
        return self._sems[provider]

    @staticmethod
    def _key(model, messages, max_tokens, temperature, tag, variant):
        blob = json.dumps([model, messages, max_tokens, temperature, tag, variant], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _with_cache_control(model, messages):
        """Anthropic models: mark the system prompt as a cache prefix. Others: plain strings."""
        if not model.startswith("anthropic/"):
            return messages
        out = []
        for m in messages:
            if m["role"] == "system" and isinstance(m["content"], str):
                out.append({"role": "system", "content": [
                    {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]})
            else:
                out.append(m)
        return out

    async def chat(self, model, messages, *, max_tokens=2000, temperature=0.8, tag="misc", variant=0,
                   extra_body=None):
        key = self._key(model, messages, max_tokens, temperature, tag, variant)
        path = os.path.join(CACHE, key[:2], key + ".json")
        if os.path.exists(path):
            try:
                rec = json.load(open(path))
                self.n_cached += 1
                return rec["content"]
            except (ValueError, KeyError):
                pass
        delay = 2.0
        last = None
        for attempt in range(8):
            try:
                kwargs = dict(model=model, messages=self._with_cache_control(model, messages),
                              max_tokens=max_tokens, extra_body=extra_body or {})
                if temperature is not None:
                    kwargs["temperature"] = temperature
                async with self.sem_for(model):
                    resp = await self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    raise RuntimeError("empty completion (finish_reason=%s)" % choice.finish_reason)
                usage = resp.usage.model_dump() if resp.usage else {}
                break
            except Exception as e:  # noqa: BLE001 -- retry anything, the proxy 5xxs and 429s
                last = e
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        else:
            raise RuntimeError("giving up after 8 attempts: %r" % (last,))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = dict(model=model, tag=tag, key=key, content=content, usage=usage, ts=time.time(),
                   finish_reason=choice.finish_reason)
        tmp = path + ".tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, path)
        self._account(model, tag, usage)
        return content

    def _account(self, model, tag, usage):
        self.n_calls += 1
        c = self.usage[(model, tag)]
        c["calls"] += 1
        c["prompt"] += usage.get("prompt_tokens", 0) or 0
        c["completion"] += usage.get("completion_tokens", 0) or 0
        det = usage.get("prompt_tokens_details") or {}
        c["cached"] += det.get("cached_tokens", 0) or 0
        cdet = usage.get("completion_tokens_details") or {}
        c["reasoning"] += cdet.get("reasoning_tokens", 0) or 0
        if usage.get("cost") is not None:
            c["cost_musd"] += int(round(float(usage["cost"]) * 1e6))
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(ts=time.time(), model=model, tag=tag, usage=usage)) + "\n")

    def report(self):
        lines = []
        for (model, tag), c in sorted(self.usage.items()):
            lines.append("%-42s %-10s calls %5d  prompt %9d (cached %9d)  completion %8d  reasoning %7d  cost $%.2f"
                         % (model, tag, c["calls"], c["prompt"], c["cached"], c["completion"], c["reasoning"],
                            estimate_cost(model, c)))
        lines.append("cache hits %d, live calls %d" % (self.n_cached, self.n_calls))
        return "\n".join(lines)


def extract_tag(text, tag):
    """The text between <tag> and </tag>, or None. Tolerates a missing closing tag at the end."""
    start = text.find("<%s>" % tag)
    if start < 0:
        return None
    start += len(tag) + 2
    end = text.find("</%s>" % tag, start)
    return text[start:end].strip() if end >= 0 else text[start:].strip()


def parse_json(text):
    """Best-effort JSON: strips code fences and anything before the first { or [."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    # The outermost value is whichever opener comes first; trying "[" before "{" would return an
    # inner list from an object that contains one.
    candidates = sorted((t.find(o), o, c) for o, c in (("[", "]"), ("{", "}")) if t.find(o) >= 0)
    for i, opener, closer in candidates:
        j = t.rfind(closer)
        if j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    raise ValueError("no JSON in reply: %r" % t[:200])


# List prices per million tokens (input, cached input, output) for the models the proxy does not
# report a cost for. OpenRouter models report `cost` in usage and are summed directly.
PRICES = {
    "anthropic/claude-haiku-4-5-20251001": (1.0, 0.10, 5.0),
    "anthropic/claude-sonnet-5": (3.0, 0.30, 15.0),
    "anthropic/claude-opus-5": (15.0, 1.50, 75.0),
}


def estimate_cost(model, c):
    if c.get("cost_musd"):
        return c["cost_musd"] / 1e6
    p = PRICES.get(model)
    if not p:
        return 0.0
    uncached = c["prompt"] - c["cached"]
    return (uncached * p[0] + c["cached"] * p[1] + c["completion"] * p[2]) / 1e6
