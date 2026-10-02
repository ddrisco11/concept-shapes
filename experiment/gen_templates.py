"""Use Gemma 3 270M (via Ollama) to write varied sentence contexts.

Gemma only supplies the linguistic context. It is asked to write sentences
containing a fixed anchor value (e.g. the year 1987); we then swap the anchor
for a slot and fill the slot programmatically, so the temporal variable is
fully controlled.
"""
import json
import random
import re
import sys

import requests

MODEL = "gemma3:270m"
URL = "http://localhost:11434/api/generate"

TOPICS = [
    "a person being born", "a building being constructed", "a company being founded",
    "a book being published", "a treaty being signed", "a photograph", "a war or battle",
    "a scientific discovery", "a sports championship", "a factory", "a shipwreck",
    "a wedding", "a law being passed", "a famous painting", "a school or university",
    "a flood or storm", "a harvest on a farm", "a railway or road", "a newspaper article",
    "an election", "a family moving to a new town", "a church or temple", "a bridge",
    "a machine being manufactured", "a song or piece of music", "a voyage or expedition",
    "a court trial", "a disease outbreak", "a shop or restaurant opening", "a letter or diary",
    "a museum exhibit", "a festival or fair", "an invention or patent", "a mine or quarry",
    "a fire in a city", "a census or survey", "a theater play", "a garden or park",
    "a bank or loan", "a death or funeral", "a map or atlas", "a hospital",
    "a novel's setting", "an old coin or stamp", "a mountain climb", "a strike by workers",
    "a retirement", "a town's anniversary", "a library", "a lighthouse",
]

KINDS = {
    # kind: (anchor string, slot, description used in the prompt)
    "year": ("1987", "{YEAR}", "the year 1987"),
    "weekday": ("Tuesday", "{DAY}", "the weekday Tuesday"),
    "month": ("March", "{MONTH}", "the month of March"),
    "hour": ("3 PM", "{HOUR}", "the time 3 PM"),
    "number": ("42", "{NUM}", "the number 42"),
}

# other temporal cues that would leak or conflict with the slot
BANNED = re.compile(
    r"\b(century|centuries|decade|ago|yesterday|tomorrow|today|tonight|morning|afternoon|"
    r"evening|night|noon|midnight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"spring|summer|autumn|fall|winter|weekend|am|pm)\b",
    re.I,
)


def ask(topic, desc, seed):
    prompt = (
        f"Write 10 different short sentences about {topic}. "
        f"Every sentence must mention {desc}. "
        "Use a different sentence structure each time. One sentence per line, no numbering."
    )
    if seed % 2:  # tiny models often drop the anchor; a few-shot variant is tried on alternate calls
        a = next(v[0] for v in KINDS.values() if v[2] == desc)
        prompt = (
            f'Every sentence below contains "{a}".\n\nExamples:\n'
            f'Everyone at the table agreed that {a} was the right answer.\n'
            f'She wrote "{a}" at the top of the page and underlined it.\n'
            f'It took a while, but in the end {a} turned out to matter.\n\n'
            f'Now write 10 new short sentences about {topic}. Each one must contain "{a}" exactly once. '
            'One sentence per line, no numbering.'
        )
    r = requests.post(URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 1.0, "top_p": 0.95, "seed": seed, "num_predict": 400},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def clean(line, anchor, slot):
    s = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"').strip()
    s = s.replace("**", "")
    if s.count(anchor) != 1:
        return None
    rest = s.replace(anchor, " ")
    if re.search(r"\d", rest) or BANNED.search(rest):
        return None
    n = len(s.split())
    if n < 5 or n > 25 or not s[0].isupper() or s[-1] not in ".!":
        return None
    # the anchor must be a standalone token ("1987's", "1987-88" rejected)
    if not re.search(rf"(?<![\w\-']){re.escape(anchor)}(?![\w\-'’])", s):
        return None
    return s.replace(anchor, slot)


def main(kind, target):
    anchor, slot, desc = KINDS[kind]
    random.seed(0)
    out, seen, calls = [], set(), 0
    while len(out) < target and calls < 40 * len(TOPICS):
        topic = TOPICS[calls % len(TOPICS)]
        text = ask(topic, desc, seed=calls)
        calls += 1
        for line in text.splitlines():
            t = clean(line, anchor, slot)
            if not t:
                continue
            key = re.sub(r"\W+", " ", t.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"template": t, "topic": topic})
        if calls % 10 == 0:
            print(f"[{kind}] calls={calls} templates={len(out)}", flush=True)
    with open(f"data/templates_{kind}.json", "w") as f:
        json.dump(out[:target], f, indent=1)
    print(f"[{kind}] done: {len(out[:target])} templates from {calls} calls")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
