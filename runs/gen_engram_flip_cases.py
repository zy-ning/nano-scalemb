"""Generate the expanded multi-domain flip-probe case set, with validation.

Each emitted case is guaranteed (against the real tokenizer) to satisfy the
single-token-flip constraints enforced by scripts.engram_flip_eval.validate_flip_alignment:
equal length, exactly one differing token position, inside the n-gram window.

Knowledge domains: capital, official language, chemical symbol, continent, planet order.
Pairs are bidirectional (A->B and B->A) and chosen so the home and flip answers differ.
A per-case `filler` puts a non-fact single-token word in the same slot (does *any* change
move the output, vs does the specific fact flip move the matching answer).

Run with the nanochat venv so the tokenizer is available:
  ../nanochat/.venv/bin/python -m runs.gen_engram_flip_cases
"""

import json
import os

from nano_scalemb.tokenizer import get_tokenizer

MAX_NGRAM = 3  # must match EngramConfig.max_ngram_size

# domain -> (template, fillers, [bidirectional (entityA, answerA, entityB, answerB)])
DOMAINS = {
    "capital": (
        "The capital of {} is",
        ["winter", "music", "copper", "tennis"],
        [
            ("France", " Paris", "Japan", " Tokyo"),
            ("Germany", " Berlin", "Italy", " Rome"),
            ("Canada", " Ottawa", "Greece", " Athens"),
            ("England", " London", "Austria", " Vienna"),
        ],
    ),
    "language": (
        "The official language of {} is",
        ["winter", "gravity", "silence", "pizza"],
        [
            ("France", " French", "Japan", " Japanese"),
            ("Germany", " German", "Italy", " Italian"),
            ("Russia", " Russian", "China", " Chinese"),
            ("Spain", " Spanish", "Poland", " Polish"),
        ],
    ),
    "element": (
        "The chemical symbol for {} is",
        ["winter", "music", "summer", "tennis"],
        [
            ("gold", " Au", "iron", " Fe"),
            ("oxygen", " O", "hydrogen", " H"),
            ("sodium", " Na", "carbon", " C"),
            ("silver", " Ag", "helium", " He"),
        ],
    ),
    "continent": (
        "The continent that contains {} is",
        ["winter", "music", "copper", "silence"],
        [
            ("Egypt", " Africa", "Japan", " Asia"),
            ("France", " Europe", "China", " Asia"),
            ("Germany", " Europe", "Kenya", " Africa"),
            ("India", " Asia", "Spain", " Europe"),
        ],
    ),
    "planet": (
        "In the solar system, the planet {} is the",
        ["winter", "music", "copper", "summer"],
        [
            ("Mercury", " first", "Venus", " second"),
            ("Earth", " third", "Mars", " fourth"),
        ],
    ),
}


def main():
    tok = get_tokenizer()
    bos = tok.get_bos_token_id()

    def ids(s):
        return tok.encode(s, prepend=bos)

    def assert_valid(tmpl, home_ent, flip_ent, home_tgt, flip_tgt, tag):
        ph, pf = ids(tmpl.format(home_ent)), ids(tmpl.format(flip_ent))
        assert len(ph) == len(pf), f"{tag}: length {len(ph)}!={len(pf)}"
        diff = [i for i, (a, b) in enumerate(zip(ph, pf)) if a != b]
        L = len(ph)
        assert len(diff) == 1, f"{tag}: {len(diff)} differing positions {diff}"
        pos = diff[0]
        assert L - MAX_NGRAM <= pos < L - 1, f"{tag}: pos {pos}/{L} out of window"
        for t in (home_tgt, flip_tgt):
            assert len(tok.encode(t)) == 1, f"{tag}: target {t!r} not single-token"

    cases = []
    for domain, (tmpl, fillers, pairs) in DOMAINS.items():
        fi = 0
        for a_ent, a_tgt, b_ent, b_tgt in pairs:
            for (home_ent, home_tgt, flip_ent, flip_tgt) in (
                (a_ent, a_tgt, b_ent, b_tgt),
                (b_ent, b_tgt, a_ent, a_tgt),
            ):
                filler = fillers[fi % len(fillers)]
                fi += 1
                tag = f"{domain}:{home_ent}->{flip_ent}"
                assert_valid(tmpl, home_ent, flip_ent, home_tgt, flip_tgt, tag)
                # filler must also be a clean single-token swap
                assert_valid(tmpl, home_ent, filler, home_tgt, home_tgt, tag + ":filler")
                cases.append(
                    {
                        "name": f"{domain}__{home_ent}_to_{flip_ent}".lower(),
                        "tier": domain,
                        "prompt": tmpl.format(home_ent),
                        "home_target": home_tgt,
                        "variants": [
                            {
                                "name": "self",
                                "engram_prompt": tmpl.format(home_ent),
                                "target": home_tgt,
                            },
                            {
                                "name": "flip",
                                "engram_prompt": tmpl.format(flip_ent),
                                "target": flip_tgt,
                            },
                            {
                                "name": "filler",
                                "engram_prompt": tmpl.format(filler),
                                "target": None,
                            },
                        ],
                    }
                )

    out = os.path.join(os.path.dirname(__file__), "engram_flip_probe_facts_cases.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    by_dom = {}
    for c in cases:
        by_dom[c["tier"]] = by_dom.get(c["tier"], 0) + 1
    print(f"Wrote {len(cases)} cases to {out}")
    print("by domain:", by_dom)


if __name__ == "__main__":
    main()
