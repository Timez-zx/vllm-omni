#!/usr/bin/env python3
"""Zero-GPU checks for the PA_SESSION patch. Run before spending a server boot.

Every failure mode this checks for is SILENT at runtime -- none of them raises. That is
why they are checked here instead of being discovered from a latency number that looks
merely disappointing:

  1. The feeder must be a NATIVE async generator. `async_omni.py:398` branches on
     `isinstance(prompt, collections.abc.AsyncGenerator)`, whose __subclasshook__ demands
     asend/athrow/aclose in addition to __aiter__/__anext__. A hand-rolled iterator class
     falls through to the one-shot per-turn path with no error, and the whole change
     becomes a no-op that reads as a null result.

  2. `<|im_end|>` and `\\n` must really be token ids 151645 and 198. Every delta after the
     first is prefixed with them, because the scheduler folds the previous segment's
     output into the prompt but drops the last sampled token -- which for the thinker is
     the EOS `<|im_end|>`. Wrong ids would corrupt the chat structure quietly.

  3. Each delta must be a self-contained `<|im_start|>user ... <|im_end|>` +
     `<|im_start|>assistant` unit. `compute_talker_prompt_ids_length` (adapter.py, the
     LIVE one under async_chunk) returns 0 for a delta with no im_start; the caller wraps
     that in max(1, ...), so the talker gets a ONE-token placeholder and the worker
     silently keeps one row of conditioning and discards the rest. Wrong audio, no error.

  4. Prefixing two tokens moves every image/audio placeholder, so the offsets must shift
     with it. Getting this wrong makes the model read image tokens at the wrong positions.
"""
from __future__ import annotations

import collections.abc
import os
import pathlib
import sys

# Import the FORK rather than whatever is installed in site-packages, so this can be
# run before (or without) an editable install.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("1. native async generator recognition")

    async def real_gen():
        yield 1

    class FakeIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    g = real_gen()
    check("an `async def ... yield` object IS an AsyncGenerator",
          isinstance(g, collections.abc.AsyncGenerator))
    check("an __aiter__/__anext__ class is NOT (would silently no-op)",
          not isinstance(FakeIter(), collections.abc.AsyncGenerator))

    print("\n2. the fork imports and carries the session-mode pieces")
    import vllm_omni.entrypoints.openai.video_stream_base as B
    loaded_from_repo = pathlib.Path(B.__file__).resolve().is_relative_to(REPO_ROOT)
    check("loaded from this checkout, not site-packages", loaded_from_repo, B.__file__)
    check("session_scoped_request is a config field",
          "session_scoped_request" in B.StreamingVideoSessionConfig.model_fields)
    check("it defaults to OFF",
          B.StreamingVideoSessionConfig().session_scoped_request is False)
    check("session chunk builder present", hasattr(B.OmniStreamingVideoHandler, "_build_session_chunk"))
    check("placeholder shifter present", hasattr(B, "_shift_mm_placeholders"))
    check("segment-boundary helper present", hasattr(B, "_segment_finish_reason"))
    check("frame downscaler present", hasattr(B, "_downscale_frame_bytes"))
    for fld in ("max_frame_width", "max_frame_height", "frame_filter_min_gap", "frame_filter_max_gap"):
        check(f"config field {fld}", fld in B.StreamingVideoSessionConfig.model_fields)
    from vllm_omni.entrypoints.openai.video_frame_filter import (
        FrameSimilarityFilter as _FrameSimilarityFilter,
    )
    check("FrameSimilarityFilter.force_next_retain present",
          hasattr(_FrameSimilarityFilter, "force_next_retain"))

    print("\n3. the two prefix token ids are really <|im_end|> and newline")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True,
            cache_dir=os.environ.get("HF_HOME", "/data/zx/hf") + "/hub",
        )
        got = tok.convert_tokens_to_ids(["<|im_end|>"])[0]
        nl = tok.encode("\n", add_special_tokens=False)
        check("<|im_end|> == 151645", got == 151645, f"got {got}")
        check("newline == [198]", nl == [198], f"got {nl}")
        ims = tok.convert_tokens_to_ids(["<|im_start|>"])[0]
        check("<|im_start|> == 151644", ims == 151644, f"got {ims}")
        for role, want in (("system", 8948), ("user", 872), ("assistant", 77091)):
            gid = tok.encode(role, add_special_tokens=False)
            check(f"role token {role!r} == {want}", gid == [want], f"got {gid}")
    except Exception as e:  # noqa: BLE001
        check("tokenizer available", False, f"{type(e).__name__}: {e}")

    print("\n4. the LIVE talker length function on realistic deltas")
    from vllm_omni.distributed.omni_connectors.adapter import (
        compute_talker_prompt_ids_length as tlen,
    )
    IM, USR, AST, END, NL = 151644, 872, 77091, 151645, 198

    def delta(n_img_tok: int, *, with_header=True, with_assistant=True, prefix_end=False):
        ids: list[int] = []
        if prefix_end:
            ids += [END, NL]
        if with_header:
            ids += [IM, USR, NL] + [0] * n_img_tok + [END, NL]
        else:
            ids += [0] * n_img_tok
        if with_assistant:
            ids += [IM, AST, NL]
        return ids

    # The function credits a user block with (next im_start index - this one), i.e. the
    # WHOLE block including its framing: im_start + role + nl + payload + im_end + nl,
    # so n + 5. The trailing assistant header contributes a flat 9 regardless of content.
    # (This is the arithmetic the first version of this test got wrong -- it guessed n + 6
    # and the mismatch looked like a code bug for a minute. Derive it, do not guess it.)
    for n in (220, 440, 660):
        L = tlen(delta(n))
        check(f"delta with {n} image tokens -> (n+5) + 9 = {n + 14}", L == n + 14, f"got {L}")
    check("prefixing <|im_end|>\\n does not change the length",
          tlen(delta(220, prefix_end=True)) == tlen(delta(220)),
          f"{tlen(delta(220, prefix_end=True))} vs {tlen(delta(220))}")
    check("NO im_start -> 0 (the silent-truncation trap)",
          tlen(delta(220, with_header=False, with_assistant=False)) == 0,
          f"got {tlen(delta(220, with_header=False, with_assistant=False))}")
    check("missing assistant header loses the +9",
          tlen(delta(220, with_assistant=False)) == 220 + 5,
          f"got {tlen(delta(220, with_assistant=False))}")
    try:
        tlen([IM])  # trailing bare im_start reads prompt_ids[s+1]
        check("trailing bare <|im_start|> raises IndexError", False, "it did not raise")
    except IndexError:
        check("trailing bare <|im_start|> raises IndexError", True)

    print("\n5. placeholder offsets shift with the prefix")
    ep = {"prompt_token_ids": [1, 2, 3],
          "mm_placeholders": {"image": [{"offset": 3, "length": 220},
                                        {"offset": 229, "length": 220}]}}
    B._shift_mm_placeholders(ep, 2)
    offs = [p["offset"] for p in ep["mm_placeholders"]["image"]]
    check("dict placeholders shifted by 2", offs == [5, 231], f"got {offs}")

    class PR:
        __slots__ = ("offset", "length")

        def __init__(self, offset, length):
            self.offset, self.length = offset, length

    ep2 = {"mm_placeholders": {"audio": [PR(10, 40)]}}
    B._shift_mm_placeholders(ep2, 2)
    check("object placeholders shifted by 2", ep2["mm_placeholders"]["audio"][0].offset == 12,
          f"got {ep2['mm_placeholders']['audio'][0].offset}")
    B._shift_mm_placeholders(ep2, 0)
    check("shift of 0 is a no-op", ep2["mm_placeholders"]["audio"][0].offset == 12)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS))
        print("Do NOT boot a server until these pass.")
        return 1
    print("all checks passed -- the delta shape is legal and the feeder will be recognised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
