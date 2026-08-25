#!/usr/bin/env python3
"""Which of the three question types arrived on `/challenge_question`.

The evaluation node publishes one question per system launch and it can be any
of the three types, so a module that only answers one of them still has to
recognise the other two rather than drive at them. Scoring is per type and the
three go to different topics:

    numerical    -> Int32   on /numerical_response          (/1)
    reference    -> Marker  on /selected_object_marker      (/2)
    instruction  -> Pose2D  on /way_point_with_heading      (/6)

This module owns only the *decision*, and the three answers now go three ways.
Instruction-following drives the trajectory here. Numerical counts here too:
eleven of the fifteen released counting questions are anchor-local — one piece
of furniture, count what is on it — which is the drive loop's own problem
shape, so it answers in-process rather than handing the budget away. Object
reference still goes to the perception + Gemini pipeline, which
`challenge_node` hands the process over to.

So this decision picks which of the team's two stacks spends the ten minutes,
it is made once from the sentence alone before either has done anything, and
it is now wrong in a new way: a counting question misread as reference leaves
this process for good and cannot come back.

Two implementations, chosen by `XIAO_HEI_CLASSIFY`:

  `claude` (default)  one short API call, ~2 s against a 600 s budget.
  `stub`              always "instruction". What shipped while the hand-off
                      did not exist yet; still useful for driving an
                      instruction question without paying for a classification.

Whatever happens, a failure falls back to "instruction". That is not a coin
flip: it keeps the question inside this process, where a wrong guess still
drives a trajectory that can score partial credit, rather than `exec`ing away
into a pipeline that cannot come back if the guess was wrong.
"""

from __future__ import annotations

import os
import sys

NUMERICAL = "numerical"
REFERENCE = "reference"
INSTRUCTION = "instruction"
KINDS = (NUMERICAL, REFERENCE, INSTRUCTION)

MODEL = os.environ.get("XIAO_HEI_CLASSIFY_MODEL", "claude-opus-5")

PROMPT = """\
You are routing one question from a robot navigation benchmark to the module \
that answers it. Reply with exactly one word and nothing else.

numerical    — asks how many of something there are. The answer is an integer.
               e.g. "How many blue chairs are between the table and the wall?"
reference    — asks to identify one specific object, uniquely picked out by \
attributes or spatial relations. The answer is that object.
               e.g. "Find the potted plant on the kitchen island closest to \
the fridge."
instruction  — asks the robot to travel, using objects to constrain the path \
it takes or where it ends up. The answer is a route.
               e.g. "Take the path near the window to the fridge."
               e.g. "Avoid the path between the two tables and go near the \
blue trash can."

Question: {q}

One word — numerical, reference, or instruction:"""


def classify_stub(question: str) -> str:
    """Everything is an instruction. See the module docstring."""
    return INSTRUCTION


def classify_claude(question: str) -> str:
    import anthropic

    client = anthropic.Anthropic(
        max_retries=int(os.environ.get("XIAO_HEI_API_MAX_RETRIES", "8")))
    # Generous ceiling for one word: on claude-opus-5 thinking is on unless it
    # is switched off, and it bills against this same limit. A classifier that
    # comes back empty because it spent the ceiling reasoning would look like a
    # model failure and be a budget failure — the same trap `ask_claude`
    # documents for the grounding call.
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": PROMPT.format(q=question)}])
    txt = "".join(b.text for b in msg.content if b.type == "text").lower()
    # Substring rather than equality: the reply is one word by instruction, not
    # by construction, and "instruction-following." should not fall through.
    for kind in KINDS:
        if kind in txt:
            return kind
    raise ValueError(f"no question type in reply {txt[:80]!r}")


def classify(question: str) -> str:
    """The question's type, never raising."""
    how = os.environ.get("XIAO_HEI_CLASSIFY", "claude").lower()
    if how != "claude":
        return classify_stub(question)
    try:
        return classify_claude(question)
    except Exception as e:                        # noqa: BLE001 — see docstring
        print(f"classify: falling back to {INSTRUCTION} after {e!r}",
              file=sys.stderr)
        return INSTRUCTION


if __name__ == "__main__":
    for q in sys.argv[1:]:
        print(f"{classify(q):12s}  {q}")
