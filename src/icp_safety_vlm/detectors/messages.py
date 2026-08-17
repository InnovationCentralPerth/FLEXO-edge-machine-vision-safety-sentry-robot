"""Fixed, non-generated GO/STOP messages, shared by every VLM backend.

Earlier versions let the VLM freely compose the sentence per frame. Two
problems with that in practice: (1) it could describe irrelevant scene
clutter (keyboards, furniture) instead of staying on-topic, and (2) on an
edge NPU (Hailo-10H) free-text generation is the expensive, latency-bound
part — real-world decode throughput for small models there is not
dramatically faster than CPU, sometimes slower (see README). Since the
task only needs a binary verdict, every backend now does the minimum work
to get that one bit right and looks up the message from here — the model
never generates prose at inference time.
"""

from __future__ import annotations

MESSAGE_GO = "You are correctly wearing your safety vest. GO!"
MESSAGE_STOP = "You are not wearing or correctly wearing a safety vest. STOP!"


def compliance_message(compliant: bool) -> str:
    return MESSAGE_GO if compliant else MESSAGE_STOP
