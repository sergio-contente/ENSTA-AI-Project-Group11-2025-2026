from __future__ import annotations


def test_base_policy_import_without_openai_when_flag_set(monkeypatch):
    # Ensure the policy init path doesn't crash due to missing OpenAI keys when using VLM-R1.
    monkeypatch.setenv("COIN_USE_VLMR1", "1")
    # Import should not raise.
    import vlfm.policy.base_objectnav_policy  # noqa: F401

