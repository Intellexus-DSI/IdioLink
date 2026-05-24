"""Smoke tests for encode_helpers relocation."""

def test_encode_queries_for_mode_importable_from_models():
    from idiolink.models.encode_helpers import encode_queries_for_mode
    assert callable(encode_queries_for_mode)
