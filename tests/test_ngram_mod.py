# SPDX-License-Identifier: Apache-2.0

from omlx.speculative.ngram_mod import EMPTY, NGramMod, NGramModSequence


def _reference_index(tokens, n_match, size):
    value = 0
    for token in tokens[:n_match]:
        value = (value * 6_364_136_223_846_793_005 + token) & ((1 << 64) - 1)
    return value % size


def test_hash_matches_llama_cpp_unsigned_size_t_math():
    mod = NGramMod(n_match=4, n_min=2, n_max=4, table_size=257)
    tokens = [2_483_199, 17, 2**31 - 1, 9, 42]
    assert mod.index(tokens) == _reference_index(tokens, 4, 257)


def test_copy_heavy_prompt_proposes_long_exact_span():
    mod = NGramMod(n_match=24, n_min=48, n_max=64)
    source = list(range(100))
    prompt = source + source
    state = mod.begin(prompt)

    assert mod.draft(state, prompt) == list(range(64))
    assert state.n_draft_last == 64


def test_short_chain_is_rejected_instead_of_partially_verified():
    mod = NGramMod(n_match=4, n_min=3, n_max=5, table_size=257)
    prompt = [1, 2, 3, 4, 9, 8]
    state = mod.begin(prompt)

    assert mod.draft(state, [1, 2, 3, 4]) == []
    assert state.n_draft_last == 0


def test_collisions_overwrite_like_the_fixed_modulo_table():
    mod = NGramMod(n_match=1, n_min=1, n_max=1, table_size=1)
    mod.add([10, 11], 0)
    mod.add([20, 12], 0)
    state = NGramModSequence()

    assert mod.used == 1
    assert mod._entries[0] == 12
    assert mod.draft(state, [99]) == [12]
    assert EMPTY == -1


def test_five_low_accept_rounds_reset_shared_table():
    mod = NGramMod(n_match=4, n_min=2, n_max=4, table_size=257)
    prompt = list(range(20)) * 2
    state = mod.begin(prompt)
    assert mod.used > 0
    state.n_draft_last = 4

    for _ in range(4):
        mod.accept(state, 0)
        assert mod.used > 0
    mod.accept(state, 0)
    assert mod.used == 0
    assert state.i_last == 0
