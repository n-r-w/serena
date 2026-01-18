from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from serena.symbol import LanguageServerSymbolRetriever, NamePathMatcher
from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language


@contextmanager
def _count_lsp_methods(language_server: SolidLanguageServer) -> Iterator[dict[str, int]]:
    call_counts: dict[str, int] = {}
    original_send_payload = language_server.server._send_payload

    def counting_send_payload(payload: dict) -> None:
        method = payload.get("method")
        if method:
            call_counts[method] = call_counts.get(method, 0) + 1
        return original_send_payload(payload)

    language_server.server._send_payload = counting_send_payload
    try:
        yield call_counts
    finally:
        language_server.server._send_payload = original_send_payload


def _duplicate_symbol(symbol):
    """Create a duplicate symbol with identical coordinates but independent symbol_root.

    The symbol_root is deep-copied to ensure tests fail if dedupe relies on object
    identity rather than (relative_path, line, column) keys.
    """
    import copy

    copied_root = copy.deepcopy(symbol.symbol_root)
    duplicate = type(symbol)(copied_root)

    assert symbol is not duplicate, "Must be distinct objects"
    assert symbol.symbol_root is not duplicate.symbol_root, "symbol_root must be independent"
    assert symbol.relative_path == duplicate.relative_path
    assert symbol.line == duplicate.line
    assert symbol.column == duplicate.column

    return duplicate


def _symbol_with_detail(symbol, detail: str):
    """Return a copy of *symbol* with UnifiedSymbolInformation.detail set to *detail*.

    Ensures symbol objects are distinct and symbol_root is deep-copied.
    """
    duplicated = _duplicate_symbol(symbol)
    duplicated.symbol_root["detail"] = detail
    return duplicated


class TestSymbolNameMatching:
    def _create_assertion_error_message(
        self,
        name_path_pattern: str,
        symbol_name_path_parts: list[str],
        is_substring_match: bool,
        expected_result: bool,
        actual_result: bool,
    ) -> str:
        """Helper to create a detailed error message for assertions."""
        qnp_repr = "/".join(symbol_name_path_parts)

        return (
            f"Pattern '{name_path_pattern}' (substring: {is_substring_match}) vs "
            f"Qualname parts {symbol_name_path_parts} (as '{qnp_repr}'). "
            f"Expected: {expected_result}, Got: {actual_result}"
        )

    @pytest.mark.parametrize(
        "name_path_pattern, symbol_name_path_parts, is_substring_match, expected",
        [
            # Exact matches, anywhere in the name (is_substring_match=False)
            pytest.param("foo", ["foo"], False, True, id="'foo' matches 'foo' exactly (simple)"),
            pytest.param("foo/", ["foo"], False, True, id="'foo/' matches 'foo' exactly (simple)"),
            pytest.param("foo", ["bar", "foo"], False, True, id="'foo' matches ['bar', 'foo'] exactly (simple, last element)"),
            pytest.param("foo", ["foobar"], False, False, id="'foo' does not match 'foobar' exactly (simple)"),
            pytest.param(
                "foo", ["bar", "foobar"], False, False, id="'foo' does not match ['bar', 'foobar'] exactly (simple, last element)"
            ),
            pytest.param(
                "foo", ["path", "to", "foo"], False, True, id="'foo' matches ['path', 'to', 'foo'] exactly (simple, last element)"
            ),
            # Exact matches, absolute patterns (is_substring_match=False)
            pytest.param("/foo", ["foo"], False, True, id="'/foo' matches ['foo'] exactly (absolute simple)"),
            pytest.param("/foo", ["foo", "bar"], False, False, id="'/foo' does not match ['foo', 'bar'] (absolute simple, len mismatch)"),
            pytest.param("/foo", ["bar"], False, False, id="'/foo' does not match ['bar'] (absolute simple, name mismatch)"),
            pytest.param(
                "/foo", ["bar", "foo"], False, False, id="'/foo' does not match ['bar', 'foo'] (absolute simple, position mismatch)"
            ),
            # Substring matches, anywhere in the name (is_substring_match=True)
            pytest.param("foo", ["foobar"], True, True, id="'foo' matches 'foobar' as substring (simple)"),
            pytest.param("foo", ["bar", "foobar"], True, True, id="'foo' matches ['bar', 'foobar'] as substring (simple, last element)"),
            pytest.param(
                "foo", ["barfoo"], True, True, id="'foo' matches 'barfoo' as substring (simple)"
            ),  # This was potentially ambiguous before
            pytest.param("foo", ["baz"], True, False, id="'foo' does not match 'baz' as substring (simple)"),
            pytest.param("foo", ["bar", "baz"], True, False, id="'foo' does not match ['bar', 'baz'] as substring (simple, last element)"),
            pytest.param("foo", ["my_foobar_func"], True, True, id="'foo' matches 'my_foobar_func' as substring (simple)"),
            pytest.param(
                "foo",
                ["ClassA", "my_foobar_method"],
                True,
                True,
                id="'foo' matches ['ClassA', 'my_foobar_method'] as substring (simple, last element)",
            ),
            pytest.param("foo", ["my_bar_func"], True, False, id="'foo' does not match 'my_bar_func' as substring (simple)"),
            # Substring matches, absolute patterns (is_substring_match=True)
            pytest.param("/foo", ["foobar"], True, True, id="'/foo' matches ['foobar'] as substring (absolute simple)"),
            pytest.param("/foo/", ["foobar"], True, True, id="'/foo/' matches ['foobar'] as substring (absolute simple, last element)"),
            pytest.param("/foo", ["barfoobaz"], True, True, id="'/foo' matches ['barfoobaz'] as substring (absolute simple)"),
            pytest.param(
                "/foo", ["foo", "bar"], True, False, id="'/foo' does not match ['foo', 'bar'] as substring (absolute simple, len mismatch)"
            ),
            pytest.param("/foo", ["bar"], True, False, id="'/foo' does not match ['bar'] (absolute simple, no substr)"),
            pytest.param(
                "/foo", ["bar", "foo"], True, False, id="'/foo' does not match ['bar', 'foo'] (absolute simple, position mismatch)"
            ),
            pytest.param(
                "/foo/", ["bar", "foo"], True, False, id="'/foo/' does not match ['bar', 'foo'] (absolute simple, position mismatch)"
            ),
        ],
    )
    def test_match_simple_name(self, name_path_pattern, symbol_name_path_parts, is_substring_match, expected):
        """Tests matching for simple names (no '/' in pattern)."""
        result = NamePathMatcher(name_path_pattern, is_substring_match).matches_components(symbol_name_path_parts, None)
        error_msg = self._create_assertion_error_message(name_path_pattern, symbol_name_path_parts, is_substring_match, expected, result)
        assert result == expected, error_msg

    @pytest.mark.parametrize(
        "name_path_pattern, symbol_name_path_parts, is_substring_match, expected",
        [
            # --- Relative patterns (suffix matching) ---
            # Exact matches, relative patterns (is_substring_match=False)
            pytest.param("bar/foo", ["bar", "foo"], False, True, id="R: 'bar/foo' matches ['bar', 'foo'] exactly"),
            pytest.param("bar/foo", ["mod", "bar", "foo"], False, True, id="R: 'bar/foo' matches ['mod', 'bar', 'foo'] exactly (suffix)"),
            pytest.param(
                "bar/foo", ["bar", "foo", "baz"], False, False, id="R: 'bar/foo' does not match ['bar', 'foo', 'baz'] (pattern shorter)"
            ),
            pytest.param("bar/foo", ["bar"], False, False, id="R: 'bar/foo' does not match ['bar'] (pattern longer)"),
            pytest.param("bar/foo", ["baz", "foo"], False, False, id="R: 'bar/foo' does not match ['baz', 'foo'] (first part mismatch)"),
            pytest.param("bar/foo", ["bar", "baz"], False, False, id="R: 'bar/foo' does not match ['bar', 'baz'] (last part mismatch)"),
            pytest.param("bar/foo", ["foo"], False, False, id="R: 'bar/foo' does not match ['foo'] (pattern longer)"),
            pytest.param(
                "bar/foo", ["other", "foo"], False, False, id="R: 'bar/foo' does not match ['other', 'foo'] (first part mismatch)"
            ),
            pytest.param(
                "bar/foo", ["bar", "otherfoo"], False, False, id="R: 'bar/foo' does not match ['bar', 'otherfoo'] (last part mismatch)"
            ),
            # Substring matches, relative patterns (is_substring_match=True)
            pytest.param("bar/foo", ["bar", "foobar"], True, True, id="R: 'bar/foo' matches ['bar', 'foobar'] as substring"),
            pytest.param(
                "bar/foo", ["mod", "bar", "foobar"], True, True, id="R: 'bar/foo' matches ['mod', 'bar', 'foobar'] as substring (suffix)"
            ),
            pytest.param("bar/foo", ["bar", "bazfoo"], True, True, id="R: 'bar/foo' matches ['bar', 'bazfoo'] as substring"),
            pytest.param("bar/fo", ["bar", "foo"], True, True, id="R: 'bar/fo' matches ['bar', 'foo'] as substring"),  # codespell:ignore
            pytest.param("bar/foo", ["bar", "baz"], True, False, id="R: 'bar/foo' does not match ['bar', 'baz'] (last no substr)"),
            pytest.param(
                "bar/foo", ["baz", "foobar"], True, False, id="R: 'bar/foo' does not match ['baz', 'foobar'] (first part mismatch)"
            ),
            pytest.param(
                "bar/foo", ["bar", "my_foobar_method"], True, True, id="R: 'bar/foo' matches ['bar', 'my_foobar_method'] as substring"
            ),
            pytest.param(
                "bar/foo",
                ["mod", "bar", "my_foobar_method"],
                True,
                True,
                id="R: 'bar/foo' matches ['mod', 'bar', 'my_foobar_method'] as substring (suffix)",
            ),
            pytest.param(
                "bar/foo",
                ["bar", "another_method"],
                True,
                False,
                id="R: 'bar/foo' does not match ['bar', 'another_method'] (last no substr)",
            ),
            pytest.param(
                "bar/foo",
                ["other", "my_foobar_method"],
                True,
                False,
                id="R: 'bar/foo' does not match ['other', 'my_foobar_method'] (first part mismatch)",
            ),
            pytest.param("bar/f", ["bar", "foo"], True, True, id="R: 'bar/f' matches ['bar', 'foo'] as substring"),
            # Exact matches, absolute patterns (is_substring_match=False)
            pytest.param("/bar/foo", ["bar", "foo"], False, True, id="A: '/bar/foo' matches ['bar', 'foo'] exactly"),
            pytest.param(
                "/bar/foo", ["bar", "foo", "baz"], False, False, id="A: '/bar/foo' does not match ['bar', 'foo', 'baz'] (pattern shorter)"
            ),
            pytest.param("/bar/foo", ["bar"], False, False, id="A: '/bar/foo' does not match ['bar'] (pattern longer)"),
            pytest.param("/bar/foo", ["baz", "foo"], False, False, id="A: '/bar/foo' does not match ['baz', 'foo'] (first part mismatch)"),
            pytest.param("/bar/foo", ["bar", "baz"], False, False, id="A: '/bar/foo' does not match ['bar', 'baz'] (last part mismatch)"),
            # Substring matches (is_substring_match=True)
            pytest.param("/bar/foo", ["bar", "foobar"], True, True, id="A: '/bar/foo' matches ['bar', 'foobar'] as substring"),
            pytest.param("/bar/foo", ["bar", "bazfoo"], True, True, id="A: '/bar/foo' matches ['bar', 'bazfoo'] as substring"),
            pytest.param("/bar/fo", ["bar", "foo"], True, True, id="A: '/bar/fo' matches ['bar', 'foo'] as substring"),  # codespell:ignore
            pytest.param("/bar/foo", ["bar", "baz"], True, False, id="A: '/bar/foo' does not match ['bar', 'baz'] (last no substr)"),
            pytest.param(
                "/bar/foo", ["baz", "foobar"], True, False, id="A: '/bar/foo' does not match ['baz', 'foobar'] (first part mismatch)"
            ),
        ],
    )
    def test_match_name_path_pattern_path_len_2(self, name_path_pattern, symbol_name_path_parts, is_substring_match, expected):
        """Tests matching for qualified names (e.g. 'module/class/func')."""
        result = NamePathMatcher(name_path_pattern, is_substring_match).matches_components(symbol_name_path_parts, None)
        error_msg = self._create_assertion_error_message(name_path_pattern, symbol_name_path_parts, is_substring_match, expected, result)
        assert result == expected, error_msg

    @pytest.mark.parametrize(
        "name_path_pattern, symbol_name_path_parts, symbol_overload_idx, expected",
        [
            pytest.param("bar/foo", ["bar", "foo"], 0, True, id="R: 'bar/foo' matches ['bar', 'foo'] with overload_index=0"),
            pytest.param("bar/foo", ["bar", "foo"], 1, True, id="R: 'bar/foo' matches ['bar', 'foo'] with overload_index=1"),
            pytest.param("bar/foo[0]", ["bar", "foo"], 0, True, id="R: 'bar/foo[0]' matches ['bar', 'foo'] with overload_index=0"),
            pytest.param("bar/foo[1]", ["bar", "foo"], 0, False, id="R: 'bar/foo[1]' does not match ['bar', 'foo'] with overload_index=0"),
        ],
    )
    def test_match_name_path_pattern_with_overload_idx(self, name_path_pattern, symbol_name_path_parts, symbol_overload_idx, expected):
        """Tests matching for qualified names (e.g. 'module/class/func')."""
        matcher = NamePathMatcher(name_path_pattern, False)
        result = matcher.matches_components(symbol_name_path_parts, symbol_overload_idx)
        error_msg = self._create_assertion_error_message(name_path_pattern, symbol_name_path_parts, False, expected, result)
        assert result == expected, error_msg


@pytest.mark.python
class TestLanguageServerSymbolRetriever:
    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info(self, language_server: SolidLanguageServer):
        symbol_retriever = LanguageServerSymbolRetriever(language_server)
        create_user_method_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        create_user_method_symbol_info = symbol_retriever.request_info_for_symbol(create_user_method_symbol)
        assert "Create a new user and store it" in create_user_method_symbol_info

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_batches_did_open_close_single_file(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test that batch API emits exactly one didOpen/didClose for multiple symbols in same file."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        symbols = [
            symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0],
            symbol_retriever.find("UserService/get_user", within_relative_path="test_repo/services.py")[0],
            symbol_retriever.find("UserService/list_users", within_relative_path="test_repo/services.py")[0],
        ]

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols)
        assert call_counts.get("textDocument/didOpen", 0) == 1
        assert call_counts.get("textDocument/didClose", 0) == 1
        assert call_counts.get("textDocument/hover", 0) == len(symbols)

        for result in results:
            assert result is None or isinstance(result, str)

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_preserves_input_order_across_files(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test that batch API preserves output ordering with interleaved multi-file input."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        services_symbol_a = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        models_symbol_a = symbol_retriever.find("User/has_role", within_relative_path="test_repo/models.py")[0]
        services_symbol_b = symbol_retriever.find("UserService/get_user", within_relative_path="test_repo/services.py")[0]
        models_symbol_b = symbol_retriever.find("Item/get_display_price", within_relative_path="test_repo/models.py")[0]

        symbols = [services_symbol_a, models_symbol_a, services_symbol_b, models_symbol_b]

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols)
        assert call_counts.get("textDocument/didOpen", 0) == 2
        assert call_counts.get("textDocument/didClose", 0) == 2

        for i, (symbol, result) in enumerate(zip(symbols, results, strict=True)):
            assert result is None or isinstance(result, str), f"Result at index {i} has unexpected type"
            if result is not None:
                assert symbol.name in result, (
                    f"Result[{i}] content does not match symbol[{i}]: "
                    f"expected symbol name '{symbol.name}' in result but got: {result[:100]}..."
                )

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_skips_invalid_symbols(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test that invalid symbols (missing relative_path/line/column) return None without LSP calls."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        valid_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        another_valid = symbol_retriever.find("User/has_role", within_relative_path="test_repo/models.py")[0]

        class InvalidSymbolMissingPath:
            relative_path = None
            line = 10
            column = 5

        class InvalidSymbolMissingLine:
            relative_path = "test_repo/models.py"
            line = None
            column = 5

        class InvalidSymbolMissingColumn:
            relative_path = "test_repo/models.py"
            line = 10
            column = None

        invalid_missing_path = InvalidSymbolMissingPath()
        invalid_missing_line = InvalidSymbolMissingLine()
        invalid_missing_column = InvalidSymbolMissingColumn()

        symbols = [
            valid_symbol,
            invalid_missing_path,  # type: ignore[list-item]
            another_valid,
            invalid_missing_line,  # type: ignore[list-item]
            invalid_missing_column,  # type: ignore[list-item]
        ]

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols), "Results must have same length as input"
        assert results[1] is None, "Invalid symbol (missing path) at index 1 must return None"
        assert results[3] is None, "Invalid symbol (missing line) at index 3 must return None"
        assert results[4] is None, "Invalid symbol (missing column) at index 4 must return None"

        assert call_counts.get("textDocument/didOpen", 0) == 2, "Only 2 valid files should be opened"
        assert call_counts.get("textDocument/didClose", 0) == 2, "Only 2 valid files should be closed"
        assert call_counts.get("textDocument/hover", 0) == 2, "Only 2 valid symbols should trigger hover"

        for valid_idx in [0, 2]:
            result = results[valid_idx]
            assert result is None or isinstance(result, str), f"Valid symbol at {valid_idx} must return str or None"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_deduplicates_hover_for_same_position(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test that duplicate symbols with same position trigger only one hover request."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        base_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]

        duplicate_1 = _duplicate_symbol(base_symbol)
        duplicate_2 = _duplicate_symbol(base_symbol)

        assert base_symbol is not duplicate_1, "Must be distinct objects"
        assert base_symbol is not duplicate_2, "Must be distinct objects"
        assert duplicate_1 is not duplicate_2, "Must be distinct objects"

        symbols = [base_symbol, duplicate_1, duplicate_2]

        unique_keys = {(s.relative_path, s.line, s.column) for s in symbols}
        expected_hover_count = len(unique_keys)
        assert expected_hover_count == 1, "Precondition: all symbols share same position"

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols), "Result length must match input length"

        assert call_counts.get("textDocument/didOpen", 0) == 1, "Single file: expect 1 didOpen"
        assert call_counts.get("textDocument/didClose", 0) == 1, "Single file: expect 1 didClose"

        assert (
            call_counts.get("textDocument/hover", 0) == expected_hover_count
        ), f"Expected {expected_hover_count} hover call for duplicates, got {call_counts.get('textDocument/hover', 0)}"

        assert results[0] == results[1] == results[2], "Duplicate positions must yield equal results"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_deduplicates_non_adjacent_duplicates(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test dedupe with interleaved duplicates [A1, B1, A2, B2]."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        symbol_a = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        symbol_b = symbol_retriever.find("UserService/get_user", within_relative_path="test_repo/services.py")[0]

        key_a = (symbol_a.relative_path, symbol_a.line, symbol_a.column)
        key_b = (symbol_b.relative_path, symbol_b.line, symbol_b.column)
        assert key_a != key_b, "Precondition: symbol_a and symbol_b must have different positions"

        a_dup = _duplicate_symbol(symbol_a)
        b_dup = _duplicate_symbol(symbol_b)

        symbols = [symbol_a, symbol_b, a_dup, b_dup]

        unique_keys = {(s.relative_path, s.line, s.column) for s in symbols}
        expected_hover_count = len(unique_keys)
        assert expected_hover_count == 2, "Precondition: expect 2 unique positions"

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols), "Result length must match input length"

        assert call_counts.get("textDocument/didOpen", 0) == 1, "Single file: expect 1 didOpen"
        assert call_counts.get("textDocument/didClose", 0) == 1, "Single file: expect 1 didClose"

        hover_count = call_counts.get("textDocument/hover", 0)
        assert hover_count == expected_hover_count, f"Expected {expected_hover_count} hover calls, got {hover_count}"

        assert results[0] == results[2], "A1 and A2 must yield equal results"
        assert results[1] == results[3], "B1 and B2 must yield equal results"
        assert results[0] != results[1], "A and B results should differ"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_deduplicates_within_file_boundary(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Test that dedupe is scoped to each file, not cross-file."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        services_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        models_symbol = symbol_retriever.find("User/has_role", within_relative_path="test_repo/models.py")[0]

        assert services_symbol.relative_path != models_symbol.relative_path, "Precondition: symbols from different files"

        services_dup = _duplicate_symbol(services_symbol)
        models_dup = _duplicate_symbol(models_symbol)

        symbols = [services_symbol, models_symbol, services_dup, models_dup]

        unique_files = {s.relative_path for s in symbols}
        unique_keys = {(s.relative_path, s.line, s.column) for s in symbols}
        expected_file_count = len(unique_files)
        expected_hover_count = len(unique_keys)

        assert expected_file_count == 2, "Precondition: expect 2 unique files"
        assert expected_hover_count == 2, "Precondition: expect 2 unique positions"

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == len(symbols), "Result length must match input length"

        did_open_count = call_counts.get("textDocument/didOpen", 0)
        did_close_count = call_counts.get("textDocument/didClose", 0)
        assert did_open_count == expected_file_count, f"Expected {expected_file_count} didOpen, got {did_open_count}"
        assert did_close_count == expected_file_count, f"Expected {expected_file_count} didClose, got {did_close_count}"

        hover_count = call_counts.get("textDocument/hover", 0)
        assert hover_count == expected_hover_count, f"Expected {expected_hover_count} hover calls, got {hover_count}"

        assert results[0] == results[2], "services_symbol duplicates must yield equal results"
        assert results[1] == results[3], "models_symbol duplicates must yield equal results"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_uses_detail_fastpath_without_hover(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """A non-empty detail value is used directly and avoids hover/didOpen/didClose."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        base_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        symbol_with_detail = _symbol_with_detail(base_symbol, "int create_user(User user)")

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols([symbol_with_detail])

        assert results == ["int create_user(User user)"], "Expected info from detail"
        assert call_counts.get("textDocument/hover", 0) == 0, "Expected 0 hover calls with detail fast-path"
        assert call_counts.get("textDocument/didOpen", 0) == 0, "Expected 0 didOpen calls with detail fast-path"
        assert call_counts.get("textDocument/didClose", 0) == 0, "Expected 0 didClose calls with detail fast-path"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_detail_whitespace_falls_back_to_hover(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Whitespace-only detail triggers the regular hover-based behavior."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        base_symbol = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        symbol_with_whitespace_detail = _symbol_with_detail(base_symbol, "   \n")

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols([symbol_with_whitespace_detail])

        assert len(results) == 1, "Result length must match input length"
        assert results[0] is None or isinstance(results[0], str), "Result must be None or str"
        assert call_counts.get("textDocument/hover", 0) == 1, "Expected 1 hover call for whitespace-only detail"
        assert call_counts.get("textDocument/didOpen", 0) == 1, "Expected 1 didOpen call for whitespace-only detail"
        assert call_counts.get("textDocument/didClose", 0) == 1, "Expected 1 didClose call for whitespace-only detail"

    @pytest.mark.parametrize("language_server", [Language.PYTHON], indirect=True)
    def test_request_info_for_symbols_mixed_detail_and_hover_preserves_order(
        self,
        language_server: SolidLanguageServer,
    ) -> None:
        """Mixing detail and hover symbols preserves input order and minimizes hover calls."""
        symbol_retriever = LanguageServerSymbolRetriever(language_server)

        symbol_a = symbol_retriever.find("UserService/create_user", within_relative_path="test_repo/services.py")[0]
        symbol_b = symbol_retriever.find("UserService/get_user", within_relative_path="test_repo/services.py")[0]

        assert symbol_a.relative_path == symbol_b.relative_path, "Precondition: same file"
        assert (symbol_a.line, symbol_a.column) != (symbol_b.line, symbol_b.column), "Precondition: different positions"

        symbol_with_detail = _symbol_with_detail(symbol_a, "DETAIL_FASTPATH_VALUE")
        symbols = [symbol_with_detail, symbol_b]

        assert not language_server.open_file_buffers, "Precondition: no files left open"

        with _count_lsp_methods(language_server) as call_counts:
            results = symbol_retriever.request_info_for_symbols(symbols)

        assert len(results) == 2, "Result length must match input length"
        assert results[0] == "DETAIL_FASTPATH_VALUE", "First result must be from detail fast-path"
        assert results[1] != "DETAIL_FASTPATH_VALUE", "Second result must not be the injected detail"
        assert results[1] is None or isinstance(results[1], str), "Second result must be None or str"
        assert call_counts.get("textDocument/hover", 0) == 1, "Expected 1 hover call for non-detail symbol"
        assert call_counts.get("textDocument/didOpen", 0) == 1, "Expected 1 didOpen call"
        assert call_counts.get("textDocument/didClose", 0) == 1, "Expected 1 didClose call"
