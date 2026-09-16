import ast

from micro_scout.data import normalize_row
from micro_scout.text import code_fingerprints, lexical_tokens, strip_python_documentation


def test_strip_documentation_preserves_runtime_strings_and_unicode():
    code = '''def café(value):
    """Find the secret target description."""
    # also remove a comment
    message = "keep this literal # content"
    return message + value
'''
    clean = strip_python_documentation(code)
    assert "secret target" not in clean
    assert "remove a comment" not in clean
    assert '"keep this literal # content"' in clean
    ast.parse(clean)


def test_nested_docstrings_are_removed():
    code = '''class C:
    """Outer text."""
    def run(self):
        """Inner text."""
        return 42
'''
    clean = strip_python_documentation(code)
    assert "Outer text" not in clean and "Inner text" not in clean
    ast.parse(clean)


def test_comment_removal_does_not_change_multiline_literal():
    code = 'def x():\n    text = """a\n# literal\nb"""\n    return text\n'
    assert "# literal" in strip_python_documentation(code)


def test_fingerprint_detects_renamed_clone():
    a = code_fingerprints("def add(a, b):\n    return a + b\n")
    b = code_fingerprints("def sum_values(x, y):\n    return x + y\n")
    assert a[0] != b[0] and a[1] == b[1]


def test_tokenizer_splits_identifiers_and_keeps_exact_name():
    assert lexical_tokens("parseHTTP get_user_id") == [
        "parsehttp",
        "parse",
        "http",
        "get_user_id",
        "get",
        "user",
        "id",
    ]


def test_dataset_normalization_uses_no_docstring_as_code():
    row = normalize_row(
        {
            "repo": "Example/Project",
            "path": "src/files.py",
            "url": "https://github.com/Example/Project/blob/abc/src/files.py#L1-L5",
            "docstring": "Read every nonempty line from the given input file.",
            "code": '''def read_lines(path):
    """Read every nonempty line from the given input file."""
    with open(path) as stream:
        return [line.strip() for line in stream if line.strip()]
''',
        }
    )
    assert row is not None
    assert row["query"] not in row["code"]
    assert row["repo"] == "example/project"
