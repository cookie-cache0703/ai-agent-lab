import pytest

from tools.calculator_tool import calculator_tool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "5"),
        ("2 * (3 + 4)", "14"),
        ("10 / 4", "2.5"),
        ("2 ** 10", "1024"),
        ("-5 + 2", "-3"),
    ],
)
def test_calculator_evaluates_arithmetic_expressions(expression, expected):
    assert calculator_tool.run({"expression": expression}) == expected


def test_calculator_reports_division_by_zero_without_raising():
    result = calculator_tool.run({"expression": "1 / 0"})

    assert result.startswith("error:")


def test_calculator_rejects_non_arithmetic_syntax_without_raising():
    result = calculator_tool.run({"expression": "__import__('os').system('echo hi')"})

    assert result.startswith("error:")
