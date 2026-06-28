"""Mini bug-fix instances for the local agent de-risk track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MiniInstance:
    instance_id: str
    difficulty: Literal["easy", "medium"]
    problem_statement: str
    files: dict[str, str]
    test_cmd: str = "python -m pytest -q"


def _repo_pytest_conftest() -> str:
    return """import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


MINI_INSTANCES: dict[str, MiniInstance] = {
    "mini_add_001": MiniInstance(
        instance_id="mini_add_001",
        difficulty="easy",
        problem_statement=(
            "The function `add` in `calc.py` fails its unit tests.\n"
            "Fix the implementation so `pytest -q` passes."
        ),
        files={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_calc.py": (
                "from calc import add\n\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n"
            ),
        },
    ),
    "mini_sign_002": MiniInstance(
        instance_id="mini_sign_002",
        difficulty="easy",
        problem_statement=(
            "`mean` in `stats.py` computes the wrong value for negative numbers.\n"
            "Fix it so all tests pass."
        ),
        files={
            "stats.py": (
                "def mean(values):\n"
                "    total = 0\n"
                "    for v in values:\n"
                "        total -= v\n"
                "    return total / len(values)\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_stats.py": (
                "from stats import mean\n\n\n"
                "def test_mean():\n"
                "    assert mean([1, 2, 3]) == 2\n"
                "    assert mean([-2, 2]) == 0\n"
            ),
        },
    ),
    "mini_offby_003": MiniInstance(
        instance_id="mini_offby_003",
        difficulty="easy",
        problem_statement=(
            "`count_up_to` in `loops.py` is off by one.\n"
            "Make `pytest -q` pass."
        ),
        files={
            "loops.py": (
                "def count_up_to(n: int) -> list[int]:\n"
                "    return list(range(n))\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_loops.py": (
                "from loops import count_up_to\n\n\n"
                "def test_count():\n"
                "    assert count_up_to(3) == [1, 2, 3]\n"
                "    assert count_up_to(1) == [1]\n"
            ),
        },
    ),
    "mini_import_004": MiniInstance(
        instance_id="mini_import_004",
        difficulty="easy",
        problem_statement=(
            "Tests import `parse_int` from `parser.py`, but the function has the wrong name.\n"
            "Fix the module so tests pass."
        ),
        files={
            "parser.py": "def parse_integer(text: str) -> int:\n    return int(text.strip())\n",
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_parser.py": (
                "from parser import parse_int\n\n\n"
                "def test_parse():\n"
                "    assert parse_int(' 42 ') == 42\n"
            ),
        },
    ),
    "mini_return_005": MiniInstance(
        instance_id="mini_return_005",
        difficulty="easy",
        problem_statement=(
            "`double` in `math_utils.py` should return twice the input but currently returns None.\n"
            "Fix the function."
        ),
        files={
            "math_utils.py": "def double(x: int) -> int:\n    x * 2\n",
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_math_utils.py": (
                "from math_utils import double\n\n\n"
                "def test_double():\n"
                "    assert double(3) == 6\n"
                "    assert double(0) == 0\n"
            ),
        },
    ),
    "mini_str_006": MiniInstance(
        instance_id="mini_str_006",
        difficulty="easy",
        problem_statement=(
            "`slugify` in `text.py` should lowercase and replace spaces with hyphens.\n"
            "The current implementation is wrong."
        ),
        files={
            "text.py": (
                "def slugify(title: str) -> str:\n"
                "    return title.upper().replace('-', ' ')\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_text.py": (
                "from text import slugify\n\n\n"
                "def test_slugify():\n"
                "    assert slugify('Hello World') == 'hello-world'\n"
            ),
        },
    ),
    "mini_dict_007": MiniInstance(
        instance_id="mini_dict_007",
        difficulty="medium",
        problem_statement=(
            "`get_default` in `config.py` should return the value for a key, "
            "or the provided default if the key is missing.\n"
            "Fix the bug."
        ),
        files={
            "config.py": (
                "def get_default(mapping: dict, key: str, default):\n"
                "    return mapping[key]\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_config.py": (
                "from config import get_default\n\n\n"
                "def test_get_default():\n"
                "    assert get_default({'a': 1}, 'a', 0) == 1\n"
                "    assert get_default({}, 'missing', 99) == 99\n"
            ),
        },
    ),
    "mini_list_008": MiniInstance(
        instance_id="mini_list_008",
        difficulty="medium",
        problem_statement=(
            "`flatten_once` in `lists.py` should concatenate one level of nested lists.\n"
            "It currently drops elements."
        ),
        files={
            "lists.py": (
                "def flatten_once(nested: list) -> list:\n"
                "    out = []\n"
                "    for item in nested:\n"
                "        if isinstance(item, list):\n"
                "            continue\n"
                "        out.append(item)\n"
                "    return out\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_lists.py": (
                "from lists import flatten_once\n\n\n"
                "def test_flatten():\n"
                "    assert flatten_once([[1, 2], [3]]) == [1, 2, 3]\n"
                "    assert flatten_once([1, [2]]) == [1, 2]\n"
            ),
        },
    ),
    "mini_cmp_009": MiniInstance(
        instance_id="mini_cmp_009",
        difficulty="medium",
        problem_statement=(
            "`is_sorted` in `validate.py` should return True when a list is non-decreasing.\n"
            "Fix the comparison logic."
        ),
        files={
            "validate.py": (
                "def is_sorted(values: list[int]) -> bool:\n"
                "    for i in range(len(values) - 1):\n"
                "        if values[i] < values[i + 1]:\n"
                "            return False\n"
                "    return True\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_validate.py": (
                "from validate import is_sorted\n\n\n"
                "def test_sorted():\n"
                "    assert is_sorted([1, 2, 2, 3]) is True\n"
                "    assert is_sorted([3, 2]) is False\n"
            ),
        },
    ),
    "mini_div_010": MiniInstance(
        instance_id="mini_div_010",
        difficulty="medium",
        problem_statement=(
            "`safe_div` in `ratio.py` should divide two numbers but return 0.0 when "
            "the denominator is zero.\n"
            "Fix the implementation."
        ),
        files={
            "ratio.py": (
                "def safe_div(numerator: float, denominator: float) -> float:\n"
                "    return numerator / denominator\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_ratio.py": (
                "from ratio import safe_div\n\n\n"
                "def test_safe_div():\n"
                "    assert safe_div(10, 2) == 5.0\n"
                "    assert safe_div(3, 0) == 0.0\n"
            ),
        },
    ),
    "mini_strip_011": MiniInstance(
        instance_id="mini_strip_011",
        difficulty="easy",
        problem_statement=(
            "`normalize` in `clean.py` should strip whitespace from both ends of a string.\n"
            "Fix the function."
        ),
        files={
            "clean.py": (
                "def normalize(text: str) -> str:\n"
                "    return text.strip(' ')\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_clean.py": (
                "from clean import normalize\n\n\n"
                "def test_normalize():\n"
                "    assert normalize('  hi\\n') == 'hi'\n"
            ),
        },
    ),
    "mini_max_012": MiniInstance(
        instance_id="mini_max_012",
        difficulty="medium",
        problem_statement=(
            "`maximum` in `select.py` should return the largest value in a non-empty list.\n"
            "The current code picks the wrong extremum."
        ),
        files={
            "select.py": (
                "def maximum(values: list[int]) -> int:\n"
                "    best = values[0]\n"
                "    for v in values[1:]:\n"
                "        if v < best:\n"
                "            best = v\n"
                "    return best\n"
            ),
            "tests/conftest.py": _repo_pytest_conftest(),
            "tests/test_select.py": (
                "from select import maximum\n\n\n"
                "def test_maximum():\n"
                "    assert maximum([1, 5, 3]) == 5\n"
                "    assert maximum([-1, -3]) == -1\n"
            ),
        },
    ),
}


def list_instance_ids() -> list[str]:
    return sorted(MINI_INSTANCES.keys())


def get_instance(instance_id: str) -> MiniInstance:
    if instance_id not in MINI_INSTANCES:
        raise KeyError(f"Unknown mini instance: {instance_id}")
    return MINI_INSTANCES[instance_id]


def load_instance_ids_from_file(path) -> list[str]:
    from pathlib import Path

    ids: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids
