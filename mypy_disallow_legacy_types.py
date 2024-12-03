import re
import sys
from collections.abc import Set

# List of files that we want to skip with this check. Currently empty.
files_to_ignore: Set[str] = {"Empty"}
file_types_to_ignore: Set[str] = {".png", ".pkl", ".pt", ".md"}
# List of disallowed types to search for.
disallowed_types = [
    "Union",
    "Optional",
    "List",
    "Dict",
    "Sequence",
    "Set",
    "Callable",
    "Iterable",
    "Hashable",
    "Generator",
    "Tuple",
    "Mapping",
    "Type",
]


type_or = "|".join(disallowed_types)
comma_separated_types = ", ".join(disallowed_types)
file_suffixes = "|".join(file_types_to_ignore)
file_type_regex = rf".*({file_suffixes})$"


def filter_files_to_ignore(file_paths: list[str]) -> list[str]:
    file_paths = [file_path for file_path in file_paths if file_path not in files_to_ignore]
    file_paths = [file_path for file_path in file_paths if not re.match(file_type_regex, file_path)]
    return file_paths


def construct_same_line_import_regex() -> str:
    return rf"from typing import ([^\n]*?, )*({type_or})(\n|, [^\n]*?\n)"


def construct_multi_line_import_regex() -> str:
    return rf"from typing import \(\n(\s{{4}}.*,\n)*\s{{4}}({type_or}),\n(\s{{4}}.*,\n)*\)$"


same_line_import_re = construct_same_line_import_regex()
multi_line_import_re = construct_multi_line_import_regex()


def discover_legacy_imports(file_paths: list[str]) -> None:
    file_paths = filter_files_to_ignore(file_paths=file_paths)
    for file_path in file_paths:
        with open(file_path, mode="r") as file_handle:
            file_contents = file_handle.read()
            same_line_match = re.search(same_line_import_re, file_contents, flags=re.MULTILINE)
            multi_line_match = re.search(multi_line_import_re, file_contents, flags=re.MULTILINE)
            if same_line_match or multi_line_match:
                raise ValueError(
                    f"A legacy mypy type is being imported in file {file_path}. "
                    f"Disallowed imports from the typing library are: {comma_separated_types}"
                )


if __name__ == "__main__":
    file_relative_paths = sys.argv[1:]
    discover_legacy_imports(file_relative_paths)
