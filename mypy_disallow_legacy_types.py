
import argparse
import re
from collections.abc import Set

# List of files that we want to skip with this check. Currently empty.
files_to_ignore: Set[str] = {"Empty"}
# List of disallowed types to search for.
disallowed_types = ["Union", "Optional", "List", "Dict", "Sequence", "Set", "Callable", "Iterable", "Hashable", "Generator", "Tuple"]


type_or = "|".join(disallowed_types)
comma_separated_types = ",\n".join(disallowed_types)

def construct_same_line_import_regex() -> str:
    return fr"from typing import [^\n]*?({type_or})[^\n]*?\n$"

def construct_multi_line_import_regex() -> str:
    return fr"from typing import \(\n(\t.*,\n)*\t({type_or}),\n(\t.*,\n)*\)$"
    

same_line_import_re = construct_same_line_import_regex()
multi_line_import_re = construct_multi_line_import_regex()


def discover_legacy_imports(file_paths: list[str]) -> None:
    file_paths = [file_path for file_path in file_paths if not file_path in files_to_ignore]
    for file_path in file_paths:
        with open(file_path, mode="r") as file_handle:
            file_contents = file_handle.read()
            same_line_match = re.match(same_line_import_re, file_contents)
            multi_line_match = re.match(multi_line_import_re, file_contents)
            if same_line_match or multi_line_match:
                raise ValueError(
                    f"A legacy mypy type is being imported in file {file_path}. "
                    f"Disallowed imports from the typing library are {comma_separated_types}"
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disallow Legacy Mypy Types")
    parser.add_argument("--file_names", action="store", type=list[str], help="List of file paths in pre-commit")
    args = parser.parse_args()

    discover_legacy_imports(args.file_names)