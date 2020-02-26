"""
Generate the "Releases" table on the main readme. Update the versions lists, run this script, and copy the output
into the markdown file.
"""
from distutils.version import LooseVersion


def table_line(version):
    return f"| **{version}**  | [source](https://github.com/Unity-Technologies/ml-agents/tree/{version}) |  [docs](https://github.com/Unity-Technologies/ml-agents/tree/{version}/docs) | [download](https://github.com/Unity-Technologies/ml-agents/archive/{version}.zip) |"  # noqa


versions = [
    "0.10.0",
    "0.10.1",
    "0.11.0",
    "0.12.0",
    "0.12.1",
    "0.13.0",
    "0.13.1",
    "0.14.0",
]

sorted_versions = sorted((LooseVersion(v) for v in versions), reverse=True)

for v in sorted_versions:
    print(table_line(str(v)))
