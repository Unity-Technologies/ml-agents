"""
Generate the "Releases" table on the main readme. Update the versions lists, run this script, and copy the output
into the markdown file.
"""
from distutils.version import LooseVersion
from datetime import datetime


def table_line(display_name, name, date, bold=False):
    bold_str = "**" if bold else ""
    return f"| **{display_name}** | {bold_str}{date}{bold_str} | {bold_str}[source](https://github.com/Unity-Technologies/ml-agents/tree/{name}){bold_str} | {bold_str}[docs](https://github.com/Unity-Technologies/ml-agents/tree/{name}/docs/Readme.md){bold_str} | {bold_str}[download](https://github.com/Unity-Technologies/ml-agents/archive/{name}.zip){bold_str} |"  # noqa


versions = [
    ["0.10.0", "September 30, 2019"],
    ["0.10.1", "October 9, 2019"],
    ["0.11.0", "November 4, 2019"],
    ["0.12.0", "December 2, 2019"],
    ["0.12.1", "December 11, 2019"],
    ["0.13.0", "January 8, 2020"],
    ["0.13.1", "January 21, 2020"],
    ["0.14.0", "February 13, 2020"],
    ["0.14.1", "February 26, 2020"],
    ["0.15.0", "March 18, 2020"],
    ["0.15.1", "March 30, 2020"],
]

MAX_DAYS = 150  # do not print releases older than this many days
sorted_versions = sorted(
    ([LooseVersion(v[0]), v[1]] for v in versions), key=lambda x: x[0], reverse=True
)

print(table_line("master (unstable)", "master", "--"))
highlight = True  # whether to bold the line or not
for version_name, version_date in sorted_versions:
    elapsed_days = (
        datetime.today() - datetime.strptime(version_date, "%B %d, %Y")
    ).days
    if elapsed_days <= MAX_DAYS:
        print(table_line(version_name, version_name, version_date, highlight))
        highlight = False  # only bold the first stable release
