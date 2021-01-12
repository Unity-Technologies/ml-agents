"""
Generate the "Releases" table on the main readme. Update the versions lists, run this script, and copy the output
into the markdown file.
"""
from distutils.version import LooseVersion
from datetime import datetime
from typing import NamedTuple


def table_line(display_name, name, date, bold=False):
    bold_str = "**" if bold else ""
    # For release_X branches, docs are on a separate tag.
    if name.startswith("release"):
        docs_name = name + "_docs"
    else:
        docs_name = name
    return f"| **{display_name}** | {bold_str}{date}{bold_str} | {bold_str}[source](https://github.com/Unity-Technologies/ml-agents/tree/{name}){bold_str} | {bold_str}[docs](https://github.com/Unity-Technologies/ml-agents/tree/{docs_name}/docs/Readme.md){bold_str} | {bold_str}[download](https://github.com/Unity-Technologies/ml-agents/archive/{name}.zip){bold_str} |"  # noqa


class ReleaseInfo(NamedTuple):
    release_tag: str
    csharp_version: str
    python_verion: str
    release_date: str

    @staticmethod
    def from_simple_tag(release_tag: str, release_date: str) -> "ReleaseInfo":
        """
        Generate the ReleaseInfo for "old style" releases, where the tag and versions
        were all the same.
        """
        return ReleaseInfo(release_tag, release_tag, release_tag, release_date)

    @property
    def loose_version(self) -> LooseVersion:
        return LooseVersion(self.python_verion)

    @property
    def elapsed_days(self) -> int:
        """
        Days since this version was released.
        :return:
        """
        return (
            datetime.today() - datetime.strptime(self.release_date, "%B %d, %Y")
        ).days

    @property
    def display_name(self) -> str:
        """
        Clean up the tag name for display, e.g. "release_1" -> "Release 1"
        :return:
        """
        return self.release_tag.replace("_", " ").title()


versions = [
    ReleaseInfo.from_simple_tag("0.10.0", "September 30, 2019"),
    ReleaseInfo.from_simple_tag("0.10.1", "October 9, 2019"),
    ReleaseInfo.from_simple_tag("0.11.0", "November 4, 2019"),
    ReleaseInfo.from_simple_tag("0.12.0", "December 2, 2019"),
    ReleaseInfo.from_simple_tag("0.12.1", "December 11, 2019"),
    ReleaseInfo.from_simple_tag("0.13.0", "January 8, 2020"),
    ReleaseInfo.from_simple_tag("0.13.1", "January 21, 2020"),
    ReleaseInfo.from_simple_tag("0.14.0", "February 13, 2020"),
    ReleaseInfo.from_simple_tag("0.14.1", "February 26, 2020"),
    ReleaseInfo.from_simple_tag("0.15.0", "March 18, 2020"),
    ReleaseInfo.from_simple_tag("0.15.1", "March 30, 2020"),
    ReleaseInfo("release_1", "1.0.0", "0.16.0", "April 30, 2020"),
    ReleaseInfo("release_2", "1.0.2", "0.16.1", "May 20, 2020"),
    ReleaseInfo("release_3", "1.1.0", "0.17.0", "June 10, 2020"),
    ReleaseInfo("release_4", "1.2.0", "0.18.0", "July 15, 2020"),
    ReleaseInfo("release_5", "1.2.1", "0.18.1", "July 31, 2020"),
    ReleaseInfo("release_6", "1.3.0", "0.19.0", "August 12, 2020"),
    ReleaseInfo("release_7", "1.4.0", "0.20.0", "September 16, 2020"),
    ReleaseInfo("release_8", "1.5.0", "0.21.0", "October 14, 2020"),
    ReleaseInfo("release_9", "1.5.0", "0.21.1", "November 4, 2020"),
    ReleaseInfo("release_10", "1.6.0", "0.22.0", "November 18, 2020"),
    ReleaseInfo("release_11", "1.7.0", "0.23.0", "December 21, 2020"),
    ReleaseInfo("release_12", "1.7.2", "0.23.0", "December 22, 2020"),
]

MAX_DAYS = 150  # do not print releases older than this many days
sorted_versions = sorted(
    versions, key=lambda x: (x.loose_version, x.csharp_version), reverse=True
)

print(table_line("master (unstable)", "master", "--"))
highlight = True  # whether to bold the line or not
for version_info in sorted_versions:
    if version_info.elapsed_days <= MAX_DAYS:
        print(
            table_line(
                version_info.display_name,
                version_info.release_tag,
                version_info.release_date,
                highlight,
            )
        )
        highlight = False  # only bold the first stable release
