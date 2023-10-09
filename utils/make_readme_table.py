"""
Generate the "Releases" table on the main readme. Update the versions lists, run this script, and copy the output
into the markdown file.
"""
from distutils.version import LooseVersion, StrictVersion
from datetime import datetime
from typing import NamedTuple
from collections import Counter

MAX_DAYS = 150  # do not print releases older than this many days


def table_line(version_info, bold=False):
    bold_str = "**" if bold else ""

    cells = [
        f"**{version_info.display_name}**",
        f"{bold_str}{version_info.release_date}{bold_str}",
        f"{bold_str}[source]({version_info.source_link}){bold_str}",
        f"{bold_str}[docs]({version_info.doc_link}){bold_str}",
        f"{bold_str}[download]({version_info.download_link}){bold_str}",
    ]
    if version_info.is_develop:
        cells.append("--")  # python
        cells.append("--")  # Unity
    else:
        cells.append(
            f"{bold_str}[{version_info.python_verion}]({version_info.pypi_link}){bold_str}"
        )
        cells.append(
            f"{bold_str}[{version_info.csharp_version}]({version_info.package_link}){bold_str}"
        )
    joined_cells = " | ".join(cells)
    return f"| {joined_cells} |"


class ReleaseInfo(NamedTuple):
    release_tag: str
    csharp_version: str
    python_verion: str
    release_date: str
    is_verified: bool = False

    @property
    def loose_version(self) -> LooseVersion:
        return LooseVersion(self.python_verion)

    @property
    def is_develop(self) -> bool:
        return self.release_tag == "develop"

    @property
    def release_datetime(self) -> datetime:
        if self.is_develop:
            return datetime.today()
        return datetime.strptime(self.release_date, "%B %d, %Y")

    @property
    def elapsed_days(self) -> int:
        """
        Days since this version was released.
        :return:
        """
        return (datetime.today() - self.release_datetime).days

    @property
    def display_name(self) -> str:
        """
        Clean up the tag name for display, e.g. "release_1" -> "Release 1"
        :return:
        """
        if self.is_verified:
            return f"Verified Package {self.csharp_version}"
        elif self.is_develop:
            return "develop (unstable)"
        else:
            return self.release_tag.replace("_", " ").title()

    @property
    def source_link(self):
        if self.is_verified:
            return f"https://github.com/Unity-Technologies/ml-agents/tree/com.unity.ml-agents_{self.csharp_version}"
        else:
            return f"https://github.com/Unity-Technologies/ml-agents/tree/{self.release_tag}"

    @property
    def download_link(self):
        if self.is_verified:
            tag = f"com.unity.ml-agents_{self.csharp_version}"
        else:
            tag = self.release_tag
        return f"https://github.com/Unity-Technologies/ml-agents/archive/{tag}.zip"

    @property
    def doc_link(self):
        if self.is_verified:
            return "https://github.com/Unity-Technologies/ml-agents/blob/release_2_verified_docs/docs/Readme.md"

        # TODO remove in favor of webdocs. commenting out for now.
        # # For release_X branches, docs are on a separate tag.
        # if self.release_tag.startswith("release"):
        #     docs_name = self.release_tag + "_docs"
        # else:
        #     docs_name = self.release_tag
        # return f"https://github.com/Unity-Technologies/ml-agents/tree/{docs_name}/docs/Readme.md"
        return "https://unity-technologies.github.io/ml-agents/"

    @property
    def package_link(self):
        try:
            v = StrictVersion(self.csharp_version).version
            return f"https://docs.unity3d.com/Packages/com.unity.ml-agents@{v[0]}.{v[1]}/manual/index.html"
        except ValueError:
            return "--"

    @property
    def pypi_link(self):
        return f"https://pypi.org/project/mlagents/{self.python_verion}/"


versions = [
    ReleaseInfo("develop", "develop", "develop", "--"),
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
    ReleaseInfo("release_13", "1.8.0", "0.24.0", "February 17, 2021"),
    ReleaseInfo("release_14", "1.8.1", "0.24.1", "March 5, 2021"),
    ReleaseInfo("release_15", "1.9.0", "0.25.0", "March 17, 2021"),
    ReleaseInfo("release_16", "1.9.1", "0.25.1", "April 13, 2021"),
    ReleaseInfo("release_17", "2.0.0", "0.26.0", "April 22, 2021"),
    ReleaseInfo("release_18", "2.1.0", "0.27.0", "June 9, 2021"),
    ReleaseInfo("release_19", "2.2.1", "0.28.0", "January 14, 2022"),
    ReleaseInfo("release_20", "2.3.0", "0.30.0", "November 21, 2022"),
    ReleaseInfo("release_21", "3.0.0", "1.0.0", "October 9, 2023"),
    # Verified releases
    # ReleaseInfo("", "1.0.8", "0.16.1", "May 26, 2021", is_verified=True),
    # ReleaseInfo("", "1.0.7", "0.16.1", "March 8, 2021", is_verified=True),
    # ReleaseInfo("", "1.0.6", "0.16.1", "November 16, 2020", is_verified=True),
    # ReleaseInfo("", "1.0.5", "0.16.1", "September 23, 2020", is_verified=True),
    # ReleaseInfo("", "1.0.4", "0.16.1", "August 20, 2020", is_verified=True),
]

sorted_versions = sorted(versions, key=lambda x: x.release_datetime, reverse=True)

highlight_versions = set()
# Highlight the most recent verified version
# disabling verified versions.
# TODO replace this table entry with released version according to
#  https://docs.unity3d.com/2022.3/Documentation/Manual/pack-safe.html
# highlight_versions.add([v for v in sorted_versions if v.is_verified][0])
# Highlight the most recent regular version
highlight_versions.add(
    [v for v in sorted_versions if (not v.is_verified and not v.is_develop)][0]
)

count_by_verified = Counter()

for version_info in sorted_versions:
    highlight = version_info in highlight_versions
    if version_info.elapsed_days > MAX_DAYS:
        # Make sure we always have at least regular and one verified entry
        if count_by_verified[version_info.is_verified] > 0:
            continue
    print(table_line(version_info, highlight))
    count_by_verified[version_info.is_verified] += 1

print("\n\n")
