from __future__ import print_function
import sys
import os

SUMMARY_XML_FILENAME = "Summary.xml"


def check_coverage(root_dir, min_percentage):
    summary_xml = None
    for dirpath, _, filenames in os.walk(root_dir):
        if SUMMARY_XML_FILENAME in filenames:
            summary_xml = os.path.join(dirpath, SUMMARY_XML_FILENAME)
            break
    if not summary_xml:
        print("Couldn't find Summary.xml in root directory")
        sys.exit(0)

    with open(summary_xml) as f:
        # Look for a line of the form
        # <Linecoverage>73.9</Linecoverage>
        lines = f.readlines()
        for l in lines:
            if "Linecoverage" in l:
                pct = l.replace("<Linecoverage>", "").replace("</Linecoverage>", "")
                pct = float(pct)
                if pct < min_percentage:
                    print(
                        "Coverage {} is below the min percentage of {}.".format(
                            pct, min_percentage
                        )
                    )
                    sys.exit(1)
                else:
                    print(
                        "Coverage {} is above the min percentage of {}.".format(
                            pct, min_percentage
                        )
                    )
                    sys.exit(0)

    # Couldn't parse the file
    print("Couldn't find Linecoverage in summary file")


def main():
    root_dir = sys.argv[1]
    min_percent = float(sys.argv[2])
    if min_percent > 0:
        # This allows us to set 0% coverage on 2018.4
        check_coverage(root_dir, min_percent)


if __name__ == "__main__":
    main()
