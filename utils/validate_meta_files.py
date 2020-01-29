import os


def main():
    asset_path = "Project/Assets"
    meta_suffix = ".meta"
    python_suffix = ".py"

    num_matched = 0

    unmatched = set()

    for root, dirs, files in os.walk(asset_path):
        dirs = set(dirs)
        files = set(files)

        combined = dirs | files
        for f in combined:
            if f.endswith(python_suffix):
                # Probably this script; skip it
                continue

            # We expect each non-.meta file to have a .meta file, and each .meta file to have a non-.meta file
            if f.endswith(meta_suffix):
                expected = f.replace(meta_suffix, "")
            else:
                expected = f + meta_suffix

            if expected not in combined:
                unmatched.add(os.path.join(root, f))
            else:
                num_matched += 1

    if unmatched:
        raise Exception(
            f"Mismatch between expected files and their .meta files: {sorted(unmatched)}"
        )

    print(f"Found {num_matched} correctly matched files")


if __name__ == "__main__":
    main()
