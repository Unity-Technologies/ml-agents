import os


def main():
    asset_paths = [
        "Project/Assets",
        "DevProject/Assets",
        "com.unity.ml-agents",
        "com.unity.ml-agents.extensions",
    ]
    meta_suffix = ".meta"
    python_suffix = ".py"
    allow_list = frozenset(
        [
            "com.unity.ml-agents/.editorconfig",
            "com.unity.ml-agents/.gitignore",
            "com.unity.ml-agents/.npmignore",
            "com.unity.ml-agents/Tests/.tests.json",
            "com.unity.ml-agents.extensions/.gitignore",
            "com.unity.ml-agents.extensions/.npmignore",
            "com.unity.ml-agents.extensions/Tests/.tests.json",
        ]
    )
    ignored_dirs = {"Documentation~"}

    num_matched = 0

    unmatched = set()

    for asset_path in asset_paths:
        for root, dirs, files in os.walk(asset_path):
            # Modifying the dirs list with topdown=True (the default) will prevent us from recursing those directories
            for ignored in ignored_dirs:
                try:
                    dirs.remove(ignored)
                except ValueError:
                    pass

            dirs = set(dirs)
            files = set(files)

            combined = dirs | files
            for f in combined:

                if f.endswith(python_suffix):
                    # Probably this script; skip it
                    continue

                full_path = os.path.join(root, f)
                if full_path in allow_list:
                    continue

                # We expect each non-.meta file to have a .meta file, and each .meta file to have a non-.meta file
                if f.endswith(meta_suffix):
                    expected = f.replace(meta_suffix, "")
                else:
                    expected = f + meta_suffix

                if expected not in combined:
                    unmatched.add(full_path)
                else:
                    num_matched += 1

    if unmatched:
        raise Exception(
            f"Mismatch between expected files and their .meta files: {sorted(unmatched)}"
        )

    print(f"Found {num_matched} correctly matched files")


if __name__ == "__main__":
    main()
