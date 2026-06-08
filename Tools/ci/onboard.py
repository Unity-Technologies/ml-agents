# Description: This script is used to onboard a project onto Wrench
import argparse
import json
import os
import shutil
import subprocess

TEMPLATE_CSPROJ = "TEMPLATE.Cookbook.csproj"
TEMPLATE_SLN = "TEMPLATE-recipes.sln"


def get_args():
    parser = argparse.ArgumentParser(description="Onboard a project onto Wrench")
    parser.add_argument(
        "--settings-name",
        required=True,
        help="The name of the settings file",
        dest="settings_name",
    )
    return parser.parse_args()


def delete_git_folder():
    print("Deleting .git folder")
    if os.path.isfile(".gitignore"):
        os.remove(".gitignore")
    if os.path.exists(".git"):
        shutil.rmtree(".git", ignore_errors=True)


def get_git_root_dir():
    git_dir = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE
    ).stdout
    return git_dir.decode("utf-8").strip()


def find_package_json_files(root_dir, initial=True):
    # walk the directory recursively and find all package.json files
    rtn_list = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "package.json":
                if initial:
                    file_location = file
                else:
                    file_location = os.path.join(root, file)
                # validate the package.json file
                with open(os.path.join(root, file_location)) as f:
                    data = json.loads(f.read())
                    if "unity" in data and "name" in data:
                        rtn_list.add(data["name"])
        for directory in dirs:
            rtn_list.update(
                find_package_json_files(os.path.join(root, directory), False)
            )

    return rtn_list


def create_package_option(name, first):
    initial_char = ""
    initial_tab = ""
    if not first:
        initial_char = ",\n"
        initial_tab = "        "
    return (
        f'{initial_char}{initial_tab}{{\n            "{name}",\n'
        f"            new PackageOptions() {{ ReleaseOptions = "
        f"new ReleaseOptions() {{ IsReleasing = true }} }}\n        }}"
    )


def update_template_variables(packages):
    template_var = "TEMPLATESettings"
    settings = get_args()
    settings_content = open(os.path.join("Settings", template_var + ".cs")).read()
    # replace the template settings with the actual settings
    settings_content = settings_content.replace(
        "TEMPLATESettings", f"{settings.settings_name}Settings"
    )
    settings_content = settings_content.replace("PACKAGES_ROOTS", ".")

    package_replace_string = ""
    first = True
    for package in packages:
        package_replace_string += create_package_option(package, first)
        first = False
    settings_content = settings_content.replace(
        '//"PACKAGES_TO_RELEASE"', package_replace_string
    )
    settings_content = settings_content.replace(
        "TEMPLATE.Cookbook.Settings", settings.settings_name + ".Cookbook.Settings"
    )

    # Write the updated settings file
    with open(
        os.path.join("Settings", settings.settings_name + "Settings.cs"), "w"
    ) as f:
        f.write(settings_content)
    # Update the Program.cs file
    program_content = open("Program.cs").read()
    with open("Program.cs", "w") as f:
        program_content = program_content.replace(
            "using TEMPLATE.Cookbook.Settings;",
            f"using {settings.settings_name}.Cookbook.Settings;",
        )
        program_content = program_content.replace(
            "TEMPLATE.Cookbook", settings.settings_name
        )
        program_content = program_content.replace("PACKAGES_ROOT", ".")
        program_content = program_content.replace(
            "TEMPLATESettings", settings.settings_name + "Settings"
        )
        f.write(program_content)
    # delete the template file
    os.remove(os.path.join("Settings", template_var + ".cs"))
    # update csproj file
    csproj_content = open(TEMPLATE_CSPROJ).read()
    with open(TEMPLATE_CSPROJ.replace("TEMPLATE", settings.settings_name), "w") as f:
        f.write(csproj_content)
    os.remove(TEMPLATE_CSPROJ)
    # update sln file
    sln_content = open(TEMPLATE_SLN).read()
    with open(TEMPLATE_SLN.replace("TEMPLATE", settings.settings_name), "w") as f:
        sln_content = sln_content.replace("TEMPLATE", settings.settings_name)
        f.write(sln_content)
    os.remove(TEMPLATE_SLN)


def update_shell_scripts(root_dir):
    split_root = os.path.normpath(root_dir).split(os.sep)
    split_cwd = os.getcwd().split(os.sep)

    relative_path = os.path.relpath(os.getcwd(), root_dir)
    csproj_path = os.path.join(
        relative_path, f"{get_args().settings_name}.Cookbook.csproj"
    )

    path_diff = len(split_cwd) - len(split_root)
    cd_string = ""
    for _ in range(path_diff):
        cd_string += "../"

    bash_content = open("regenerate.bat").read()
    bash_content = bash_content.replace("STEPS_TO_ROOT", cd_string)
    bash_content = bash_content.replace("PATH_TO_CSPROJ", csproj_path)
    with open("regenerate.bat", "w") as f:
        f.write(bash_content)

    shell_content = open("regenerate.sh").read()
    shell_content = shell_content.replace("STEPS_TO_ROOT", cd_string)
    shell_content = shell_content.replace("PATH_TO_CSPROJ", csproj_path)
    with open("regenerate.sh", "w") as f:
        f.write(shell_content)


def main():
    print("Starting process of onboarding")
    # delete git folder
    delete_git_folder()
    # get root folder
    root_dir = get_git_root_dir()
    # find all appropriate package.json files
    package_files = find_package_json_files(root_dir)
    packages = set()
    # process out any testing packages
    for package in package_files:
        if ".tests" in package:
            continue
        packages.add(package)
    # Replace Template naming with actual naming
    update_template_variables(packages)
    update_shell_scripts(root_dir)


if "__main__" in __name__:
    main()
