using RecipeEngine.Api.Dependencies;
using RecipeEngine.Api.Settings;
using RecipeEngine.Modules.Wrench.Models;
using RecipeEngine.Modules.Wrench.Settings;

namespace MLAgents.Cookbook.Settings;

public class MLAgentsSettings : AnnotatedSettingsBase
{
    // Path from the root of the repository where packages are located.
    readonly string[] PackagesRootPaths = {"."};

    // update this to list all packages in this repo that you want to release.
    Dictionary<string, PackageOptions> PackageOptions = new()
    {
        {
            "com.unity.ml-agents",
            new PackageOptions()
            {
                ReleaseOptions = new ReleaseOptions() { IsReleasing = true },
                PackJobOptions = new PackJobOptions() { Dependencies = new List<Dependency>() { new Dependency("com.unity.ml-agents-pack", "sign_macos") } }
            }
        }
    };

    public MLAgentsSettings()
    {
        Wrench = new WrenchSettings(
            PackagesRootPaths,
            PackageOptions
        );
    }

    public WrenchSettings Wrench { get; private set; }
}
