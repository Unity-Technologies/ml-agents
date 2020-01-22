using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
    /// <summary>
    /// Interface for Package Manager UI Extension
    /// </summary>
    public interface IPackageManagerExtension
    {
        /// <summary>
        /// Creates the extension UI visual element.
        /// </summary>
        /// <returns>A visual element that represents the UI or null if none</returns>
        VisualElement CreateExtensionUI();
        
        /// <summary>
        /// Called by the Package Manager UI when the package selection changed.
        /// </summary>
        /// <param name="packageInfo">The newly selected package information (can be null)</param>
        void OnPackageSelectionChange(PackageManager.PackageInfo packageInfo);
        
        /// <summary>
        /// Called by the Package Manager UI when a package is added or updated.
        /// </summary>
        /// <param name="packageInfo">The package information</param>
        void OnPackageAddedOrUpdated(PackageManager.PackageInfo packageInfo);
        
        /// <summary>
        /// Called by the Package Manager UI when a package is removed.
        /// </summary>
        /// <param name="packageInfo">The package information</param>
        void OnPackageRemoved(PackageManager.PackageInfo packageInfo);
    }
}