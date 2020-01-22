using System;
using System.Text.RegularExpressions;
using UnityEngine.Experimental.UIElements;

namespace UnityEditor.PackageManager.UI
{
    internal static class PackageFiltering
    {
        private static bool FilterByText(PackageInfo info, string text)
        {
            if (info == null)
                return false;
            
            if (info.Name.IndexOf(text, StringComparison.CurrentCultureIgnoreCase) >= 0)
                return true;

            if (!string.IsNullOrEmpty(info.DisplayName) && info.DisplayName.IndexOf(text, StringComparison.CurrentCultureIgnoreCase) >= 0)
                return true;

            if (!info.IsBuiltIn)
            {
                var prerelease = text.StartsWith("-") ? text.Substring(1) : text;
                if (info.Version != null && info.Version.Prerelease.IndexOf(prerelease, StringComparison.CurrentCultureIgnoreCase) >= 0)
                    return true;
    
                if (info.VersionWithoutTag.StartsWith(text, StringComparison.CurrentCultureIgnoreCase))
                    return true;

                if (info.IsPreview)
                {
                    if (PackageTag.preview.ToString().IndexOf(text, StringComparison.CurrentCultureIgnoreCase) >= 0)
                        return true;
                }

                if (info.IsVerified)
                {
                    if (PackageTag.verified.ToString().IndexOf(text, StringComparison.CurrentCultureIgnoreCase) >= 0)
                        return true;
                }
            }

            return false;
        }

        internal static bool FilterByText(Package package, string text)
        {
            if (string.IsNullOrEmpty(text))
                return true;
            
            var trimText = text.Trim(' ', '\t');
            trimText = Regex.Replace(trimText, @"[ ]{2,}", " ");
            return string.IsNullOrEmpty(trimText) || FilterByText(package.Current ?? package.Latest, trimText);
        }

        private static bool FilterByText(PackageItem item, string text)
        {
            return item.package != null && FilterByText(item.package, text);
        }

        public static void FilterPackageList(PackageList packageList)
        {
            PackageItem firstItem = null;
            PackageItem lastItem = null;
            var selectedItemInFilter = false;
            var selectedItem = packageList.selectedItem;
            var packageItems = packageList.Query<PackageItem>().ToList();
            foreach (var item in packageItems)
            {
                if (FilterByText(item, PackageSearchFilter.Instance.SearchText))
                {
                    if (firstItem == null)
                        firstItem = item;
                    if (item == selectedItem)
                        selectedItemInFilter = true;
                    
                    UIUtils.SetElementDisplay(item, true);
                    
                    if (lastItem != null)
                        lastItem.nextItem = item;
                
                    item.previousItem = lastItem;
                    item.nextItem = null;
                    
                    lastItem = item;
                }
                else
                    UIUtils.SetElementDisplay(item, false);
            }

            if (firstItem == null)
                packageList.ShowNoResults();
            else
                packageList.ShowResults(selectedItemInFilter ? selectedItem : firstItem);
        }
    }
}