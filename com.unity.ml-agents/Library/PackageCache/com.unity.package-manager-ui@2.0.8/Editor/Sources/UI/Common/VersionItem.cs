namespace UnityEditor.PackageManager.UI
{
    internal class VersionItem
    {
        internal PackageInfo Version;

        public string MenuName { get; set; }
        
        // Base label
        public string Label
        {
            get
            {
                if (Version == null)
                    return MenuName;
                
                var label = Version.VersionWithoutTag;

                return label;
            }
        }
        
        public string DropdownLabel
        {
            get
            {
                if (Version == null)
                    return MenuName;

                var label = MenuName + Label;
                
                if (Version.IsLocal)
                    label += " - local";
                if (Version.IsCurrent)
                    label += " - current";
                if (Version.IsVerified)
                    label += " - verified";
                else if (!string.IsNullOrEmpty(Version.Version.Prerelease))
                    label += string.Format(" - {0}", Version.Version.Prerelease);
                else if (Version.IsPreview)
                    label += " - preview";

                return label;
            }
        }

        public override string ToString()
        {
            return DropdownLabel;
        }
        
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(obj, null)) return false;
            if (ReferenceEquals(this, obj)) return true;

            var other = (VersionItem)obj;
            return Version == other.Version;
        }

        public override int GetHashCode()
        {
            return Version.GetHashCode();
        }        
    }
}