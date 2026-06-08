// Add deprecation banner with link
document.addEventListener('DOMContentLoaded', function() {
    // Create the banner element
    const banner = document.createElement('a');
    banner.href = 'https://docs.unity3d.com/Packages/com.unity.ml-agents@latest';
    banner.target = '_blank';
    banner.className = 'deprecation-banner';
    banner.innerHTML = '⚠️ DEPRECATED: This documentation has moved to Unity Package Documentation - Click here to view the latest documentation';
    
    // Insert at the top of the body
    document.body.insertBefore(banner, document.body.firstChild);
});
