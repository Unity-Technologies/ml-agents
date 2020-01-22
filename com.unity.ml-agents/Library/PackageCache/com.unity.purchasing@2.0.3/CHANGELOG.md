[2.0.0] 2018-02-07
Fixed issue with IAP_PURCHASING flag not set on project load

[2.0.1] 2018-02-14
Fixed issue where importing the asset store package would fail due to importer settings.

[2.0.2] 2018-06-12
Fixed issue where TypeLoadException occured while using "UnityEngine.Purchasing" because SimpleJson was not found. fogbugzId: 1035663/

[2.0.3] 2018-06-14
Fixed issue related to 2.0.2 that caused new projects to not compile in the editor. 
Engine dll is enabled for editor by default.
Removed meta data that disabled engine dll for windows store.
