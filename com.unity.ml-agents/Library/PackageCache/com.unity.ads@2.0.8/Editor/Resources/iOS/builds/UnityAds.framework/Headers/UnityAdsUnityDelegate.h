#import "UnityAds.h"
NS_ASSUME_NONNULL_BEGIN
@protocol UnityAdsUnityDelegate <UnityAdsDelegate>
/**
 *  Called when an in-app purchase is initiated from an ad.
 *
 *  @param eventString The string provided via the ad.
 */
- (void)unityAdsDidInitiatePurchase:(NSString *)eventString;
@end
NS_ASSUME_NONNULL_END
