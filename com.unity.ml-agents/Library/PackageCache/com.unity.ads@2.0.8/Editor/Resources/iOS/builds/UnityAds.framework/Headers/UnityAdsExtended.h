#import "UnityAds.h"

NS_ASSUME_NONNULL_BEGIN
@protocol UnityAdsExtendedDelegate <UnityAdsDelegate>
/**
 *  Called when a click event happens.
 *
 *  @param placementId The ID of the placement that was clicked.
 */
- (void)unityAdsDidClick:(NSString *)placementId;

/**
 *  Called when a placement changes state.
 *
 *  @param placementId The ID of the placement that changed state.
 *  @param oldState The state before the change.
 *  @param newState The state after the change.
 */
- (void)unityAdsPlacementStateChanged:(NSString *)placementId oldState:(UnityAdsPlacementState)oldState newState:(UnityAdsPlacementState)newState;
@end
NS_ASSUME_NONNULL_END
