#import "UADSMetaData.h"

@interface UADSInAppPurchaseMetaData : UADSMetaData

- (void)setProductId:(NSString *)productId;
- (void)setPrice:(NSNumber *)price;
- (void)setCurrency:(NSString *)currency;
- (void)setReceiptPurchaseData:(NSString *)receiptPurchaseData;
- (void)setSignature:(NSString *)signature;

@end
