@interface UADSJsonStorage : NSObject

@property (nonatomic, strong) NSMutableDictionary *storageContents;

- (BOOL)set:(NSString *)key value:(id)value;
- (id)getValueForKey:(NSString *)key;
- (BOOL)deleteKey:(NSString *)key;
- (NSArray *)getKeys:(NSString *)key recursive:(BOOL)recursive;
- (void)clearData;
- (BOOL)initData;
- (BOOL)hasData;

@end
