
import pre_processing

mex = pre_processing.MexPreprocessing()
class_length = len(mex.activityList)

all_data = mex.get_data()
all_data = mex.get_features(all_data, increment=2, window=5)

features, labels = mex.flatten(all_data, dct_length=60)
