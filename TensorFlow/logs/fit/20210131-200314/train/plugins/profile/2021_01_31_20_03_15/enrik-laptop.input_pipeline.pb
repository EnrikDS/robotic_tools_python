	�����8�?�����8�?!�����8�?	
���@
���@!
���@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�����8�?E�k���?AH,�
�?Y"ߥ�%�?*	X9�ȦE@2F
Iterator::Model
����?!�#�T�F@)jg��R�?1��4!�8<@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat��?�@��?!D"
�:@)ZEh�Ʌ?1��f���8@:Preprocessing2S
Iterator::Model::ParallelMap6Y���}?!n#��0@)6Y���}?1n#��0@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate���!�?!a7�d+4@)v8�Jw�y?1�A?�#-@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip#LQ.�_�?!l��i�{K@)���}��f?1=���xp@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�j+���c?!Z��yf@)�j+���c?1Z��yf@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��	L�uK?!�i����?)��	L�uK?1�i����?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�������?!~�66@)�h9�CmK?1@jT���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2B10.4 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	E�k���?E�k���?!E�k���?      ��!       "      ��!       *      ��!       2	H,�
�?H,�
�?!H,�
�?:      ��!       B      ��!       J	"ߥ�%�?"ߥ�%�?!"ߥ�%�?R      ��!       Z	"ߥ�%�?"ߥ�%�?!"ߥ�%�?JCPU_ONLY