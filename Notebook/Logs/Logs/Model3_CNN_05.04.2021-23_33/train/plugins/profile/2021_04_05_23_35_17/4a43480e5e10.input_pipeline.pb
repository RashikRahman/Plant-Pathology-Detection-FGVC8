	;?i??6g@;?i??6g@!;?i??6g@	R??S?f??R??S?f??!R??S?f??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6;?i??6g@,I???0 @1?7>??e@A?c"??<??I???e?N @Yђ?????*	? ?rh?\@2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchBA)Z???!)????yK@)BA)Z???1)????yK@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?L?T??!??2?ݟU@)??}?Az??1#??i??@:Preprocessing2F
Iterator::Model???-I??!      Y@)W?'???1?j?+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9R??S?f??I ?Ǭ
?@QϪ?؋?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,I???0 @,I???0 @!,I???0 @      ??!       "	?7>??e@?7>??e@!?7>??e@*      ??!       2	?c"??<???c"??<??!?c"??<??:	???e?N @???e?N @!???e?N @B      ??!       J	ђ?????ђ?????!ђ?????R      ??!       Z	ђ?????ђ?????!ђ?????b      ??!       JGPUYR??S?f??b q ?Ǭ
?@yϪ?؋?W@