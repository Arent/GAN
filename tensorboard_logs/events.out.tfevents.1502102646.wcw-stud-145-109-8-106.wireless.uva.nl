       ŁK"	  bÖAbrain.Event:2.iť ˇ     O	>bÖA"ď
ů
ConstConst*
dtype0*ż
valueľB˛B1/Users/Arent/image_net/apple/n07739125_10006.JPEGB1/Users/Arent/image_net/apple/n07739125_10022.JPEGB1/Users/Arent/image_net/apple/n07739125_10025.JPEGB0/Users/Arent/image_net/apple/n07739125_1003.JPEGB1/Users/Arent/image_net/apple/n07739125_10031.JPEGB1/Users/Arent/image_net/apple/n07739125_10033.JPEGB1/Users/Arent/image_net/apple/n07739125_10045.JPEGB1/Users/Arent/image_net/apple/n07739125_10057.JPEGB1/Users/Arent/image_net/apple/n07739125_10066.JPEGB1/Users/Arent/image_net/apple/n07739125_10069.JPEGB1/Users/Arent/image_net/apple/n07739125_10071.JPEGB1/Users/Arent/image_net/apple/n07739125_10077.JPEGB1/Users/Arent/image_net/apple/n07739125_10078.JPEGB1/Users/Arent/image_net/apple/n07739125_10083.JPEGB1/Users/Arent/image_net/apple/n07739125_10091.JPEGB1/Users/Arent/image_net/apple/n07739125_10093.JPEGB1/Users/Arent/image_net/apple/n07739125_10094.JPEGB1/Users/Arent/image_net/apple/n07739125_10098.JPEGB1/Users/Arent/image_net/apple/n07739125_10102.JPEGB1/Users/Arent/image_net/apple/n07739125_10108.JPEGB0/Users/Arent/image_net/apple/n07739125_1011.JPEGB1/Users/Arent/image_net/apple/n07739125_10118.JPEGB1/Users/Arent/image_net/apple/n07739125_10135.JPEGB0/Users/Arent/image_net/apple/n07739125_1015.JPEGB1/Users/Arent/image_net/apple/n07739125_10161.JPEGB1/Users/Arent/image_net/apple/n07739125_10169.JPEGB1/Users/Arent/image_net/apple/n07739125_10179.JPEGB1/Users/Arent/image_net/apple/n07739125_10203.JPEGB1/Users/Arent/image_net/apple/n07739125_10208.JPEGB1/Users/Arent/image_net/apple/n07739125_10211.JPEGB1/Users/Arent/image_net/apple/n07739125_10222.JPEG*
_output_shapes
:
Ş
Const_1Const*
dtype0*î
valueäBáBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBapple*
_output_shapes
:
ă
DynamicPartition/partitionsConst*
dtype0*
valueB"|                                                                                                                       *
_output_shapes
:

DynamicPartitionDynamicPartitionConstDynamicPartition/partitions*
num_partitions*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
DynamicPartition_1/partitionsConst*
dtype0*
valueB"|                                                                                                                       *
_output_shapes
:
Ą
DynamicPartition_1DynamicPartitionConst_1DynamicPartition_1/partitions*
num_partitions*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
d
input_producer/ShapeShapeDynamicPartition*
out_type0*
T0*
_output_shapes
:
l
"input_producer/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
$input_producer/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
n
$input_producer/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ä
input_producer/strided_sliceStridedSliceinput_producer/Shape"input_producer/strided_slice/stack$input_producer/strided_slice/stack_1$input_producer/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
k
)input_producer/input_producer/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
k
)input_producer/input_producer/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ń
#input_producer/input_producer/rangeRange)input_producer/input_producer/range/startinput_producer/strided_slice)input_producer/input_producer/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
+input_producer/input_producer/RandomShuffleRandomShuffle#input_producer/input_producer/range*
seed2 *

seed *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
0input_producer/input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 

1input_producer/input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
	container *
shared_name *
_output_shapes
: 
ˇ
8input_producer/input_producer/limit_epochs/epochs/AssignAssign1input_producer/input_producer/limit_epochs/epochs0input_producer/input_producer/limit_epochs/Const*
validate_shape(*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
Ü
6input_producer/input_producer/limit_epochs/epochs/readIdentity1input_producer/input_producer/limit_epochs/epochs*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
č
4input_producer/input_producer/limit_epochs/CountUpTo	CountUpTo1input_producer/input_producer/limit_epochs/epochs*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Č
*input_producer/input_producer/limit_epochsIdentity+input_producer/input_producer/RandomShuffle5^input_producer/input_producer/limit_epochs/CountUpTo*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
input_producer/input_producerFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
Ę
8input_producer/input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producer/input_producer*input_producer/input_producer/limit_epochs*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2

2input_producer/input_producer/input_producer_CloseQueueCloseV2input_producer/input_producer*
cancel_pending_enqueues( 

4input_producer/input_producer/input_producer_Close_1QueueCloseV2input_producer/input_producer*
cancel_pending_enqueues(
w
1input_producer/input_producer/input_producer_SizeQueueSizeV2input_producer/input_producer*
_output_shapes
: 

"input_producer/input_producer/CastCast1input_producer/input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
h
#input_producer/input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 

!input_producer/input_producer/mulMul"input_producer/input_producer/Cast#input_producer/input_producer/mul/y*
T0*
_output_shapes
: 
¨
6input_producer/input_producer/fraction_of_32_full/tagsConst*
dtype0*B
value9B7 B1input_producer/input_producer/fraction_of_32_full*
_output_shapes
: 
ž
1input_producer/input_producer/fraction_of_32_fullScalarSummary6input_producer/input_producer/fraction_of_32_full/tags!input_producer/input_producer/mul*
T0*
_output_shapes
: 
Ł
%input_producer/input_producer_DequeueQueueDequeueV2input_producer/input_producer*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*
_output_shapes
: 
§
input_producer/GatherGatherDynamicPartition%input_producer/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
Ť
input_producer/Gather_1GatherDynamicPartition_1%input_producer/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
h
input_producer_1/ShapeShapeDynamicPartition:1*
out_type0*
T0*
_output_shapes
:
n
$input_producer_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
p
&input_producer_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
p
&input_producer_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Î
input_producer_1/strided_sliceStridedSliceinput_producer_1/Shape$input_producer_1/strided_slice/stack&input_producer_1/strided_slice/stack_1&input_producer_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
m
+input_producer_1/input_producer/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
m
+input_producer_1/input_producer/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ů
%input_producer_1/input_producer/rangeRange+input_producer_1/input_producer/range/startinput_producer_1/strided_slice+input_producer_1/input_producer/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
-input_producer_1/input_producer/RandomShuffleRandomShuffle%input_producer_1/input_producer/range*
seed2 *

seed *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
2input_producer_1/input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 

3input_producer_1/input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
	container *
shared_name *
_output_shapes
: 
ż
:input_producer_1/input_producer/limit_epochs/epochs/AssignAssign3input_producer_1/input_producer/limit_epochs/epochs2input_producer_1/input_producer/limit_epochs/Const*
validate_shape(*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
â
8input_producer_1/input_producer/limit_epochs/epochs/readIdentity3input_producer_1/input_producer/limit_epochs/epochs*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
î
6input_producer_1/input_producer/limit_epochs/CountUpTo	CountUpTo3input_producer_1/input_producer/limit_epochs/epochs*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Î
,input_producer_1/input_producer/limit_epochsIdentity-input_producer_1/input_producer/RandomShuffle7^input_producer_1/input_producer/limit_epochs/CountUpTo*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
input_producer_1/input_producerFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
: *
component_types
2*
	container *
shared_name 
Đ
:input_producer_1/input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producer_1/input_producer,input_producer_1/input_producer/limit_epochs*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2

4input_producer_1/input_producer/input_producer_CloseQueueCloseV2input_producer_1/input_producer*
cancel_pending_enqueues( 

6input_producer_1/input_producer/input_producer_Close_1QueueCloseV2input_producer_1/input_producer*
cancel_pending_enqueues(
{
3input_producer_1/input_producer/input_producer_SizeQueueSizeV2input_producer_1/input_producer*
_output_shapes
: 

$input_producer_1/input_producer/CastCast3input_producer_1/input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
j
%input_producer_1/input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 

#input_producer_1/input_producer/mulMul$input_producer_1/input_producer/Cast%input_producer_1/input_producer/mul/y*
T0*
_output_shapes
: 
Ź
8input_producer_1/input_producer/fraction_of_32_full/tagsConst*
dtype0*D
value;B9 B3input_producer_1/input_producer/fraction_of_32_full*
_output_shapes
: 
Ä
3input_producer_1/input_producer/fraction_of_32_fullScalarSummary8input_producer_1/input_producer/fraction_of_32_full/tags#input_producer_1/input_producer/mul*
T0*
_output_shapes
: 
§
'input_producer_1/input_producer_DequeueQueueDequeueV2input_producer_1/input_producer*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*
_output_shapes
: 
­
input_producer_1/GatherGatherDynamicPartition:1'input_producer_1/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
ą
input_producer_1/Gather_1GatherDynamicPartition_1:1'input_producer_1/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
C
ReadFileReadFileinput_producer/Gather*
_output_shapes
: 
Ů

DecodeJpeg
DecodeJpegReadFile*
ratio*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
try_recover_truncated( *
acceptable_fraction%  ?*
channels*

dct_method *
fancy_upscaling(
P
ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 


ExpandDims
ExpandDims
DecodeJpegExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
U
sizeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
x
ResizeBilinearResizeBilinear
ExpandDimssize*
align_corners( *
T0*&
_output_shapes
:@@
f
SqueezeSqueezeResizeBilinear*
squeeze_dims
 *
T0*"
_output_shapes
:@@
J
div/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
K
divRealDivSqueezediv/y*
T0*"
_output_shapes
:@@
J
sub/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
C
subSubdivsub/y*
T0*"
_output_shapes
:@@
J
mul/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
C
mulMulsubmul/y*
T0*"
_output_shapes
:@@
G

ReadFile_1ReadFileinput_producer_1/Gather*
_output_shapes
: 
Ý
DecodeJpeg_1
DecodeJpeg
ReadFile_1*
ratio*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
try_recover_truncated( *
acceptable_fraction%  ?*
channels*

dct_method *
fancy_upscaling(
R
ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 

ExpandDims_1
ExpandDimsDecodeJpeg_1ExpandDims_1/dim*

Tdim0*
T0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
size_1Const*
dtype0*
valueB"@   @   *
_output_shapes
:
~
ResizeBilinear_1ResizeBilinearExpandDims_1size_1*
align_corners( *
T0*&
_output_shapes
:@@
j
	Squeeze_1SqueezeResizeBilinear_1*
squeeze_dims
 *
T0*"
_output_shapes
:@@
L
div_1/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
Q
div_1RealDiv	Squeeze_1div_1/y*
T0*"
_output_shapes
:@@
L
sub_1/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
I
sub_1Subdiv_1sub_1/y*
T0*"
_output_shapes
:@@
L
mul_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
I
mul_1Mulsub_1mul_1/y*
T0*"
_output_shapes
:@@
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
¤
batch/fifo_queueFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
:@@: *
component_types
2*
	container *
shared_name 

batch/fifo_queue_enqueueQueueEnqueueV2batch/fifo_queuemulinput_producer/Gather_1*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
x
batch/fraction_of_32_full/tagsConst*
dtype0**
value!B Bbatch/fraction_of_32_full*
_output_shapes
: 
v
batch/fraction_of_32_fullScalarSummarybatch/fraction_of_32_full/tags	batch/mul*
T0*
_output_shapes
: 
I
batch/nConst*
dtype0*
value	B :*
_output_shapes
: 

batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*,
_output_shapes
:@@:
O
batch_1/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ś
batch_1/fifo_queueFIFOQueueV2*
capacity *
_output_shapes
: *
shapes
:@@: *
component_types
2*
	container *
shared_name 

batch_1/fifo_queue_enqueueQueueEnqueueV2batch_1/fifo_queuemul_1input_producer_1/Gather_1*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2
[
batch_1/fifo_queue_CloseQueueCloseV2batch_1/fifo_queue*
cancel_pending_enqueues( 
]
batch_1/fifo_queue_Close_1QueueCloseV2batch_1/fifo_queue*
cancel_pending_enqueues(
R
batch_1/fifo_queue_SizeQueueSizeV2batch_1/fifo_queue*
_output_shapes
: 
]
batch_1/CastCastbatch_1/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
R
batch_1/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
P
batch_1/mulMulbatch_1/Castbatch_1/mul/y*
T0*
_output_shapes
: 
|
 batch_1/fraction_of_32_full/tagsConst*
dtype0*,
value#B! Bbatch_1/fraction_of_32_full*
_output_shapes
: 
|
batch_1/fraction_of_32_fullScalarSummary batch_1/fraction_of_32_full/tagsbatch_1/mul*
T0*
_output_shapes
: 
K
	batch_1/nConst*
dtype0*
value	B :*
_output_shapes
: 
 
batch_1QueueDequeueManyV2batch_1/fifo_queue	batch_1/n*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*,
_output_shapes
:@@:
c
model/PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
m
model/Placeholder_1Placeholder*
dtype0*
shape: */
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
\
model/ShapeShapemodel/Placeholder*
out_type0*
T0*
_output_shapes
:
c
model/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
e
model/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
e
model/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

model/strided_sliceStridedSlicemodel/Shapemodel/strided_slice/stackmodel/strided_slice/stack_1model/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
a
model/generator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
a
model/generator/Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
a
model/generator/Reshape/shape/3Const*
dtype0*
value	B :d*
_output_shapes
: 
×
model/generator/Reshape/shapePackmodel/strided_slicemodel/generator/Reshape/shape/1model/generator/Reshape/shape/2model/generator/Reshape/shape/3*
N*
T0*
_output_shapes
:*

axis 

model/generator/ReshapeReshapemodel/Placeholdermodel/generator/Reshape/shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
T0*
Tshape0

*model/generator/layer1/random_normal/shapeConst*
dtype0*%
valueB"         d   *
_output_shapes
:
n
)model/generator/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
9model/generator/layer1/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:d
É
(model/generator/layer1/random_normal/mulMul9model/generator/layer1/random_normal/RandomStandardNormal+model/generator/layer1/random_normal/stddev*
T0*'
_output_shapes
:d
˛
$model/generator/layer1/random_normalAdd(model/generator/layer1/random_normal/mul)model/generator/layer1/random_normal/mean*
T0*'
_output_shapes
:d
¤
model/generator/layer1/weights
VariableV2*
dtype0*
shape:d*
	container *
shared_name *'
_output_shapes
:d

%model/generator/layer1/weights/AssignAssignmodel/generator/layer1/weights$model/generator/layer1/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:d
´
#model/generator/layer1/weights/readIdentitymodel/generator/layer1/weights*1
_class'
%#loc:@model/generator/layer1/weights*
T0*'
_output_shapes
:d
s
model/generator/layer1/ShapeShapemodel/generator/Reshape*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer1/strided_sliceStridedSlicemodel/generator/layer1/Shape*model/generator/layer1/strided_slice/stack,model/generator/layer1/strided_slice/stack_1,model/generator/layer1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer1/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer1/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer1/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer1/conv2d_transpose/output_shapePack$model/generator/layer1/strided_slice6model/generator/layer1/conv2d_transpose/output_shape/16model/generator/layer1/conv2d_transpose/output_shape/26model/generator/layer1/conv2d_transpose/output_shape/3*
N*
T0*
_output_shapes
:*

axis 
ß
'model/generator/layer1/conv2d_transposeConv2DBackpropInput4model/generator/layer1/conv2d_transpose/output_shape#model/generator/layer1/weights/readmodel/generator/Reshape*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0

,model/generator/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer1/random_normal_1/mulMul;model/generator/layer1/random_normal_1/RandomStandardNormal-model/generator/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer1/random_normal_1Add*model/generator/layer1/random_normal_1/mul+model/generator/layer1/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer1/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:
ř
"model/generator/layer1/bias/AssignAssignmodel/generator/layer1/bias&model/generator/layer1/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer1/bias/readIdentitymodel/generator/layer1/bias*.
_class$
" loc:@model/generator/layer1/bias*
T0*#
_output_shapes
:
§
model/generator/layer1/addAdd'model/generator/layer1/conv2d_transpose model/generator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer1/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer1/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
shared_name 

,model/generator/layer1/BatchNorm/beta/AssignAssign%model/generator/layer1/BatchNorm/beta7model/generator/layer1/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer1/BatchNorm/beta/readIdentity%model/generator/layer1/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer1/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer1/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer1/BatchNorm/gamma/AssignAssign&model/generator/layer1/BatchNorm/gamma8model/generator/layer1/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer1/BatchNorm/gamma/readIdentity&model/generator/layer1/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer1/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer1/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer1/BatchNorm/moving_mean/AssignAssign,model/generator/layer1/BatchNorm/moving_mean>model/generator/layer1/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer1/BatchNorm/moving_mean/readIdentity,model/generator/layer1/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer1/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer1/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer1/BatchNorm/moving_variance/AssignAssign0model/generator/layer1/BatchNorm/moving_varianceBmodel/generator/layer1/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer1/BatchNorm/moving_variance/readIdentity0model/generator/layer1/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer1/BatchNorm/moments/MeanMeanmodel/generator/layer1/add?model/generator/layer1/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ś
5model/generator/layer1/BatchNorm/moments/StopGradientStopGradient-model/generator/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer1/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
×
Bmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer1/add5model/generator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer1/add5model/generator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
Ź
Wmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
y
.model/generator/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer1/BatchNorm/moments/ReshapeReshape5model/generator/layer1/BatchNorm/moments/StopGradient.model/generator/layer1/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ŕ
:model/generator/layer1/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer1/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer1/BatchNorm/moments/normalize/meanAdd?model/generator/layer1/BatchNorm/moments/normalize/shifted_mean0model/generator/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer1/BatchNorm/moments/normalize/MulMulEmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer1/BatchNorm/moments/normalize/SquareSquare?model/generator/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer1/BatchNorm/moments/normalize/varianceSub6model/generator/layer1/BatchNorm/moments/normalize/Mul9model/generator/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer1/BatchNorm/AssignMovingAvg/subSub1model/generator/layer1/BatchNorm/moving_mean/read7model/generator/layer1/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer1/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer1/BatchNorm/AssignMovingAvg/sub6model/generator/layer1/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer1/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer1/BatchNorm/moving_mean4model/generator/layer1/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer1/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer1/BatchNorm/moving_variance/read;model/generator/layer1/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer1/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer1/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer1/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer1/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer1/BatchNorm/moving_variance6model/generator/layer1/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer1/BatchNorm/batchnorm/addAdd;model/generator/layer1/BatchNorm/moments/normalize/variance0model/generator/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer1/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer1/BatchNorm/batchnorm/mulMul0model/generator/layer1/BatchNorm/batchnorm/Rsqrt+model/generator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer1/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer1/add.model/generator/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer1/BatchNorm/batchnorm/mul_2Mul7model/generator/layer1/BatchNorm/moments/normalize/mean.model/generator/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer1/BatchNorm/batchnorm/subSub*model/generator/layer1/BatchNorm/beta/read0model/generator/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer1/BatchNorm/batchnorm/add_1Add0model/generator/layer1/BatchNorm/batchnorm/mul_1.model/generator/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer1/ReluRelu0model/generator/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
)model/generator/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer2/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ę
(model/generator/layer2/random_normal/mulMul9model/generator/layer2/random_normal/RandomStandardNormal+model/generator/layer2/random_normal/stddev*
T0*(
_output_shapes
:
ł
$model/generator/layer2/random_normalAdd(model/generator/layer2/random_normal/mul)model/generator/layer2/random_normal/mean*
T0*(
_output_shapes
:
Ś
model/generator/layer2/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *(
_output_shapes
:

%model/generator/layer2/weights/AssignAssignmodel/generator/layer2/weights$model/generator/layer2/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:
ľ
#model/generator/layer2/weights/readIdentitymodel/generator/layer2/weights*1
_class'
%#loc:@model/generator/layer2/weights*
T0*(
_output_shapes
:
w
model/generator/layer2/ShapeShapemodel/generator/layer1/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer2/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer2/strided_sliceStridedSlicemodel/generator/layer2/Shape*model/generator/layer2/strided_slice/stack,model/generator/layer2/strided_slice/stack_1,model/generator/layer2/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer2/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer2/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer2/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer2/conv2d_transpose/output_shapePack$model/generator/layer2/strided_slice6model/generator/layer2/conv2d_transpose/output_shape/16model/generator/layer2/conv2d_transpose/output_shape/26model/generator/layer2/conv2d_transpose/output_shape/3*
N*
T0*
_output_shapes
:*

axis 
â
'model/generator/layer2/conv2d_transposeConv2DBackpropInput4model/generator/layer2/conv2d_transpose/output_shape#model/generator/layer2/weights/readmodel/generator/layer1/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer2/random_normal_1/mulMul;model/generator/layer2/random_normal_1/RandomStandardNormal-model/generator/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer2/random_normal_1Add*model/generator/layer2/random_normal_1/mul+model/generator/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer2/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:
ř
"model/generator/layer2/bias/AssignAssignmodel/generator/layer2/bias&model/generator/layer2/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer2/bias/readIdentitymodel/generator/layer2/bias*.
_class$
" loc:@model/generator/layer2/bias*
T0*#
_output_shapes
:
§
model/generator/layer2/addAdd'model/generator/layer2/conv2d_transpose model/generator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer2/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer2/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
shared_name 

,model/generator/layer2/BatchNorm/beta/AssignAssign%model/generator/layer2/BatchNorm/beta7model/generator/layer2/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer2/BatchNorm/beta/readIdentity%model/generator/layer2/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer2/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer2/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer2/BatchNorm/gamma/AssignAssign&model/generator/layer2/BatchNorm/gamma8model/generator/layer2/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer2/BatchNorm/gamma/readIdentity&model/generator/layer2/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer2/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer2/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer2/BatchNorm/moving_mean/AssignAssign,model/generator/layer2/BatchNorm/moving_mean>model/generator/layer2/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer2/BatchNorm/moving_mean/readIdentity,model/generator/layer2/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer2/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer2/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer2/BatchNorm/moving_variance/AssignAssign0model/generator/layer2/BatchNorm/moving_varianceBmodel/generator/layer2/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer2/BatchNorm/moving_variance/readIdentity0model/generator/layer2/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer2/BatchNorm/moments/MeanMeanmodel/generator/layer2/add?model/generator/layer2/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ś
5model/generator/layer2/BatchNorm/moments/StopGradientStopGradient-model/generator/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer2/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
×
Bmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer2/add5model/generator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer2/add5model/generator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
Ź
Wmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
y
.model/generator/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer2/BatchNorm/moments/ReshapeReshape5model/generator/layer2/BatchNorm/moments/StopGradient.model/generator/layer2/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ŕ
:model/generator/layer2/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer2/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer2/BatchNorm/moments/normalize/meanAdd?model/generator/layer2/BatchNorm/moments/normalize/shifted_mean0model/generator/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer2/BatchNorm/moments/normalize/MulMulEmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer2/BatchNorm/moments/normalize/SquareSquare?model/generator/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer2/BatchNorm/moments/normalize/varianceSub6model/generator/layer2/BatchNorm/moments/normalize/Mul9model/generator/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer2/BatchNorm/AssignMovingAvg/subSub1model/generator/layer2/BatchNorm/moving_mean/read7model/generator/layer2/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer2/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer2/BatchNorm/AssignMovingAvg/sub6model/generator/layer2/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer2/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer2/BatchNorm/moving_mean4model/generator/layer2/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer2/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer2/BatchNorm/moving_variance/read;model/generator/layer2/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer2/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer2/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer2/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer2/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer2/BatchNorm/moving_variance6model/generator/layer2/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer2/BatchNorm/batchnorm/addAdd;model/generator/layer2/BatchNorm/moments/normalize/variance0model/generator/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer2/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer2/BatchNorm/batchnorm/mulMul0model/generator/layer2/BatchNorm/batchnorm/Rsqrt+model/generator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer2/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer2/add.model/generator/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer2/BatchNorm/batchnorm/mul_2Mul7model/generator/layer2/BatchNorm/moments/normalize/mean.model/generator/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer2/BatchNorm/batchnorm/subSub*model/generator/layer2/BatchNorm/beta/read0model/generator/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer2/BatchNorm/batchnorm/add_1Add0model/generator/layer2/BatchNorm/batchnorm/mul_1.model/generator/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer2/ReluRelu0model/generator/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
)model/generator/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer3/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ę
(model/generator/layer3/random_normal/mulMul9model/generator/layer3/random_normal/RandomStandardNormal+model/generator/layer3/random_normal/stddev*
T0*(
_output_shapes
:
ł
$model/generator/layer3/random_normalAdd(model/generator/layer3/random_normal/mul)model/generator/layer3/random_normal/mean*
T0*(
_output_shapes
:
Ś
model/generator/layer3/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *(
_output_shapes
:

%model/generator/layer3/weights/AssignAssignmodel/generator/layer3/weights$model/generator/layer3/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:
ľ
#model/generator/layer3/weights/readIdentitymodel/generator/layer3/weights*1
_class'
%#loc:@model/generator/layer3/weights*
T0*(
_output_shapes
:
w
model/generator/layer3/ShapeShapemodel/generator/layer2/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer3/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer3/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer3/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer3/strided_sliceStridedSlicemodel/generator/layer3/Shape*model/generator/layer3/strided_slice/stack,model/generator/layer3/strided_slice/stack_1,model/generator/layer3/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer3/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer3/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer3/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer3/conv2d_transpose/output_shapePack$model/generator/layer3/strided_slice6model/generator/layer3/conv2d_transpose/output_shape/16model/generator/layer3/conv2d_transpose/output_shape/26model/generator/layer3/conv2d_transpose/output_shape/3*
N*
T0*
_output_shapes
:*

axis 
â
'model/generator/layer3/conv2d_transposeConv2DBackpropInput4model/generator/layer3/conv2d_transpose/output_shape#model/generator/layer3/weights/readmodel/generator/layer2/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer3/random_normal_1/mulMul;model/generator/layer3/random_normal_1/RandomStandardNormal-model/generator/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer3/random_normal_1Add*model/generator/layer3/random_normal_1/mul+model/generator/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer3/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:
ř
"model/generator/layer3/bias/AssignAssignmodel/generator/layer3/bias&model/generator/layer3/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer3/bias/readIdentitymodel/generator/layer3/bias*.
_class$
" loc:@model/generator/layer3/bias*
T0*#
_output_shapes
:
§
model/generator/layer3/addAdd'model/generator/layer3/conv2d_transpose model/generator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer3/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer3/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
shared_name 

,model/generator/layer3/BatchNorm/beta/AssignAssign%model/generator/layer3/BatchNorm/beta7model/generator/layer3/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer3/BatchNorm/beta/readIdentity%model/generator/layer3/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer3/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer3/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer3/BatchNorm/gamma/AssignAssign&model/generator/layer3/BatchNorm/gamma8model/generator/layer3/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer3/BatchNorm/gamma/readIdentity&model/generator/layer3/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer3/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer3/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer3/BatchNorm/moving_mean/AssignAssign,model/generator/layer3/BatchNorm/moving_mean>model/generator/layer3/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer3/BatchNorm/moving_mean/readIdentity,model/generator/layer3/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer3/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer3/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer3/BatchNorm/moving_variance/AssignAssign0model/generator/layer3/BatchNorm/moving_varianceBmodel/generator/layer3/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer3/BatchNorm/moving_variance/readIdentity0model/generator/layer3/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer3/BatchNorm/moments/MeanMeanmodel/generator/layer3/add?model/generator/layer3/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ś
5model/generator/layer3/BatchNorm/moments/StopGradientStopGradient-model/generator/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer3/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
×
Bmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer3/add5model/generator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer3/add5model/generator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
Ź
Wmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
y
.model/generator/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer3/BatchNorm/moments/ReshapeReshape5model/generator/layer3/BatchNorm/moments/StopGradient.model/generator/layer3/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ŕ
:model/generator/layer3/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer3/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer3/BatchNorm/moments/normalize/meanAdd?model/generator/layer3/BatchNorm/moments/normalize/shifted_mean0model/generator/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer3/BatchNorm/moments/normalize/MulMulEmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer3/BatchNorm/moments/normalize/SquareSquare?model/generator/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer3/BatchNorm/moments/normalize/varianceSub6model/generator/layer3/BatchNorm/moments/normalize/Mul9model/generator/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer3/BatchNorm/AssignMovingAvg/subSub1model/generator/layer3/BatchNorm/moving_mean/read7model/generator/layer3/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer3/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer3/BatchNorm/AssignMovingAvg/sub6model/generator/layer3/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer3/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer3/BatchNorm/moving_mean4model/generator/layer3/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer3/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer3/BatchNorm/moving_variance/read;model/generator/layer3/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer3/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer3/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer3/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer3/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer3/BatchNorm/moving_variance6model/generator/layer3/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer3/BatchNorm/batchnorm/addAdd;model/generator/layer3/BatchNorm/moments/normalize/variance0model/generator/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer3/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer3/BatchNorm/batchnorm/mulMul0model/generator/layer3/BatchNorm/batchnorm/Rsqrt+model/generator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer3/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer3/add.model/generator/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer3/BatchNorm/batchnorm/mul_2Mul7model/generator/layer3/BatchNorm/moments/normalize/mean.model/generator/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer3/BatchNorm/batchnorm/subSub*model/generator/layer3/BatchNorm/beta/read0model/generator/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer3/BatchNorm/batchnorm/add_1Add0model/generator/layer3/BatchNorm/batchnorm/mul_1.model/generator/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer3/ReluRelu0model/generator/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer4/random_normal/shapeConst*
dtype0*%
valueB"              *
_output_shapes
:
n
)model/generator/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer4/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:  
Ę
(model/generator/layer4/random_normal/mulMul9model/generator/layer4/random_normal/RandomStandardNormal+model/generator/layer4/random_normal/stddev*
T0*(
_output_shapes
:  
ł
$model/generator/layer4/random_normalAdd(model/generator/layer4/random_normal/mul)model/generator/layer4/random_normal/mean*
T0*(
_output_shapes
:  
Ś
model/generator/layer4/weights
VariableV2*
dtype0*
shape:  *
	container *
shared_name *(
_output_shapes
:  

%model/generator/layer4/weights/AssignAssignmodel/generator/layer4/weights$model/generator/layer4/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:  
ľ
#model/generator/layer4/weights/readIdentitymodel/generator/layer4/weights*1
_class'
%#loc:@model/generator/layer4/weights*
T0*(
_output_shapes
:  
w
model/generator/layer4/ShapeShapemodel/generator/layer3/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer4/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer4/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer4/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer4/strided_sliceStridedSlicemodel/generator/layer4/Shape*model/generator/layer4/strided_slice/stack,model/generator/layer4/strided_slice/stack_1,model/generator/layer4/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer4/conv2d_transpose/output_shape/1Const*
dtype0*
value	B : *
_output_shapes
: 
x
6model/generator/layer4/conv2d_transpose/output_shape/2Const*
dtype0*
value	B : *
_output_shapes
: 
y
6model/generator/layer4/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer4/conv2d_transpose/output_shapePack$model/generator/layer4/strided_slice6model/generator/layer4/conv2d_transpose/output_shape/16model/generator/layer4/conv2d_transpose/output_shape/26model/generator/layer4/conv2d_transpose/output_shape/3*
N*
T0*
_output_shapes
:*

axis 
â
'model/generator/layer4/conv2d_transposeConv2DBackpropInput4model/generator/layer4/conv2d_transpose/output_shape#model/generator/layer4/weights/readmodel/generator/layer3/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
p
+model/generator/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
Ë
*model/generator/layer4/random_normal_1/mulMul;model/generator/layer4/random_normal_1/RandomStandardNormal-model/generator/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:  
´
&model/generator/layer4/random_normal_1Add*model/generator/layer4/random_normal_1/mul+model/generator/layer4/random_normal_1/mean*
T0*#
_output_shapes
:  

model/generator/layer4/bias
VariableV2*
dtype0*
shape:  *
	container *
shared_name *#
_output_shapes
:  
ř
"model/generator/layer4/bias/AssignAssignmodel/generator/layer4/bias&model/generator/layer4/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:  
§
 model/generator/layer4/bias/readIdentitymodel/generator/layer4/bias*.
_class$
" loc:@model/generator/layer4/bias*
T0*#
_output_shapes
:  
§
model/generator/layer4/addAdd'model/generator/layer4/conv2d_transpose model/generator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ŕ
7model/generator/layer4/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer4/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
shared_name 

,model/generator/layer4/BatchNorm/beta/AssignAssign%model/generator/layer4/BatchNorm/beta7model/generator/layer4/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer4/BatchNorm/beta/readIdentity%model/generator/layer4/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer4/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer4/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer4/BatchNorm/gamma/AssignAssign&model/generator/layer4/BatchNorm/gamma8model/generator/layer4/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer4/BatchNorm/gamma/readIdentity&model/generator/layer4/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer4/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer4/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer4/BatchNorm/moving_mean/AssignAssign,model/generator/layer4/BatchNorm/moving_mean>model/generator/layer4/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer4/BatchNorm/moving_mean/readIdentity,model/generator/layer4/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer4/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer4/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer4/BatchNorm/moving_variance/AssignAssign0model/generator/layer4/BatchNorm/moving_varianceBmodel/generator/layer4/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer4/BatchNorm/moving_variance/readIdentity0model/generator/layer4/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer4/BatchNorm/moments/MeanMeanmodel/generator/layer4/add?model/generator/layer4/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ś
5model/generator/layer4/BatchNorm/moments/StopGradientStopGradient-model/generator/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer4/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
×
Bmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer4/add5model/generator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ó
Pmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer4/add5model/generator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
­
Xmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
Ź
Wmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
y
.model/generator/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer4/BatchNorm/moments/ReshapeReshape5model/generator/layer4/BatchNorm/moments/StopGradient.model/generator/layer4/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ŕ
:model/generator/layer4/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer4/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer4/BatchNorm/moments/normalize/meanAdd?model/generator/layer4/BatchNorm/moments/normalize/shifted_mean0model/generator/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer4/BatchNorm/moments/normalize/MulMulEmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer4/BatchNorm/moments/normalize/SquareSquare?model/generator/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer4/BatchNorm/moments/normalize/varianceSub6model/generator/layer4/BatchNorm/moments/normalize/Mul9model/generator/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer4/BatchNorm/AssignMovingAvg/subSub1model/generator/layer4/BatchNorm/moving_mean/read7model/generator/layer4/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer4/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer4/BatchNorm/AssignMovingAvg/sub6model/generator/layer4/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer4/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer4/BatchNorm/moving_mean4model/generator/layer4/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer4/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer4/BatchNorm/moving_variance/read;model/generator/layer4/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer4/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer4/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer4/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer4/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer4/BatchNorm/moving_variance6model/generator/layer4/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer4/BatchNorm/batchnorm/addAdd;model/generator/layer4/BatchNorm/moments/normalize/variance0model/generator/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer4/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer4/BatchNorm/batchnorm/mulMul0model/generator/layer4/BatchNorm/batchnorm/Rsqrt+model/generator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer4/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer4/add.model/generator/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ć
0model/generator/layer4/BatchNorm/batchnorm/mul_2Mul7model/generator/layer4/BatchNorm/moments/normalize/mean.model/generator/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer4/BatchNorm/batchnorm/subSub*model/generator/layer4/BatchNorm/beta/read0model/generator/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer4/BatchNorm/batchnorm/add_1Add0model/generator/layer4/BatchNorm/batchnorm/mul_1.model/generator/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

model/generator/layer4/ReluRelu0model/generator/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

*model/generator/layer5/random_normal/shapeConst*
dtype0*%
valueB"@   @         *
_output_shapes
:
n
)model/generator/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
9model/generator/layer5/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:@@
É
(model/generator/layer5/random_normal/mulMul9model/generator/layer5/random_normal/RandomStandardNormal+model/generator/layer5/random_normal/stddev*
T0*'
_output_shapes
:@@
˛
$model/generator/layer5/random_normalAdd(model/generator/layer5/random_normal/mul)model/generator/layer5/random_normal/mean*
T0*'
_output_shapes
:@@
¤
model/generator/layer5/weights
VariableV2*
dtype0*
shape:@@*
	container *
shared_name *'
_output_shapes
:@@

%model/generator/layer5/weights/AssignAssignmodel/generator/layer5/weights$model/generator/layer5/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer5/weights*
use_locking(*
T0*'
_output_shapes
:@@
´
#model/generator/layer5/weights/readIdentitymodel/generator/layer5/weights*1
_class'
%#loc:@model/generator/layer5/weights*
T0*'
_output_shapes
:@@
w
model/generator/layer5/ShapeShapemodel/generator/layer4/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer5/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer5/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer5/strided_sliceStridedSlicemodel/generator/layer5/Shape*model/generator/layer5/strided_slice/stack,model/generator/layer5/strided_slice/stack_1,model/generator/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer5/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :@*
_output_shapes
: 
x
6model/generator/layer5/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :@*
_output_shapes
: 
x
6model/generator/layer5/conv2d_transpose/output_shape/3Const*
dtype0*
value	B :*
_output_shapes
: 
Ä
4model/generator/layer5/conv2d_transpose/output_shapePack$model/generator/layer5/strided_slice6model/generator/layer5/conv2d_transpose/output_shape/16model/generator/layer5/conv2d_transpose/output_shape/26model/generator/layer5/conv2d_transpose/output_shape/3*
N*
T0*
_output_shapes
:*

axis 
â
'model/generator/layer5/conv2d_transposeConv2DBackpropInput4model/generator/layer5/conv2d_transpose/output_shape#model/generator/layer5/weights/readmodel/generator/layer4/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer5/random_normal_1/shapeConst*
dtype0*!
valueB"@   @      *
_output_shapes
:
p
+model/generator/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ô
;model/generator/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*"
_output_shapes
:@@
Ę
*model/generator/layer5/random_normal_1/mulMul;model/generator/layer5/random_normal_1/RandomStandardNormal-model/generator/layer5/random_normal_1/stddev*
T0*"
_output_shapes
:@@
ł
&model/generator/layer5/random_normal_1Add*model/generator/layer5/random_normal_1/mul+model/generator/layer5/random_normal_1/mean*
T0*"
_output_shapes
:@@

model/generator/layer5/bias
VariableV2*
dtype0*
shape:@@*
	container *
shared_name *"
_output_shapes
:@@
÷
"model/generator/layer5/bias/AssignAssignmodel/generator/layer5/bias&model/generator/layer5/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer5/bias*
use_locking(*
T0*"
_output_shapes
:@@
Ś
 model/generator/layer5/bias/readIdentitymodel/generator/layer5/bias*.
_class$
" loc:@model/generator/layer5/bias*
T0*"
_output_shapes
:@@
Ś
model/generator/layer5/addAdd'model/generator/layer5/conv2d_transpose model/generator/layer5/bias/read*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ž
7model/generator/layer5/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
valueB*    *
_output_shapes
:
Ë
%model/generator/layer5/BatchNorm/beta
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
shared_name 

,model/generator/layer5/BatchNorm/beta/AssignAssign%model/generator/layer5/BatchNorm/beta7model/generator/layer5/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:
ź
*model/generator/layer5/BatchNorm/beta/readIdentity%model/generator/layer5/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
T0*
_output_shapes
:
Ŕ
8model/generator/layer5/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
valueB*  ?*
_output_shapes
:
Í
&model/generator/layer5/BatchNorm/gamma
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
shared_name 
˘
-model/generator/layer5/BatchNorm/gamma/AssignAssign&model/generator/layer5/BatchNorm/gamma8model/generator/layer5/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:
ż
+model/generator/layer5/BatchNorm/gamma/readIdentity&model/generator/layer5/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
T0*
_output_shapes
:
Ě
>model/generator/layer5/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
valueB*    *
_output_shapes
:
Ů
,model/generator/layer5/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
shared_name 
ş
3model/generator/layer5/BatchNorm/moving_mean/AssignAssign,model/generator/layer5/BatchNorm/moving_mean>model/generator/layer5/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:
Ń
1model/generator/layer5/BatchNorm/moving_mean/readIdentity,model/generator/layer5/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ô
Bmodel/generator/layer5/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes
:
á
0model/generator/layer5/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
shared_name 
Ę
7model/generator/layer5/BatchNorm/moving_variance/AssignAssign0model/generator/layer5/BatchNorm/moving_varianceBmodel/generator/layer5/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:
Ý
5model/generator/layer5/BatchNorm/moving_variance/readIdentity0model/generator/layer5/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

?model/generator/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ŕ
-model/generator/layer5/BatchNorm/moments/MeanMeanmodel/generator/layer5/add?model/generator/layer5/BatchNorm/moments/Mean/reduction_indices*&
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ľ
5model/generator/layer5/BatchNorm/moments/StopGradientStopGradient-model/generator/layer5/BatchNorm/moments/Mean*
T0*&
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer5/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ö
Bmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer5/add5model/generator/layer5/BatchNorm/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ň
Pmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer5/add5model/generator/layer5/BatchNorm/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
­
Xmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
­
Fmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ź
Wmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
š
Emodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
x
.model/generator/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ő
0model/generator/layer5/BatchNorm/moments/ReshapeReshape5model/generator/layer5/BatchNorm/moments/StopGradient.model/generator/layer5/BatchNorm/moments/Shape*
_output_shapes
:*
T0*
Tshape0
Ŕ
:model/generator/layer5/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ď
?model/generator/layer5/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
Ö
7model/generator/layer5/BatchNorm/moments/normalize/meanAdd?model/generator/layer5/BatchNorm/moments/normalize/shifted_mean0model/generator/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
ĺ
6model/generator/layer5/BatchNorm/moments/normalize/MulMulEmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
Š
9model/generator/layer5/BatchNorm/moments/normalize/SquareSquare?model/generator/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
Ú
;model/generator/layer5/BatchNorm/moments/normalize/varianceSub6model/generator/layer5/BatchNorm/moments/normalize/Mul9model/generator/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
ź
6model/generator/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer5/BatchNorm/AssignMovingAvg/subSub1model/generator/layer5/BatchNorm/moving_mean/read7model/generator/layer5/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

4model/generator/layer5/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer5/BatchNorm/AssignMovingAvg/sub6model/generator/layer5/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

0model/generator/layer5/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer5/BatchNorm/moving_mean4model/generator/layer5/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Â
8model/generator/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer5/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer5/BatchNorm/moving_variance/read;model/generator/layer5/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

6model/generator/layer5/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer5/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer5/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Ś
2model/generator/layer5/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer5/BatchNorm/moving_variance6model/generator/layer5/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
u
0model/generator/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
É
.model/generator/layer5/BatchNorm/batchnorm/addAdd;model/generator/layer5/BatchNorm/moments/normalize/variance0model/generator/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

0model/generator/layer5/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
š
.model/generator/layer5/BatchNorm/batchnorm/mulMul0model/generator/layer5/BatchNorm/batchnorm/Rsqrt+model/generator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
˝
0model/generator/layer5/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer5/add.model/generator/layer5/BatchNorm/batchnorm/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ĺ
0model/generator/layer5/BatchNorm/batchnorm/mul_2Mul7model/generator/layer5/BatchNorm/moments/normalize/mean.model/generator/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
¸
.model/generator/layer5/BatchNorm/batchnorm/subSub*model/generator/layer5/BatchNorm/beta/read0model/generator/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
Ó
0model/generator/layer5/BatchNorm/batchnorm/add_1Add0model/generator/layer5/BatchNorm/batchnorm/mul_1.model/generator/layer5/BatchNorm/batchnorm/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@

model/generator/layer5/ReluRelu0model/generator/layer5/BatchNorm/batchnorm/add_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
`
model/Shape_1Shapemodel/Placeholder_1*
out_type0*
T0*
_output_shapes
:
e
model/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
g
model/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
g
model/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ą
model/strided_slice_1StridedSlicemodel/Shape_1model/strided_slice_1/stackmodel/strided_slice_1/stack_1model/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask

.model/discriminator/layer1/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
=model/discriminator/layer1/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:
Ő
,model/discriminator/layer1/random_normal/mulMul=model/discriminator/layer1/random_normal/RandomStandardNormal/model/discriminator/layer1/random_normal/stddev*
T0*'
_output_shapes
:
ž
(model/discriminator/layer1/random_normalAdd,model/discriminator/layer1/random_normal/mul-model/discriminator/layer1/random_normal/mean*
T0*'
_output_shapes
:
¨
"model/discriminator/layer1/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *'
_output_shapes
:

)model/discriminator/layer1/weights/AssignAssign"model/discriminator/layer1/weights(model/discriminator/layer1/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:
Ŕ
'model/discriminator/layer1/weights/readIdentity"model/discriminator/layer1/weights*5
_class+
)'loc:@model/discriminator/layer1/weights*
T0*'
_output_shapes
:

0model/discriminator/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
t
/model/discriminator/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
×
.model/discriminator/layer1/random_normal_1/mulMul?model/discriminator/layer1/random_normal_1/RandomStandardNormal1model/discriminator/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:  
Ŕ
*model/discriminator/layer1/random_normal_1Add.model/discriminator/layer1/random_normal_1/mul/model/discriminator/layer1/random_normal_1/mean*
T0*#
_output_shapes
:  

model/discriminator/layer1/bias
VariableV2*
dtype0*
shape:  *
	container *
shared_name *#
_output_shapes
:  

&model/discriminator/layer1/bias/AssignAssignmodel/discriminator/layer1/bias*model/discriminator/layer1/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:  
ł
$model/discriminator/layer1/bias/readIdentitymodel/discriminator/layer1/bias*2
_class(
&$loc:@model/discriminator/layer1/bias*
T0*#
_output_shapes
:  
ű
!model/discriminator/layer1/Conv2DConv2Dmodel/Placeholder_1'model/discriminator/layer1/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer1/addAdd!model/discriminator/layer1/Conv2D$model/discriminator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
;model/discriminator/layer1/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer1/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer1/BatchNorm/beta/AssignAssign)model/discriminator/layer1/BatchNorm/beta;model/discriminator/layer1/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer1/BatchNorm/beta/readIdentity)model/discriminator/layer1/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer1/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer1/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer1/BatchNorm/gamma/AssignAssign*model/discriminator/layer1/BatchNorm/gamma<model/discriminator/layer1/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer1/BatchNorm/gamma/readIdentity*model/discriminator/layer1/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer1/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer1/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer1/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer1/BatchNorm/moving_meanBmodel/discriminator/layer1/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer1/BatchNorm/moving_mean/readIdentity0model/discriminator/layer1/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer1/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer1/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer1/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer1/BatchNorm/moving_varianceFmodel/discriminator/layer1/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer1/BatchNorm/moving_variance/readIdentity4model/discriminator/layer1/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer1/BatchNorm/moments/MeanMeanmodel/discriminator/layer1/addCmodel/discriminator/layer1/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ž
9model/discriminator/layer1/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer1/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ă
Fmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer1/add9model/discriminator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
˙
Tmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer1/add9model/discriminator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ą
\model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
°
[model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
}
2model/discriminator/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer1/BatchNorm/moments/ReshapeReshape9model/discriminator/layer1/BatchNorm/moments/StopGradient2model/discriminator/layer1/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Đ
>model/discriminator/layer1/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer1/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer1/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer1/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer1/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer1/BatchNorm/moments/normalize/Mul=model/discriminator/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer1/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer1/BatchNorm/moving_mean/read;model/discriminator/layer1/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer1/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer1/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer1/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer1/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer1/BatchNorm/moving_mean8model/discriminator/layer1/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer1/BatchNorm/moving_variance/read?model/discriminator/layer1/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer1/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer1/BatchNorm/moving_variance:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer1/BatchNorm/batchnorm/addAdd?model/discriminator/layer1/BatchNorm/moments/normalize/variance4model/discriminator/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer1/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer1/BatchNorm/batchnorm/mulMul4model/discriminator/layer1/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer1/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer1/add2model/discriminator/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ň
4model/discriminator/layer1/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer1/BatchNorm/moments/normalize/mean2model/discriminator/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer1/BatchNorm/batchnorm/subSub.model/discriminator/layer1/BatchNorm/beta/read4model/discriminator/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer1/BatchNorm/batchnorm/add_1Add4model/discriminator/layer1/BatchNorm/batchnorm/mul_12model/discriminator/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
e
 model/discriminator/layer1/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer1/mulMul model/discriminator/layer1/mul/x4model/discriminator/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ž
"model/discriminator/layer1/MaximumMaximum4model/discriminator/layer1/BatchNorm/batchnorm/add_1model/discriminator/layer1/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

.model/discriminator/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer2/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer2/random_normal/mulMul=model/discriminator/layer2/random_normal/RandomStandardNormal/model/discriminator/layer2/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer2/random_normalAdd,model/discriminator/layer2/random_normal/mul-model/discriminator/layer2/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer2/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *(
_output_shapes
:

)model/discriminator/layer2/weights/AssignAssign"model/discriminator/layer2/weights(model/discriminator/layer2/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer2/weights/readIdentity"model/discriminator/layer2/weights*5
_class+
)'loc:@model/discriminator/layer2/weights*
T0*(
_output_shapes
:

0model/discriminator/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer2/random_normal_1/mulMul?model/discriminator/layer2/random_normal_1/RandomStandardNormal1model/discriminator/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer2/random_normal_1Add.model/discriminator/layer2/random_normal_1/mul/model/discriminator/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer2/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:

&model/discriminator/layer2/bias/AssignAssignmodel/discriminator/layer2/bias*model/discriminator/layer2/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer2/bias/readIdentitymodel/discriminator/layer2/bias*2
_class(
&$loc:@model/discriminator/layer2/bias*
T0*#
_output_shapes
:

!model/discriminator/layer2/Conv2DConv2D"model/discriminator/layer1/Maximum'model/discriminator/layer2/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer2/addAdd!model/discriminator/layer2/Conv2D$model/discriminator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer2/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer2/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer2/BatchNorm/beta/AssignAssign)model/discriminator/layer2/BatchNorm/beta;model/discriminator/layer2/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer2/BatchNorm/beta/readIdentity)model/discriminator/layer2/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer2/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer2/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer2/BatchNorm/gamma/AssignAssign*model/discriminator/layer2/BatchNorm/gamma<model/discriminator/layer2/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer2/BatchNorm/gamma/readIdentity*model/discriminator/layer2/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer2/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer2/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer2/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer2/BatchNorm/moving_meanBmodel/discriminator/layer2/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer2/BatchNorm/moving_mean/readIdentity0model/discriminator/layer2/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer2/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer2/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer2/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer2/BatchNorm/moving_varianceFmodel/discriminator/layer2/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer2/BatchNorm/moving_variance/readIdentity4model/discriminator/layer2/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer2/BatchNorm/moments/MeanMeanmodel/discriminator/layer2/addCmodel/discriminator/layer2/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ž
9model/discriminator/layer2/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer2/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ă
Fmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer2/add9model/discriminator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer2/add9model/discriminator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
°
[model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
}
2model/discriminator/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer2/BatchNorm/moments/ReshapeReshape9model/discriminator/layer2/BatchNorm/moments/StopGradient2model/discriminator/layer2/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Đ
>model/discriminator/layer2/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer2/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer2/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer2/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer2/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer2/BatchNorm/moments/normalize/Mul=model/discriminator/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer2/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer2/BatchNorm/moving_mean/read;model/discriminator/layer2/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer2/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer2/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer2/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer2/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer2/BatchNorm/moving_mean8model/discriminator/layer2/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer2/BatchNorm/moving_variance/read?model/discriminator/layer2/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer2/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer2/BatchNorm/moving_variance:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer2/BatchNorm/batchnorm/addAdd?model/discriminator/layer2/BatchNorm/moments/normalize/variance4model/discriminator/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer2/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer2/BatchNorm/batchnorm/mulMul4model/discriminator/layer2/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer2/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer2/add2model/discriminator/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer2/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer2/BatchNorm/moments/normalize/mean2model/discriminator/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer2/BatchNorm/batchnorm/subSub.model/discriminator/layer2/BatchNorm/beta/read4model/discriminator/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer2/BatchNorm/batchnorm/add_1Add4model/discriminator/layer2/BatchNorm/batchnorm/mul_12model/discriminator/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer2/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer2/mulMul model/discriminator/layer2/mul/x4model/discriminator/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer2/MaximumMaximum4model/discriminator/layer2/BatchNorm/batchnorm/add_1model/discriminator/layer2/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

.model/discriminator/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer3/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer3/random_normal/mulMul=model/discriminator/layer3/random_normal/RandomStandardNormal/model/discriminator/layer3/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer3/random_normalAdd,model/discriminator/layer3/random_normal/mul-model/discriminator/layer3/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer3/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *(
_output_shapes
:

)model/discriminator/layer3/weights/AssignAssign"model/discriminator/layer3/weights(model/discriminator/layer3/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer3/weights/readIdentity"model/discriminator/layer3/weights*5
_class+
)'loc:@model/discriminator/layer3/weights*
T0*(
_output_shapes
:

0model/discriminator/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer3/random_normal_1/mulMul?model/discriminator/layer3/random_normal_1/RandomStandardNormal1model/discriminator/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer3/random_normal_1Add.model/discriminator/layer3/random_normal_1/mul/model/discriminator/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer3/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:

&model/discriminator/layer3/bias/AssignAssignmodel/discriminator/layer3/bias*model/discriminator/layer3/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer3/bias/readIdentitymodel/discriminator/layer3/bias*2
_class(
&$loc:@model/discriminator/layer3/bias*
T0*#
_output_shapes
:

!model/discriminator/layer3/Conv2DConv2D"model/discriminator/layer2/Maximum'model/discriminator/layer3/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer3/addAdd!model/discriminator/layer3/Conv2D$model/discriminator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer3/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer3/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer3/BatchNorm/beta/AssignAssign)model/discriminator/layer3/BatchNorm/beta;model/discriminator/layer3/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer3/BatchNorm/beta/readIdentity)model/discriminator/layer3/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer3/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer3/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer3/BatchNorm/gamma/AssignAssign*model/discriminator/layer3/BatchNorm/gamma<model/discriminator/layer3/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer3/BatchNorm/gamma/readIdentity*model/discriminator/layer3/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer3/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer3/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer3/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer3/BatchNorm/moving_meanBmodel/discriminator/layer3/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer3/BatchNorm/moving_mean/readIdentity0model/discriminator/layer3/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer3/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer3/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer3/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer3/BatchNorm/moving_varianceFmodel/discriminator/layer3/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer3/BatchNorm/moving_variance/readIdentity4model/discriminator/layer3/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer3/BatchNorm/moments/MeanMeanmodel/discriminator/layer3/addCmodel/discriminator/layer3/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ž
9model/discriminator/layer3/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer3/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ă
Fmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer3/add9model/discriminator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer3/add9model/discriminator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
°
[model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
}
2model/discriminator/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer3/BatchNorm/moments/ReshapeReshape9model/discriminator/layer3/BatchNorm/moments/StopGradient2model/discriminator/layer3/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Đ
>model/discriminator/layer3/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer3/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer3/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer3/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer3/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer3/BatchNorm/moments/normalize/Mul=model/discriminator/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer3/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer3/BatchNorm/moving_mean/read;model/discriminator/layer3/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer3/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer3/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer3/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer3/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer3/BatchNorm/moving_mean8model/discriminator/layer3/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer3/BatchNorm/moving_variance/read?model/discriminator/layer3/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer3/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer3/BatchNorm/moving_variance:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer3/BatchNorm/batchnorm/addAdd?model/discriminator/layer3/BatchNorm/moments/normalize/variance4model/discriminator/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer3/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer3/BatchNorm/batchnorm/mulMul4model/discriminator/layer3/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer3/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer3/add2model/discriminator/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer3/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer3/BatchNorm/moments/normalize/mean2model/discriminator/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer3/BatchNorm/batchnorm/subSub.model/discriminator/layer3/BatchNorm/beta/read4model/discriminator/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer3/BatchNorm/batchnorm/add_1Add4model/discriminator/layer3/BatchNorm/batchnorm/mul_12model/discriminator/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer3/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer3/mulMul model/discriminator/layer3/mul/x4model/discriminator/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer3/MaximumMaximum4model/discriminator/layer3/BatchNorm/batchnorm/add_1model/discriminator/layer3/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

.model/discriminator/layer4/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer4/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer4/random_normal/mulMul=model/discriminator/layer4/random_normal/RandomStandardNormal/model/discriminator/layer4/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer4/random_normalAdd,model/discriminator/layer4/random_normal/mul-model/discriminator/layer4/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer4/weights
VariableV2*
dtype0*
shape:*
	container *
shared_name *(
_output_shapes
:

)model/discriminator/layer4/weights/AssignAssign"model/discriminator/layer4/weights(model/discriminator/layer4/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer4/weights/readIdentity"model/discriminator/layer4/weights*5
_class+
)'loc:@model/discriminator/layer4/weights*
T0*(
_output_shapes
:

0model/discriminator/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer4/random_normal_1/mulMul?model/discriminator/layer4/random_normal_1/RandomStandardNormal1model/discriminator/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer4/random_normal_1Add.model/discriminator/layer4/random_normal_1/mul/model/discriminator/layer4/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer4/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *#
_output_shapes
:

&model/discriminator/layer4/bias/AssignAssignmodel/discriminator/layer4/bias*model/discriminator/layer4/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer4/bias/readIdentitymodel/discriminator/layer4/bias*2
_class(
&$loc:@model/discriminator/layer4/bias*
T0*#
_output_shapes
:

!model/discriminator/layer4/Conv2DConv2D"model/discriminator/layer3/Maximum'model/discriminator/layer4/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer4/addAdd!model/discriminator/layer4/Conv2D$model/discriminator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer4/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer4/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer4/BatchNorm/beta/AssignAssign)model/discriminator/layer4/BatchNorm/beta;model/discriminator/layer4/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer4/BatchNorm/beta/readIdentity)model/discriminator/layer4/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer4/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer4/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer4/BatchNorm/gamma/AssignAssign*model/discriminator/layer4/BatchNorm/gamma<model/discriminator/layer4/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer4/BatchNorm/gamma/readIdentity*model/discriminator/layer4/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer4/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer4/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer4/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer4/BatchNorm/moving_meanBmodel/discriminator/layer4/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer4/BatchNorm/moving_mean/readIdentity0model/discriminator/layer4/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer4/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer4/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer4/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer4/BatchNorm/moving_varianceFmodel/discriminator/layer4/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer4/BatchNorm/moving_variance/readIdentity4model/discriminator/layer4/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer4/BatchNorm/moments/MeanMeanmodel/discriminator/layer4/addCmodel/discriminator/layer4/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
Ž
9model/discriminator/layer4/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer4/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ă
Fmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer4/add9model/discriminator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer4/add9model/discriminator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
°
[model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
}
2model/discriminator/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer4/BatchNorm/moments/ReshapeReshape9model/discriminator/layer4/BatchNorm/moments/StopGradient2model/discriminator/layer4/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Đ
>model/discriminator/layer4/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer4/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer4/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer4/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer4/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer4/BatchNorm/moments/normalize/Mul=model/discriminator/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer4/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer4/BatchNorm/moving_mean/read;model/discriminator/layer4/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer4/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer4/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer4/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer4/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer4/BatchNorm/moving_mean8model/discriminator/layer4/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer4/BatchNorm/moving_variance/read?model/discriminator/layer4/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer4/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer4/BatchNorm/moving_variance:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer4/BatchNorm/batchnorm/addAdd?model/discriminator/layer4/BatchNorm/moments/normalize/variance4model/discriminator/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer4/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer4/BatchNorm/batchnorm/mulMul4model/discriminator/layer4/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer4/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer4/add2model/discriminator/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer4/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer4/BatchNorm/moments/normalize/mean2model/discriminator/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer4/BatchNorm/batchnorm/subSub.model/discriminator/layer4/BatchNorm/beta/read4model/discriminator/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer4/BatchNorm/batchnorm/add_1Add4model/discriminator/layer4/BatchNorm/batchnorm/mul_12model/discriminator/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer4/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer4/mulMul model/discriminator/layer4/mul/x4model/discriminator/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer4/MaximumMaximum4model/discriminator/layer4/BatchNorm/batchnorm/add_1model/discriminator/layer4/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 model/discriminator/layer5/ShapeShape"model/discriminator/layer4/Maximum*
out_type0*
T0*
_output_shapes
:
x
.model/discriminator/layer5/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
z
0model/discriminator/layer5/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
z
0model/discriminator/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

(model/discriminator/layer5/strided_sliceStridedSlice model/discriminator/layer5/Shape.model/discriminator/layer5/strided_slice/stack0model/discriminator/layer5/strided_slice/stack_10model/discriminator/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
j
 model/discriminator/layer5/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ą
model/discriminator/layer5/ProdProd(model/discriminator/layer5/strided_slice model/discriminator/layer5/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
˘
(model/discriminator/layer5/Reshape/shapePackmodel/strided_slice_1model/discriminator/layer5/Prod*
N*
T0*
_output_shapes
:*

axis 
Ä
"model/discriminator/layer5/ReshapeReshape"model/discriminator/layer4/Maximum(model/discriminator/layer5/Reshape/shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

.model/discriminator/layer5/random_normal/shapeConst*
dtype0*
valueB" @     *
_output_shapes
:
r
-model/discriminator/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
=model/discriminator/layer5/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:

Î
,model/discriminator/layer5/random_normal/mulMul=model/discriminator/layer5/random_normal/RandomStandardNormal/model/discriminator/layer5/random_normal/stddev*
T0* 
_output_shapes
:

ˇ
(model/discriminator/layer5/random_normalAdd,model/discriminator/layer5/random_normal/mul-model/discriminator/layer5/random_normal/mean*
T0* 
_output_shapes
:


"model/discriminator/layer5/weights
VariableV2*
dtype0*
shape:
*
	container *
shared_name * 
_output_shapes
:


)model/discriminator/layer5/weights/AssignAssign"model/discriminator/layer5/weights(model/discriminator/layer5/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer5/weights*
use_locking(*
T0* 
_output_shapes
:

š
'model/discriminator/layer5/weights/readIdentity"model/discriminator/layer5/weights*5
_class+
)'loc:@model/discriminator/layer5/weights*
T0* 
_output_shapes
:

z
0model/discriminator/layer5/random_normal_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
t
/model/discriminator/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ô
?model/discriminator/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
Î
.model/discriminator/layer5/random_normal_1/mulMul?model/discriminator/layer5/random_normal_1/RandomStandardNormal1model/discriminator/layer5/random_normal_1/stddev*
T0*
_output_shapes
:
ˇ
*model/discriminator/layer5/random_normal_1Add.model/discriminator/layer5/random_normal_1/mul/model/discriminator/layer5/random_normal_1/mean*
T0*
_output_shapes
:

model/discriminator/layer5/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
˙
&model/discriminator/layer5/bias/AssignAssignmodel/discriminator/layer5/bias*model/discriminator/layer5/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer5/bias*
use_locking(*
T0*
_output_shapes
:
Ş
$model/discriminator/layer5/bias/readIdentitymodel/discriminator/layer5/bias*2
_class(
&$loc:@model/discriminator/layer5/bias*
T0*
_output_shapes
:
Đ
!model/discriminator/layer5/MatMulMatMul"model/discriminator/layer5/Reshape'model/discriminator/layer5/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
model/discriminator/layer5/addAdd!model/discriminator/layer5/MatMul$model/discriminator/layer5/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
;model/discriminator/layer5/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
valueB*    *
_output_shapes
:
Ó
)model/discriminator/layer5/BatchNorm/beta
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
shared_name 
Ž
0model/discriminator/layer5/BatchNorm/beta/AssignAssign)model/discriminator/layer5/BatchNorm/beta;model/discriminator/layer5/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:
Č
.model/discriminator/layer5/BatchNorm/beta/readIdentity)model/discriminator/layer5/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
T0*
_output_shapes
:
Č
<model/discriminator/layer5/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
valueB*  ?*
_output_shapes
:
Ő
*model/discriminator/layer5/BatchNorm/gamma
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
shared_name 
˛
1model/discriminator/layer5/BatchNorm/gamma/AssignAssign*model/discriminator/layer5/BatchNorm/gamma<model/discriminator/layer5/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:
Ë
/model/discriminator/layer5/BatchNorm/gamma/readIdentity*model/discriminator/layer5/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
T0*
_output_shapes
:
Ô
Bmodel/discriminator/layer5/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB*    *
_output_shapes
:
á
0model/discriminator/layer5/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
shared_name 
Ę
7model/discriminator/layer5/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer5/BatchNorm/moving_meanBmodel/discriminator/layer5/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:
Ý
5model/discriminator/layer5/BatchNorm/moving_mean/readIdentity0model/discriminator/layer5/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ü
Fmodel/discriminator/layer5/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes
:
é
4model/discriminator/layer5/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
shared_name 
Ú
;model/discriminator/layer5/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer5/BatchNorm/moving_varianceFmodel/discriminator/layer5/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:
é
9model/discriminator/layer5/BatchNorm/moving_variance/readIdentity4model/discriminator/layer5/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

Cmodel/discriminator/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ç
1model/discriminator/layer5/BatchNorm/moments/MeanMean!model/discriminator/layer5/MatMulCmodel/discriminator/layer5/BatchNorm/moments/Mean/reduction_indices*
_output_shapes

:*
T0*
	keep_dims(*

Tidx0
Ľ
9model/discriminator/layer5/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer5/BatchNorm/moments/Mean*
T0*
_output_shapes

:
Š
Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/ShapeShape!model/discriminator/layer5/MatMul*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:

Qmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*
valueB: *
_output_shapes
:
Â
Imodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ý
Fmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SubSub!model/discriminator/layer5/MatMul9model/discriminator/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Tmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference!model/discriminator/layer5/MatMul9model/discriminator/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
\model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
š
Jmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ľ
[model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
Ĺ
Imodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
|
2model/discriminator/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
á
4model/discriminator/layer5/BatchNorm/moments/ReshapeReshape9model/discriminator/layer5/BatchNorm/moments/StopGradient2model/discriminator/layer5/BatchNorm/moments/Shape*
_output_shapes
:*
T0*
Tshape0
Đ
>model/discriminator/layer5/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ű
Cmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
â
;model/discriminator/layer5/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
ń
:model/discriminator/layer5/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
ą
=model/discriminator/layer5/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
ć
?model/discriminator/layer5/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer5/BatchNorm/moments/normalize/Mul=model/discriminator/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
Ä
:model/discriminator/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer5/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer5/BatchNorm/moving_mean/read;model/discriminator/layer5/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

8model/discriminator/layer5/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer5/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer5/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ş
4model/discriminator/layer5/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer5/BatchNorm/moving_mean8model/discriminator/layer5/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Ę
<model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ť
:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer5/BatchNorm/moving_variance/read?model/discriminator/layer5/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Š
:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
ś
6model/discriminator/layer5/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer5/BatchNorm/moving_variance:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
y
4model/discriminator/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ő
2model/discriminator/layer5/BatchNorm/batchnorm/addAdd?model/discriminator/layer5/BatchNorm/moments/normalize/variance4model/discriminator/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

4model/discriminator/layer5/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
Ĺ
2model/discriminator/layer5/BatchNorm/batchnorm/mulMul4model/discriminator/layer5/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
Ä
4model/discriminator/layer5/BatchNorm/batchnorm/mul_1Mul!model/discriminator/layer5/MatMul2model/discriminator/layer5/BatchNorm/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
4model/discriminator/layer5/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer5/BatchNorm/moments/normalize/mean2model/discriminator/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
Ä
2model/discriminator/layer5/BatchNorm/batchnorm/subSub.model/discriminator/layer5/BatchNorm/beta/read4model/discriminator/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
×
4model/discriminator/layer5/BatchNorm/batchnorm/add_1Add4model/discriminator/layer5/BatchNorm/batchnorm/mul_12model/discriminator/layer5/BatchNorm/batchnorm/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/discriminator/layer5/TanhTanh4model/discriminator/layer5/BatchNorm/batchnorm/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
model/Shape_2Shapemodel/generator/layer5/Relu*
out_type0*
T0*
_output_shapes
:
e
model/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
g
model/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
g
model/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ą
model/strided_slice_2StridedSlicemodel/Shape_2model/strided_slice_2/stackmodel/strided_slice_2/stack_1model/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask

0model/discriminator_1/layer1/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
?model/discriminator_1/layer1/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:
Ű
.model/discriminator_1/layer1/random_normal/mulMul?model/discriminator_1/layer1/random_normal/RandomStandardNormal1model/discriminator_1/layer1/random_normal/stddev*
T0*'
_output_shapes
:
Ä
*model/discriminator_1/layer1/random_normalAdd.model/discriminator_1/layer1/random_normal/mul/model/discriminator_1/layer1/random_normal/mean*
T0*'
_output_shapes
:

2model/discriminator_1/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
v
1model/discriminator_1/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
Ý
0model/discriminator_1/layer1/random_normal_1/mulMulAmodel/discriminator_1/layer1/random_normal_1/RandomStandardNormal3model/discriminator_1/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:  
Ć
,model/discriminator_1/layer1/random_normal_1Add0model/discriminator_1/layer1/random_normal_1/mul1model/discriminator_1/layer1/random_normal_1/mean*
T0*#
_output_shapes
:  

#model/discriminator_1/layer1/Conv2DConv2Dmodel/generator/layer5/Relu'model/discriminator/layer1/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer1/addAdd#model/discriminator_1/layer1/Conv2D$model/discriminator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

Emodel/discriminator_1/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer1/BatchNorm/moments/MeanMean model/discriminator_1/layer1/addEmodel/discriminator_1/layer1/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
˛
;model/discriminator_1/layer1/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer1/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
é
Hmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer1/add;model/discriminator_1/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

Vmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer1/add;model/discriminator_1/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ł
^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
˛
]model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0

4model/discriminator_1/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer1/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer1/BatchNorm/moments/StopGradient4model/discriminator_1/layer1/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ř
@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer1/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer1/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer1/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer1/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer1/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer1/BatchNorm/moving_mean/read=model/discriminator_1/layer1/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer1/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer1/BatchNorm/moving_mean:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer1/BatchNorm/moving_variance/readAmodel/discriminator_1/layer1/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer1/BatchNorm/moving_variance<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer1/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer1/BatchNorm/moments/normalize/variance6model/discriminator_1/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer1/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer1/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer1/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer1/add4model/discriminator_1/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ř
6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer1/BatchNorm/moments/normalize/mean4model/discriminator_1/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer1/BatchNorm/batchnorm/subSub.model/discriminator/layer1/BatchNorm/beta/read6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_14model/discriminator_1/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
g
"model/discriminator_1/layer1/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer1/mulMul"model/discriminator_1/layer1/mul/x6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ä
$model/discriminator_1/layer1/MaximumMaximum6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1 model/discriminator_1/layer1/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

0model/discriminator_1/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer2/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer2/random_normal/mulMul?model/discriminator_1/layer2/random_normal/RandomStandardNormal1model/discriminator_1/layer2/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer2/random_normalAdd.model/discriminator_1/layer2/random_normal/mul/model/discriminator_1/layer2/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer2/random_normal_1/mulMulAmodel/discriminator_1/layer2/random_normal_1/RandomStandardNormal3model/discriminator_1/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer2/random_normal_1Add0model/discriminator_1/layer2/random_normal_1/mul1model/discriminator_1/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer2/Conv2DConv2D$model/discriminator_1/layer1/Maximum'model/discriminator/layer2/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer2/addAdd#model/discriminator_1/layer2/Conv2D$model/discriminator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer2/BatchNorm/moments/MeanMean model/discriminator_1/layer2/addEmodel/discriminator_1/layer2/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
˛
;model/discriminator_1/layer2/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer2/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
é
Hmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer2/add;model/discriminator_1/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer2/add;model/discriminator_1/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
˛
]model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0

4model/discriminator_1/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer2/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer2/BatchNorm/moments/StopGradient4model/discriminator_1/layer2/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ř
@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer2/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer2/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer2/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer2/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer2/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer2/BatchNorm/moving_mean/read=model/discriminator_1/layer2/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer2/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer2/BatchNorm/moving_mean:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer2/BatchNorm/moving_variance/readAmodel/discriminator_1/layer2/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer2/BatchNorm/moving_variance<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer2/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer2/BatchNorm/moments/normalize/variance6model/discriminator_1/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer2/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer2/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer2/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer2/add4model/discriminator_1/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer2/BatchNorm/moments/normalize/mean4model/discriminator_1/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer2/BatchNorm/batchnorm/subSub.model/discriminator/layer2/BatchNorm/beta/read6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_14model/discriminator_1/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer2/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer2/mulMul"model/discriminator_1/layer2/mul/x6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer2/MaximumMaximum6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1 model/discriminator_1/layer2/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

0model/discriminator_1/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer3/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer3/random_normal/mulMul?model/discriminator_1/layer3/random_normal/RandomStandardNormal1model/discriminator_1/layer3/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer3/random_normalAdd.model/discriminator_1/layer3/random_normal/mul/model/discriminator_1/layer3/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer3/random_normal_1/mulMulAmodel/discriminator_1/layer3/random_normal_1/RandomStandardNormal3model/discriminator_1/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer3/random_normal_1Add0model/discriminator_1/layer3/random_normal_1/mul1model/discriminator_1/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer3/Conv2DConv2D$model/discriminator_1/layer2/Maximum'model/discriminator/layer3/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer3/addAdd#model/discriminator_1/layer3/Conv2D$model/discriminator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer3/BatchNorm/moments/MeanMean model/discriminator_1/layer3/addEmodel/discriminator_1/layer3/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
˛
;model/discriminator_1/layer3/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer3/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
é
Hmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer3/add;model/discriminator_1/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer3/add;model/discriminator_1/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
˛
]model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0

4model/discriminator_1/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer3/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer3/BatchNorm/moments/StopGradient4model/discriminator_1/layer3/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ř
@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer3/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer3/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer3/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer3/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer3/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer3/BatchNorm/moving_mean/read=model/discriminator_1/layer3/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer3/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer3/BatchNorm/moving_mean:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer3/BatchNorm/moving_variance/readAmodel/discriminator_1/layer3/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer3/BatchNorm/moving_variance<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer3/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer3/BatchNorm/moments/normalize/variance6model/discriminator_1/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer3/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer3/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer3/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer3/add4model/discriminator_1/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer3/BatchNorm/moments/normalize/mean4model/discriminator_1/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer3/BatchNorm/batchnorm/subSub.model/discriminator/layer3/BatchNorm/beta/read6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_14model/discriminator_1/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer3/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer3/mulMul"model/discriminator_1/layer3/mul/x6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer3/MaximumMaximum6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1 model/discriminator_1/layer3/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

0model/discriminator_1/layer4/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer4/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer4/random_normal/mulMul?model/discriminator_1/layer4/random_normal/RandomStandardNormal1model/discriminator_1/layer4/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer4/random_normalAdd.model/discriminator_1/layer4/random_normal/mul/model/discriminator_1/layer4/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer4/random_normal_1/mulMulAmodel/discriminator_1/layer4/random_normal_1/RandomStandardNormal3model/discriminator_1/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer4/random_normal_1Add0model/discriminator_1/layer4/random_normal_1/mul1model/discriminator_1/layer4/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer4/Conv2DConv2D$model/discriminator_1/layer3/Maximum'model/discriminator/layer4/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer4/addAdd#model/discriminator_1/layer4/Conv2D$model/discriminator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer4/BatchNorm/moments/MeanMean model/discriminator_1/layer4/addEmodel/discriminator_1/layer4/BatchNorm/moments/Mean/reduction_indices*'
_output_shapes
:*
T0*
	keep_dims(*

Tidx0
˛
;model/discriminator_1/layer4/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer4/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
é
Hmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer4/add;model/discriminator_1/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer4/add;model/discriminator_1/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0
˛
]model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:*
T0*
	keep_dims( *

Tidx0

4model/discriminator_1/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer4/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer4/BatchNorm/moments/StopGradient4model/discriminator_1/layer4/BatchNorm/moments/Shape*
_output_shapes	
:*
T0*
Tshape0
Ř
@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer4/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer4/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer4/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer4/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer4/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer4/BatchNorm/moving_mean/read=model/discriminator_1/layer4/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer4/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer4/BatchNorm/moving_mean:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer4/BatchNorm/moving_variance/readAmodel/discriminator_1/layer4/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer4/BatchNorm/moving_variance<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer4/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer4/BatchNorm/moments/normalize/variance6model/discriminator_1/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer4/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer4/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer4/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer4/add4model/discriminator_1/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer4/BatchNorm/moments/normalize/mean4model/discriminator_1/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer4/BatchNorm/batchnorm/subSub.model/discriminator/layer4/BatchNorm/beta/read6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_14model/discriminator_1/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer4/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer4/mulMul"model/discriminator_1/layer4/mul/x6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer4/MaximumMaximum6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1 model/discriminator_1/layer4/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

"model/discriminator_1/layer5/ShapeShape$model/discriminator_1/layer4/Maximum*
out_type0*
T0*
_output_shapes
:
z
0model/discriminator_1/layer5/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
|
2model/discriminator_1/layer5/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
|
2model/discriminator_1/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

*model/discriminator_1/layer5/strided_sliceStridedSlice"model/discriminator_1/layer5/Shape0model/discriminator_1/layer5/strided_slice/stack2model/discriminator_1/layer5/strided_slice/stack_12model/discriminator_1/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
l
"model/discriminator_1/layer5/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ˇ
!model/discriminator_1/layer5/ProdProd*model/discriminator_1/layer5/strided_slice"model/discriminator_1/layer5/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ś
*model/discriminator_1/layer5/Reshape/shapePackmodel/strided_slice_2!model/discriminator_1/layer5/Prod*
N*
T0*
_output_shapes
:*

axis 
Ę
$model/discriminator_1/layer5/ReshapeReshape$model/discriminator_1/layer4/Maximum*model/discriminator_1/layer5/Reshape/shape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

0model/discriminator_1/layer5/random_normal/shapeConst*
dtype0*
valueB" @     *
_output_shapes
:
t
/model/discriminator_1/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ú
?model/discriminator_1/layer5/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:

Ô
.model/discriminator_1/layer5/random_normal/mulMul?model/discriminator_1/layer5/random_normal/RandomStandardNormal1model/discriminator_1/layer5/random_normal/stddev*
T0* 
_output_shapes
:

˝
*model/discriminator_1/layer5/random_normalAdd.model/discriminator_1/layer5/random_normal/mul/model/discriminator_1/layer5/random_normal/mean*
T0* 
_output_shapes
:

|
2model/discriminator_1/layer5/random_normal_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
v
1model/discriminator_1/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ř
Amodel/discriminator_1/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
Ô
0model/discriminator_1/layer5/random_normal_1/mulMulAmodel/discriminator_1/layer5/random_normal_1/RandomStandardNormal3model/discriminator_1/layer5/random_normal_1/stddev*
T0*
_output_shapes
:
˝
,model/discriminator_1/layer5/random_normal_1Add0model/discriminator_1/layer5/random_normal_1/mul1model/discriminator_1/layer5/random_normal_1/mean*
T0*
_output_shapes
:
Ô
#model/discriminator_1/layer5/MatMulMatMul$model/discriminator_1/layer5/Reshape'model/discriminator/layer5/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
 model/discriminator_1/layer5/addAdd#model/discriminator_1/layer5/MatMul$model/discriminator/layer5/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
í
3model/discriminator_1/layer5/BatchNorm/moments/MeanMean#model/discriminator_1/layer5/MatMulEmodel/discriminator_1/layer5/BatchNorm/moments/Mean/reduction_indices*
_output_shapes

:*
T0*
	keep_dims(*

Tidx0
Š
;model/discriminator_1/layer5/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer5/BatchNorm/moments/Mean*
T0*
_output_shapes

:
­
Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/ShapeShape#model/discriminator_1/layer5/MatMul*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:

Smodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*
valueB: *
_output_shapes
:
Č
Kmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
ă
Hmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SubSub#model/discriminator_1/layer5/MatMul;model/discriminator_1/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Vmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference#model/discriminator_1/layer5/MatMul;model/discriminator_1/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ż
Lmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
§
]model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
Ë
Kmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
~
4model/discriminator_1/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
ç
6model/discriminator_1/layer5/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer5/BatchNorm/moments/StopGradient4model/discriminator_1/layer5/BatchNorm/moments/Shape*
_output_shapes
:*
T0*
Tshape0
Ř
@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
č
=model/discriminator_1/layer5/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
÷
<model/discriminator_1/layer5/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
ľ
?model/discriminator_1/layer5/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
ě
Amodel/discriminator_1/layer5/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer5/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
Ć
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ą
:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer5/BatchNorm/moving_mean/read=model/discriminator_1/layer5/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ľ
:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ž
6model/discriminator_1/layer5/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer5/BatchNorm/moving_mean:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Ě
>model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ż
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer5/BatchNorm/moving_variance/readAmodel/discriminator_1/layer5/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Ż
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
ş
8model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer5/BatchNorm/moving_variance<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
{
6model/discriminator_1/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ű
4model/discriminator_1/layer5/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer5/BatchNorm/moments/normalize/variance6model/discriminator_1/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

6model/discriminator_1/layer5/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
É
4model/discriminator_1/layer5/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer5/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
Ę
6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_1Mul#model/discriminator_1/layer5/MatMul4model/discriminator_1/layer5/BatchNorm/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer5/BatchNorm/moments/normalize/mean4model/discriminator_1/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
Č
4model/discriminator_1/layer5/BatchNorm/batchnorm/subSub.model/discriminator/layer5/BatchNorm/beta/read6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
Ý
6model/discriminator_1/layer5/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_14model/discriminator_1/layer5/BatchNorm/batchnorm/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!model/discriminator_1/layer5/TanhTanh6model/discriminator_1/layer5/BatchNorm/batchnorm/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
model/ones_like/ShapeShapemodel/discriminator/layer5/Tanh*
out_type0*
T0*
_output_shapes
:
Z
model/ones_like/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
w
model/ones_likeFillmodel/ones_like/Shapemodel/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
model/clip_by_value/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value/MinimumMinimummodel/discriminator/layer5/Tanhmodel/clip_by_value/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
model/clip_by_value/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_valueMaximummodel/clip_by_value/Minimummodel/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
	model/LogLogmodel/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
	model/mulMulmodel/ones_like	model/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
model/sub/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
`
	model/subSubmodel/sub/xmodel/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
h
model/sub_1Submodel/sub_1/xmodel/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_1Logmodel/sub_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/mul_1Mul	model/submodel/Log_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
	model/addAdd	model/mulmodel/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
	model/NegNeg	model/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
h

model/MeanMean	model/Negmodel/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
v
model/discriminator_real/tagsConst*
dtype0*)
value B Bmodel/discriminator_real*
_output_shapes
: 
u
model/discriminator_realScalarSummarymodel/discriminator_real/tags
model/Mean*
T0*
_output_shapes
: 
p
model/zeros_like	ZerosLikemodel/discriminator/layer5/Tanh*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/clip_by_value_1/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value_1/MinimumMinimum!model/discriminator_1/layer5/Tanhmodel/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/clip_by_value_1/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_value_1Maximummodel/clip_by_value_1/Minimummodel/clip_by_value_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
model/Log_2Logmodel/clip_by_value_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
model/mul_2Mulmodel/zeros_likemodel/Log_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_2/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
e
model/sub_2Submodel/sub_2/xmodel/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_3/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
model/sub_3Submodel/sub_3/xmodel/clip_by_value_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_3Logmodel/sub_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/mul_3Mulmodel/sub_2model/Log_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/add_1Addmodel/mul_2model/mul_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Neg_1Negmodel/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/Const_1Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_1Meanmodel/Neg_1model/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
v
model/discriminator_fake/tagsConst*
dtype0*)
value B Bmodel/discriminator_fake*
_output_shapes
: 
w
model/discriminator_fakeScalarSummarymodel/discriminator_fake/tagsmodel/Mean_1*
T0*
_output_shapes
: 
^
model/Const_2Const*
dtype0*
valueB"       *
_output_shapes
:
l
model/Mean_2Mean	model/Negmodel/Const_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
^
model/Const_3Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_3Meanmodel/Neg_1model/Const_3*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
O
model/add_2Addmodel/Mean_2model/Mean_3*
T0*
_output_shapes
: 
p
model/discriminator_2/tagsConst*
dtype0*&
valueB Bmodel/discriminator_2*
_output_shapes
: 
p
model/discriminator_2ScalarSummarymodel/discriminator_2/tagsmodel/add_2*
T0*
_output_shapes
: 
x
model/ones_like_1/ShapeShape!model/discriminator_1/layer5/Tanh*
out_type0*
T0*
_output_shapes
:
\
model/ones_like_1/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
}
model/ones_like_1Fillmodel/ones_like_1/Shapemodel/ones_like_1/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/clip_by_value_2/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value_2/MinimumMinimum!model/discriminator_1/layer5/Tanhmodel/clip_by_value_2/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/clip_by_value_2/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_value_2Maximummodel/clip_by_value_2/Minimummodel/clip_by_value_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
model/Log_4Logmodel/clip_by_value_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/mul_4Mulmodel/ones_like_1model/Log_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_4/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
model/sub_4Submodel/sub_4/xmodel/ones_like_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_5/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
model/sub_5Submodel/sub_5/xmodel/clip_by_value_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_5Logmodel/sub_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/mul_5Mulmodel/sub_4model/Log_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/add_3Addmodel/mul_4model/mul_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Neg_2Negmodel/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/Const_4Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_4Meanmodel/Neg_2model/Const_4*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
h
model/generator_1/tagsConst*
dtype0*"
valueB Bmodel/generator_1*
_output_shapes
: 
i
model/generator_1ScalarSummarymodel/generator_1/tagsmodel/Mean_4*
T0*
_output_shapes
: 
Ç
model/Merge/MergeSummaryMergeSummary1input_producer/input_producer/fraction_of_32_full3input_producer_1/input_producer/fraction_of_32_fullbatch/fraction_of_32_fullbatch_1/fraction_of_32_fullmodel/discriminator_realmodel/discriminator_fakemodel/discriminator_2model/generator_1*
_output_shapes
: *
N
]
model/merged_summariesIdentitymodel/Merge/MergeSummary*
T0*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/SaveV2/tensor_namesConst*
dtype0*Ĺ
valueťB¸<B)model/discriminator/layer1/BatchNorm/betaB*model/discriminator/layer1/BatchNorm/gammaB0model/discriminator/layer1/BatchNorm/moving_meanB4model/discriminator/layer1/BatchNorm/moving_varianceBmodel/discriminator/layer1/biasB"model/discriminator/layer1/weightsB)model/discriminator/layer2/BatchNorm/betaB*model/discriminator/layer2/BatchNorm/gammaB0model/discriminator/layer2/BatchNorm/moving_meanB4model/discriminator/layer2/BatchNorm/moving_varianceBmodel/discriminator/layer2/biasB"model/discriminator/layer2/weightsB)model/discriminator/layer3/BatchNorm/betaB*model/discriminator/layer3/BatchNorm/gammaB0model/discriminator/layer3/BatchNorm/moving_meanB4model/discriminator/layer3/BatchNorm/moving_varianceBmodel/discriminator/layer3/biasB"model/discriminator/layer3/weightsB)model/discriminator/layer4/BatchNorm/betaB*model/discriminator/layer4/BatchNorm/gammaB0model/discriminator/layer4/BatchNorm/moving_meanB4model/discriminator/layer4/BatchNorm/moving_varianceBmodel/discriminator/layer4/biasB"model/discriminator/layer4/weightsB)model/discriminator/layer5/BatchNorm/betaB*model/discriminator/layer5/BatchNorm/gammaB0model/discriminator/layer5/BatchNorm/moving_meanB4model/discriminator/layer5/BatchNorm/moving_varianceBmodel/discriminator/layer5/biasB"model/discriminator/layer5/weightsB%model/generator/layer1/BatchNorm/betaB&model/generator/layer1/BatchNorm/gammaB,model/generator/layer1/BatchNorm/moving_meanB0model/generator/layer1/BatchNorm/moving_varianceBmodel/generator/layer1/biasBmodel/generator/layer1/weightsB%model/generator/layer2/BatchNorm/betaB&model/generator/layer2/BatchNorm/gammaB,model/generator/layer2/BatchNorm/moving_meanB0model/generator/layer2/BatchNorm/moving_varianceBmodel/generator/layer2/biasBmodel/generator/layer2/weightsB%model/generator/layer3/BatchNorm/betaB&model/generator/layer3/BatchNorm/gammaB,model/generator/layer3/BatchNorm/moving_meanB0model/generator/layer3/BatchNorm/moving_varianceBmodel/generator/layer3/biasBmodel/generator/layer3/weightsB%model/generator/layer4/BatchNorm/betaB&model/generator/layer4/BatchNorm/gammaB,model/generator/layer4/BatchNorm/moving_meanB0model/generator/layer4/BatchNorm/moving_varianceBmodel/generator/layer4/biasBmodel/generator/layer4/weightsB%model/generator/layer5/BatchNorm/betaB&model/generator/layer5/BatchNorm/gammaB,model/generator/layer5/BatchNorm/moving_meanB0model/generator/layer5/BatchNorm/moving_varianceBmodel/generator/layer5/biasBmodel/generator/layer5/weights*
_output_shapes
:<
Ţ
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:<
Ő
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices)model/discriminator/layer1/BatchNorm/beta*model/discriminator/layer1/BatchNorm/gamma0model/discriminator/layer1/BatchNorm/moving_mean4model/discriminator/layer1/BatchNorm/moving_variancemodel/discriminator/layer1/bias"model/discriminator/layer1/weights)model/discriminator/layer2/BatchNorm/beta*model/discriminator/layer2/BatchNorm/gamma0model/discriminator/layer2/BatchNorm/moving_mean4model/discriminator/layer2/BatchNorm/moving_variancemodel/discriminator/layer2/bias"model/discriminator/layer2/weights)model/discriminator/layer3/BatchNorm/beta*model/discriminator/layer3/BatchNorm/gamma0model/discriminator/layer3/BatchNorm/moving_mean4model/discriminator/layer3/BatchNorm/moving_variancemodel/discriminator/layer3/bias"model/discriminator/layer3/weights)model/discriminator/layer4/BatchNorm/beta*model/discriminator/layer4/BatchNorm/gamma0model/discriminator/layer4/BatchNorm/moving_mean4model/discriminator/layer4/BatchNorm/moving_variancemodel/discriminator/layer4/bias"model/discriminator/layer4/weights)model/discriminator/layer5/BatchNorm/beta*model/discriminator/layer5/BatchNorm/gamma0model/discriminator/layer5/BatchNorm/moving_mean4model/discriminator/layer5/BatchNorm/moving_variancemodel/discriminator/layer5/bias"model/discriminator/layer5/weights%model/generator/layer1/BatchNorm/beta&model/generator/layer1/BatchNorm/gamma,model/generator/layer1/BatchNorm/moving_mean0model/generator/layer1/BatchNorm/moving_variancemodel/generator/layer1/biasmodel/generator/layer1/weights%model/generator/layer2/BatchNorm/beta&model/generator/layer2/BatchNorm/gamma,model/generator/layer2/BatchNorm/moving_mean0model/generator/layer2/BatchNorm/moving_variancemodel/generator/layer2/biasmodel/generator/layer2/weights%model/generator/layer3/BatchNorm/beta&model/generator/layer3/BatchNorm/gamma,model/generator/layer3/BatchNorm/moving_mean0model/generator/layer3/BatchNorm/moving_variancemodel/generator/layer3/biasmodel/generator/layer3/weights%model/generator/layer4/BatchNorm/beta&model/generator/layer4/BatchNorm/gamma,model/generator/layer4/BatchNorm/moving_mean0model/generator/layer4/BatchNorm/moving_variancemodel/generator/layer4/biasmodel/generator/layer4/weights%model/generator/layer5/BatchNorm/beta&model/generator/layer5/BatchNorm/gamma,model/generator/layer5/BatchNorm/moving_mean0model/generator/layer5/BatchNorm/moving_variancemodel/generator/layer5/biasmodel/generator/layer5/weights*J
dtypes@
>2<
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 

save/RestoreV2/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer1/BatchNorm/beta*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/AssignAssign)model/discriminator/layer1/BatchNorm/betasave/RestoreV2*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_1/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer1/BatchNorm/gamma*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_1Assign*model/discriminator/layer1/BatchNorm/gammasave/RestoreV2_1*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_2/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer1/BatchNorm/moving_mean*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ď
save/Assign_2Assign0model/discriminator/layer1/BatchNorm/moving_meansave/RestoreV2_2*
validate_shape(*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_3/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer1/BatchNorm/moving_variance*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
÷
save/Assign_3Assign4model/discriminator/layer1/BatchNorm/moving_variancesave/RestoreV2_3*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_4/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer1/bias*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ő
save/Assign_4Assignmodel/discriminator/layer1/biassave/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:  

save/RestoreV2_5/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer1/weights*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_5Assign"model/discriminator/layer1/weightssave/RestoreV2_5*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:

save/RestoreV2_6/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer2/BatchNorm/beta*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
á
save/Assign_6Assign)model/discriminator/layer2/BatchNorm/betasave/RestoreV2_6*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_7/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer2/BatchNorm/gamma*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_7Assign*model/discriminator/layer2/BatchNorm/gammasave/RestoreV2_7*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_8/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer2/BatchNorm/moving_mean*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ď
save/Assign_8Assign0model/discriminator/layer2/BatchNorm/moving_meansave/RestoreV2_8*
validate_shape(*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_9/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer2/BatchNorm/moving_variance*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
÷
save/Assign_9Assign4model/discriminator/layer2/BatchNorm/moving_variancesave/RestoreV2_9*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_10/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer2/bias*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_10Assignmodel/discriminator/layer2/biassave/RestoreV2_10*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_11/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer2/weights*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_11Assign"model/discriminator/layer2/weightssave/RestoreV2_11*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_12/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer3/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_12Assign)model/discriminator/layer3/BatchNorm/betasave/RestoreV2_12*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_13/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer3/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ĺ
save/Assign_13Assign*model/discriminator/layer3/BatchNorm/gammasave/RestoreV2_13*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_14/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer3/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_14Assign0model/discriminator/layer3/BatchNorm/moving_meansave/RestoreV2_14*
validate_shape(*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_15/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer3/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_15Assign4model/discriminator/layer3/BatchNorm/moving_variancesave/RestoreV2_15*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_16/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer3/bias*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_16Assignmodel/discriminator/layer3/biassave/RestoreV2_16*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_17/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer3/weights*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_17Assign"model/discriminator/layer3/weightssave/RestoreV2_17*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_18/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer4/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_18Assign)model/discriminator/layer4/BatchNorm/betasave/RestoreV2_18*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_19/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer4/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
ĺ
save/Assign_19Assign*model/discriminator/layer4/BatchNorm/gammasave/RestoreV2_19*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_20/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer4/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_20Assign0model/discriminator/layer4/BatchNorm/moving_meansave/RestoreV2_20*
validate_shape(*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_21/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer4/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_21Assign4model/discriminator/layer4/BatchNorm/moving_variancesave/RestoreV2_21*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_22/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer4/bias*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_22Assignmodel/discriminator/layer4/biassave/RestoreV2_22*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_23/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer4/weights*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_23Assign"model/discriminator/layer4/weightssave/RestoreV2_23*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_24/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer5/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_24Assign)model/discriminator/layer5/BatchNorm/betasave/RestoreV2_24*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_25/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer5/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save/Assign_25Assign*model/discriminator/layer5/BatchNorm/gammasave/RestoreV2_25*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_26/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer5/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_26Assign0model/discriminator/layer5/BatchNorm/moving_meansave/RestoreV2_26*
validate_shape(*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_27/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer5/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ř
save/Assign_27Assign4model/discriminator/layer5/BatchNorm/moving_variancesave/RestoreV2_27*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_28/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer5/bias*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_28Assignmodel/discriminator/layer5/biassave/RestoreV2_28*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer5/bias*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_29/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer5/weights*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_29Assign"model/discriminator/layer5/weightssave/RestoreV2_29*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer5/weights*
use_locking(*
T0* 
_output_shapes
:


save/RestoreV2_30/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer1/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_30Assign%model/generator/layer1/BatchNorm/betasave/RestoreV2_30*
validate_shape(*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_31/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer1/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_31Assign&model/generator/layer1/BatchNorm/gammasave/RestoreV2_31*
validate_shape(*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_32/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer1/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_32Assign,model/generator/layer1/BatchNorm/moving_meansave/RestoreV2_32*
validate_shape(*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_33/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer1/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_33Assign0model/generator/layer1/BatchNorm/moving_variancesave/RestoreV2_33*
validate_shape(*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_34/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer1/bias*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_34Assignmodel/generator/layer1/biassave/RestoreV2_34*
validate_shape(*.
_class$
" loc:@model/generator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_35/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer1/weights*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_35Assignmodel/generator/layer1/weightssave/RestoreV2_35*
validate_shape(*1
_class'
%#loc:@model/generator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:d

save/RestoreV2_36/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer2/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_36Assign%model/generator/layer2/BatchNorm/betasave/RestoreV2_36*
validate_shape(*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_37/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer2/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_37Assign&model/generator/layer2/BatchNorm/gammasave/RestoreV2_37*
validate_shape(*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_38/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer2/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_38Assign,model/generator/layer2/BatchNorm/moving_meansave/RestoreV2_38*
validate_shape(*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_39/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer2/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_39Assign0model/generator/layer2/BatchNorm/moving_variancesave/RestoreV2_39*
validate_shape(*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_40/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer2/bias*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_40Assignmodel/generator/layer2/biassave/RestoreV2_40*
validate_shape(*.
_class$
" loc:@model/generator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_41/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer2/weights*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_41Assignmodel/generator/layer2/weightssave/RestoreV2_41*
validate_shape(*1
_class'
%#loc:@model/generator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_42/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer3/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_42Assign%model/generator/layer3/BatchNorm/betasave/RestoreV2_42*
validate_shape(*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_43/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer3/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_43Assign&model/generator/layer3/BatchNorm/gammasave/RestoreV2_43*
validate_shape(*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_44/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer3/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_44/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_44Assign,model/generator/layer3/BatchNorm/moving_meansave/RestoreV2_44*
validate_shape(*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_45/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer3/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_45Assign0model/generator/layer3/BatchNorm/moving_variancesave/RestoreV2_45*
validate_shape(*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_46/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer3/bias*
_output_shapes
:
k
"save/RestoreV2_46/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_46Assignmodel/generator/layer3/biassave/RestoreV2_46*
validate_shape(*.
_class$
" loc:@model/generator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_47/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer3/weights*
_output_shapes
:
k
"save/RestoreV2_47/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_47Assignmodel/generator/layer3/weightssave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@model/generator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_48/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer4/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_48/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_48Assign%model/generator/layer4/BatchNorm/betasave/RestoreV2_48*
validate_shape(*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_49/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer4/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_49/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_49Assign&model/generator/layer4/BatchNorm/gammasave/RestoreV2_49*
validate_shape(*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_50/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer4/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_50Assign,model/generator/layer4/BatchNorm/moving_meansave/RestoreV2_50*
validate_shape(*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_51/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer4/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_51Assign0model/generator/layer4/BatchNorm/moving_variancesave/RestoreV2_51*
validate_shape(*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_52/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer4/bias*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_52Assignmodel/generator/layer4/biassave/RestoreV2_52*
validate_shape(*.
_class$
" loc:@model/generator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:  

save/RestoreV2_53/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer4/weights*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_53Assignmodel/generator/layer4/weightssave/RestoreV2_53*
validate_shape(*1
_class'
%#loc:@model/generator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:  

save/RestoreV2_54/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer5/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_54Assign%model/generator/layer5/BatchNorm/betasave/RestoreV2_54*
validate_shape(*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_55/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer5/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save/Assign_55Assign&model/generator/layer5/BatchNorm/gammasave/RestoreV2_55*
validate_shape(*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_56/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer5/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
č
save/Assign_56Assign,model/generator/layer5/BatchNorm/moving_meansave/RestoreV2_56*
validate_shape(*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_57/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer5/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_57Assign0model/generator/layer5/BatchNorm/moving_variancesave/RestoreV2_57*
validate_shape(*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_58/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer5/bias*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_58Assignmodel/generator/layer5/biassave/RestoreV2_58*
validate_shape(*.
_class$
" loc:@model/generator/layer5/bias*
use_locking(*
T0*"
_output_shapes
:@@

save/RestoreV2_59/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer5/weights*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_59Assignmodel/generator/layer5/weightssave/RestoreV2_59*
validate_shape(*1
_class'
%#loc:@model/generator/layer5/weights*
use_locking(*
T0*'
_output_shapes
:@@

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59"|§Ź_ö     aăÔ	xĺbÖAJéŠ
*ę)
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignSub
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
É
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
î
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
F
	CountUpTo
ref"T
output"T"
limitint"
Ttype:
2	
Ë

DecodeJpeg
contents	
image"
channelsint "
ratioint"
fancy_upscalingbool("!
try_recover_truncatedbool( "#
acceptable_fractionfloat%  ?"

dct_methodstring 
q
DynamicPartition	
data"T

partitions
outputs"T*num_partitions"
num_partitionsint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ž
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint˙˙˙˙˙˙˙˙˙"
	containerstring "
shared_namestring 
4
Fill
dims

value"T
output"T"	
Ttype

Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
B
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙

QueueDequeueV2

handle

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
y
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
&
QueueSizeV2

handle
size
Y
RandomShuffle

value"T
output"T"
seedint "
seed2int "	
Ttype

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
&
ReadFile
filename
contents
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
A
Relu
features"T
activations"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
ResizeBilinear
images"T
size
resized_images"
Ttype:

2	"
align_cornersbool( 
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
-
Rsqrt
x"T
y"T"
Ttype:	
2
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Square
x"T
y"T"
Ttype:
	2	
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.1.02v1.1.0-rc0-61-g1ec6ed5ď
ů
ConstConst*
dtype0*ż
valueľB˛B1/Users/Arent/image_net/apple/n07739125_10006.JPEGB1/Users/Arent/image_net/apple/n07739125_10022.JPEGB1/Users/Arent/image_net/apple/n07739125_10025.JPEGB0/Users/Arent/image_net/apple/n07739125_1003.JPEGB1/Users/Arent/image_net/apple/n07739125_10031.JPEGB1/Users/Arent/image_net/apple/n07739125_10033.JPEGB1/Users/Arent/image_net/apple/n07739125_10045.JPEGB1/Users/Arent/image_net/apple/n07739125_10057.JPEGB1/Users/Arent/image_net/apple/n07739125_10066.JPEGB1/Users/Arent/image_net/apple/n07739125_10069.JPEGB1/Users/Arent/image_net/apple/n07739125_10071.JPEGB1/Users/Arent/image_net/apple/n07739125_10077.JPEGB1/Users/Arent/image_net/apple/n07739125_10078.JPEGB1/Users/Arent/image_net/apple/n07739125_10083.JPEGB1/Users/Arent/image_net/apple/n07739125_10091.JPEGB1/Users/Arent/image_net/apple/n07739125_10093.JPEGB1/Users/Arent/image_net/apple/n07739125_10094.JPEGB1/Users/Arent/image_net/apple/n07739125_10098.JPEGB1/Users/Arent/image_net/apple/n07739125_10102.JPEGB1/Users/Arent/image_net/apple/n07739125_10108.JPEGB0/Users/Arent/image_net/apple/n07739125_1011.JPEGB1/Users/Arent/image_net/apple/n07739125_10118.JPEGB1/Users/Arent/image_net/apple/n07739125_10135.JPEGB0/Users/Arent/image_net/apple/n07739125_1015.JPEGB1/Users/Arent/image_net/apple/n07739125_10161.JPEGB1/Users/Arent/image_net/apple/n07739125_10169.JPEGB1/Users/Arent/image_net/apple/n07739125_10179.JPEGB1/Users/Arent/image_net/apple/n07739125_10203.JPEGB1/Users/Arent/image_net/apple/n07739125_10208.JPEGB1/Users/Arent/image_net/apple/n07739125_10211.JPEGB1/Users/Arent/image_net/apple/n07739125_10222.JPEG*
_output_shapes
:
Ş
Const_1Const*
dtype0*î
valueäBáBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBappleBapple*
_output_shapes
:
ă
DynamicPartition/partitionsConst*
dtype0*
valueB"|                                                                                                                       *
_output_shapes
:

DynamicPartitionDynamicPartitionConstDynamicPartition/partitions*
num_partitions*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ĺ
DynamicPartition_1/partitionsConst*
dtype0*
valueB"|                                                                                                                       *
_output_shapes
:
Ą
DynamicPartition_1DynamicPartitionConst_1DynamicPartition_1/partitions*
num_partitions*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
d
input_producer/ShapeShapeDynamicPartition*
out_type0*
T0*
_output_shapes
:
l
"input_producer/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
n
$input_producer/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
n
$input_producer/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ä
input_producer/strided_sliceStridedSliceinput_producer/Shape"input_producer/strided_slice/stack$input_producer/strided_slice/stack_1$input_producer/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
k
)input_producer/input_producer/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
k
)input_producer/input_producer/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ń
#input_producer/input_producer/rangeRange)input_producer/input_producer/range/startinput_producer/strided_slice)input_producer/input_producer/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
+input_producer/input_producer/RandomShuffleRandomShuffle#input_producer/input_producer/range*
seed2 *

seed *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
0input_producer/input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 

1input_producer/input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
shared_name *
	container *
_output_shapes
: 
ˇ
8input_producer/input_producer/limit_epochs/epochs/AssignAssign1input_producer/input_producer/limit_epochs/epochs0input_producer/input_producer/limit_epochs/Const*
validate_shape(*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
Ü
6input_producer/input_producer/limit_epochs/epochs/readIdentity1input_producer/input_producer/limit_epochs/epochs*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
č
4input_producer/input_producer/limit_epochs/CountUpTo	CountUpTo1input_producer/input_producer/limit_epochs/epochs*D
_class:
86loc:@input_producer/input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Č
*input_producer/input_producer/limit_epochsIdentity+input_producer/input_producer/RandomShuffle5^input_producer/input_producer/limit_epochs/CountUpTo*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
input_producer/input_producerFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
Ę
8input_producer/input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producer/input_producer*input_producer/input_producer/limit_epochs*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2

2input_producer/input_producer/input_producer_CloseQueueCloseV2input_producer/input_producer*
cancel_pending_enqueues( 

4input_producer/input_producer/input_producer_Close_1QueueCloseV2input_producer/input_producer*
cancel_pending_enqueues(
w
1input_producer/input_producer/input_producer_SizeQueueSizeV2input_producer/input_producer*
_output_shapes
: 

"input_producer/input_producer/CastCast1input_producer/input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
h
#input_producer/input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 

!input_producer/input_producer/mulMul"input_producer/input_producer/Cast#input_producer/input_producer/mul/y*
T0*
_output_shapes
: 
¨
6input_producer/input_producer/fraction_of_32_full/tagsConst*
dtype0*B
value9B7 B1input_producer/input_producer/fraction_of_32_full*
_output_shapes
: 
ž
1input_producer/input_producer/fraction_of_32_fullScalarSummary6input_producer/input_producer/fraction_of_32_full/tags!input_producer/input_producer/mul*
T0*
_output_shapes
: 
Ł
%input_producer/input_producer_DequeueQueueDequeueV2input_producer/input_producer*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*
_output_shapes
: 
§
input_producer/GatherGatherDynamicPartition%input_producer/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
Ť
input_producer/Gather_1GatherDynamicPartition_1%input_producer/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
h
input_producer_1/ShapeShapeDynamicPartition:1*
out_type0*
T0*
_output_shapes
:
n
$input_producer_1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
p
&input_producer_1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
p
&input_producer_1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Î
input_producer_1/strided_sliceStridedSliceinput_producer_1/Shape$input_producer_1/strided_slice/stack&input_producer_1/strided_slice/stack_1&input_producer_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
m
+input_producer_1/input_producer/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
m
+input_producer_1/input_producer/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
Ů
%input_producer_1/input_producer/rangeRange+input_producer_1/input_producer/range/startinput_producer_1/strided_slice+input_producer_1/input_producer/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
-input_producer_1/input_producer/RandomShuffleRandomShuffle%input_producer_1/input_producer/range*
seed2 *

seed *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
2input_producer_1/input_producer/limit_epochs/ConstConst*
dtype0	*
value	B	 R *
_output_shapes
: 

3input_producer_1/input_producer/limit_epochs/epochs
VariableV2*
dtype0	*
shape: *
shared_name *
	container *
_output_shapes
: 
ż
:input_producer_1/input_producer/limit_epochs/epochs/AssignAssign3input_producer_1/input_producer/limit_epochs/epochs2input_producer_1/input_producer/limit_epochs/Const*
validate_shape(*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
use_locking(*
T0	*
_output_shapes
: 
â
8input_producer_1/input_producer/limit_epochs/epochs/readIdentity3input_producer_1/input_producer/limit_epochs/epochs*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
T0	*
_output_shapes
: 
î
6input_producer_1/input_producer/limit_epochs/CountUpTo	CountUpTo3input_producer_1/input_producer/limit_epochs/epochs*F
_class<
:8loc:@input_producer_1/input_producer/limit_epochs/epochs*
limit*
T0	*
_output_shapes
: 
Î
,input_producer_1/input_producer/limit_epochsIdentity-input_producer_1/input_producer/RandomShuffle7^input_producer_1/input_producer/limit_epochs/CountUpTo*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
input_producer_1/input_producerFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
: *
	container *
shared_name 
Đ
:input_producer_1/input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producer_1/input_producer,input_producer_1/input_producer/limit_epochs*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2

4input_producer_1/input_producer/input_producer_CloseQueueCloseV2input_producer_1/input_producer*
cancel_pending_enqueues( 

6input_producer_1/input_producer/input_producer_Close_1QueueCloseV2input_producer_1/input_producer*
cancel_pending_enqueues(
{
3input_producer_1/input_producer/input_producer_SizeQueueSizeV2input_producer_1/input_producer*
_output_shapes
: 

$input_producer_1/input_producer/CastCast3input_producer_1/input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
j
%input_producer_1/input_producer/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 

#input_producer_1/input_producer/mulMul$input_producer_1/input_producer/Cast%input_producer_1/input_producer/mul/y*
T0*
_output_shapes
: 
Ź
8input_producer_1/input_producer/fraction_of_32_full/tagsConst*
dtype0*D
value;B9 B3input_producer_1/input_producer/fraction_of_32_full*
_output_shapes
: 
Ä
3input_producer_1/input_producer/fraction_of_32_fullScalarSummary8input_producer_1/input_producer/fraction_of_32_full/tags#input_producer_1/input_producer/mul*
T0*
_output_shapes
: 
§
'input_producer_1/input_producer_DequeueQueueDequeueV2input_producer_1/input_producer*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*
_output_shapes
: 
­
input_producer_1/GatherGatherDynamicPartition:1'input_producer_1/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
ą
input_producer_1/Gather_1GatherDynamicPartition_1:1'input_producer_1/input_producer_Dequeue*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
: 
C
ReadFileReadFileinput_producer/Gather*
_output_shapes
: 
Ů

DecodeJpeg
DecodeJpegReadFile*
ratio*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
try_recover_truncated( *
acceptable_fraction%  ?*
channels*

dct_method *
fancy_upscaling(
P
ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 


ExpandDims
ExpandDims
DecodeJpegExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
U
sizeConst*
dtype0*
valueB"@   @   *
_output_shapes
:
x
ResizeBilinearResizeBilinear
ExpandDimssize*
align_corners( *
T0*&
_output_shapes
:@@
f
SqueezeSqueezeResizeBilinear*
squeeze_dims
 *
T0*"
_output_shapes
:@@
J
div/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
K
divRealDivSqueezediv/y*
T0*"
_output_shapes
:@@
J
sub/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
C
subSubdivsub/y*
T0*"
_output_shapes
:@@
J
mul/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
C
mulMulsubmul/y*
T0*"
_output_shapes
:@@
G

ReadFile_1ReadFileinput_producer_1/Gather*
_output_shapes
: 
Ý
DecodeJpeg_1
DecodeJpeg
ReadFile_1*
ratio*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
try_recover_truncated( *
acceptable_fraction%  ?*
channels*

dct_method *
fancy_upscaling(
R
ExpandDims_1/dimConst*
dtype0*
value	B : *
_output_shapes
: 

ExpandDims_1
ExpandDimsDecodeJpeg_1ExpandDims_1/dim*

Tdim0*
T0*8
_output_shapes&
$:"˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
W
size_1Const*
dtype0*
valueB"@   @   *
_output_shapes
:
~
ResizeBilinear_1ResizeBilinearExpandDims_1size_1*
align_corners( *
T0*&
_output_shapes
:@@
j
	Squeeze_1SqueezeResizeBilinear_1*
squeeze_dims
 *
T0*"
_output_shapes
:@@
L
div_1/yConst*
dtype0*
valueB
 *  C*
_output_shapes
: 
Q
div_1RealDiv	Squeeze_1div_1/y*
T0*"
_output_shapes
:@@
L
sub_1/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
I
sub_1Subdiv_1sub_1/y*
T0*"
_output_shapes
:@@
L
mul_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
I
mul_1Mulsub_1mul_1/y*
T0*"
_output_shapes
:@@
M
batch/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
¤
batch/fifo_queueFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
:@@: *
	container *
shared_name 

batch/fifo_queue_enqueueQueueEnqueueV2batch/fifo_queuemulinput_producer/Gather_1*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2
W
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue*
cancel_pending_enqueues( 
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
x
batch/fraction_of_32_full/tagsConst*
dtype0**
value!B Bbatch/fraction_of_32_full*
_output_shapes
: 
v
batch/fraction_of_32_fullScalarSummarybatch/fraction_of_32_full/tags	batch/mul*
T0*
_output_shapes
: 
I
batch/nConst*
dtype0*
value	B :*
_output_shapes
: 

batchQueueDequeueManyV2batch/fifo_queuebatch/n*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*,
_output_shapes
:@@:
O
batch_1/ConstConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
Ś
batch_1/fifo_queueFIFOQueueV2*
capacity *
component_types
2*
_output_shapes
: *
shapes
:@@: *
	container *
shared_name 

batch_1/fifo_queue_enqueueQueueEnqueueV2batch_1/fifo_queuemul_1input_producer_1/Gather_1*

timeout_ms˙˙˙˙˙˙˙˙˙*
Tcomponents
2
[
batch_1/fifo_queue_CloseQueueCloseV2batch_1/fifo_queue*
cancel_pending_enqueues( 
]
batch_1/fifo_queue_Close_1QueueCloseV2batch_1/fifo_queue*
cancel_pending_enqueues(
R
batch_1/fifo_queue_SizeQueueSizeV2batch_1/fifo_queue*
_output_shapes
: 
]
batch_1/CastCastbatch_1/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
R
batch_1/mul/yConst*
dtype0*
valueB
 *   =*
_output_shapes
: 
P
batch_1/mulMulbatch_1/Castbatch_1/mul/y*
T0*
_output_shapes
: 
|
 batch_1/fraction_of_32_full/tagsConst*
dtype0*,
value#B! Bbatch_1/fraction_of_32_full*
_output_shapes
: 
|
batch_1/fraction_of_32_fullScalarSummary batch_1/fraction_of_32_full/tagsbatch_1/mul*
T0*
_output_shapes
: 
K
	batch_1/nConst*
dtype0*
value	B :*
_output_shapes
: 
 
batch_1QueueDequeueManyV2batch_1/fifo_queue	batch_1/n*

timeout_ms˙˙˙˙˙˙˙˙˙*
component_types
2*,
_output_shapes
:@@:
c
model/PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
m
model/Placeholder_1Placeholder*
dtype0*
shape: */
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
\
model/ShapeShapemodel/Placeholder*
out_type0*
T0*
_output_shapes
:
c
model/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
e
model/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
e
model/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

model/strided_sliceStridedSlicemodel/Shapemodel/strided_slice/stackmodel/strided_slice/stack_1model/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
a
model/generator/Reshape/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
a
model/generator/Reshape/shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
a
model/generator/Reshape/shape/3Const*
dtype0*
value	B :d*
_output_shapes
: 
×
model/generator/Reshape/shapePackmodel/strided_slicemodel/generator/Reshape/shape/1model/generator/Reshape/shape/2model/generator/Reshape/shape/3*
_output_shapes
:*

axis *
T0*
N

model/generator/ReshapeReshapemodel/Placeholdermodel/generator/Reshape/shape*
Tshape0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d

*model/generator/layer1/random_normal/shapeConst*
dtype0*%
valueB"         d   *
_output_shapes
:
n
)model/generator/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
9model/generator/layer1/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:d
É
(model/generator/layer1/random_normal/mulMul9model/generator/layer1/random_normal/RandomStandardNormal+model/generator/layer1/random_normal/stddev*
T0*'
_output_shapes
:d
˛
$model/generator/layer1/random_normalAdd(model/generator/layer1/random_normal/mul)model/generator/layer1/random_normal/mean*
T0*'
_output_shapes
:d
¤
model/generator/layer1/weights
VariableV2*
dtype0*
shape:d*
shared_name *
	container *'
_output_shapes
:d

%model/generator/layer1/weights/AssignAssignmodel/generator/layer1/weights$model/generator/layer1/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:d
´
#model/generator/layer1/weights/readIdentitymodel/generator/layer1/weights*1
_class'
%#loc:@model/generator/layer1/weights*
T0*'
_output_shapes
:d
s
model/generator/layer1/ShapeShapemodel/generator/Reshape*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer1/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer1/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer1/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer1/strided_sliceStridedSlicemodel/generator/layer1/Shape*model/generator/layer1/strided_slice/stack,model/generator/layer1/strided_slice/stack_1,model/generator/layer1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer1/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer1/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer1/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer1/conv2d_transpose/output_shapePack$model/generator/layer1/strided_slice6model/generator/layer1/conv2d_transpose/output_shape/16model/generator/layer1/conv2d_transpose/output_shape/26model/generator/layer1/conv2d_transpose/output_shape/3*
_output_shapes
:*

axis *
T0*
N
ß
'model/generator/layer1/conv2d_transposeConv2DBackpropInput4model/generator/layer1/conv2d_transpose/output_shape#model/generator/layer1/weights/readmodel/generator/Reshape*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0

,model/generator/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer1/random_normal_1/mulMul;model/generator/layer1/random_normal_1/RandomStandardNormal-model/generator/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer1/random_normal_1Add*model/generator/layer1/random_normal_1/mul+model/generator/layer1/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer1/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:
ř
"model/generator/layer1/bias/AssignAssignmodel/generator/layer1/bias&model/generator/layer1/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer1/bias/readIdentitymodel/generator/layer1/bias*.
_class$
" loc:@model/generator/layer1/bias*
T0*#
_output_shapes
:
§
model/generator/layer1/addAdd'model/generator/layer1/conv2d_transpose model/generator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer1/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer1/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
shared_name 

,model/generator/layer1/BatchNorm/beta/AssignAssign%model/generator/layer1/BatchNorm/beta7model/generator/layer1/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer1/BatchNorm/beta/readIdentity%model/generator/layer1/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer1/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer1/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer1/BatchNorm/gamma/AssignAssign&model/generator/layer1/BatchNorm/gamma8model/generator/layer1/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer1/BatchNorm/gamma/readIdentity&model/generator/layer1/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer1/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer1/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer1/BatchNorm/moving_mean/AssignAssign,model/generator/layer1/BatchNorm/moving_mean>model/generator/layer1/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer1/BatchNorm/moving_mean/readIdentity,model/generator/layer1/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer1/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer1/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer1/BatchNorm/moving_variance/AssignAssign0model/generator/layer1/BatchNorm/moving_varianceBmodel/generator/layer1/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer1/BatchNorm/moving_variance/readIdentity0model/generator/layer1/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer1/BatchNorm/moments/MeanMeanmodel/generator/layer1/add?model/generator/layer1/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ś
5model/generator/layer1/BatchNorm/moments/StopGradientStopGradient-model/generator/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer1/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
×
Bmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer1/add5model/generator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer1/add5model/generator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
Ź
Wmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
y
.model/generator/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer1/BatchNorm/moments/ReshapeReshape5model/generator/layer1/BatchNorm/moments/StopGradient.model/generator/layer1/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ŕ
:model/generator/layer1/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer1/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer1/BatchNorm/moments/normalize/meanAdd?model/generator/layer1/BatchNorm/moments/normalize/shifted_mean0model/generator/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer1/BatchNorm/moments/normalize/MulMulEmodel/generator/layer1/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer1/BatchNorm/moments/normalize/SquareSquare?model/generator/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer1/BatchNorm/moments/normalize/varianceSub6model/generator/layer1/BatchNorm/moments/normalize/Mul9model/generator/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer1/BatchNorm/AssignMovingAvg/subSub1model/generator/layer1/BatchNorm/moving_mean/read7model/generator/layer1/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer1/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer1/BatchNorm/AssignMovingAvg/sub6model/generator/layer1/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer1/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer1/BatchNorm/moving_mean4model/generator/layer1/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer1/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer1/BatchNorm/moving_variance/read;model/generator/layer1/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer1/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer1/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer1/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer1/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer1/BatchNorm/moving_variance6model/generator/layer1/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer1/BatchNorm/batchnorm/addAdd;model/generator/layer1/BatchNorm/moments/normalize/variance0model/generator/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer1/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer1/BatchNorm/batchnorm/mulMul0model/generator/layer1/BatchNorm/batchnorm/Rsqrt+model/generator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer1/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer1/add.model/generator/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer1/BatchNorm/batchnorm/mul_2Mul7model/generator/layer1/BatchNorm/moments/normalize/mean.model/generator/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer1/BatchNorm/batchnorm/subSub*model/generator/layer1/BatchNorm/beta/read0model/generator/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer1/BatchNorm/batchnorm/add_1Add0model/generator/layer1/BatchNorm/batchnorm/mul_1.model/generator/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer1/ReluRelu0model/generator/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
)model/generator/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer2/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ę
(model/generator/layer2/random_normal/mulMul9model/generator/layer2/random_normal/RandomStandardNormal+model/generator/layer2/random_normal/stddev*
T0*(
_output_shapes
:
ł
$model/generator/layer2/random_normalAdd(model/generator/layer2/random_normal/mul)model/generator/layer2/random_normal/mean*
T0*(
_output_shapes
:
Ś
model/generator/layer2/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *(
_output_shapes
:

%model/generator/layer2/weights/AssignAssignmodel/generator/layer2/weights$model/generator/layer2/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:
ľ
#model/generator/layer2/weights/readIdentitymodel/generator/layer2/weights*1
_class'
%#loc:@model/generator/layer2/weights*
T0*(
_output_shapes
:
w
model/generator/layer2/ShapeShapemodel/generator/layer1/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer2/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer2/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer2/strided_sliceStridedSlicemodel/generator/layer2/Shape*model/generator/layer2/strided_slice/stack,model/generator/layer2/strided_slice/stack_1,model/generator/layer2/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer2/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer2/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer2/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer2/conv2d_transpose/output_shapePack$model/generator/layer2/strided_slice6model/generator/layer2/conv2d_transpose/output_shape/16model/generator/layer2/conv2d_transpose/output_shape/26model/generator/layer2/conv2d_transpose/output_shape/3*
_output_shapes
:*

axis *
T0*
N
â
'model/generator/layer2/conv2d_transposeConv2DBackpropInput4model/generator/layer2/conv2d_transpose/output_shape#model/generator/layer2/weights/readmodel/generator/layer1/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer2/random_normal_1/mulMul;model/generator/layer2/random_normal_1/RandomStandardNormal-model/generator/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer2/random_normal_1Add*model/generator/layer2/random_normal_1/mul+model/generator/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer2/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:
ř
"model/generator/layer2/bias/AssignAssignmodel/generator/layer2/bias&model/generator/layer2/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer2/bias/readIdentitymodel/generator/layer2/bias*.
_class$
" loc:@model/generator/layer2/bias*
T0*#
_output_shapes
:
§
model/generator/layer2/addAdd'model/generator/layer2/conv2d_transpose model/generator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer2/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer2/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
shared_name 

,model/generator/layer2/BatchNorm/beta/AssignAssign%model/generator/layer2/BatchNorm/beta7model/generator/layer2/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer2/BatchNorm/beta/readIdentity%model/generator/layer2/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer2/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer2/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer2/BatchNorm/gamma/AssignAssign&model/generator/layer2/BatchNorm/gamma8model/generator/layer2/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer2/BatchNorm/gamma/readIdentity&model/generator/layer2/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer2/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer2/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer2/BatchNorm/moving_mean/AssignAssign,model/generator/layer2/BatchNorm/moving_mean>model/generator/layer2/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer2/BatchNorm/moving_mean/readIdentity,model/generator/layer2/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer2/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer2/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer2/BatchNorm/moving_variance/AssignAssign0model/generator/layer2/BatchNorm/moving_varianceBmodel/generator/layer2/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer2/BatchNorm/moving_variance/readIdentity0model/generator/layer2/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer2/BatchNorm/moments/MeanMeanmodel/generator/layer2/add?model/generator/layer2/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ś
5model/generator/layer2/BatchNorm/moments/StopGradientStopGradient-model/generator/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer2/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
×
Bmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer2/add5model/generator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer2/add5model/generator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
Ź
Wmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
y
.model/generator/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer2/BatchNorm/moments/ReshapeReshape5model/generator/layer2/BatchNorm/moments/StopGradient.model/generator/layer2/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ŕ
:model/generator/layer2/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer2/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer2/BatchNorm/moments/normalize/meanAdd?model/generator/layer2/BatchNorm/moments/normalize/shifted_mean0model/generator/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer2/BatchNorm/moments/normalize/MulMulEmodel/generator/layer2/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer2/BatchNorm/moments/normalize/SquareSquare?model/generator/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer2/BatchNorm/moments/normalize/varianceSub6model/generator/layer2/BatchNorm/moments/normalize/Mul9model/generator/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer2/BatchNorm/AssignMovingAvg/subSub1model/generator/layer2/BatchNorm/moving_mean/read7model/generator/layer2/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer2/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer2/BatchNorm/AssignMovingAvg/sub6model/generator/layer2/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer2/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer2/BatchNorm/moving_mean4model/generator/layer2/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer2/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer2/BatchNorm/moving_variance/read;model/generator/layer2/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer2/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer2/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer2/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer2/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer2/BatchNorm/moving_variance6model/generator/layer2/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer2/BatchNorm/batchnorm/addAdd;model/generator/layer2/BatchNorm/moments/normalize/variance0model/generator/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer2/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer2/BatchNorm/batchnorm/mulMul0model/generator/layer2/BatchNorm/batchnorm/Rsqrt+model/generator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer2/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer2/add.model/generator/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer2/BatchNorm/batchnorm/mul_2Mul7model/generator/layer2/BatchNorm/moments/normalize/mean.model/generator/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer2/BatchNorm/batchnorm/subSub*model/generator/layer2/BatchNorm/beta/read0model/generator/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer2/BatchNorm/batchnorm/add_1Add0model/generator/layer2/BatchNorm/batchnorm/mul_1.model/generator/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer2/ReluRelu0model/generator/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
n
)model/generator/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer3/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ę
(model/generator/layer3/random_normal/mulMul9model/generator/layer3/random_normal/RandomStandardNormal+model/generator/layer3/random_normal/stddev*
T0*(
_output_shapes
:
ł
$model/generator/layer3/random_normalAdd(model/generator/layer3/random_normal/mul)model/generator/layer3/random_normal/mean*
T0*(
_output_shapes
:
Ś
model/generator/layer3/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *(
_output_shapes
:

%model/generator/layer3/weights/AssignAssignmodel/generator/layer3/weights$model/generator/layer3/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:
ľ
#model/generator/layer3/weights/readIdentitymodel/generator/layer3/weights*1
_class'
%#loc:@model/generator/layer3/weights*
T0*(
_output_shapes
:
w
model/generator/layer3/ShapeShapemodel/generator/layer2/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer3/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer3/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer3/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer3/strided_sliceStridedSlicemodel/generator/layer3/Shape*model/generator/layer3/strided_slice/stack,model/generator/layer3/strided_slice/stack_1,model/generator/layer3/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer3/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :*
_output_shapes
: 
x
6model/generator/layer3/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :*
_output_shapes
: 
y
6model/generator/layer3/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer3/conv2d_transpose/output_shapePack$model/generator/layer3/strided_slice6model/generator/layer3/conv2d_transpose/output_shape/16model/generator/layer3/conv2d_transpose/output_shape/26model/generator/layer3/conv2d_transpose/output_shape/3*
_output_shapes
:*

axis *
T0*
N
â
'model/generator/layer3/conv2d_transposeConv2DBackpropInput4model/generator/layer3/conv2d_transpose/output_shape#model/generator/layer3/weights/readmodel/generator/layer2/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
p
+model/generator/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ë
*model/generator/layer3/random_normal_1/mulMul;model/generator/layer3/random_normal_1/RandomStandardNormal-model/generator/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
´
&model/generator/layer3/random_normal_1Add*model/generator/layer3/random_normal_1/mul+model/generator/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

model/generator/layer3/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:
ř
"model/generator/layer3/bias/AssignAssignmodel/generator/layer3/bias&model/generator/layer3/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:
§
 model/generator/layer3/bias/readIdentitymodel/generator/layer3/bias*.
_class$
" loc:@model/generator/layer3/bias*
T0*#
_output_shapes
:
§
model/generator/layer3/addAdd'model/generator/layer3/conv2d_transpose model/generator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
7model/generator/layer3/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer3/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
shared_name 

,model/generator/layer3/BatchNorm/beta/AssignAssign%model/generator/layer3/BatchNorm/beta7model/generator/layer3/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer3/BatchNorm/beta/readIdentity%model/generator/layer3/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer3/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer3/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer3/BatchNorm/gamma/AssignAssign&model/generator/layer3/BatchNorm/gamma8model/generator/layer3/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer3/BatchNorm/gamma/readIdentity&model/generator/layer3/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer3/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer3/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer3/BatchNorm/moving_mean/AssignAssign,model/generator/layer3/BatchNorm/moving_mean>model/generator/layer3/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer3/BatchNorm/moving_mean/readIdentity,model/generator/layer3/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer3/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer3/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer3/BatchNorm/moving_variance/AssignAssign0model/generator/layer3/BatchNorm/moving_varianceBmodel/generator/layer3/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer3/BatchNorm/moving_variance/readIdentity0model/generator/layer3/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer3/BatchNorm/moments/MeanMeanmodel/generator/layer3/add?model/generator/layer3/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ś
5model/generator/layer3/BatchNorm/moments/StopGradientStopGradient-model/generator/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer3/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
×
Bmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer3/add5model/generator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
Pmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer3/add5model/generator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Xmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
Ź
Wmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
y
.model/generator/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer3/BatchNorm/moments/ReshapeReshape5model/generator/layer3/BatchNorm/moments/StopGradient.model/generator/layer3/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ŕ
:model/generator/layer3/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer3/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer3/BatchNorm/moments/normalize/meanAdd?model/generator/layer3/BatchNorm/moments/normalize/shifted_mean0model/generator/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer3/BatchNorm/moments/normalize/MulMulEmodel/generator/layer3/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer3/BatchNorm/moments/normalize/SquareSquare?model/generator/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer3/BatchNorm/moments/normalize/varianceSub6model/generator/layer3/BatchNorm/moments/normalize/Mul9model/generator/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer3/BatchNorm/AssignMovingAvg/subSub1model/generator/layer3/BatchNorm/moving_mean/read7model/generator/layer3/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer3/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer3/BatchNorm/AssignMovingAvg/sub6model/generator/layer3/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer3/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer3/BatchNorm/moving_mean4model/generator/layer3/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer3/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer3/BatchNorm/moving_variance/read;model/generator/layer3/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer3/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer3/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer3/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer3/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer3/BatchNorm/moving_variance6model/generator/layer3/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer3/BatchNorm/batchnorm/addAdd;model/generator/layer3/BatchNorm/moments/normalize/variance0model/generator/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer3/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer3/BatchNorm/batchnorm/mulMul0model/generator/layer3/BatchNorm/batchnorm/Rsqrt+model/generator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer3/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer3/add.model/generator/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
0model/generator/layer3/BatchNorm/batchnorm/mul_2Mul7model/generator/layer3/BatchNorm/moments/normalize/mean.model/generator/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer3/BatchNorm/batchnorm/subSub*model/generator/layer3/BatchNorm/beta/read0model/generator/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer3/BatchNorm/batchnorm/add_1Add0model/generator/layer3/BatchNorm/batchnorm/mul_1.model/generator/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/generator/layer3/ReluRelu0model/generator/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

*model/generator/layer4/random_normal/shapeConst*
dtype0*%
valueB"              *
_output_shapes
:
n
)model/generator/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
9model/generator/layer4/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:  
Ę
(model/generator/layer4/random_normal/mulMul9model/generator/layer4/random_normal/RandomStandardNormal+model/generator/layer4/random_normal/stddev*
T0*(
_output_shapes
:  
ł
$model/generator/layer4/random_normalAdd(model/generator/layer4/random_normal/mul)model/generator/layer4/random_normal/mean*
T0*(
_output_shapes
:  
Ś
model/generator/layer4/weights
VariableV2*
dtype0*
shape:  *
shared_name *
	container *(
_output_shapes
:  

%model/generator/layer4/weights/AssignAssignmodel/generator/layer4/weights$model/generator/layer4/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:  
ľ
#model/generator/layer4/weights/readIdentitymodel/generator/layer4/weights*1
_class'
%#loc:@model/generator/layer4/weights*
T0*(
_output_shapes
:  
w
model/generator/layer4/ShapeShapemodel/generator/layer3/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer4/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer4/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer4/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer4/strided_sliceStridedSlicemodel/generator/layer4/Shape*model/generator/layer4/strided_slice/stack,model/generator/layer4/strided_slice/stack_1,model/generator/layer4/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer4/conv2d_transpose/output_shape/1Const*
dtype0*
value	B : *
_output_shapes
: 
x
6model/generator/layer4/conv2d_transpose/output_shape/2Const*
dtype0*
value	B : *
_output_shapes
: 
y
6model/generator/layer4/conv2d_transpose/output_shape/3Const*
dtype0*
value
B :*
_output_shapes
: 
Ä
4model/generator/layer4/conv2d_transpose/output_shapePack$model/generator/layer4/strided_slice6model/generator/layer4/conv2d_transpose/output_shape/16model/generator/layer4/conv2d_transpose/output_shape/26model/generator/layer4/conv2d_transpose/output_shape/3*
_output_shapes
:*

axis *
T0*
N
â
'model/generator/layer4/conv2d_transposeConv2DBackpropInput4model/generator/layer4/conv2d_transpose/output_shape#model/generator/layer4/weights/readmodel/generator/layer3/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
p
+model/generator/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
;model/generator/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
Ë
*model/generator/layer4/random_normal_1/mulMul;model/generator/layer4/random_normal_1/RandomStandardNormal-model/generator/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:  
´
&model/generator/layer4/random_normal_1Add*model/generator/layer4/random_normal_1/mul+model/generator/layer4/random_normal_1/mean*
T0*#
_output_shapes
:  

model/generator/layer4/bias
VariableV2*
dtype0*
shape:  *
shared_name *
	container *#
_output_shapes
:  
ř
"model/generator/layer4/bias/AssignAssignmodel/generator/layer4/bias&model/generator/layer4/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:  
§
 model/generator/layer4/bias/readIdentitymodel/generator/layer4/bias*.
_class$
" loc:@model/generator/layer4/bias*
T0*#
_output_shapes
:  
§
model/generator/layer4/addAdd'model/generator/layer4/conv2d_transpose model/generator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ŕ
7model/generator/layer4/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Í
%model/generator/layer4/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
shared_name 

,model/generator/layer4/BatchNorm/beta/AssignAssign%model/generator/layer4/BatchNorm/beta7model/generator/layer4/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
˝
*model/generator/layer4/BatchNorm/beta/readIdentity%model/generator/layer4/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
T0*
_output_shapes	
:
Â
8model/generator/layer4/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
Ď
&model/generator/layer4/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
shared_name 
Ł
-model/generator/layer4/BatchNorm/gamma/AssignAssign&model/generator/layer4/BatchNorm/gamma8model/generator/layer4/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ŕ
+model/generator/layer4/BatchNorm/gamma/readIdentity&model/generator/layer4/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
T0*
_output_shapes	
:
Î
>model/generator/layer4/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
Ű
,model/generator/layer4/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
shared_name 
ť
3model/generator/layer4/BatchNorm/moving_mean/AssignAssign,model/generator/layer4/BatchNorm/moving_mean>model/generator/layer4/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ň
1model/generator/layer4/BatchNorm/moving_mean/readIdentity,model/generator/layer4/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ö
Bmodel/generator/layer4/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ă
0model/generator/layer4/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
shared_name 
Ë
7model/generator/layer4/BatchNorm/moving_variance/AssignAssign0model/generator/layer4/BatchNorm/moving_varianceBmodel/generator/layer4/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/generator/layer4/BatchNorm/moving_variance/readIdentity0model/generator/layer4/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

?model/generator/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
á
-model/generator/layer4/BatchNorm/moments/MeanMeanmodel/generator/layer4/add?model/generator/layer4/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ś
5model/generator/layer4/BatchNorm/moments/StopGradientStopGradient-model/generator/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer4/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
×
Bmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer4/add5model/generator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ó
Pmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer4/add5model/generator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
­
Xmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ž
Fmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
Ź
Wmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Emodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
y
.model/generator/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ö
0model/generator/layer4/BatchNorm/moments/ReshapeReshape5model/generator/layer4/BatchNorm/moments/StopGradient.model/generator/layer4/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ŕ
:model/generator/layer4/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
đ
?model/generator/layer4/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
×
7model/generator/layer4/BatchNorm/moments/normalize/meanAdd?model/generator/layer4/BatchNorm/moments/normalize/shifted_mean0model/generator/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ć
6model/generator/layer4/BatchNorm/moments/normalize/MulMulEmodel/generator/layer4/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
Ş
9model/generator/layer4/BatchNorm/moments/normalize/SquareSquare?model/generator/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
Ű
;model/generator/layer4/BatchNorm/moments/normalize/varianceSub6model/generator/layer4/BatchNorm/moments/normalize/Mul9model/generator/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
ź
6model/generator/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer4/BatchNorm/AssignMovingAvg/subSub1model/generator/layer4/BatchNorm/moving_mean/read7model/generator/layer4/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:

4model/generator/layer4/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer4/BatchNorm/AssignMovingAvg/sub6model/generator/layer4/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:

0model/generator/layer4/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer4/BatchNorm/moving_mean4model/generator/layer4/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Â
8model/generator/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer4/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer4/BatchNorm/moving_variance/read;model/generator/layer4/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

6model/generator/layer4/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer4/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer4/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
§
2model/generator/layer4/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer4/BatchNorm/moving_variance6model/generator/layer4/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
u
0model/generator/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ę
.model/generator/layer4/BatchNorm/batchnorm/addAdd;model/generator/layer4/BatchNorm/moments/normalize/variance0model/generator/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

0model/generator/layer4/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
ş
.model/generator/layer4/BatchNorm/batchnorm/mulMul0model/generator/layer4/BatchNorm/batchnorm/Rsqrt+model/generator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
ž
0model/generator/layer4/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer4/add.model/generator/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ć
0model/generator/layer4/BatchNorm/batchnorm/mul_2Mul7model/generator/layer4/BatchNorm/moments/normalize/mean.model/generator/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
š
.model/generator/layer4/BatchNorm/batchnorm/subSub*model/generator/layer4/BatchNorm/beta/read0model/generator/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
Ô
0model/generator/layer4/BatchNorm/batchnorm/add_1Add0model/generator/layer4/BatchNorm/batchnorm/mul_1.model/generator/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

model/generator/layer4/ReluRelu0model/generator/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

*model/generator/layer5/random_normal/shapeConst*
dtype0*%
valueB"@   @         *
_output_shapes
:
n
)model/generator/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
+model/generator/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ő
9model/generator/layer5/random_normal/RandomStandardNormalRandomStandardNormal*model/generator/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:@@
É
(model/generator/layer5/random_normal/mulMul9model/generator/layer5/random_normal/RandomStandardNormal+model/generator/layer5/random_normal/stddev*
T0*'
_output_shapes
:@@
˛
$model/generator/layer5/random_normalAdd(model/generator/layer5/random_normal/mul)model/generator/layer5/random_normal/mean*
T0*'
_output_shapes
:@@
¤
model/generator/layer5/weights
VariableV2*
dtype0*
shape:@@*
shared_name *
	container *'
_output_shapes
:@@

%model/generator/layer5/weights/AssignAssignmodel/generator/layer5/weights$model/generator/layer5/random_normal*
validate_shape(*1
_class'
%#loc:@model/generator/layer5/weights*
use_locking(*
T0*'
_output_shapes
:@@
´
#model/generator/layer5/weights/readIdentitymodel/generator/layer5/weights*1
_class'
%#loc:@model/generator/layer5/weights*
T0*'
_output_shapes
:@@
w
model/generator/layer5/ShapeShapemodel/generator/layer4/Relu*
out_type0*
T0*
_output_shapes
:
t
*model/generator/layer5/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
v
,model/generator/layer5/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
v
,model/generator/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ě
$model/generator/layer5/strided_sliceStridedSlicemodel/generator/layer5/Shape*model/generator/layer5/strided_slice/stack,model/generator/layer5/strided_slice/stack_1,model/generator/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
x
6model/generator/layer5/conv2d_transpose/output_shape/1Const*
dtype0*
value	B :@*
_output_shapes
: 
x
6model/generator/layer5/conv2d_transpose/output_shape/2Const*
dtype0*
value	B :@*
_output_shapes
: 
x
6model/generator/layer5/conv2d_transpose/output_shape/3Const*
dtype0*
value	B :*
_output_shapes
: 
Ä
4model/generator/layer5/conv2d_transpose/output_shapePack$model/generator/layer5/strided_slice6model/generator/layer5/conv2d_transpose/output_shape/16model/generator/layer5/conv2d_transpose/output_shape/26model/generator/layer5/conv2d_transpose/output_shape/3*
_output_shapes
:*

axis *
T0*
N
â
'model/generator/layer5/conv2d_transposeConv2DBackpropInput4model/generator/layer5/conv2d_transpose/output_shape#model/generator/layer5/weights/readmodel/generator/layer4/Relu*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0

,model/generator/layer5/random_normal_1/shapeConst*
dtype0*!
valueB"@   @      *
_output_shapes
:
p
+model/generator/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
-model/generator/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ô
;model/generator/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal,model/generator/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*"
_output_shapes
:@@
Ę
*model/generator/layer5/random_normal_1/mulMul;model/generator/layer5/random_normal_1/RandomStandardNormal-model/generator/layer5/random_normal_1/stddev*
T0*"
_output_shapes
:@@
ł
&model/generator/layer5/random_normal_1Add*model/generator/layer5/random_normal_1/mul+model/generator/layer5/random_normal_1/mean*
T0*"
_output_shapes
:@@

model/generator/layer5/bias
VariableV2*
dtype0*
shape:@@*
shared_name *
	container *"
_output_shapes
:@@
÷
"model/generator/layer5/bias/AssignAssignmodel/generator/layer5/bias&model/generator/layer5/random_normal_1*
validate_shape(*.
_class$
" loc:@model/generator/layer5/bias*
use_locking(*
T0*"
_output_shapes
:@@
Ś
 model/generator/layer5/bias/readIdentitymodel/generator/layer5/bias*.
_class$
" loc:@model/generator/layer5/bias*
T0*"
_output_shapes
:@@
Ś
model/generator/layer5/addAdd'model/generator/layer5/conv2d_transpose model/generator/layer5/bias/read*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ž
7model/generator/layer5/BatchNorm/beta/Initializer/ConstConst*
dtype0*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
valueB*    *
_output_shapes
:
Ë
%model/generator/layer5/BatchNorm/beta
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
shared_name 

,model/generator/layer5/BatchNorm/beta/AssignAssign%model/generator/layer5/BatchNorm/beta7model/generator/layer5/BatchNorm/beta/Initializer/Const*
validate_shape(*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:
ź
*model/generator/layer5/BatchNorm/beta/readIdentity%model/generator/layer5/BatchNorm/beta*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
T0*
_output_shapes
:
Ŕ
8model/generator/layer5/BatchNorm/gamma/Initializer/ConstConst*
dtype0*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
valueB*  ?*
_output_shapes
:
Í
&model/generator/layer5/BatchNorm/gamma
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
shared_name 
˘
-model/generator/layer5/BatchNorm/gamma/AssignAssign&model/generator/layer5/BatchNorm/gamma8model/generator/layer5/BatchNorm/gamma/Initializer/Const*
validate_shape(*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:
ż
+model/generator/layer5/BatchNorm/gamma/readIdentity&model/generator/layer5/BatchNorm/gamma*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
T0*
_output_shapes
:
Ě
>model/generator/layer5/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
valueB*    *
_output_shapes
:
Ů
,model/generator/layer5/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
shared_name 
ş
3model/generator/layer5/BatchNorm/moving_mean/AssignAssign,model/generator/layer5/BatchNorm/moving_mean>model/generator/layer5/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:
Ń
1model/generator/layer5/BatchNorm/moving_mean/readIdentity,model/generator/layer5/BatchNorm/moving_mean*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ô
Bmodel/generator/layer5/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes
:
á
0model/generator/layer5/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
shared_name 
Ę
7model/generator/layer5/BatchNorm/moving_variance/AssignAssign0model/generator/layer5/BatchNorm/moving_varianceBmodel/generator/layer5/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:
Ý
5model/generator/layer5/BatchNorm/moving_variance/readIdentity0model/generator/layer5/BatchNorm/moving_variance*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

?model/generator/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ŕ
-model/generator/layer5/BatchNorm/moments/MeanMeanmodel/generator/layer5/add?model/generator/layer5/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*&
_output_shapes
:
Ľ
5model/generator/layer5/BatchNorm/moments/StopGradientStopGradient-model/generator/layer5/BatchNorm/moments/Mean*
T0*&
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/generator/layer5/add*
out_type0*
T0*
_output_shapes
:
Ĺ
Cmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/CastCastDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
˘
Mmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ś
Emodel/generator/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherCmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/CastMmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:

Dmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/countProdEmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/GatherDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ö
Bmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SubSubmodel/generator/layer5/add5model/generator/layer5/BatchNorm/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ň
Pmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/generator/layer5/add5model/generator/layer5/BatchNorm/moments/StopGradient*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
­
Xmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
­
Fmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumBmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SubXmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ź
Wmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
š
Emodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumPmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceWmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
x
.model/generator/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
Ő
0model/generator/layer5/BatchNorm/moments/ReshapeReshape5model/generator/layer5/BatchNorm/moments/StopGradient.model/generator/layer5/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes
:
Ŕ
:model/generator/layer5/BatchNorm/moments/normalize/divisor
ReciprocalDmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/countG^model/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssF^model/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ď
?model/generator/layer5/BatchNorm/moments/normalize/shifted_meanMulFmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss:model/generator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
Ö
7model/generator/layer5/BatchNorm/moments/normalize/meanAdd?model/generator/layer5/BatchNorm/moments/normalize/shifted_mean0model/generator/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
ĺ
6model/generator/layer5/BatchNorm/moments/normalize/MulMulEmodel/generator/layer5/BatchNorm/moments/sufficient_statistics/var_ss:model/generator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
Š
9model/generator/layer5/BatchNorm/moments/normalize/SquareSquare?model/generator/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
Ú
;model/generator/layer5/BatchNorm/moments/normalize/varianceSub6model/generator/layer5/BatchNorm/moments/normalize/Mul9model/generator/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
ź
6model/generator/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

4model/generator/layer5/BatchNorm/AssignMovingAvg/subSub1model/generator/layer5/BatchNorm/moving_mean/read7model/generator/layer5/BatchNorm/moments/normalize/mean*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

4model/generator/layer5/BatchNorm/AssignMovingAvg/mulMul4model/generator/layer5/BatchNorm/AssignMovingAvg/sub6model/generator/layer5/BatchNorm/AssignMovingAvg/decay*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

0model/generator/layer5/BatchNorm/AssignMovingAvg	AssignSub,model/generator/layer5/BatchNorm/moving_mean4model/generator/layer5/BatchNorm/AssignMovingAvg/mul*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Â
8model/generator/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 

6model/generator/layer5/BatchNorm/AssignMovingAvg_1/subSub5model/generator/layer5/BatchNorm/moving_variance/read;model/generator/layer5/BatchNorm/moments/normalize/variance*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

6model/generator/layer5/BatchNorm/AssignMovingAvg_1/mulMul6model/generator/layer5/BatchNorm/AssignMovingAvg_1/sub8model/generator/layer5/BatchNorm/AssignMovingAvg_1/decay*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Ś
2model/generator/layer5/BatchNorm/AssignMovingAvg_1	AssignSub0model/generator/layer5/BatchNorm/moving_variance6model/generator/layer5/BatchNorm/AssignMovingAvg_1/mul*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
u
0model/generator/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
É
.model/generator/layer5/BatchNorm/batchnorm/addAdd;model/generator/layer5/BatchNorm/moments/normalize/variance0model/generator/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

0model/generator/layer5/BatchNorm/batchnorm/RsqrtRsqrt.model/generator/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
š
.model/generator/layer5/BatchNorm/batchnorm/mulMul0model/generator/layer5/BatchNorm/batchnorm/Rsqrt+model/generator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
˝
0model/generator/layer5/BatchNorm/batchnorm/mul_1Mulmodel/generator/layer5/add.model/generator/layer5/BatchNorm/batchnorm/mul*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ĺ
0model/generator/layer5/BatchNorm/batchnorm/mul_2Mul7model/generator/layer5/BatchNorm/moments/normalize/mean.model/generator/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
¸
.model/generator/layer5/BatchNorm/batchnorm/subSub*model/generator/layer5/BatchNorm/beta/read0model/generator/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
Ó
0model/generator/layer5/BatchNorm/batchnorm/add_1Add0model/generator/layer5/BatchNorm/batchnorm/mul_1.model/generator/layer5/BatchNorm/batchnorm/sub*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@

model/generator/layer5/ReluRelu0model/generator/layer5/BatchNorm/batchnorm/add_1*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
`
model/Shape_1Shapemodel/Placeholder_1*
out_type0*
T0*
_output_shapes
:
e
model/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
g
model/strided_slice_1/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
g
model/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ą
model/strided_slice_1StridedSlicemodel/Shape_1model/strided_slice_1/stackmodel/strided_slice_1/stack_1model/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask

.model/discriminator/layer1/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
=model/discriminator/layer1/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:
Ő
,model/discriminator/layer1/random_normal/mulMul=model/discriminator/layer1/random_normal/RandomStandardNormal/model/discriminator/layer1/random_normal/stddev*
T0*'
_output_shapes
:
ž
(model/discriminator/layer1/random_normalAdd,model/discriminator/layer1/random_normal/mul-model/discriminator/layer1/random_normal/mean*
T0*'
_output_shapes
:
¨
"model/discriminator/layer1/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *'
_output_shapes
:

)model/discriminator/layer1/weights/AssignAssign"model/discriminator/layer1/weights(model/discriminator/layer1/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:
Ŕ
'model/discriminator/layer1/weights/readIdentity"model/discriminator/layer1/weights*5
_class+
)'loc:@model/discriminator/layer1/weights*
T0*'
_output_shapes
:

0model/discriminator/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
t
/model/discriminator/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
×
.model/discriminator/layer1/random_normal_1/mulMul?model/discriminator/layer1/random_normal_1/RandomStandardNormal1model/discriminator/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:  
Ŕ
*model/discriminator/layer1/random_normal_1Add.model/discriminator/layer1/random_normal_1/mul/model/discriminator/layer1/random_normal_1/mean*
T0*#
_output_shapes
:  

model/discriminator/layer1/bias
VariableV2*
dtype0*
shape:  *
shared_name *
	container *#
_output_shapes
:  

&model/discriminator/layer1/bias/AssignAssignmodel/discriminator/layer1/bias*model/discriminator/layer1/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:  
ł
$model/discriminator/layer1/bias/readIdentitymodel/discriminator/layer1/bias*2
_class(
&$loc:@model/discriminator/layer1/bias*
T0*#
_output_shapes
:  
ű
!model/discriminator/layer1/Conv2DConv2Dmodel/Placeholder_1'model/discriminator/layer1/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer1/addAdd!model/discriminator/layer1/Conv2D$model/discriminator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
;model/discriminator/layer1/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer1/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer1/BatchNorm/beta/AssignAssign)model/discriminator/layer1/BatchNorm/beta;model/discriminator/layer1/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer1/BatchNorm/beta/readIdentity)model/discriminator/layer1/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer1/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer1/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer1/BatchNorm/gamma/AssignAssign*model/discriminator/layer1/BatchNorm/gamma<model/discriminator/layer1/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer1/BatchNorm/gamma/readIdentity*model/discriminator/layer1/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer1/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer1/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer1/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer1/BatchNorm/moving_meanBmodel/discriminator/layer1/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer1/BatchNorm/moving_mean/readIdentity0model/discriminator/layer1/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer1/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer1/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer1/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer1/BatchNorm/moving_varianceFmodel/discriminator/layer1/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer1/BatchNorm/moving_variance/readIdentity4model/discriminator/layer1/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer1/BatchNorm/moments/MeanMeanmodel/discriminator/layer1/addCmodel/discriminator/layer1/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ž
9model/discriminator/layer1/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer1/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ă
Fmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer1/add9model/discriminator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
˙
Tmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer1/add9model/discriminator/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ą
\model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
°
[model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
}
2model/discriminator/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer1/BatchNorm/moments/ReshapeReshape9model/discriminator/layer1/BatchNorm/moments/StopGradient2model/discriminator/layer1/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Đ
>model/discriminator/layer1/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer1/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer1/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer1/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer1/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer1/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer1/BatchNorm/moments/normalize/Mul=model/discriminator/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer1/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer1/BatchNorm/moving_mean/read;model/discriminator/layer1/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer1/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer1/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer1/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer1/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer1/BatchNorm/moving_mean8model/discriminator/layer1/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer1/BatchNorm/moving_variance/read?model/discriminator/layer1/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer1/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer1/BatchNorm/moving_variance:model/discriminator/layer1/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer1/BatchNorm/batchnorm/addAdd?model/discriminator/layer1/BatchNorm/moments/normalize/variance4model/discriminator/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer1/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer1/BatchNorm/batchnorm/mulMul4model/discriminator/layer1/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer1/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer1/add2model/discriminator/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ň
4model/discriminator/layer1/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer1/BatchNorm/moments/normalize/mean2model/discriminator/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer1/BatchNorm/batchnorm/subSub.model/discriminator/layer1/BatchNorm/beta/read4model/discriminator/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer1/BatchNorm/batchnorm/add_1Add4model/discriminator/layer1/BatchNorm/batchnorm/mul_12model/discriminator/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
e
 model/discriminator/layer1/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer1/mulMul model/discriminator/layer1/mul/x4model/discriminator/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ž
"model/discriminator/layer1/MaximumMaximum4model/discriminator/layer1/BatchNorm/batchnorm/add_1model/discriminator/layer1/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

.model/discriminator/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer2/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer2/random_normal/mulMul=model/discriminator/layer2/random_normal/RandomStandardNormal/model/discriminator/layer2/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer2/random_normalAdd,model/discriminator/layer2/random_normal/mul-model/discriminator/layer2/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer2/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *(
_output_shapes
:

)model/discriminator/layer2/weights/AssignAssign"model/discriminator/layer2/weights(model/discriminator/layer2/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer2/weights/readIdentity"model/discriminator/layer2/weights*5
_class+
)'loc:@model/discriminator/layer2/weights*
T0*(
_output_shapes
:

0model/discriminator/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer2/random_normal_1/mulMul?model/discriminator/layer2/random_normal_1/RandomStandardNormal1model/discriminator/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer2/random_normal_1Add.model/discriminator/layer2/random_normal_1/mul/model/discriminator/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer2/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:

&model/discriminator/layer2/bias/AssignAssignmodel/discriminator/layer2/bias*model/discriminator/layer2/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer2/bias/readIdentitymodel/discriminator/layer2/bias*2
_class(
&$loc:@model/discriminator/layer2/bias*
T0*#
_output_shapes
:

!model/discriminator/layer2/Conv2DConv2D"model/discriminator/layer1/Maximum'model/discriminator/layer2/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer2/addAdd!model/discriminator/layer2/Conv2D$model/discriminator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer2/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer2/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer2/BatchNorm/beta/AssignAssign)model/discriminator/layer2/BatchNorm/beta;model/discriminator/layer2/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer2/BatchNorm/beta/readIdentity)model/discriminator/layer2/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer2/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer2/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer2/BatchNorm/gamma/AssignAssign*model/discriminator/layer2/BatchNorm/gamma<model/discriminator/layer2/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer2/BatchNorm/gamma/readIdentity*model/discriminator/layer2/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer2/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer2/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer2/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer2/BatchNorm/moving_meanBmodel/discriminator/layer2/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer2/BatchNorm/moving_mean/readIdentity0model/discriminator/layer2/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer2/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer2/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer2/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer2/BatchNorm/moving_varianceFmodel/discriminator/layer2/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer2/BatchNorm/moving_variance/readIdentity4model/discriminator/layer2/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer2/BatchNorm/moments/MeanMeanmodel/discriminator/layer2/addCmodel/discriminator/layer2/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ž
9model/discriminator/layer2/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer2/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ă
Fmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer2/add9model/discriminator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer2/add9model/discriminator/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
°
[model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
}
2model/discriminator/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer2/BatchNorm/moments/ReshapeReshape9model/discriminator/layer2/BatchNorm/moments/StopGradient2model/discriminator/layer2/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Đ
>model/discriminator/layer2/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer2/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer2/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer2/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer2/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer2/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer2/BatchNorm/moments/normalize/Mul=model/discriminator/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer2/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer2/BatchNorm/moving_mean/read;model/discriminator/layer2/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer2/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer2/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer2/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer2/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer2/BatchNorm/moving_mean8model/discriminator/layer2/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer2/BatchNorm/moving_variance/read?model/discriminator/layer2/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer2/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer2/BatchNorm/moving_variance:model/discriminator/layer2/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer2/BatchNorm/batchnorm/addAdd?model/discriminator/layer2/BatchNorm/moments/normalize/variance4model/discriminator/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer2/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer2/BatchNorm/batchnorm/mulMul4model/discriminator/layer2/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer2/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer2/add2model/discriminator/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer2/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer2/BatchNorm/moments/normalize/mean2model/discriminator/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer2/BatchNorm/batchnorm/subSub.model/discriminator/layer2/BatchNorm/beta/read4model/discriminator/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer2/BatchNorm/batchnorm/add_1Add4model/discriminator/layer2/BatchNorm/batchnorm/mul_12model/discriminator/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer2/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer2/mulMul model/discriminator/layer2/mul/x4model/discriminator/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer2/MaximumMaximum4model/discriminator/layer2/BatchNorm/batchnorm/add_1model/discriminator/layer2/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

.model/discriminator/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer3/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer3/random_normal/mulMul=model/discriminator/layer3/random_normal/RandomStandardNormal/model/discriminator/layer3/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer3/random_normalAdd,model/discriminator/layer3/random_normal/mul-model/discriminator/layer3/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer3/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *(
_output_shapes
:

)model/discriminator/layer3/weights/AssignAssign"model/discriminator/layer3/weights(model/discriminator/layer3/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer3/weights/readIdentity"model/discriminator/layer3/weights*5
_class+
)'loc:@model/discriminator/layer3/weights*
T0*(
_output_shapes
:

0model/discriminator/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer3/random_normal_1/mulMul?model/discriminator/layer3/random_normal_1/RandomStandardNormal1model/discriminator/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer3/random_normal_1Add.model/discriminator/layer3/random_normal_1/mul/model/discriminator/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer3/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:

&model/discriminator/layer3/bias/AssignAssignmodel/discriminator/layer3/bias*model/discriminator/layer3/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer3/bias/readIdentitymodel/discriminator/layer3/bias*2
_class(
&$loc:@model/discriminator/layer3/bias*
T0*#
_output_shapes
:

!model/discriminator/layer3/Conv2DConv2D"model/discriminator/layer2/Maximum'model/discriminator/layer3/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer3/addAdd!model/discriminator/layer3/Conv2D$model/discriminator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer3/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer3/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer3/BatchNorm/beta/AssignAssign)model/discriminator/layer3/BatchNorm/beta;model/discriminator/layer3/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer3/BatchNorm/beta/readIdentity)model/discriminator/layer3/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer3/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer3/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer3/BatchNorm/gamma/AssignAssign*model/discriminator/layer3/BatchNorm/gamma<model/discriminator/layer3/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer3/BatchNorm/gamma/readIdentity*model/discriminator/layer3/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer3/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer3/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer3/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer3/BatchNorm/moving_meanBmodel/discriminator/layer3/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer3/BatchNorm/moving_mean/readIdentity0model/discriminator/layer3/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer3/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer3/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer3/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer3/BatchNorm/moving_varianceFmodel/discriminator/layer3/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer3/BatchNorm/moving_variance/readIdentity4model/discriminator/layer3/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer3/BatchNorm/moments/MeanMeanmodel/discriminator/layer3/addCmodel/discriminator/layer3/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ž
9model/discriminator/layer3/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer3/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ă
Fmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer3/add9model/discriminator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer3/add9model/discriminator/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
°
[model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
}
2model/discriminator/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer3/BatchNorm/moments/ReshapeReshape9model/discriminator/layer3/BatchNorm/moments/StopGradient2model/discriminator/layer3/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Đ
>model/discriminator/layer3/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer3/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer3/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer3/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer3/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer3/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer3/BatchNorm/moments/normalize/Mul=model/discriminator/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer3/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer3/BatchNorm/moving_mean/read;model/discriminator/layer3/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer3/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer3/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer3/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer3/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer3/BatchNorm/moving_mean8model/discriminator/layer3/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer3/BatchNorm/moving_variance/read?model/discriminator/layer3/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer3/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer3/BatchNorm/moving_variance:model/discriminator/layer3/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer3/BatchNorm/batchnorm/addAdd?model/discriminator/layer3/BatchNorm/moments/normalize/variance4model/discriminator/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer3/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer3/BatchNorm/batchnorm/mulMul4model/discriminator/layer3/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer3/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer3/add2model/discriminator/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer3/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer3/BatchNorm/moments/normalize/mean2model/discriminator/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer3/BatchNorm/batchnorm/subSub.model/discriminator/layer3/BatchNorm/beta/read4model/discriminator/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer3/BatchNorm/batchnorm/add_1Add4model/discriminator/layer3/BatchNorm/batchnorm/mul_12model/discriminator/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer3/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer3/mulMul model/discriminator/layer3/mul/x4model/discriminator/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer3/MaximumMaximum4model/discriminator/layer3/BatchNorm/batchnorm/add_1model/discriminator/layer3/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

.model/discriminator/layer4/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
r
-model/discriminator/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ţ
=model/discriminator/layer4/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ö
,model/discriminator/layer4/random_normal/mulMul=model/discriminator/layer4/random_normal/RandomStandardNormal/model/discriminator/layer4/random_normal/stddev*
T0*(
_output_shapes
:
ż
(model/discriminator/layer4/random_normalAdd,model/discriminator/layer4/random_normal/mul-model/discriminator/layer4/random_normal/mean*
T0*(
_output_shapes
:
Ş
"model/discriminator/layer4/weights
VariableV2*
dtype0*
shape:*
shared_name *
	container *(
_output_shapes
:

)model/discriminator/layer4/weights/AssignAssign"model/discriminator/layer4/weights(model/discriminator/layer4/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:
Á
'model/discriminator/layer4/weights/readIdentity"model/discriminator/layer4/weights*5
_class+
)'loc:@model/discriminator/layer4/weights*
T0*(
_output_shapes
:

0model/discriminator/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
t
/model/discriminator/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ý
?model/discriminator/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
×
.model/discriminator/layer4/random_normal_1/mulMul?model/discriminator/layer4/random_normal_1/RandomStandardNormal1model/discriminator/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:
Ŕ
*model/discriminator/layer4/random_normal_1Add.model/discriminator/layer4/random_normal_1/mul/model/discriminator/layer4/random_normal_1/mean*
T0*#
_output_shapes
:

model/discriminator/layer4/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *#
_output_shapes
:

&model/discriminator/layer4/bias/AssignAssignmodel/discriminator/layer4/bias*model/discriminator/layer4/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:
ł
$model/discriminator/layer4/bias/readIdentitymodel/discriminator/layer4/bias*2
_class(
&$loc:@model/discriminator/layer4/bias*
T0*#
_output_shapes
:

!model/discriminator/layer4/Conv2DConv2D"model/discriminator/layer3/Maximum'model/discriminator/layer4/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Š
model/discriminator/layer4/addAdd!model/discriminator/layer4/Conv2D$model/discriminator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
;model/discriminator/layer4/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
valueB*    *
_output_shapes	
:
Ő
)model/discriminator/layer4/BatchNorm/beta
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
shared_name 
Ż
0model/discriminator/layer4/BatchNorm/beta/AssignAssign)model/discriminator/layer4/BatchNorm/beta;model/discriminator/layer4/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:
É
.model/discriminator/layer4/BatchNorm/beta/readIdentity)model/discriminator/layer4/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
T0*
_output_shapes	
:
Ę
<model/discriminator/layer4/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
valueB*  ?*
_output_shapes	
:
×
*model/discriminator/layer4/BatchNorm/gamma
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
shared_name 
ł
1model/discriminator/layer4/BatchNorm/gamma/AssignAssign*model/discriminator/layer4/BatchNorm/gamma<model/discriminator/layer4/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:
Ě
/model/discriminator/layer4/BatchNorm/gamma/readIdentity*model/discriminator/layer4/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
T0*
_output_shapes	
:
Ö
Bmodel/discriminator/layer4/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB*    *
_output_shapes	
:
ă
0model/discriminator/layer4/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
shared_name 
Ë
7model/discriminator/layer4/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer4/BatchNorm/moving_meanBmodel/discriminator/layer4/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:
Ţ
5model/discriminator/layer4/BatchNorm/moving_mean/readIdentity0model/discriminator/layer4/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ţ
Fmodel/discriminator/layer4/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes	
:
ë
4model/discriminator/layer4/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
shared_name 
Ű
;model/discriminator/layer4/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer4/BatchNorm/moving_varianceFmodel/discriminator/layer4/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:
ę
9model/discriminator/layer4/BatchNorm/moving_variance/readIdentity4model/discriminator/layer4/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:

Cmodel/discriminator/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
í
1model/discriminator/layer4/BatchNorm/moments/MeanMeanmodel/discriminator/layer4/addCmodel/discriminator/layer4/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
Ž
9model/discriminator/layer4/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ś
Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/ShapeShapemodel/discriminator/layer4/add*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
Ś
Qmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Â
Imodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ă
Fmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SubSubmodel/discriminator/layer4/add9model/discriminator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Tmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencemodel/discriminator/layer4/add9model/discriminator/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
\model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ş
Jmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
°
[model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ć
Imodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
}
2model/discriminator/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
â
4model/discriminator/layer4/BatchNorm/moments/ReshapeReshape9model/discriminator/layer4/BatchNorm/moments/StopGradient2model/discriminator/layer4/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Đ
>model/discriminator/layer4/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ü
Cmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ă
;model/discriminator/layer4/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ň
:model/discriminator/layer4/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer4/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
˛
=model/discriminator/layer4/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
ç
?model/discriminator/layer4/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer4/BatchNorm/moments/normalize/Mul=model/discriminator/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ä
:model/discriminator/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer4/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer4/BatchNorm/moving_mean/read;model/discriminator/layer4/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
 
8model/discriminator/layer4/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer4/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer4/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ť
4model/discriminator/layer4/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer4/BatchNorm/moving_mean8model/discriminator/layer4/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ę
<model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ź
:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer4/BatchNorm/moving_variance/read?model/discriminator/layer4/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
Ş
:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ˇ
6model/discriminator/layer4/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer4/BatchNorm/moving_variance:model/discriminator/layer4/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
y
4model/discriminator/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ö
2model/discriminator/layer4/BatchNorm/batchnorm/addAdd?model/discriminator/layer4/BatchNorm/moments/normalize/variance4model/discriminator/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

4model/discriminator/layer4/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ć
2model/discriminator/layer4/BatchNorm/batchnorm/mulMul4model/discriminator/layer4/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Ę
4model/discriminator/layer4/BatchNorm/batchnorm/mul_1Mulmodel/discriminator/layer4/add2model/discriminator/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
4model/discriminator/layer4/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer4/BatchNorm/moments/normalize/mean2model/discriminator/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
Ĺ
2model/discriminator/layer4/BatchNorm/batchnorm/subSub.model/discriminator/layer4/BatchNorm/beta/read4model/discriminator/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ŕ
4model/discriminator/layer4/BatchNorm/batchnorm/add_1Add4model/discriminator/layer4/BatchNorm/batchnorm/mul_12model/discriminator/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 model/discriminator/layer4/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
¸
model/discriminator/layer4/mulMul model/discriminator/layer4/mul/x4model/discriminator/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
"model/discriminator/layer4/MaximumMaximum4model/discriminator/layer4/BatchNorm/batchnorm/add_1model/discriminator/layer4/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

 model/discriminator/layer5/ShapeShape"model/discriminator/layer4/Maximum*
out_type0*
T0*
_output_shapes
:
x
.model/discriminator/layer5/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
z
0model/discriminator/layer5/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
z
0model/discriminator/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

(model/discriminator/layer5/strided_sliceStridedSlice model/discriminator/layer5/Shape.model/discriminator/layer5/strided_slice/stack0model/discriminator/layer5/strided_slice/stack_10model/discriminator/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
j
 model/discriminator/layer5/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ą
model/discriminator/layer5/ProdProd(model/discriminator/layer5/strided_slice model/discriminator/layer5/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
˘
(model/discriminator/layer5/Reshape/shapePackmodel/strided_slice_1model/discriminator/layer5/Prod*
_output_shapes
:*

axis *
T0*
N
Ä
"model/discriminator/layer5/ReshapeReshape"model/discriminator/layer4/Maximum(model/discriminator/layer5/Reshape/shape*
Tshape0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

.model/discriminator/layer5/random_normal/shapeConst*
dtype0*
valueB" @     *
_output_shapes
:
r
-model/discriminator/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
t
/model/discriminator/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ö
=model/discriminator/layer5/random_normal/RandomStandardNormalRandomStandardNormal.model/discriminator/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:

Î
,model/discriminator/layer5/random_normal/mulMul=model/discriminator/layer5/random_normal/RandomStandardNormal/model/discriminator/layer5/random_normal/stddev*
T0* 
_output_shapes
:

ˇ
(model/discriminator/layer5/random_normalAdd,model/discriminator/layer5/random_normal/mul-model/discriminator/layer5/random_normal/mean*
T0* 
_output_shapes
:


"model/discriminator/layer5/weights
VariableV2*
dtype0*
shape:
*
shared_name *
	container * 
_output_shapes
:


)model/discriminator/layer5/weights/AssignAssign"model/discriminator/layer5/weights(model/discriminator/layer5/random_normal*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer5/weights*
use_locking(*
T0* 
_output_shapes
:

š
'model/discriminator/layer5/weights/readIdentity"model/discriminator/layer5/weights*5
_class+
)'loc:@model/discriminator/layer5/weights*
T0* 
_output_shapes
:

z
0model/discriminator/layer5/random_normal_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
t
/model/discriminator/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ô
?model/discriminator/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal0model/discriminator/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
Î
.model/discriminator/layer5/random_normal_1/mulMul?model/discriminator/layer5/random_normal_1/RandomStandardNormal1model/discriminator/layer5/random_normal_1/stddev*
T0*
_output_shapes
:
ˇ
*model/discriminator/layer5/random_normal_1Add.model/discriminator/layer5/random_normal_1/mul/model/discriminator/layer5/random_normal_1/mean*
T0*
_output_shapes
:

model/discriminator/layer5/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
˙
&model/discriminator/layer5/bias/AssignAssignmodel/discriminator/layer5/bias*model/discriminator/layer5/random_normal_1*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer5/bias*
use_locking(*
T0*
_output_shapes
:
Ş
$model/discriminator/layer5/bias/readIdentitymodel/discriminator/layer5/bias*2
_class(
&$loc:@model/discriminator/layer5/bias*
T0*
_output_shapes
:
Đ
!model/discriminator/layer5/MatMulMatMul"model/discriminator/layer5/Reshape'model/discriminator/layer5/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
model/discriminator/layer5/addAdd!model/discriminator/layer5/MatMul$model/discriminator/layer5/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
;model/discriminator/layer5/BatchNorm/beta/Initializer/ConstConst*
dtype0*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
valueB*    *
_output_shapes
:
Ó
)model/discriminator/layer5/BatchNorm/beta
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
shared_name 
Ž
0model/discriminator/layer5/BatchNorm/beta/AssignAssign)model/discriminator/layer5/BatchNorm/beta;model/discriminator/layer5/BatchNorm/beta/Initializer/Const*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:
Č
.model/discriminator/layer5/BatchNorm/beta/readIdentity)model/discriminator/layer5/BatchNorm/beta*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
T0*
_output_shapes
:
Č
<model/discriminator/layer5/BatchNorm/gamma/Initializer/ConstConst*
dtype0*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
valueB*  ?*
_output_shapes
:
Ő
*model/discriminator/layer5/BatchNorm/gamma
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
shared_name 
˛
1model/discriminator/layer5/BatchNorm/gamma/AssignAssign*model/discriminator/layer5/BatchNorm/gamma<model/discriminator/layer5/BatchNorm/gamma/Initializer/Const*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:
Ë
/model/discriminator/layer5/BatchNorm/gamma/readIdentity*model/discriminator/layer5/BatchNorm/gamma*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
T0*
_output_shapes
:
Ô
Bmodel/discriminator/layer5/BatchNorm/moving_mean/Initializer/ConstConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB*    *
_output_shapes
:
á
0model/discriminator/layer5/BatchNorm/moving_mean
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
shared_name 
Ę
7model/discriminator/layer5/BatchNorm/moving_mean/AssignAssign0model/discriminator/layer5/BatchNorm/moving_meanBmodel/discriminator/layer5/BatchNorm/moving_mean/Initializer/Const*
validate_shape(*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:
Ý
5model/discriminator/layer5/BatchNorm/moving_mean/readIdentity0model/discriminator/layer5/BatchNorm/moving_mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ü
Fmodel/discriminator/layer5/BatchNorm/moving_variance/Initializer/ConstConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB*  ?*
_output_shapes
:
é
4model/discriminator/layer5/BatchNorm/moving_variance
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
shared_name 
Ú
;model/discriminator/layer5/BatchNorm/moving_variance/AssignAssign4model/discriminator/layer5/BatchNorm/moving_varianceFmodel/discriminator/layer5/BatchNorm/moving_variance/Initializer/Const*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:
é
9model/discriminator/layer5/BatchNorm/moving_variance/readIdentity4model/discriminator/layer5/BatchNorm/moving_variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:

Cmodel/discriminator/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ç
1model/discriminator/layer5/BatchNorm/moments/MeanMean!model/discriminator/layer5/MatMulCmodel/discriminator/layer5/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*
_output_shapes

:
Ľ
9model/discriminator/layer5/BatchNorm/moments/StopGradientStopGradient1model/discriminator/layer5/BatchNorm/moments/Mean*
T0*
_output_shapes

:
Š
Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/ShapeShape!model/discriminator/layer5/MatMul*
out_type0*
T0*
_output_shapes
:
Í
Gmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/CastCastHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:

Qmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*
valueB: *
_output_shapes
:
Â
Imodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherGmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/CastQmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Ł
Hmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/countProdImodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/GatherHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ý
Fmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SubSub!model/discriminator/layer5/MatMul9model/discriminator/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Tmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference!model/discriminator/layer5/MatMul9model/discriminator/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
\model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
š
Jmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumFmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/Sub\model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ľ
[model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
Ĺ
Imodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumTmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifference[model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
|
2model/discriminator/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
á
4model/discriminator/layer5/BatchNorm/moments/ReshapeReshape9model/discriminator/layer5/BatchNorm/moments/StopGradient2model/discriminator/layer5/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes
:
Đ
>model/discriminator/layer5/BatchNorm/moments/normalize/divisor
ReciprocalHmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/countK^model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ssJ^model/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
ű
Cmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_meanMulJmodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/mean_ss>model/discriminator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
â
;model/discriminator/layer5/BatchNorm/moments/normalize/meanAddCmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_mean4model/discriminator/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
ń
:model/discriminator/layer5/BatchNorm/moments/normalize/MulMulImodel/discriminator/layer5/BatchNorm/moments/sufficient_statistics/var_ss>model/discriminator/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
ą
=model/discriminator/layer5/BatchNorm/moments/normalize/SquareSquareCmodel/discriminator/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
ć
?model/discriminator/layer5/BatchNorm/moments/normalize/varianceSub:model/discriminator/layer5/BatchNorm/moments/normalize/Mul=model/discriminator/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
Ä
:model/discriminator/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 

8model/discriminator/layer5/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer5/BatchNorm/moving_mean/read;model/discriminator/layer5/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:

8model/discriminator/layer5/BatchNorm/AssignMovingAvg/mulMul8model/discriminator/layer5/BatchNorm/AssignMovingAvg/sub:model/discriminator/layer5/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ş
4model/discriminator/layer5/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer5/BatchNorm/moving_mean8model/discriminator/layer5/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Ę
<model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ť
:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer5/BatchNorm/moving_variance/read?model/discriminator/layer5/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Š
:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/mulMul:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/sub<model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
ś
6model/discriminator/layer5/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer5/BatchNorm/moving_variance:model/discriminator/layer5/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
y
4model/discriminator/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ő
2model/discriminator/layer5/BatchNorm/batchnorm/addAdd?model/discriminator/layer5/BatchNorm/moments/normalize/variance4model/discriminator/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

4model/discriminator/layer5/BatchNorm/batchnorm/RsqrtRsqrt2model/discriminator/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
Ĺ
2model/discriminator/layer5/BatchNorm/batchnorm/mulMul4model/discriminator/layer5/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
Ä
4model/discriminator/layer5/BatchNorm/batchnorm/mul_1Mul!model/discriminator/layer5/MatMul2model/discriminator/layer5/BatchNorm/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ń
4model/discriminator/layer5/BatchNorm/batchnorm/mul_2Mul;model/discriminator/layer5/BatchNorm/moments/normalize/mean2model/discriminator/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
Ä
2model/discriminator/layer5/BatchNorm/batchnorm/subSub.model/discriminator/layer5/BatchNorm/beta/read4model/discriminator/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
×
4model/discriminator/layer5/BatchNorm/batchnorm/add_1Add4model/discriminator/layer5/BatchNorm/batchnorm/mul_12model/discriminator/layer5/BatchNorm/batchnorm/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

model/discriminator/layer5/TanhTanh4model/discriminator/layer5/BatchNorm/batchnorm/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
model/Shape_2Shapemodel/generator/layer5/Relu*
out_type0*
T0*
_output_shapes
:
e
model/strided_slice_2/stackConst*
dtype0*
valueB: *
_output_shapes
:
g
model/strided_slice_2/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
g
model/strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ą
model/strided_slice_2StridedSlicemodel/Shape_2model/strided_slice_2/stackmodel/strided_slice_2/stack_1model/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask

0model/discriminator_1/layer1/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer1/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer1/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
?model/discriminator_1/layer1/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer1/random_normal/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:
Ű
.model/discriminator_1/layer1/random_normal/mulMul?model/discriminator_1/layer1/random_normal/RandomStandardNormal1model/discriminator_1/layer1/random_normal/stddev*
T0*'
_output_shapes
:
Ä
*model/discriminator_1/layer1/random_normalAdd.model/discriminator_1/layer1/random_normal/mul/model/discriminator_1/layer1/random_normal/mean*
T0*'
_output_shapes
:

2model/discriminator_1/layer1/random_normal_1/shapeConst*
dtype0*!
valueB"           *
_output_shapes
:
v
1model/discriminator_1/layer1/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer1/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer1/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer1/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:  
Ý
0model/discriminator_1/layer1/random_normal_1/mulMulAmodel/discriminator_1/layer1/random_normal_1/RandomStandardNormal3model/discriminator_1/layer1/random_normal_1/stddev*
T0*#
_output_shapes
:  
Ć
,model/discriminator_1/layer1/random_normal_1Add0model/discriminator_1/layer1/random_normal_1/mul1model/discriminator_1/layer1/random_normal_1/mean*
T0*#
_output_shapes
:  

#model/discriminator_1/layer1/Conv2DConv2Dmodel/generator/layer5/Relu'model/discriminator/layer1/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer1/addAdd#model/discriminator_1/layer1/Conv2D$model/discriminator/layer1/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

Emodel/discriminator_1/layer1/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer1/BatchNorm/moments/MeanMean model/discriminator_1/layer1/addEmodel/discriminator_1/layer1/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
˛
;model/discriminator_1/layer1/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer1/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer1/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
é
Hmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer1/add;model/discriminator_1/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

Vmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer1/add;model/discriminator_1/layer1/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ł
^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
˛
]model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:

4model/discriminator_1/layer1/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer1/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer1/BatchNorm/moments/StopGradient4model/discriminator_1/layer1/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ř
@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer1/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer1/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer1/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer1/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer1/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer1/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer1/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer1/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer1/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer1/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer1/BatchNorm/moving_mean/read=model/discriminator_1/layer1/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer1/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer1/BatchNorm/moving_mean:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer1/BatchNorm/moving_variance/readAmodel/discriminator_1/layer1/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer1/BatchNorm/moving_variance<model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer1/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer1/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer1/BatchNorm/moments/normalize/variance6model/discriminator_1/layer1/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer1/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer1/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer1/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer1/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer1/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer1/add4model/discriminator_1/layer1/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ř
6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer1/BatchNorm/moments/normalize/mean4model/discriminator_1/layer1/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer1/BatchNorm/batchnorm/subSub.model/discriminator/layer1/BatchNorm/beta/read6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer1/BatchNorm/batchnorm/mul_14model/discriminator_1/layer1/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
g
"model/discriminator_1/layer1/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer1/mulMul"model/discriminator_1/layer1/mul/x6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ä
$model/discriminator_1/layer1/MaximumMaximum6model/discriminator_1/layer1/BatchNorm/batchnorm/add_1 model/discriminator_1/layer1/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙  

0model/discriminator_1/layer2/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer2/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer2/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer2/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer2/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer2/random_normal/mulMul?model/discriminator_1/layer2/random_normal/RandomStandardNormal1model/discriminator_1/layer2/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer2/random_normalAdd.model/discriminator_1/layer2/random_normal/mul/model/discriminator_1/layer2/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer2/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer2/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer2/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer2/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer2/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer2/random_normal_1/mulMulAmodel/discriminator_1/layer2/random_normal_1/RandomStandardNormal3model/discriminator_1/layer2/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer2/random_normal_1Add0model/discriminator_1/layer2/random_normal_1/mul1model/discriminator_1/layer2/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer2/Conv2DConv2D$model/discriminator_1/layer1/Maximum'model/discriminator/layer2/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer2/addAdd#model/discriminator_1/layer2/Conv2D$model/discriminator/layer2/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer2/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer2/BatchNorm/moments/MeanMean model/discriminator_1/layer2/addEmodel/discriminator_1/layer2/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
˛
;model/discriminator_1/layer2/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer2/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer2/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
é
Hmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer2/add;model/discriminator_1/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer2/add;model/discriminator_1/layer2/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
˛
]model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:

4model/discriminator_1/layer2/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer2/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer2/BatchNorm/moments/StopGradient4model/discriminator_1/layer2/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ř
@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer2/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer2/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer2/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer2/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer2/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer2/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer2/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer2/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer2/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer2/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer2/BatchNorm/moving_mean/read=model/discriminator_1/layer2/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer2/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer2/BatchNorm/moving_mean:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer2/BatchNorm/moving_variance/readAmodel/discriminator_1/layer2/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer2/BatchNorm/moving_variance<model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer2/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer2/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer2/BatchNorm/moments/normalize/variance6model/discriminator_1/layer2/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer2/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer2/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer2/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer2/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer2/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer2/add4model/discriminator_1/layer2/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer2/BatchNorm/moments/normalize/mean4model/discriminator_1/layer2/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer2/BatchNorm/batchnorm/subSub.model/discriminator/layer2/BatchNorm/beta/read6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer2/BatchNorm/batchnorm/mul_14model/discriminator_1/layer2/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer2/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer2/mulMul"model/discriminator_1/layer2/mul/x6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer2/MaximumMaximum6model/discriminator_1/layer2/BatchNorm/batchnorm/add_1 model/discriminator_1/layer2/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

0model/discriminator_1/layer3/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer3/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer3/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer3/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer3/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer3/random_normal/mulMul?model/discriminator_1/layer3/random_normal/RandomStandardNormal1model/discriminator_1/layer3/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer3/random_normalAdd.model/discriminator_1/layer3/random_normal/mul/model/discriminator_1/layer3/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer3/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer3/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer3/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer3/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer3/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer3/random_normal_1/mulMulAmodel/discriminator_1/layer3/random_normal_1/RandomStandardNormal3model/discriminator_1/layer3/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer3/random_normal_1Add0model/discriminator_1/layer3/random_normal_1/mul1model/discriminator_1/layer3/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer3/Conv2DConv2D$model/discriminator_1/layer2/Maximum'model/discriminator/layer3/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer3/addAdd#model/discriminator_1/layer3/Conv2D$model/discriminator/layer3/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer3/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer3/BatchNorm/moments/MeanMean model/discriminator_1/layer3/addEmodel/discriminator_1/layer3/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
˛
;model/discriminator_1/layer3/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer3/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer3/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
é
Hmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer3/add;model/discriminator_1/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer3/add;model/discriminator_1/layer3/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
˛
]model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:

4model/discriminator_1/layer3/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer3/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer3/BatchNorm/moments/StopGradient4model/discriminator_1/layer3/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ř
@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer3/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer3/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer3/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer3/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer3/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer3/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer3/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer3/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer3/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer3/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer3/BatchNorm/moving_mean/read=model/discriminator_1/layer3/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer3/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer3/BatchNorm/moving_mean:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer3/BatchNorm/moving_variance/readAmodel/discriminator_1/layer3/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer3/BatchNorm/moving_variance<model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer3/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer3/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer3/BatchNorm/moments/normalize/variance6model/discriminator_1/layer3/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer3/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer3/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer3/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer3/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer3/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer3/add4model/discriminator_1/layer3/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer3/BatchNorm/moments/normalize/mean4model/discriminator_1/layer3/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer3/BatchNorm/batchnorm/subSub.model/discriminator/layer3/BatchNorm/beta/read6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer3/BatchNorm/batchnorm/mul_14model/discriminator_1/layer3/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer3/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer3/mulMul"model/discriminator_1/layer3/mul/x6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer3/MaximumMaximum6model/discriminator_1/layer3/BatchNorm/batchnorm/add_1 model/discriminator_1/layer3/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

0model/discriminator_1/layer4/random_normal/shapeConst*
dtype0*%
valueB"            *
_output_shapes
:
t
/model/discriminator_1/layer4/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer4/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
â
?model/discriminator_1/layer4/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer4/random_normal/shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:
Ü
.model/discriminator_1/layer4/random_normal/mulMul?model/discriminator_1/layer4/random_normal/RandomStandardNormal1model/discriminator_1/layer4/random_normal/stddev*
T0*(
_output_shapes
:
Ĺ
*model/discriminator_1/layer4/random_normalAdd.model/discriminator_1/layer4/random_normal/mul/model/discriminator_1/layer4/random_normal/mean*
T0*(
_output_shapes
:

2model/discriminator_1/layer4/random_normal_1/shapeConst*
dtype0*!
valueB"         *
_output_shapes
:
v
1model/discriminator_1/layer4/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer4/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
á
Amodel/discriminator_1/layer4/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer4/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*#
_output_shapes
:
Ý
0model/discriminator_1/layer4/random_normal_1/mulMulAmodel/discriminator_1/layer4/random_normal_1/RandomStandardNormal3model/discriminator_1/layer4/random_normal_1/stddev*
T0*#
_output_shapes
:
Ć
,model/discriminator_1/layer4/random_normal_1Add0model/discriminator_1/layer4/random_normal_1/mul1model/discriminator_1/layer4/random_normal_1/mean*
T0*#
_output_shapes
:

#model/discriminator_1/layer4/Conv2DConv2D$model/discriminator_1/layer3/Maximum'model/discriminator/layer4/weights/read*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
­
 model/discriminator_1/layer4/addAdd#model/discriminator_1/layer4/Conv2D$model/discriminator/layer4/bias/read*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer4/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
ó
3model/discriminator_1/layer4/BatchNorm/moments/MeanMean model/discriminator_1/layer4/addEmodel/discriminator_1/layer4/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*'
_output_shapes
:
˛
;model/discriminator_1/layer4/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer4/BatchNorm/moments/Mean*
T0*'
_output_shapes
:
Ş
Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/ShapeShape model/discriminator_1/layer4/add*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:
¨
Smodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Č
Kmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
é
Hmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SubSub model/discriminator_1/layer4/add;model/discriminator_1/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference model/discriminator_1/layer4/add;model/discriminator_1/layer4/BatchNorm/moments/StopGradient*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ŕ
Lmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:
˛
]model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*!
valueB"          *
_output_shapes
:
Ě
Kmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes	
:

4model/discriminator_1/layer4/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
č
6model/discriminator_1/layer4/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer4/BatchNorm/moments/StopGradient4model/discriminator_1/layer4/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes	
:
Ř
@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
é
=model/discriminator_1/layer4/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer4/BatchNorm/moments/Reshape*
T0*
_output_shapes	
:
ř
<model/discriminator_1/layer4/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer4/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer4/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes	
:
ś
?model/discriminator_1/layer4/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer4/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes	
:
í
Amodel/discriminator_1/layer4/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer4/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer4/BatchNorm/moments/normalize/Square*
T0*
_output_shapes	
:
Ć
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
˘
:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer4/BatchNorm/moving_mean/read=model/discriminator_1/layer4/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ś
:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
T0*
_output_shapes	
:
Ż
6model/discriminator_1/layer4/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer4/BatchNorm/moving_mean:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes	
:
Ě
>model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
°
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer4/BatchNorm/moving_variance/readAmodel/discriminator_1/layer4/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
°
<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
T0*
_output_shapes	
:
ť
8model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer4/BatchNorm/moving_variance<model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes	
:
{
6model/discriminator_1/layer4/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ü
4model/discriminator_1/layer4/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer4/BatchNorm/moments/normalize/variance6model/discriminator_1/layer4/BatchNorm/batchnorm/add/y*
T0*
_output_shapes	
:

6model/discriminator_1/layer4/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer4/BatchNorm/batchnorm/add*
T0*
_output_shapes	
:
Ę
4model/discriminator_1/layer4/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer4/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer4/BatchNorm/gamma/read*
T0*
_output_shapes	
:
Đ
6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_1Mul model/discriminator_1/layer4/add4model/discriminator_1/layer4/BatchNorm/batchnorm/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer4/BatchNorm/moments/normalize/mean4model/discriminator_1/layer4/BatchNorm/batchnorm/mul*
T0*
_output_shapes	
:
É
4model/discriminator_1/layer4/BatchNorm/batchnorm/subSub.model/discriminator/layer4/BatchNorm/beta/read6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes	
:
ć
6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer4/BatchNorm/batchnorm/mul_14model/discriminator_1/layer4/BatchNorm/batchnorm/sub*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
"model/discriminator_1/layer4/mul/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
ž
 model/discriminator_1/layer4/mulMul"model/discriminator_1/layer4/mul/x6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
$model/discriminator_1/layer4/MaximumMaximum6model/discriminator_1/layer4/BatchNorm/batchnorm/add_1 model/discriminator_1/layer4/mul*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙

"model/discriminator_1/layer5/ShapeShape$model/discriminator_1/layer4/Maximum*
out_type0*
T0*
_output_shapes
:
z
0model/discriminator_1/layer5/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:
|
2model/discriminator_1/layer5/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
|
2model/discriminator_1/layer5/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

*model/discriminator_1/layer5/strided_sliceStridedSlice"model/discriminator_1/layer5/Shape0model/discriminator_1/layer5/strided_slice/stack2model/discriminator_1/layer5/strided_slice/stack_12model/discriminator_1/layer5/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
l
"model/discriminator_1/layer5/ConstConst*
dtype0*
valueB: *
_output_shapes
:
ˇ
!model/discriminator_1/layer5/ProdProd*model/discriminator_1/layer5/strided_slice"model/discriminator_1/layer5/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Ś
*model/discriminator_1/layer5/Reshape/shapePackmodel/strided_slice_2!model/discriminator_1/layer5/Prod*
_output_shapes
:*

axis *
T0*
N
Ę
$model/discriminator_1/layer5/ReshapeReshape$model/discriminator_1/layer4/Maximum*model/discriminator_1/layer5/Reshape/shape*
Tshape0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

0model/discriminator_1/layer5/random_normal/shapeConst*
dtype0*
valueB" @     *
_output_shapes
:
t
/model/discriminator_1/layer5/random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
1model/discriminator_1/layer5/random_normal/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ú
?model/discriminator_1/layer5/random_normal/RandomStandardNormalRandomStandardNormal0model/discriminator_1/layer5/random_normal/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:

Ô
.model/discriminator_1/layer5/random_normal/mulMul?model/discriminator_1/layer5/random_normal/RandomStandardNormal1model/discriminator_1/layer5/random_normal/stddev*
T0* 
_output_shapes
:

˝
*model/discriminator_1/layer5/random_normalAdd.model/discriminator_1/layer5/random_normal/mul/model/discriminator_1/layer5/random_normal/mean*
T0* 
_output_shapes
:

|
2model/discriminator_1/layer5/random_normal_1/shapeConst*
dtype0*
valueB:*
_output_shapes
:
v
1model/discriminator_1/layer5/random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
x
3model/discriminator_1/layer5/random_normal_1/stddevConst*
dtype0*
valueB
 *
×#<*
_output_shapes
: 
Ř
Amodel/discriminator_1/layer5/random_normal_1/RandomStandardNormalRandomStandardNormal2model/discriminator_1/layer5/random_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:
Ô
0model/discriminator_1/layer5/random_normal_1/mulMulAmodel/discriminator_1/layer5/random_normal_1/RandomStandardNormal3model/discriminator_1/layer5/random_normal_1/stddev*
T0*
_output_shapes
:
˝
,model/discriminator_1/layer5/random_normal_1Add0model/discriminator_1/layer5/random_normal_1/mul1model/discriminator_1/layer5/random_normal_1/mean*
T0*
_output_shapes
:
Ô
#model/discriminator_1/layer5/MatMulMatMul$model/discriminator_1/layer5/Reshape'model/discriminator/layer5/weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
 model/discriminator_1/layer5/addAdd#model/discriminator_1/layer5/MatMul$model/discriminator/layer5/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Emodel/discriminator_1/layer5/BatchNorm/moments/Mean/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
í
3model/discriminator_1/layer5/BatchNorm/moments/MeanMean#model/discriminator_1/layer5/MatMulEmodel/discriminator_1/layer5/BatchNorm/moments/Mean/reduction_indices*

Tidx0*
T0*
	keep_dims(*
_output_shapes

:
Š
;model/discriminator_1/layer5/BatchNorm/moments/StopGradientStopGradient3model/discriminator_1/layer5/BatchNorm/moments/Mean*
T0*
_output_shapes

:
­
Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/ShapeShape#model/discriminator_1/layer5/MatMul*
out_type0*
T0*
_output_shapes
:
Ń
Imodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/CastCastJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Shape*

DstT0*

SrcT0*
_output_shapes
:

Smodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Gather/indicesConst*
dtype0*
valueB: *
_output_shapes
:
Č
Kmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/GatherGatherImodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/CastSmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Gather/indices*
validate_indices(*
Tparams0*
Tindices0*
_output_shapes
:

Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Š
Jmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/countProdKmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/GatherJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
ă
Hmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SubSub#model/discriminator_1/layer5/MatMul;model/discriminator_1/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
Vmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifferenceSquaredDifference#model/discriminator_1/layer5/MatMul;model/discriminator_1/layer5/BatchNorm/moments/StopGradient*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
ż
Lmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ssSumHmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/Sub^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
§
]model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
Ë
Kmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ssSumVmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/SquaredDifference]model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
~
4model/discriminator_1/layer5/BatchNorm/moments/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
ç
6model/discriminator_1/layer5/BatchNorm/moments/ReshapeReshape;model/discriminator_1/layer5/BatchNorm/moments/StopGradient4model/discriminator_1/layer5/BatchNorm/moments/Shape*
Tshape0*
T0*
_output_shapes
:
Ř
@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor
ReciprocalJmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/countM^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ssL^model/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 

Emodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_meanMulLmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/mean_ss@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
č
=model/discriminator_1/layer5/BatchNorm/moments/normalize/meanAddEmodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_mean6model/discriminator_1/layer5/BatchNorm/moments/Reshape*
T0*
_output_shapes
:
÷
<model/discriminator_1/layer5/BatchNorm/moments/normalize/MulMulKmodel/discriminator_1/layer5/BatchNorm/moments/sufficient_statistics/var_ss@model/discriminator_1/layer5/BatchNorm/moments/normalize/divisor*
T0*
_output_shapes
:
ľ
?model/discriminator_1/layer5/BatchNorm/moments/normalize/SquareSquareEmodel/discriminator_1/layer5/BatchNorm/moments/normalize/shifted_mean*
T0*
_output_shapes
:
ě
Amodel/discriminator_1/layer5/BatchNorm/moments/normalize/varianceSub<model/discriminator_1/layer5/BatchNorm/moments/normalize/Mul?model/discriminator_1/layer5/BatchNorm/moments/normalize/Square*
T0*
_output_shapes
:
Ć
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/decayConst*
dtype0*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ą
:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/subSub5model/discriminator/layer5/BatchNorm/moving_mean/read=model/discriminator_1/layer5/BatchNorm/moments/normalize/mean*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ľ
:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/mulMul:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/sub<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/decay*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
T0*
_output_shapes
:
Ž
6model/discriminator_1/layer5/BatchNorm/AssignMovingAvg	AssignSub0model/discriminator/layer5/BatchNorm/moving_mean:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg/mul*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking( *
T0*
_output_shapes
:
Ě
>model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/decayConst*
dtype0*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
valueB
 *ÍĚĚ=*
_output_shapes
: 
Ż
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/subSub9model/discriminator/layer5/BatchNorm/moving_variance/readAmodel/discriminator_1/layer5/BatchNorm/moments/normalize/variance*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
Ż
<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/mulMul<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/sub>model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/decay*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
T0*
_output_shapes
:
ş
8model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1	AssignSub4model/discriminator/layer5/BatchNorm/moving_variance<model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1/mul*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking( *
T0*
_output_shapes
:
{
6model/discriminator_1/layer5/BatchNorm/batchnorm/add/yConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
Ű
4model/discriminator_1/layer5/BatchNorm/batchnorm/addAddAmodel/discriminator_1/layer5/BatchNorm/moments/normalize/variance6model/discriminator_1/layer5/BatchNorm/batchnorm/add/y*
T0*
_output_shapes
:

6model/discriminator_1/layer5/BatchNorm/batchnorm/RsqrtRsqrt4model/discriminator_1/layer5/BatchNorm/batchnorm/add*
T0*
_output_shapes
:
É
4model/discriminator_1/layer5/BatchNorm/batchnorm/mulMul6model/discriminator_1/layer5/BatchNorm/batchnorm/Rsqrt/model/discriminator/layer5/BatchNorm/gamma/read*
T0*
_output_shapes
:
Ę
6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_1Mul#model/discriminator_1/layer5/MatMul4model/discriminator_1/layer5/BatchNorm/batchnorm/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_2Mul=model/discriminator_1/layer5/BatchNorm/moments/normalize/mean4model/discriminator_1/layer5/BatchNorm/batchnorm/mul*
T0*
_output_shapes
:
Č
4model/discriminator_1/layer5/BatchNorm/batchnorm/subSub.model/discriminator/layer5/BatchNorm/beta/read6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_2*
T0*
_output_shapes
:
Ý
6model/discriminator_1/layer5/BatchNorm/batchnorm/add_1Add6model/discriminator_1/layer5/BatchNorm/batchnorm/mul_14model/discriminator_1/layer5/BatchNorm/batchnorm/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

!model/discriminator_1/layer5/TanhTanh6model/discriminator_1/layer5/BatchNorm/batchnorm/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
model/ones_like/ShapeShapemodel/discriminator/layer5/Tanh*
out_type0*
T0*
_output_shapes
:
Z
model/ones_like/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
w
model/ones_likeFillmodel/ones_like/Shapemodel/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
model/clip_by_value/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value/MinimumMinimummodel/discriminator/layer5/Tanhmodel/clip_by_value/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
model/clip_by_value/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_valueMaximummodel/clip_by_value/Minimummodel/clip_by_value/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
	model/LogLogmodel/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
	model/mulMulmodel/ones_like	model/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
model/sub/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
`
	model/subSubmodel/sub/xmodel/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
h
model/sub_1Submodel/sub_1/xmodel/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_1Logmodel/sub_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/mul_1Mul	model/submodel/Log_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
	model/addAdd	model/mulmodel/mul_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
	model/NegNeg	model/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
h

model/MeanMean	model/Negmodel/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
v
model/discriminator_real/tagsConst*
dtype0*)
value B Bmodel/discriminator_real*
_output_shapes
: 
u
model/discriminator_realScalarSummarymodel/discriminator_real/tags
model/Mean*
T0*
_output_shapes
: 
p
model/zeros_like	ZerosLikemodel/discriminator/layer5/Tanh*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/clip_by_value_1/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value_1/MinimumMinimum!model/discriminator_1/layer5/Tanhmodel/clip_by_value_1/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/clip_by_value_1/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_value_1Maximummodel/clip_by_value_1/Minimummodel/clip_by_value_1/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
model/Log_2Logmodel/clip_by_value_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
model/mul_2Mulmodel/zeros_likemodel/Log_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_2/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
e
model/sub_2Submodel/sub_2/xmodel/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_3/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
model/sub_3Submodel/sub_3/xmodel/clip_by_value_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_3Logmodel/sub_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/mul_3Mulmodel/sub_2model/Log_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/add_1Addmodel/mul_2model/mul_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Neg_1Negmodel/add_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/Const_1Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_1Meanmodel/Neg_1model/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
v
model/discriminator_fake/tagsConst*
dtype0*)
value B Bmodel/discriminator_fake*
_output_shapes
: 
w
model/discriminator_fakeScalarSummarymodel/discriminator_fake/tagsmodel/Mean_1*
T0*
_output_shapes
: 
^
model/Const_2Const*
dtype0*
valueB"       *
_output_shapes
:
l
model/Mean_2Mean	model/Negmodel/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
^
model/Const_3Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_3Meanmodel/Neg_1model/Const_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
O
model/add_2Addmodel/Mean_2model/Mean_3*
T0*
_output_shapes
: 
p
model/discriminator_2/tagsConst*
dtype0*&
valueB Bmodel/discriminator_2*
_output_shapes
: 
p
model/discriminator_2ScalarSummarymodel/discriminator_2/tagsmodel/add_2*
T0*
_output_shapes
: 
x
model/ones_like_1/ShapeShape!model/discriminator_1/layer5/Tanh*
out_type0*
T0*
_output_shapes
:
\
model/ones_like_1/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
}
model/ones_like_1Fillmodel/ones_like_1/Shapemodel/ones_like_1/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/clip_by_value_2/Minimum/yConst*
dtype0*
valueB
 *ţ˙?*
_output_shapes
: 

model/clip_by_value_2/MinimumMinimum!model/discriminator_1/layer5/Tanhmodel/clip_by_value_2/Minimum/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
model/clip_by_value_2/yConst*
dtype0*
valueB
 *żÖ3*
_output_shapes
: 

model/clip_by_value_2Maximummodel/clip_by_value_2/Minimummodel/clip_by_value_2/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
model/Log_4Logmodel/clip_by_value_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
model/mul_4Mulmodel/ones_like_1model/Log_4*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_4/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
model/sub_4Submodel/sub_4/xmodel/ones_like_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
model/sub_5/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
model/sub_5Submodel/sub_5/xmodel/clip_by_value_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Log_5Logmodel/sub_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/mul_5Mulmodel/sub_4model/Log_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/add_3Addmodel/mul_4model/mul_5*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
model/Neg_2Negmodel/add_3*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
model/Const_4Const*
dtype0*
valueB"       *
_output_shapes
:
n
model/Mean_4Meanmodel/Neg_2model/Const_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
h
model/generator_1/tagsConst*
dtype0*"
valueB Bmodel/generator_1*
_output_shapes
: 
i
model/generator_1ScalarSummarymodel/generator_1/tagsmodel/Mean_4*
T0*
_output_shapes
: 
Ç
model/Merge/MergeSummaryMergeSummary1input_producer/input_producer/fraction_of_32_full3input_producer_1/input_producer/fraction_of_32_fullbatch/fraction_of_32_fullbatch_1/fraction_of_32_fullmodel/discriminator_realmodel/discriminator_fakemodel/discriminator_2model/generator_1*
N*
_output_shapes
: 
]
model/merged_summariesIdentitymodel/Merge/MergeSummary*
T0*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/SaveV2/tensor_namesConst*
dtype0*Ĺ
valueťB¸<B)model/discriminator/layer1/BatchNorm/betaB*model/discriminator/layer1/BatchNorm/gammaB0model/discriminator/layer1/BatchNorm/moving_meanB4model/discriminator/layer1/BatchNorm/moving_varianceBmodel/discriminator/layer1/biasB"model/discriminator/layer1/weightsB)model/discriminator/layer2/BatchNorm/betaB*model/discriminator/layer2/BatchNorm/gammaB0model/discriminator/layer2/BatchNorm/moving_meanB4model/discriminator/layer2/BatchNorm/moving_varianceBmodel/discriminator/layer2/biasB"model/discriminator/layer2/weightsB)model/discriminator/layer3/BatchNorm/betaB*model/discriminator/layer3/BatchNorm/gammaB0model/discriminator/layer3/BatchNorm/moving_meanB4model/discriminator/layer3/BatchNorm/moving_varianceBmodel/discriminator/layer3/biasB"model/discriminator/layer3/weightsB)model/discriminator/layer4/BatchNorm/betaB*model/discriminator/layer4/BatchNorm/gammaB0model/discriminator/layer4/BatchNorm/moving_meanB4model/discriminator/layer4/BatchNorm/moving_varianceBmodel/discriminator/layer4/biasB"model/discriminator/layer4/weightsB)model/discriminator/layer5/BatchNorm/betaB*model/discriminator/layer5/BatchNorm/gammaB0model/discriminator/layer5/BatchNorm/moving_meanB4model/discriminator/layer5/BatchNorm/moving_varianceBmodel/discriminator/layer5/biasB"model/discriminator/layer5/weightsB%model/generator/layer1/BatchNorm/betaB&model/generator/layer1/BatchNorm/gammaB,model/generator/layer1/BatchNorm/moving_meanB0model/generator/layer1/BatchNorm/moving_varianceBmodel/generator/layer1/biasBmodel/generator/layer1/weightsB%model/generator/layer2/BatchNorm/betaB&model/generator/layer2/BatchNorm/gammaB,model/generator/layer2/BatchNorm/moving_meanB0model/generator/layer2/BatchNorm/moving_varianceBmodel/generator/layer2/biasBmodel/generator/layer2/weightsB%model/generator/layer3/BatchNorm/betaB&model/generator/layer3/BatchNorm/gammaB,model/generator/layer3/BatchNorm/moving_meanB0model/generator/layer3/BatchNorm/moving_varianceBmodel/generator/layer3/biasBmodel/generator/layer3/weightsB%model/generator/layer4/BatchNorm/betaB&model/generator/layer4/BatchNorm/gammaB,model/generator/layer4/BatchNorm/moving_meanB0model/generator/layer4/BatchNorm/moving_varianceBmodel/generator/layer4/biasBmodel/generator/layer4/weightsB%model/generator/layer5/BatchNorm/betaB&model/generator/layer5/BatchNorm/gammaB,model/generator/layer5/BatchNorm/moving_meanB0model/generator/layer5/BatchNorm/moving_varianceBmodel/generator/layer5/biasBmodel/generator/layer5/weights*
_output_shapes
:<
Ţ
save/SaveV2/shape_and_slicesConst*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:<
Ő
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices)model/discriminator/layer1/BatchNorm/beta*model/discriminator/layer1/BatchNorm/gamma0model/discriminator/layer1/BatchNorm/moving_mean4model/discriminator/layer1/BatchNorm/moving_variancemodel/discriminator/layer1/bias"model/discriminator/layer1/weights)model/discriminator/layer2/BatchNorm/beta*model/discriminator/layer2/BatchNorm/gamma0model/discriminator/layer2/BatchNorm/moving_mean4model/discriminator/layer2/BatchNorm/moving_variancemodel/discriminator/layer2/bias"model/discriminator/layer2/weights)model/discriminator/layer3/BatchNorm/beta*model/discriminator/layer3/BatchNorm/gamma0model/discriminator/layer3/BatchNorm/moving_mean4model/discriminator/layer3/BatchNorm/moving_variancemodel/discriminator/layer3/bias"model/discriminator/layer3/weights)model/discriminator/layer4/BatchNorm/beta*model/discriminator/layer4/BatchNorm/gamma0model/discriminator/layer4/BatchNorm/moving_mean4model/discriminator/layer4/BatchNorm/moving_variancemodel/discriminator/layer4/bias"model/discriminator/layer4/weights)model/discriminator/layer5/BatchNorm/beta*model/discriminator/layer5/BatchNorm/gamma0model/discriminator/layer5/BatchNorm/moving_mean4model/discriminator/layer5/BatchNorm/moving_variancemodel/discriminator/layer5/bias"model/discriminator/layer5/weights%model/generator/layer1/BatchNorm/beta&model/generator/layer1/BatchNorm/gamma,model/generator/layer1/BatchNorm/moving_mean0model/generator/layer1/BatchNorm/moving_variancemodel/generator/layer1/biasmodel/generator/layer1/weights%model/generator/layer2/BatchNorm/beta&model/generator/layer2/BatchNorm/gamma,model/generator/layer2/BatchNorm/moving_mean0model/generator/layer2/BatchNorm/moving_variancemodel/generator/layer2/biasmodel/generator/layer2/weights%model/generator/layer3/BatchNorm/beta&model/generator/layer3/BatchNorm/gamma,model/generator/layer3/BatchNorm/moving_mean0model/generator/layer3/BatchNorm/moving_variancemodel/generator/layer3/biasmodel/generator/layer3/weights%model/generator/layer4/BatchNorm/beta&model/generator/layer4/BatchNorm/gamma,model/generator/layer4/BatchNorm/moving_mean0model/generator/layer4/BatchNorm/moving_variancemodel/generator/layer4/biasmodel/generator/layer4/weights%model/generator/layer5/BatchNorm/beta&model/generator/layer5/BatchNorm/gamma,model/generator/layer5/BatchNorm/moving_mean0model/generator/layer5/BatchNorm/moving_variancemodel/generator/layer5/biasmodel/generator/layer5/weights*J
dtypes@
>2<
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 

save/RestoreV2/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer1/BatchNorm/beta*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/AssignAssign)model/discriminator/layer1/BatchNorm/betasave/RestoreV2*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_1/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer1/BatchNorm/gamma*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_1Assign*model/discriminator/layer1/BatchNorm/gammasave/RestoreV2_1*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_2/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer1/BatchNorm/moving_mean*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
ď
save/Assign_2Assign0model/discriminator/layer1/BatchNorm/moving_meansave/RestoreV2_2*
validate_shape(*C
_class9
75loc:@model/discriminator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_3/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer1/BatchNorm/moving_variance*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
÷
save/Assign_3Assign4model/discriminator/layer1/BatchNorm/moving_variancesave/RestoreV2_3*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_4/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer1/bias*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ő
save/Assign_4Assignmodel/discriminator/layer1/biassave/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:  

save/RestoreV2_5/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer1/weights*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_5Assign"model/discriminator/layer1/weightssave/RestoreV2_5*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:

save/RestoreV2_6/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer2/BatchNorm/beta*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
á
save/Assign_6Assign)model/discriminator/layer2/BatchNorm/betasave/RestoreV2_6*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_7/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer2/BatchNorm/gamma*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_7Assign*model/discriminator/layer2/BatchNorm/gammasave/RestoreV2_7*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_8/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer2/BatchNorm/moving_mean*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
ď
save/Assign_8Assign0model/discriminator/layer2/BatchNorm/moving_meansave/RestoreV2_8*
validate_shape(*C
_class9
75loc:@model/discriminator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_9/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer2/BatchNorm/moving_variance*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
÷
save/Assign_9Assign4model/discriminator/layer2/BatchNorm/moving_variancesave/RestoreV2_9*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_10/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer2/bias*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_10Assignmodel/discriminator/layer2/biassave/RestoreV2_10*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_11/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer2/weights*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_11Assign"model/discriminator/layer2/weightssave/RestoreV2_11*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_12/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer3/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_12Assign)model/discriminator/layer3/BatchNorm/betasave/RestoreV2_12*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_13/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer3/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ĺ
save/Assign_13Assign*model/discriminator/layer3/BatchNorm/gammasave/RestoreV2_13*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_14/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer3/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_14Assign0model/discriminator/layer3/BatchNorm/moving_meansave/RestoreV2_14*
validate_shape(*C
_class9
75loc:@model/discriminator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_15/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer3/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_15Assign4model/discriminator/layer3/BatchNorm/moving_variancesave/RestoreV2_15*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_16/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer3/bias*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_16Assignmodel/discriminator/layer3/biassave/RestoreV2_16*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_17/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer3/weights*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_17Assign"model/discriminator/layer3/weightssave/RestoreV2_17*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_18/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer4/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
ă
save/Assign_18Assign)model/discriminator/layer4/BatchNorm/betasave/RestoreV2_18*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_19/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer4/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
ĺ
save/Assign_19Assign*model/discriminator/layer4/BatchNorm/gammasave/RestoreV2_19*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_20/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer4/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_20Assign0model/discriminator/layer4/BatchNorm/moving_meansave/RestoreV2_20*
validate_shape(*C
_class9
75loc:@model/discriminator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_21/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer4/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ů
save/Assign_21Assign4model/discriminator/layer4/BatchNorm/moving_variancesave/RestoreV2_21*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_22/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer4/bias*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_22Assignmodel/discriminator/layer4/biassave/RestoreV2_22*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_23/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer4/weights*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_23Assign"model/discriminator/layer4/weightssave/RestoreV2_23*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_24/tensor_namesConst*
dtype0*>
value5B3B)model/discriminator/layer5/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
â
save/Assign_24Assign)model/discriminator/layer5/BatchNorm/betasave/RestoreV2_24*
validate_shape(*<
_class2
0.loc:@model/discriminator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_25/tensor_namesConst*
dtype0*?
value6B4B*model/discriminator/layer5/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save/Assign_25Assign*model/discriminator/layer5/BatchNorm/gammasave/RestoreV2_25*
validate_shape(*=
_class3
1/loc:@model/discriminator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_26/tensor_namesConst*
dtype0*E
value<B:B0model/discriminator/layer5/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_26Assign0model/discriminator/layer5/BatchNorm/moving_meansave/RestoreV2_26*
validate_shape(*C
_class9
75loc:@model/discriminator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_27/tensor_namesConst*
dtype0*I
value@B>B4model/discriminator/layer5/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ř
save/Assign_27Assign4model/discriminator/layer5/BatchNorm/moving_variancesave/RestoreV2_27*
validate_shape(*G
_class=
;9loc:@model/discriminator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_28/tensor_namesConst*
dtype0*4
value+B)Bmodel/discriminator/layer5/bias*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_28Assignmodel/discriminator/layer5/biassave/RestoreV2_28*
validate_shape(*2
_class(
&$loc:@model/discriminator/layer5/bias*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_29/tensor_namesConst*
dtype0*7
value.B,B"model/discriminator/layer5/weights*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_29Assign"model/discriminator/layer5/weightssave/RestoreV2_29*
validate_shape(*5
_class+
)'loc:@model/discriminator/layer5/weights*
use_locking(*
T0* 
_output_shapes
:


save/RestoreV2_30/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer1/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_30Assign%model/generator/layer1/BatchNorm/betasave/RestoreV2_30*
validate_shape(*8
_class.
,*loc:@model/generator/layer1/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_31/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer1/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_31Assign&model/generator/layer1/BatchNorm/gammasave/RestoreV2_31*
validate_shape(*9
_class/
-+loc:@model/generator/layer1/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_32/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer1/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_32Assign,model/generator/layer1/BatchNorm/moving_meansave/RestoreV2_32*
validate_shape(*?
_class5
31loc:@model/generator/layer1/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_33/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer1/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_33Assign0model/generator/layer1/BatchNorm/moving_variancesave/RestoreV2_33*
validate_shape(*C
_class9
75loc:@model/generator/layer1/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_34/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer1/bias*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_34Assignmodel/generator/layer1/biassave/RestoreV2_34*
validate_shape(*.
_class$
" loc:@model/generator/layer1/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_35/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer1/weights*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_35Assignmodel/generator/layer1/weightssave/RestoreV2_35*
validate_shape(*1
_class'
%#loc:@model/generator/layer1/weights*
use_locking(*
T0*'
_output_shapes
:d

save/RestoreV2_36/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer2/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_36Assign%model/generator/layer2/BatchNorm/betasave/RestoreV2_36*
validate_shape(*8
_class.
,*loc:@model/generator/layer2/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_37/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer2/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_37Assign&model/generator/layer2/BatchNorm/gammasave/RestoreV2_37*
validate_shape(*9
_class/
-+loc:@model/generator/layer2/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_38/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer2/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_38Assign,model/generator/layer2/BatchNorm/moving_meansave/RestoreV2_38*
validate_shape(*?
_class5
31loc:@model/generator/layer2/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_39/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer2/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_39Assign0model/generator/layer2/BatchNorm/moving_variancesave/RestoreV2_39*
validate_shape(*C
_class9
75loc:@model/generator/layer2/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_40/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer2/bias*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_40Assignmodel/generator/layer2/biassave/RestoreV2_40*
validate_shape(*.
_class$
" loc:@model/generator/layer2/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_41/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer2/weights*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_41Assignmodel/generator/layer2/weightssave/RestoreV2_41*
validate_shape(*1
_class'
%#loc:@model/generator/layer2/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_42/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer3/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_42Assign%model/generator/layer3/BatchNorm/betasave/RestoreV2_42*
validate_shape(*8
_class.
,*loc:@model/generator/layer3/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_43/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer3/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_43Assign&model/generator/layer3/BatchNorm/gammasave/RestoreV2_43*
validate_shape(*9
_class/
-+loc:@model/generator/layer3/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_44/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer3/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_44/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_44Assign,model/generator/layer3/BatchNorm/moving_meansave/RestoreV2_44*
validate_shape(*?
_class5
31loc:@model/generator/layer3/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_45/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer3/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_45Assign0model/generator/layer3/BatchNorm/moving_variancesave/RestoreV2_45*
validate_shape(*C
_class9
75loc:@model/generator/layer3/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_46/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer3/bias*
_output_shapes
:
k
"save/RestoreV2_46/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_46Assignmodel/generator/layer3/biassave/RestoreV2_46*
validate_shape(*.
_class$
" loc:@model/generator/layer3/bias*
use_locking(*
T0*#
_output_shapes
:

save/RestoreV2_47/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer3/weights*
_output_shapes
:
k
"save/RestoreV2_47/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_47Assignmodel/generator/layer3/weightssave/RestoreV2_47*
validate_shape(*1
_class'
%#loc:@model/generator/layer3/weights*
use_locking(*
T0*(
_output_shapes
:

save/RestoreV2_48/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer4/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_48/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ű
save/Assign_48Assign%model/generator/layer4/BatchNorm/betasave/RestoreV2_48*
validate_shape(*8
_class.
,*loc:@model/generator/layer4/BatchNorm/beta*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_49/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer4/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_49/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
Ý
save/Assign_49Assign&model/generator/layer4/BatchNorm/gammasave/RestoreV2_49*
validate_shape(*9
_class/
-+loc:@model/generator/layer4/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_50/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer4/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save/Assign_50Assign,model/generator/layer4/BatchNorm/moving_meansave/RestoreV2_50*
validate_shape(*?
_class5
31loc:@model/generator/layer4/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_51/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer4/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save/Assign_51Assign0model/generator/layer4/BatchNorm/moving_variancesave/RestoreV2_51*
validate_shape(*C
_class9
75loc:@model/generator/layer4/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes	
:

save/RestoreV2_52/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer4/bias*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
Ď
save/Assign_52Assignmodel/generator/layer4/biassave/RestoreV2_52*
validate_shape(*.
_class$
" loc:@model/generator/layer4/bias*
use_locking(*
T0*#
_output_shapes
:  

save/RestoreV2_53/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer4/weights*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_53Assignmodel/generator/layer4/weightssave/RestoreV2_53*
validate_shape(*1
_class'
%#loc:@model/generator/layer4/weights*
use_locking(*
T0*(
_output_shapes
:  

save/RestoreV2_54/tensor_namesConst*
dtype0*:
value1B/B%model/generator/layer5/BatchNorm/beta*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ú
save/Assign_54Assign%model/generator/layer5/BatchNorm/betasave/RestoreV2_54*
validate_shape(*8
_class.
,*loc:@model/generator/layer5/BatchNorm/beta*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_55/tensor_namesConst*
dtype0*;
value2B0B&model/generator/layer5/BatchNorm/gamma*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save/Assign_55Assign&model/generator/layer5/BatchNorm/gammasave/RestoreV2_55*
validate_shape(*9
_class/
-+loc:@model/generator/layer5/BatchNorm/gamma*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_56/tensor_namesConst*
dtype0*A
value8B6B,model/generator/layer5/BatchNorm/moving_mean*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
č
save/Assign_56Assign,model/generator/layer5/BatchNorm/moving_meansave/RestoreV2_56*
validate_shape(*?
_class5
31loc:@model/generator/layer5/BatchNorm/moving_mean*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_57/tensor_namesConst*
dtype0*E
value<B:B0model/generator/layer5/BatchNorm/moving_variance*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
đ
save/Assign_57Assign0model/generator/layer5/BatchNorm/moving_variancesave/RestoreV2_57*
validate_shape(*C
_class9
75loc:@model/generator/layer5/BatchNorm/moving_variance*
use_locking(*
T0*
_output_shapes
:

save/RestoreV2_58/tensor_namesConst*
dtype0*0
value'B%Bmodel/generator/layer5/bias*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
Î
save/Assign_58Assignmodel/generator/layer5/biassave/RestoreV2_58*
validate_shape(*.
_class$
" loc:@model/generator/layer5/bias*
use_locking(*
T0*"
_output_shapes
:@@

save/RestoreV2_59/tensor_namesConst*
dtype0*3
value*B(Bmodel/generator/layer5/weights*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
Ů
save/Assign_59Assignmodel/generator/layer5/weightssave/RestoreV2_59*
validate_shape(*1
_class'
%#loc:@model/generator/layer5/weights*
use_locking(*
T0*'
_output_shapes
:@@

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59""×(
trainable_variablesż(ź(
p
 model/generator/layer1/weights:0%model/generator/layer1/weights/Assign%model/generator/layer1/weights/read:0
g
model/generator/layer1/bias:0"model/generator/layer1/bias/Assign"model/generator/layer1/bias/read:0

'model/generator/layer1/BatchNorm/beta:0,model/generator/layer1/BatchNorm/beta/Assign,model/generator/layer1/BatchNorm/beta/read:0

(model/generator/layer1/BatchNorm/gamma:0-model/generator/layer1/BatchNorm/gamma/Assign-model/generator/layer1/BatchNorm/gamma/read:0
p
 model/generator/layer2/weights:0%model/generator/layer2/weights/Assign%model/generator/layer2/weights/read:0
g
model/generator/layer2/bias:0"model/generator/layer2/bias/Assign"model/generator/layer2/bias/read:0

'model/generator/layer2/BatchNorm/beta:0,model/generator/layer2/BatchNorm/beta/Assign,model/generator/layer2/BatchNorm/beta/read:0

(model/generator/layer2/BatchNorm/gamma:0-model/generator/layer2/BatchNorm/gamma/Assign-model/generator/layer2/BatchNorm/gamma/read:0
p
 model/generator/layer3/weights:0%model/generator/layer3/weights/Assign%model/generator/layer3/weights/read:0
g
model/generator/layer3/bias:0"model/generator/layer3/bias/Assign"model/generator/layer3/bias/read:0

'model/generator/layer3/BatchNorm/beta:0,model/generator/layer3/BatchNorm/beta/Assign,model/generator/layer3/BatchNorm/beta/read:0

(model/generator/layer3/BatchNorm/gamma:0-model/generator/layer3/BatchNorm/gamma/Assign-model/generator/layer3/BatchNorm/gamma/read:0
p
 model/generator/layer4/weights:0%model/generator/layer4/weights/Assign%model/generator/layer4/weights/read:0
g
model/generator/layer4/bias:0"model/generator/layer4/bias/Assign"model/generator/layer4/bias/read:0

'model/generator/layer4/BatchNorm/beta:0,model/generator/layer4/BatchNorm/beta/Assign,model/generator/layer4/BatchNorm/beta/read:0

(model/generator/layer4/BatchNorm/gamma:0-model/generator/layer4/BatchNorm/gamma/Assign-model/generator/layer4/BatchNorm/gamma/read:0
p
 model/generator/layer5/weights:0%model/generator/layer5/weights/Assign%model/generator/layer5/weights/read:0
g
model/generator/layer5/bias:0"model/generator/layer5/bias/Assign"model/generator/layer5/bias/read:0

'model/generator/layer5/BatchNorm/beta:0,model/generator/layer5/BatchNorm/beta/Assign,model/generator/layer5/BatchNorm/beta/read:0

(model/generator/layer5/BatchNorm/gamma:0-model/generator/layer5/BatchNorm/gamma/Assign-model/generator/layer5/BatchNorm/gamma/read:0
|
$model/discriminator/layer1/weights:0)model/discriminator/layer1/weights/Assign)model/discriminator/layer1/weights/read:0
s
!model/discriminator/layer1/bias:0&model/discriminator/layer1/bias/Assign&model/discriminator/layer1/bias/read:0

+model/discriminator/layer1/BatchNorm/beta:00model/discriminator/layer1/BatchNorm/beta/Assign0model/discriminator/layer1/BatchNorm/beta/read:0

,model/discriminator/layer1/BatchNorm/gamma:01model/discriminator/layer1/BatchNorm/gamma/Assign1model/discriminator/layer1/BatchNorm/gamma/read:0
|
$model/discriminator/layer2/weights:0)model/discriminator/layer2/weights/Assign)model/discriminator/layer2/weights/read:0
s
!model/discriminator/layer2/bias:0&model/discriminator/layer2/bias/Assign&model/discriminator/layer2/bias/read:0

+model/discriminator/layer2/BatchNorm/beta:00model/discriminator/layer2/BatchNorm/beta/Assign0model/discriminator/layer2/BatchNorm/beta/read:0

,model/discriminator/layer2/BatchNorm/gamma:01model/discriminator/layer2/BatchNorm/gamma/Assign1model/discriminator/layer2/BatchNorm/gamma/read:0
|
$model/discriminator/layer3/weights:0)model/discriminator/layer3/weights/Assign)model/discriminator/layer3/weights/read:0
s
!model/discriminator/layer3/bias:0&model/discriminator/layer3/bias/Assign&model/discriminator/layer3/bias/read:0

+model/discriminator/layer3/BatchNorm/beta:00model/discriminator/layer3/BatchNorm/beta/Assign0model/discriminator/layer3/BatchNorm/beta/read:0

,model/discriminator/layer3/BatchNorm/gamma:01model/discriminator/layer3/BatchNorm/gamma/Assign1model/discriminator/layer3/BatchNorm/gamma/read:0
|
$model/discriminator/layer4/weights:0)model/discriminator/layer4/weights/Assign)model/discriminator/layer4/weights/read:0
s
!model/discriminator/layer4/bias:0&model/discriminator/layer4/bias/Assign&model/discriminator/layer4/bias/read:0

+model/discriminator/layer4/BatchNorm/beta:00model/discriminator/layer4/BatchNorm/beta/Assign0model/discriminator/layer4/BatchNorm/beta/read:0

,model/discriminator/layer4/BatchNorm/gamma:01model/discriminator/layer4/BatchNorm/gamma/Assign1model/discriminator/layer4/BatchNorm/gamma/read:0
|
$model/discriminator/layer5/weights:0)model/discriminator/layer5/weights/Assign)model/discriminator/layer5/weights/read:0
s
!model/discriminator/layer5/bias:0&model/discriminator/layer5/bias/Assign&model/discriminator/layer5/bias/read:0

+model/discriminator/layer5/BatchNorm/beta:00model/discriminator/layer5/BatchNorm/beta/Assign0model/discriminator/layer5/BatchNorm/beta/read:0

,model/discriminator/layer5/BatchNorm/gamma:01model/discriminator/layer5/BatchNorm/gamma/Assign1model/discriminator/layer5/BatchNorm/gamma/read:0"C
	variablesóBđB
p
 model/generator/layer1/weights:0%model/generator/layer1/weights/Assign%model/generator/layer1/weights/read:0
g
model/generator/layer1/bias:0"model/generator/layer1/bias/Assign"model/generator/layer1/bias/read:0

'model/generator/layer1/BatchNorm/beta:0,model/generator/layer1/BatchNorm/beta/Assign,model/generator/layer1/BatchNorm/beta/read:0

(model/generator/layer1/BatchNorm/gamma:0-model/generator/layer1/BatchNorm/gamma/Assign-model/generator/layer1/BatchNorm/gamma/read:0

.model/generator/layer1/BatchNorm/moving_mean:03model/generator/layer1/BatchNorm/moving_mean/Assign3model/generator/layer1/BatchNorm/moving_mean/read:0
Ś
2model/generator/layer1/BatchNorm/moving_variance:07model/generator/layer1/BatchNorm/moving_variance/Assign7model/generator/layer1/BatchNorm/moving_variance/read:0
p
 model/generator/layer2/weights:0%model/generator/layer2/weights/Assign%model/generator/layer2/weights/read:0
g
model/generator/layer2/bias:0"model/generator/layer2/bias/Assign"model/generator/layer2/bias/read:0

'model/generator/layer2/BatchNorm/beta:0,model/generator/layer2/BatchNorm/beta/Assign,model/generator/layer2/BatchNorm/beta/read:0

(model/generator/layer2/BatchNorm/gamma:0-model/generator/layer2/BatchNorm/gamma/Assign-model/generator/layer2/BatchNorm/gamma/read:0

.model/generator/layer2/BatchNorm/moving_mean:03model/generator/layer2/BatchNorm/moving_mean/Assign3model/generator/layer2/BatchNorm/moving_mean/read:0
Ś
2model/generator/layer2/BatchNorm/moving_variance:07model/generator/layer2/BatchNorm/moving_variance/Assign7model/generator/layer2/BatchNorm/moving_variance/read:0
p
 model/generator/layer3/weights:0%model/generator/layer3/weights/Assign%model/generator/layer3/weights/read:0
g
model/generator/layer3/bias:0"model/generator/layer3/bias/Assign"model/generator/layer3/bias/read:0

'model/generator/layer3/BatchNorm/beta:0,model/generator/layer3/BatchNorm/beta/Assign,model/generator/layer3/BatchNorm/beta/read:0

(model/generator/layer3/BatchNorm/gamma:0-model/generator/layer3/BatchNorm/gamma/Assign-model/generator/layer3/BatchNorm/gamma/read:0

.model/generator/layer3/BatchNorm/moving_mean:03model/generator/layer3/BatchNorm/moving_mean/Assign3model/generator/layer3/BatchNorm/moving_mean/read:0
Ś
2model/generator/layer3/BatchNorm/moving_variance:07model/generator/layer3/BatchNorm/moving_variance/Assign7model/generator/layer3/BatchNorm/moving_variance/read:0
p
 model/generator/layer4/weights:0%model/generator/layer4/weights/Assign%model/generator/layer4/weights/read:0
g
model/generator/layer4/bias:0"model/generator/layer4/bias/Assign"model/generator/layer4/bias/read:0

'model/generator/layer4/BatchNorm/beta:0,model/generator/layer4/BatchNorm/beta/Assign,model/generator/layer4/BatchNorm/beta/read:0

(model/generator/layer4/BatchNorm/gamma:0-model/generator/layer4/BatchNorm/gamma/Assign-model/generator/layer4/BatchNorm/gamma/read:0

.model/generator/layer4/BatchNorm/moving_mean:03model/generator/layer4/BatchNorm/moving_mean/Assign3model/generator/layer4/BatchNorm/moving_mean/read:0
Ś
2model/generator/layer4/BatchNorm/moving_variance:07model/generator/layer4/BatchNorm/moving_variance/Assign7model/generator/layer4/BatchNorm/moving_variance/read:0
p
 model/generator/layer5/weights:0%model/generator/layer5/weights/Assign%model/generator/layer5/weights/read:0
g
model/generator/layer5/bias:0"model/generator/layer5/bias/Assign"model/generator/layer5/bias/read:0

'model/generator/layer5/BatchNorm/beta:0,model/generator/layer5/BatchNorm/beta/Assign,model/generator/layer5/BatchNorm/beta/read:0

(model/generator/layer5/BatchNorm/gamma:0-model/generator/layer5/BatchNorm/gamma/Assign-model/generator/layer5/BatchNorm/gamma/read:0

.model/generator/layer5/BatchNorm/moving_mean:03model/generator/layer5/BatchNorm/moving_mean/Assign3model/generator/layer5/BatchNorm/moving_mean/read:0
Ś
2model/generator/layer5/BatchNorm/moving_variance:07model/generator/layer5/BatchNorm/moving_variance/Assign7model/generator/layer5/BatchNorm/moving_variance/read:0
|
$model/discriminator/layer1/weights:0)model/discriminator/layer1/weights/Assign)model/discriminator/layer1/weights/read:0
s
!model/discriminator/layer1/bias:0&model/discriminator/layer1/bias/Assign&model/discriminator/layer1/bias/read:0

+model/discriminator/layer1/BatchNorm/beta:00model/discriminator/layer1/BatchNorm/beta/Assign0model/discriminator/layer1/BatchNorm/beta/read:0

,model/discriminator/layer1/BatchNorm/gamma:01model/discriminator/layer1/BatchNorm/gamma/Assign1model/discriminator/layer1/BatchNorm/gamma/read:0
Ś
2model/discriminator/layer1/BatchNorm/moving_mean:07model/discriminator/layer1/BatchNorm/moving_mean/Assign7model/discriminator/layer1/BatchNorm/moving_mean/read:0
˛
6model/discriminator/layer1/BatchNorm/moving_variance:0;model/discriminator/layer1/BatchNorm/moving_variance/Assign;model/discriminator/layer1/BatchNorm/moving_variance/read:0
|
$model/discriminator/layer2/weights:0)model/discriminator/layer2/weights/Assign)model/discriminator/layer2/weights/read:0
s
!model/discriminator/layer2/bias:0&model/discriminator/layer2/bias/Assign&model/discriminator/layer2/bias/read:0

+model/discriminator/layer2/BatchNorm/beta:00model/discriminator/layer2/BatchNorm/beta/Assign0model/discriminator/layer2/BatchNorm/beta/read:0

,model/discriminator/layer2/BatchNorm/gamma:01model/discriminator/layer2/BatchNorm/gamma/Assign1model/discriminator/layer2/BatchNorm/gamma/read:0
Ś
2model/discriminator/layer2/BatchNorm/moving_mean:07model/discriminator/layer2/BatchNorm/moving_mean/Assign7model/discriminator/layer2/BatchNorm/moving_mean/read:0
˛
6model/discriminator/layer2/BatchNorm/moving_variance:0;model/discriminator/layer2/BatchNorm/moving_variance/Assign;model/discriminator/layer2/BatchNorm/moving_variance/read:0
|
$model/discriminator/layer3/weights:0)model/discriminator/layer3/weights/Assign)model/discriminator/layer3/weights/read:0
s
!model/discriminator/layer3/bias:0&model/discriminator/layer3/bias/Assign&model/discriminator/layer3/bias/read:0

+model/discriminator/layer3/BatchNorm/beta:00model/discriminator/layer3/BatchNorm/beta/Assign0model/discriminator/layer3/BatchNorm/beta/read:0

,model/discriminator/layer3/BatchNorm/gamma:01model/discriminator/layer3/BatchNorm/gamma/Assign1model/discriminator/layer3/BatchNorm/gamma/read:0
Ś
2model/discriminator/layer3/BatchNorm/moving_mean:07model/discriminator/layer3/BatchNorm/moving_mean/Assign7model/discriminator/layer3/BatchNorm/moving_mean/read:0
˛
6model/discriminator/layer3/BatchNorm/moving_variance:0;model/discriminator/layer3/BatchNorm/moving_variance/Assign;model/discriminator/layer3/BatchNorm/moving_variance/read:0
|
$model/discriminator/layer4/weights:0)model/discriminator/layer4/weights/Assign)model/discriminator/layer4/weights/read:0
s
!model/discriminator/layer4/bias:0&model/discriminator/layer4/bias/Assign&model/discriminator/layer4/bias/read:0

+model/discriminator/layer4/BatchNorm/beta:00model/discriminator/layer4/BatchNorm/beta/Assign0model/discriminator/layer4/BatchNorm/beta/read:0

,model/discriminator/layer4/BatchNorm/gamma:01model/discriminator/layer4/BatchNorm/gamma/Assign1model/discriminator/layer4/BatchNorm/gamma/read:0
Ś
2model/discriminator/layer4/BatchNorm/moving_mean:07model/discriminator/layer4/BatchNorm/moving_mean/Assign7model/discriminator/layer4/BatchNorm/moving_mean/read:0
˛
6model/discriminator/layer4/BatchNorm/moving_variance:0;model/discriminator/layer4/BatchNorm/moving_variance/Assign;model/discriminator/layer4/BatchNorm/moving_variance/read:0
|
$model/discriminator/layer5/weights:0)model/discriminator/layer5/weights/Assign)model/discriminator/layer5/weights/read:0
s
!model/discriminator/layer5/bias:0&model/discriminator/layer5/bias/Assign&model/discriminator/layer5/bias/read:0

+model/discriminator/layer5/BatchNorm/beta:00model/discriminator/layer5/BatchNorm/beta/Assign0model/discriminator/layer5/BatchNorm/beta/read:0

,model/discriminator/layer5/BatchNorm/gamma:01model/discriminator/layer5/BatchNorm/gamma/Assign1model/discriminator/layer5/BatchNorm/gamma/read:0
Ś
2model/discriminator/layer5/BatchNorm/moving_mean:07model/discriminator/layer5/BatchNorm/moving_mean/Assign7model/discriminator/layer5/BatchNorm/moving_mean/read:0
˛
6model/discriminator/layer5/BatchNorm/moving_variance:0;model/discriminator/layer5/BatchNorm/moving_variance/Assign;model/discriminator/layer5/BatchNorm/moving_variance/read:0"Ź

update_ops

2model/generator/layer1/BatchNorm/AssignMovingAvg:0
4model/generator/layer1/BatchNorm/AssignMovingAvg_1:0
2model/generator/layer2/BatchNorm/AssignMovingAvg:0
4model/generator/layer2/BatchNorm/AssignMovingAvg_1:0
2model/generator/layer3/BatchNorm/AssignMovingAvg:0
4model/generator/layer3/BatchNorm/AssignMovingAvg_1:0
2model/generator/layer4/BatchNorm/AssignMovingAvg:0
4model/generator/layer4/BatchNorm/AssignMovingAvg_1:0
2model/generator/layer5/BatchNorm/AssignMovingAvg:0
4model/generator/layer5/BatchNorm/AssignMovingAvg_1:0
6model/discriminator/layer1/BatchNorm/AssignMovingAvg:0
8model/discriminator/layer1/BatchNorm/AssignMovingAvg_1:0
6model/discriminator/layer2/BatchNorm/AssignMovingAvg:0
8model/discriminator/layer2/BatchNorm/AssignMovingAvg_1:0
6model/discriminator/layer3/BatchNorm/AssignMovingAvg:0
8model/discriminator/layer3/BatchNorm/AssignMovingAvg_1:0
6model/discriminator/layer4/BatchNorm/AssignMovingAvg:0
8model/discriminator/layer4/BatchNorm/AssignMovingAvg_1:0
6model/discriminator/layer5/BatchNorm/AssignMovingAvg:0
8model/discriminator/layer5/BatchNorm/AssignMovingAvg_1:0
8model/discriminator_1/layer1/BatchNorm/AssignMovingAvg:0
:model/discriminator_1/layer1/BatchNorm/AssignMovingAvg_1:0
8model/discriminator_1/layer2/BatchNorm/AssignMovingAvg:0
:model/discriminator_1/layer2/BatchNorm/AssignMovingAvg_1:0
8model/discriminator_1/layer3/BatchNorm/AssignMovingAvg:0
:model/discriminator_1/layer3/BatchNorm/AssignMovingAvg_1:0
8model/discriminator_1/layer4/BatchNorm/AssignMovingAvg:0
:model/discriminator_1/layer4/BatchNorm/AssignMovingAvg_1:0
8model/discriminator_1/layer5/BatchNorm/AssignMovingAvg:0
:model/discriminator_1/layer5/BatchNorm/AssignMovingAvg_1:0"
local_variablesn
l
3input_producer/input_producer/limit_epochs/epochs:0
5input_producer_1/input_producer/limit_epochs/epochs:0"ý
queue_runnersëč
Ć
input_producer/input_producer8input_producer/input_producer/input_producer_EnqueueMany2input_producer/input_producer/input_producer_Close"4input_producer/input_producer/input_producer_Close_1*
Î
input_producer_1/input_producer:input_producer_1/input_producer/input_producer_EnqueueMany4input_producer_1/input_producer/input_producer_Close"6input_producer_1/input_producer/input_producer_Close_1*
a
batch/fifo_queuebatch/fifo_queue_enqueuebatch/fifo_queue_Close"batch/fifo_queue_Close_1*
i
batch_1/fifo_queuebatch_1/fifo_queue_enqueuebatch_1/fifo_queue_Close"batch_1/fifo_queue_Close_1*"
	summaries

3input_producer/input_producer/fraction_of_32_full:0
5input_producer_1/input_producer/fraction_of_32_full:0
batch/fraction_of_32_full:0
batch_1/fraction_of_32_full:0
model/discriminator_real:0
model/discriminator_fake:0
model/discriminator_2:0
model/generator_1:0"
model_variablesů
ö
'model/generator/layer1/BatchNorm/beta:0
(model/generator/layer1/BatchNorm/gamma:0
.model/generator/layer1/BatchNorm/moving_mean:0
2model/generator/layer1/BatchNorm/moving_variance:0
'model/generator/layer2/BatchNorm/beta:0
(model/generator/layer2/BatchNorm/gamma:0
.model/generator/layer2/BatchNorm/moving_mean:0
2model/generator/layer2/BatchNorm/moving_variance:0
'model/generator/layer3/BatchNorm/beta:0
(model/generator/layer3/BatchNorm/gamma:0
.model/generator/layer3/BatchNorm/moving_mean:0
2model/generator/layer3/BatchNorm/moving_variance:0
'model/generator/layer4/BatchNorm/beta:0
(model/generator/layer4/BatchNorm/gamma:0
.model/generator/layer4/BatchNorm/moving_mean:0
2model/generator/layer4/BatchNorm/moving_variance:0
'model/generator/layer5/BatchNorm/beta:0
(model/generator/layer5/BatchNorm/gamma:0
.model/generator/layer5/BatchNorm/moving_mean:0
2model/generator/layer5/BatchNorm/moving_variance:0
+model/discriminator/layer1/BatchNorm/beta:0
,model/discriminator/layer1/BatchNorm/gamma:0
2model/discriminator/layer1/BatchNorm/moving_mean:0
6model/discriminator/layer1/BatchNorm/moving_variance:0
+model/discriminator/layer2/BatchNorm/beta:0
,model/discriminator/layer2/BatchNorm/gamma:0
2model/discriminator/layer2/BatchNorm/moving_mean:0
6model/discriminator/layer2/BatchNorm/moving_variance:0
+model/discriminator/layer3/BatchNorm/beta:0
,model/discriminator/layer3/BatchNorm/gamma:0
2model/discriminator/layer3/BatchNorm/moving_mean:0
6model/discriminator/layer3/BatchNorm/moving_variance:0
+model/discriminator/layer4/BatchNorm/beta:0
,model/discriminator/layer4/BatchNorm/gamma:0
2model/discriminator/layer4/BatchNorm/moving_mean:0
6model/discriminator/layer4/BatchNorm/moving_variance:0
+model/discriminator/layer5/BatchNorm/beta:0
,model/discriminator/layer5/BatchNorm/gamma:0
2model/discriminator/layer5/BatchNorm/moving_mean:0
6model/discriminator/layer5/BatchNorm/moving_variance:0içB      Ă˙ 	ő­3bÖA*ś
8
1input_producer/input_producer/fraction_of_32_full    
:
3input_producer_1/input_producer/fraction_of_32_full    
 
batch/fraction_of_32_full   >
"
batch_1/fraction_of_32_full   >

model/discriminator_real``3A

model/discriminator_fake-×?

model/discriminator_2Ó­<A

model/generator_1ZDAăÜ