πΛ,
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ό)

words_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Σ*+
shared_namewords_embedding/embeddings

.words_embedding/embeddings/Read/ReadVariableOpReadVariableOpwords_embedding/embeddings* 
_output_shapes
:
Σ*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0

main_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namemain_output/kernel
z
&main_output/kernel/Read/ReadVariableOpReadVariableOpmain_output/kernel*
_output_shapes
:	*
dtype0
x
main_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namemain_output/bias
q
$main_output/bias/Read/ReadVariableOpReadVariableOpmain_output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

words_lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namewords_lstm/lstm_cell/kernel

/words_lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpwords_lstm/lstm_cell/kernel* 
_output_shapes
:
*
dtype0
¨
%words_lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%words_lstm/lstm_cell/recurrent_kernel
‘
9words_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp%words_lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

words_lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namewords_lstm/lstm_cell/bias

-words_lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpwords_lstm/lstm_cell/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0
d
total_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_15
]
total_15/Read/ReadVariableOpReadVariableOptotal_15*
_output_shapes
: *
dtype0
d
count_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_15
]
count_15/Read/ReadVariableOpReadVariableOpcount_15*
_output_shapes
: *
dtype0
 
!Adam/words_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Σ*2
shared_name#!Adam/words_embedding/embeddings/m

5Adam/words_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp!Adam/words_embedding/embeddings/m* 
_output_shapes
:
Σ*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/main_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameAdam/main_output/kernel/m

-Adam/main_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/main_output/kernel/m*
_output_shapes
:	*
dtype0

Adam/main_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/main_output/bias/m

+Adam/main_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/main_output/bias/m*
_output_shapes
:*
dtype0
’
"Adam/words_lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/words_lstm/lstm_cell/kernel/m

6Adam/words_lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/words_lstm/lstm_cell/kernel/m* 
_output_shapes
:
*
dtype0
Ά
,Adam/words_lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/words_lstm/lstm_cell/recurrent_kernel/m
―
@Adam/words_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/words_lstm/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

 Adam/words_lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/words_lstm/lstm_cell/bias/m

4Adam/words_lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp Adam/words_lstm/lstm_cell/bias/m*
_output_shapes	
:*
dtype0
 
!Adam/words_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Σ*2
shared_name#!Adam/words_embedding/embeddings/v

5Adam/words_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp!Adam/words_embedding/embeddings/v* 
_output_shapes
:
Σ*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/main_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameAdam/main_output/kernel/v

-Adam/main_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/main_output/kernel/v*
_output_shapes
:	*
dtype0

Adam/main_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/main_output/bias/v

+Adam/main_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/main_output/bias/v*
_output_shapes
:*
dtype0
’
"Adam/words_lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adam/words_lstm/lstm_cell/kernel/v

6Adam/words_lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/words_lstm/lstm_cell/kernel/v* 
_output_shapes
:
*
dtype0
Ά
,Adam/words_lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/words_lstm/lstm_cell/recurrent_kernel/v
―
@Adam/words_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/words_lstm/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

 Adam/words_lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/words_lstm/lstm_cell/bias/v

4Adam/words_lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp Adam/words_lstm/lstm_cell/bias/v*
_output_shapes	
:*
dtype0
¦
$Adam/words_embedding/embeddings/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Σ*5
shared_name&$Adam/words_embedding/embeddings/vhat

8Adam/words_embedding/embeddings/vhat/Read/ReadVariableOpReadVariableOp$Adam/words_embedding/embeddings/vhat* 
_output_shapes
:
Σ*
dtype0

Adam/dense/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense/kernel/vhat

*Adam/dense/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/vhat* 
_output_shapes
:
*
dtype0

Adam/dense/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense/bias/vhat
z
(Adam/dense/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/dense/bias/vhat*
_output_shapes	
:*
dtype0

Adam/main_output/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/main_output/kernel/vhat

0Adam/main_output/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/main_output/kernel/vhat*
_output_shapes
:	*
dtype0

Adam/main_output/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/main_output/bias/vhat

.Adam/main_output/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/main_output/bias/vhat*
_output_shapes
:*
dtype0
¨
%Adam/words_lstm/lstm_cell/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/words_lstm/lstm_cell/kernel/vhat
‘
9Adam/words_lstm/lstm_cell/kernel/vhat/Read/ReadVariableOpReadVariableOp%Adam/words_lstm/lstm_cell/kernel/vhat* 
_output_shapes
:
*
dtype0
Ό
/Adam/words_lstm/lstm_cell/recurrent_kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*@
shared_name1/Adam/words_lstm/lstm_cell/recurrent_kernel/vhat
΅
CAdam/words_lstm/lstm_cell/recurrent_kernel/vhat/Read/ReadVariableOpReadVariableOp/Adam/words_lstm/lstm_cell/recurrent_kernel/vhat* 
_output_shapes
:
*
dtype0

#Adam/words_lstm/lstm_cell/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/words_lstm/lstm_cell/bias/vhat

7Adam/words_lstm/lstm_cell/bias/vhat/Read/ReadVariableOpReadVariableOp#Adam/words_lstm/lstm_cell/bias/vhat*
_output_shapes	
:*
dtype0

NoOpNoOp
¬_
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*η^
valueέ^BΪ^ BΣ^
Α
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
Θ
.iter

/beta_1

0beta_2
	1decay
2learning_ratemΒmΓmΔ(mΕ)mΖ3mΗ4mΘ5mΙvΚvΛvΜ(vΝ)vΞ3vΟ4vΠ5vΡvhat?vhatΣvhatΤ(vhatΥ)vhatΦ3vhatΧ4vhatΨ5vhatΩ
 
8
0
31
42
53
4
5
(6
)7
8
0
31
42
53
4
5
(6
)7
­
6metrics

7layers

regularization_losses
	variables
trainable_variables
8non_trainable_variables
9layer_metrics
:layer_regularization_losses
 
jh
VARIABLE_VALUEwords_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
;metrics

<layers
regularization_losses
trainable_variables
	variables
=non_trainable_variables
>layer_metrics
?layer_regularization_losses
~

3kernel
4recurrent_kernel
5bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
 
 

30
41
52

30
41
52
Ή

Dstates
Emetrics

Flayers
regularization_losses
trainable_variables
	variables
Gnon_trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
 
 
 
­
Jmetrics

Klayers
regularization_losses
trainable_variables
	variables
Lnon_trainable_variables
Mlayer_metrics
Nlayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Ometrics

Players
 regularization_losses
!trainable_variables
"	variables
Qnon_trainable_variables
Rlayer_metrics
Slayer_regularization_losses
 
 
 
­
Tmetrics

Ulayers
$regularization_losses
%trainable_variables
&	variables
Vnon_trainable_variables
Wlayer_metrics
Xlayer_regularization_losses
^\
VARIABLE_VALUEmain_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmain_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­
Ymetrics

Zlayers
*regularization_losses
+trainable_variables
,	variables
[non_trainable_variables
\layer_metrics
]layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEwords_lstm/lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%words_lstm/lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEwords_lstm/lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
v
^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
l14
m15
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 

30
41
52

30
41
52
­
nmetrics

olayers
@regularization_losses
Atrainable_variables
B	variables
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	stotal
	tcount
u	variables
v	keras_api
D
	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api
E
	|total
	}count
~
_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
I

total

 count
‘
_fn_kwargs
’	variables
£	keras_api
I

€total

₯count
¦
_fn_kwargs
§	variables
¨	keras_api
I

©total

ͺcount
«
_fn_kwargs
¬	variables
­	keras_api
I

?total

―count
°
_fn_kwargs
±	variables
²	keras_api
I

³total

΄count
΅
_fn_kwargs
Ά	variables
·	keras_api
I

Έtotal

Ήcount
Ί
_fn_kwargs
»	variables
Ό	keras_api
I

½total

Ύcount
Ώ
_fn_kwargs
ΐ	variables
Α	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

u	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

w0
x1

z	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

’	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

€0
₯1

§	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE
 

©0
ͺ1

¬	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
―1

±	variables
SQ
VARIABLE_VALUEtotal_135keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_135keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUE
 

³0
΄1

Ά	variables
SQ
VARIABLE_VALUEtotal_145keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_145keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUE
 

Έ0
Ή1

»	variables
SQ
VARIABLE_VALUEtotal_155keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_155keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUE
 

½0
Ύ1

ΐ	variables

VARIABLE_VALUE!Adam/words_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/main_output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/words_lstm/lstm_cell/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/words_lstm/lstm_cell/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/words_lstm/lstm_cell/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/words_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/main_output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/words_lstm/lstm_cell/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/words_lstm/lstm_cell/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/words_lstm/lstm_cell/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/words_embedding/embeddings/vhatYlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/main_output/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/main_output/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE%Adam/words_lstm/lstm_cell/kernel/vhatEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/words_lstm/lstm_cell/recurrent_kernel/vhatEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE#Adam/words_lstm/lstm_cell/bias/vhatEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE

%serving_default_layout_features_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_words_inputPlaceholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
ͺ
StatefulPartitionedCallStatefulPartitionedCall%serving_default_layout_features_inputserving_default_words_inputwords_embedding/embeddingswords_lstm/lstm_cell/kernelwords_lstm/lstm_cell/bias%words_lstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasmain_output/kernelmain_output/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_5984168
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ή
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.words_embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp&main_output/kernel/Read/ReadVariableOp$main_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/words_lstm/lstm_cell/kernel/Read/ReadVariableOp9words_lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp-words_lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOptotal_15/Read/ReadVariableOpcount_15/Read/ReadVariableOp5Adam/words_embedding/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp-Adam/main_output/kernel/m/Read/ReadVariableOp+Adam/main_output/bias/m/Read/ReadVariableOp6Adam/words_lstm/lstm_cell/kernel/m/Read/ReadVariableOp@Adam/words_lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp4Adam/words_lstm/lstm_cell/bias/m/Read/ReadVariableOp5Adam/words_embedding/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp-Adam/main_output/kernel/v/Read/ReadVariableOp+Adam/main_output/bias/v/Read/ReadVariableOp6Adam/words_lstm/lstm_cell/kernel/v/Read/ReadVariableOp@Adam/words_lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp4Adam/words_lstm/lstm_cell/bias/v/Read/ReadVariableOp8Adam/words_embedding/embeddings/vhat/Read/ReadVariableOp*Adam/dense/kernel/vhat/Read/ReadVariableOp(Adam/dense/bias/vhat/Read/ReadVariableOp0Adam/main_output/kernel/vhat/Read/ReadVariableOp.Adam/main_output/bias/vhat/Read/ReadVariableOp9Adam/words_lstm/lstm_cell/kernel/vhat/Read/ReadVariableOpCAdam/words_lstm/lstm_cell/recurrent_kernel/vhat/Read/ReadVariableOp7Adam/words_lstm/lstm_cell/bias/vhat/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_5986816
υ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamewords_embedding/embeddingsdense/kernel
dense/biasmain_output/kernelmain_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratewords_lstm/lstm_cell/kernel%words_lstm/lstm_cell/recurrent_kernelwords_lstm/lstm_cell/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10total_11count_11total_12count_12total_13count_13total_14count_14total_15count_15!Adam/words_embedding/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/main_output/kernel/mAdam/main_output/bias/m"Adam/words_lstm/lstm_cell/kernel/m,Adam/words_lstm/lstm_cell/recurrent_kernel/m Adam/words_lstm/lstm_cell/bias/m!Adam/words_embedding/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/main_output/kernel/vAdam/main_output/bias/v"Adam/words_lstm/lstm_cell/kernel/v,Adam/words_lstm/lstm_cell/recurrent_kernel/v Adam/words_lstm/lstm_cell/bias/v$Adam/words_embedding/embeddings/vhatAdam/dense/kernel/vhatAdam/dense/bias/vhatAdam/main_output/kernel/vhatAdam/main_output/bias/vhat%Adam/words_lstm/lstm_cell/kernel/vhat/Adam/words_lstm/lstm_cell/recurrent_kernel/vhat#Adam/words_lstm/lstm_cell/bias/vhat*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_5987033ο'
ͺ
Θ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984117

inputs
inputs_1
words_embedding_5984094
words_lstm_5984097
words_lstm_5984099
words_lstm_5984101
dense_5984105
dense_5984107
main_output_5984111
main_output_5984113
identity’dense/StatefulPartitionedCall’#main_output/StatefulPartitionedCall’'words_embedding/StatefulPartitionedCall’"words_lstm/StatefulPartitionedCall€
'words_embedding/StatefulPartitionedCallStatefulPartitionedCallinputswords_embedding_5984094*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_words_embedding_layer_call_and_return_conditional_losses_59832302)
'words_embedding/StatefulPartitionedCallβ
"words_lstm/StatefulPartitionedCallStatefulPartitionedCall0words_embedding/StatefulPartitionedCall:output:0words_lstm_5984097words_lstm_5984099words_lstm_5984101*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59838792$
"words_lstm/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCall+words_lstm/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_59839162
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_5984105dense_5984107*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_59839362
dense/StatefulPartitionedCallσ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839692
dropout/PartitionedCallΐ
#main_output/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0main_output_5984111main_output_5984113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_59839932%
#main_output/StatefulPartitionedCall
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^main_output/StatefulPartitionedCall(^words_embedding/StatefulPartitionedCall#^words_lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2R
'words_embedding/StatefulPartitionedCall'words_embedding/StatefulPartitionedCall2H
"words_lstm/StatefulPartitionedCall"words_lstm/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
«ύ
ρ
"__inference__wrapped_model_5982440
words_input
layout_features_input9
5functional_1_words_embedding_embedding_lookup_5982166C
?functional_1_words_lstm_lstm_cell_split_readvariableop_resourceE
Afunctional_1_words_lstm_lstm_cell_split_1_readvariableop_resource=
9functional_1_words_lstm_lstm_cell_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource;
7functional_1_main_output_matmul_readvariableop_resource<
8functional_1_main_output_biasadd_readvariableop_resource
identity’functional_1/words_lstm/whileΙ
-functional_1/words_embedding/embedding_lookupResourceGather5functional_1_words_embedding_embedding_lookup_5982166words_input*
Tindices0*H
_class>
<:loc:@functional_1/words_embedding/embedding_lookup/5982166*,
_output_shapes
:?????????d*
dtype02/
-functional_1/words_embedding/embedding_lookup΅
6functional_1/words_embedding/embedding_lookup/IdentityIdentity6functional_1/words_embedding/embedding_lookup:output:0*
T0*H
_class>
<:loc:@functional_1/words_embedding/embedding_lookup/5982166*,
_output_shapes
:?????????d28
6functional_1/words_embedding/embedding_lookup/Identityψ
8functional_1/words_embedding/embedding_lookup/Identity_1Identity?functional_1/words_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????d2:
8functional_1/words_embedding/embedding_lookup/Identity_1―
functional_1/words_lstm/ShapeShapeAfunctional_1/words_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
functional_1/words_lstm/Shape€
+functional_1/words_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/words_lstm/strided_slice/stack¨
-functional_1/words_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/words_lstm/strided_slice/stack_1¨
-functional_1/words_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/words_lstm/strided_slice/stack_2ς
%functional_1/words_lstm/strided_sliceStridedSlice&functional_1/words_lstm/Shape:output:04functional_1/words_lstm/strided_slice/stack:output:06functional_1/words_lstm/strided_slice/stack_1:output:06functional_1/words_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/words_lstm/strided_slice
#functional_1/words_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2%
#functional_1/words_lstm/zeros/mul/yΜ
!functional_1/words_lstm/zeros/mulMul.functional_1/words_lstm/strided_slice:output:0,functional_1/words_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2#
!functional_1/words_lstm/zeros/mul
$functional_1/words_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2&
$functional_1/words_lstm/zeros/Less/yΗ
"functional_1/words_lstm/zeros/LessLess%functional_1/words_lstm/zeros/mul:z:0-functional_1/words_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2$
"functional_1/words_lstm/zeros/Less
&functional_1/words_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&functional_1/words_lstm/zeros/packed/1γ
$functional_1/words_lstm/zeros/packedPack.functional_1/words_lstm/strided_slice:output:0/functional_1/words_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$functional_1/words_lstm/zeros/packed
#functional_1/words_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#functional_1/words_lstm/zeros/ConstΦ
functional_1/words_lstm/zerosFill-functional_1/words_lstm/zeros/packed:output:0,functional_1/words_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
functional_1/words_lstm/zeros
%functional_1/words_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2'
%functional_1/words_lstm/zeros_1/mul/y?
#functional_1/words_lstm/zeros_1/mulMul.functional_1/words_lstm/strided_slice:output:0.functional_1/words_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/words_lstm/zeros_1/mul
&functional_1/words_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2(
&functional_1/words_lstm/zeros_1/Less/yΟ
$functional_1/words_lstm/zeros_1/LessLess'functional_1/words_lstm/zeros_1/mul:z:0/functional_1/words_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2&
$functional_1/words_lstm/zeros_1/Less
(functional_1/words_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2*
(functional_1/words_lstm/zeros_1/packed/1ι
&functional_1/words_lstm/zeros_1/packedPack.functional_1/words_lstm/strided_slice:output:01functional_1/words_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2(
&functional_1/words_lstm/zeros_1/packed
%functional_1/words_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%functional_1/words_lstm/zeros_1/Constή
functional_1/words_lstm/zeros_1Fill/functional_1/words_lstm/zeros_1/packed:output:0.functional_1/words_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2!
functional_1/words_lstm/zeros_1₯
&functional_1/words_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&functional_1/words_lstm/transpose/permώ
!functional_1/words_lstm/transpose	TransposeAfunctional_1/words_embedding/embedding_lookup/Identity_1:output:0/functional_1/words_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:d?????????2#
!functional_1/words_lstm/transpose
functional_1/words_lstm/Shape_1Shape%functional_1/words_lstm/transpose:y:0*
T0*
_output_shapes
:2!
functional_1/words_lstm/Shape_1¨
-functional_1/words_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/words_lstm/strided_slice_1/stack¬
/functional_1/words_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/words_lstm/strided_slice_1/stack_1¬
/functional_1/words_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/words_lstm/strided_slice_1/stack_2ώ
'functional_1/words_lstm/strided_slice_1StridedSlice(functional_1/words_lstm/Shape_1:output:06functional_1/words_lstm/strided_slice_1/stack:output:08functional_1/words_lstm/strided_slice_1/stack_1:output:08functional_1/words_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/words_lstm/strided_slice_1΅
3functional_1/words_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3functional_1/words_lstm/TensorArrayV2/element_shape
%functional_1/words_lstm/TensorArrayV2TensorListReserve<functional_1/words_lstm/TensorArrayV2/element_shape:output:00functional_1/words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/words_lstm/TensorArrayV2ο
Mfunctional_1/words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2O
Mfunctional_1/words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeΨ
?functional_1/words_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%functional_1/words_lstm/transpose:y:0Vfunctional_1/words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?functional_1/words_lstm/TensorArrayUnstack/TensorListFromTensor¨
-functional_1/words_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/words_lstm/strided_slice_2/stack¬
/functional_1/words_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/words_lstm/strided_slice_2/stack_1¬
/functional_1/words_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/words_lstm/strided_slice_2/stack_2
'functional_1/words_lstm/strided_slice_2StridedSlice%functional_1/words_lstm/transpose:y:06functional_1/words_lstm/strided_slice_2/stack:output:08functional_1/words_lstm/strided_slice_2/stack_1:output:08functional_1/words_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2)
'functional_1/words_lstm/strided_slice_2Ζ
1functional_1/words_lstm/lstm_cell/ones_like/ShapeShape0functional_1/words_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:23
1functional_1/words_lstm/lstm_cell/ones_like/Shape«
1functional_1/words_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1functional_1/words_lstm/lstm_cell/ones_like/Const
+functional_1/words_lstm/lstm_cell/ones_likeFill:functional_1/words_lstm/lstm_cell/ones_like/Shape:output:0:functional_1/words_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/ones_likeΐ
3functional_1/words_lstm/lstm_cell/ones_like_1/ShapeShape&functional_1/words_lstm/zeros:output:0*
T0*
_output_shapes
:25
3functional_1/words_lstm/lstm_cell/ones_like_1/Shape―
3functional_1/words_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3functional_1/words_lstm/lstm_cell/ones_like_1/Const
-functional_1/words_lstm/lstm_cell/ones_like_1Fill<functional_1/words_lstm/lstm_cell/ones_like_1/Shape:output:0<functional_1/words_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/lstm_cell/ones_like_1π
%functional_1/words_lstm/lstm_cell/mulMul0functional_1/words_lstm/strided_slice_2:output:04functional_1/words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2'
%functional_1/words_lstm/lstm_cell/mulτ
'functional_1/words_lstm/lstm_cell/mul_1Mul0functional_1/words_lstm/strided_slice_2:output:04functional_1/words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_1τ
'functional_1/words_lstm/lstm_cell/mul_2Mul0functional_1/words_lstm/strided_slice_2:output:04functional_1/words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_2τ
'functional_1/words_lstm/lstm_cell/mul_3Mul0functional_1/words_lstm/strided_slice_2:output:04functional_1/words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_3
'functional_1/words_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'functional_1/words_lstm/lstm_cell/Const¨
1functional_1/words_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1functional_1/words_lstm/lstm_cell/split/split_dimς
6functional_1/words_lstm/lstm_cell/split/ReadVariableOpReadVariableOp?functional_1_words_lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype028
6functional_1/words_lstm/lstm_cell/split/ReadVariableOp·
'functional_1/words_lstm/lstm_cell/splitSplit:functional_1/words_lstm/lstm_cell/split/split_dim:output:0>functional_1/words_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2)
'functional_1/words_lstm/lstm_cell/splitξ
(functional_1/words_lstm/lstm_cell/MatMulMatMul)functional_1/words_lstm/lstm_cell/mul:z:00functional_1/words_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2*
(functional_1/words_lstm/lstm_cell/MatMulτ
*functional_1/words_lstm/lstm_cell/MatMul_1MatMul+functional_1/words_lstm/lstm_cell/mul_1:z:00functional_1/words_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_1τ
*functional_1/words_lstm/lstm_cell/MatMul_2MatMul+functional_1/words_lstm/lstm_cell/mul_2:z:00functional_1/words_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_2τ
*functional_1/words_lstm/lstm_cell/MatMul_3MatMul+functional_1/words_lstm/lstm_cell/mul_3:z:00functional_1/words_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_3
)functional_1/words_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/words_lstm/lstm_cell/Const_1¬
3functional_1/words_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/words_lstm/lstm_cell/split_1/split_dimσ
8functional_1/words_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOpAfunctional_1_words_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/words_lstm/lstm_cell/split_1/ReadVariableOp«
)functional_1/words_lstm/lstm_cell/split_1Split<functional_1/words_lstm/lstm_cell/split_1/split_dim:output:0@functional_1/words_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2+
)functional_1/words_lstm/lstm_cell/split_1ό
)functional_1/words_lstm/lstm_cell/BiasAddBiasAdd2functional_1/words_lstm/lstm_cell/MatMul:product:02functional_1/words_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2+
)functional_1/words_lstm/lstm_cell/BiasAdd
+functional_1/words_lstm/lstm_cell/BiasAdd_1BiasAdd4functional_1/words_lstm/lstm_cell/MatMul_1:product:02functional_1/words_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/BiasAdd_1
+functional_1/words_lstm/lstm_cell/BiasAdd_2BiasAdd4functional_1/words_lstm/lstm_cell/MatMul_2:product:02functional_1/words_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/BiasAdd_2
+functional_1/words_lstm/lstm_cell/BiasAdd_3BiasAdd4functional_1/words_lstm/lstm_cell/MatMul_3:product:02functional_1/words_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/BiasAdd_3μ
'functional_1/words_lstm/lstm_cell/mul_4Mul&functional_1/words_lstm/zeros:output:06functional_1/words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_4μ
'functional_1/words_lstm/lstm_cell/mul_5Mul&functional_1/words_lstm/zeros:output:06functional_1/words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_5μ
'functional_1/words_lstm/lstm_cell/mul_6Mul&functional_1/words_lstm/zeros:output:06functional_1/words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_6μ
'functional_1/words_lstm/lstm_cell/mul_7Mul&functional_1/words_lstm/zeros:output:06functional_1/words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_7ΰ
0functional_1/words_lstm/lstm_cell/ReadVariableOpReadVariableOp9functional_1_words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype022
0functional_1/words_lstm/lstm_cell/ReadVariableOpΏ
5functional_1/words_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5functional_1/words_lstm/lstm_cell/strided_slice/stackΓ
7functional_1/words_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7functional_1/words_lstm/lstm_cell/strided_slice/stack_1Γ
7functional_1/words_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/words_lstm/lstm_cell/strided_slice/stack_2Κ
/functional_1/words_lstm/lstm_cell/strided_sliceStridedSlice8functional_1/words_lstm/lstm_cell/ReadVariableOp:value:0>functional_1/words_lstm/lstm_cell/strided_slice/stack:output:0@functional_1/words_lstm/lstm_cell/strided_slice/stack_1:output:0@functional_1/words_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask21
/functional_1/words_lstm/lstm_cell/strided_sliceό
*functional_1/words_lstm/lstm_cell/MatMul_4MatMul+functional_1/words_lstm/lstm_cell/mul_4:z:08functional_1/words_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_4τ
%functional_1/words_lstm/lstm_cell/addAddV22functional_1/words_lstm/lstm_cell/BiasAdd:output:04functional_1/words_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2'
%functional_1/words_lstm/lstm_cell/addΏ
)functional_1/words_lstm/lstm_cell/SigmoidSigmoid)functional_1/words_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2+
)functional_1/words_lstm/lstm_cell/Sigmoidδ
2functional_1/words_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp9functional_1_words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype024
2functional_1/words_lstm/lstm_cell/ReadVariableOp_1Γ
7functional_1/words_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7functional_1/words_lstm/lstm_cell/strided_slice_1/stackΗ
9functional_1/words_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9functional_1/words_lstm/lstm_cell/strided_slice_1/stack_1Η
9functional_1/words_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9functional_1/words_lstm/lstm_cell/strided_slice_1/stack_2Φ
1functional_1/words_lstm/lstm_cell/strided_slice_1StridedSlice:functional_1/words_lstm/lstm_cell/ReadVariableOp_1:value:0@functional_1/words_lstm/lstm_cell/strided_slice_1/stack:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_1/stack_1:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask23
1functional_1/words_lstm/lstm_cell/strided_slice_1ώ
*functional_1/words_lstm/lstm_cell/MatMul_5MatMul+functional_1/words_lstm/lstm_cell/mul_5:z:0:functional_1/words_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_5ϊ
'functional_1/words_lstm/lstm_cell/add_1AddV24functional_1/words_lstm/lstm_cell/BiasAdd_1:output:04functional_1/words_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/add_1Ε
+functional_1/words_lstm/lstm_cell/Sigmoid_1Sigmoid+functional_1/words_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/Sigmoid_1η
'functional_1/words_lstm/lstm_cell/mul_8Mul/functional_1/words_lstm/lstm_cell/Sigmoid_1:y:0(functional_1/words_lstm/zeros_1:output:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_8δ
2functional_1/words_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp9functional_1_words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype024
2functional_1/words_lstm/lstm_cell/ReadVariableOp_2Γ
7functional_1/words_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7functional_1/words_lstm/lstm_cell/strided_slice_2/stackΗ
9functional_1/words_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9functional_1/words_lstm/lstm_cell/strided_slice_2/stack_1Η
9functional_1/words_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9functional_1/words_lstm/lstm_cell/strided_slice_2/stack_2Φ
1functional_1/words_lstm/lstm_cell/strided_slice_2StridedSlice:functional_1/words_lstm/lstm_cell/ReadVariableOp_2:value:0@functional_1/words_lstm/lstm_cell/strided_slice_2/stack:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_2/stack_1:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask23
1functional_1/words_lstm/lstm_cell/strided_slice_2ώ
*functional_1/words_lstm/lstm_cell/MatMul_6MatMul+functional_1/words_lstm/lstm_cell/mul_6:z:0:functional_1/words_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_6ϊ
'functional_1/words_lstm/lstm_cell/add_2AddV24functional_1/words_lstm/lstm_cell/BiasAdd_2:output:04functional_1/words_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/add_2Έ
&functional_1/words_lstm/lstm_cell/TanhTanh+functional_1/words_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2(
&functional_1/words_lstm/lstm_cell/Tanhη
'functional_1/words_lstm/lstm_cell/mul_9Mul-functional_1/words_lstm/lstm_cell/Sigmoid:y:0*functional_1/words_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/mul_9θ
'functional_1/words_lstm/lstm_cell/add_3AddV2+functional_1/words_lstm/lstm_cell/mul_8:z:0+functional_1/words_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/add_3δ
2functional_1/words_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp9functional_1_words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype024
2functional_1/words_lstm/lstm_cell/ReadVariableOp_3Γ
7functional_1/words_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       29
7functional_1/words_lstm/lstm_cell/strided_slice_3/stackΗ
9functional_1/words_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9functional_1/words_lstm/lstm_cell/strided_slice_3/stack_1Η
9functional_1/words_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9functional_1/words_lstm/lstm_cell/strided_slice_3/stack_2Φ
1functional_1/words_lstm/lstm_cell/strided_slice_3StridedSlice:functional_1/words_lstm/lstm_cell/ReadVariableOp_3:value:0@functional_1/words_lstm/lstm_cell/strided_slice_3/stack:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_3/stack_1:output:0Bfunctional_1/words_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask23
1functional_1/words_lstm/lstm_cell/strided_slice_3ώ
*functional_1/words_lstm/lstm_cell/MatMul_7MatMul+functional_1/words_lstm/lstm_cell/mul_7:z:0:functional_1/words_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2,
*functional_1/words_lstm/lstm_cell/MatMul_7ϊ
'functional_1/words_lstm/lstm_cell/add_4AddV24functional_1/words_lstm/lstm_cell/BiasAdd_3:output:04functional_1/words_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2)
'functional_1/words_lstm/lstm_cell/add_4Ε
+functional_1/words_lstm/lstm_cell/Sigmoid_2Sigmoid+functional_1/words_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/lstm_cell/Sigmoid_2Ό
(functional_1/words_lstm/lstm_cell/Tanh_1Tanh+functional_1/words_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2*
(functional_1/words_lstm/lstm_cell/Tanh_1ν
(functional_1/words_lstm/lstm_cell/mul_10Mul/functional_1/words_lstm/lstm_cell/Sigmoid_2:y:0,functional_1/words_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2*
(functional_1/words_lstm/lstm_cell/mul_10Ώ
5functional_1/words_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5functional_1/words_lstm/TensorArrayV2_1/element_shape
'functional_1/words_lstm/TensorArrayV2_1TensorListReserve>functional_1/words_lstm/TensorArrayV2_1/element_shape:output:00functional_1/words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'functional_1/words_lstm/TensorArrayV2_1~
functional_1/words_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
functional_1/words_lstm/time―
0functional_1/words_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0functional_1/words_lstm/while/maximum_iterations
*functional_1/words_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/words_lstm/while/loop_counterΙ
functional_1/words_lstm/whileWhile3functional_1/words_lstm/while/loop_counter:output:09functional_1/words_lstm/while/maximum_iterations:output:0%functional_1/words_lstm/time:output:00functional_1/words_lstm/TensorArrayV2_1:handle:0&functional_1/words_lstm/zeros:output:0(functional_1/words_lstm/zeros_1:output:00functional_1/words_lstm/strided_slice_1:output:0Ofunctional_1/words_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0?functional_1_words_lstm_lstm_cell_split_readvariableop_resourceAfunctional_1_words_lstm_lstm_cell_split_1_readvariableop_resource9functional_1_words_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*6
body.R,
*functional_1_words_lstm_while_body_5982287*6
cond.R,
*functional_1_words_lstm_while_cond_5982286*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
functional_1/words_lstm/whileε
Hfunctional_1/words_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hfunctional_1/words_lstm/TensorArrayV2Stack/TensorListStack/element_shapeΙ
:functional_1/words_lstm/TensorArrayV2Stack/TensorListStackTensorListStack&functional_1/words_lstm/while:output:3Qfunctional_1/words_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02<
:functional_1/words_lstm/TensorArrayV2Stack/TensorListStack±
-functional_1/words_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-functional_1/words_lstm/strided_slice_3/stack¬
/functional_1/words_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/words_lstm/strided_slice_3/stack_1¬
/functional_1/words_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/words_lstm/strided_slice_3/stack_2«
'functional_1/words_lstm/strided_slice_3StridedSliceCfunctional_1/words_lstm/TensorArrayV2Stack/TensorListStack:tensor:06functional_1/words_lstm/strided_slice_3/stack:output:08functional_1/words_lstm/strided_slice_3/stack_1:output:08functional_1/words_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2)
'functional_1/words_lstm/strided_slice_3©
(functional_1/words_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2*
(functional_1/words_lstm/transpose_1/perm
#functional_1/words_lstm/transpose_1	TransposeCfunctional_1/words_lstm/TensorArrayV2Stack/TensorListStack:tensor:01functional_1/words_lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2%
#functional_1/words_lstm/transpose_1
functional_1/words_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2!
functional_1/words_lstm/runtime
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis
functional_1/concatenate/concatConcatV20functional_1/words_lstm/strided_slice_3:output:0layout_features_input-functional_1/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????2!
functional_1/concatenate/concatΘ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpΟ
functional_1/dense/MatMulMatMul(functional_1/concatenate/concat:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
functional_1/dense/MatMulΖ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpΞ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
functional_1/dense/BiasAdd
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
functional_1/dense/Relu€
functional_1/dropout/IdentityIdentity%functional_1/dense/Relu:activations:0*
T0*(
_output_shapes
:?????????2
functional_1/dropout/IdentityΩ
.functional_1/main_output/MatMul/ReadVariableOpReadVariableOp7functional_1_main_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype020
.functional_1/main_output/MatMul/ReadVariableOpή
functional_1/main_output/MatMulMatMul&functional_1/dropout/Identity:output:06functional_1/main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
functional_1/main_output/MatMulΧ
/functional_1/main_output/BiasAdd/ReadVariableOpReadVariableOp8functional_1_main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/functional_1/main_output/BiasAdd/ReadVariableOpε
 functional_1/main_output/BiasAddBiasAdd)functional_1/main_output/MatMul:product:07functional_1/main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 functional_1/main_output/BiasAdd¬
 functional_1/main_output/SoftmaxSoftmax)functional_1/main_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 functional_1/main_output/Softmax
IdentityIdentity*functional_1/main_output/Softmax:softmax:0^functional_1/words_lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2>
functional_1/words_lstm/whilefunctional_1/words_lstm/while:T P
'
_output_shapes
:?????????d
%
_user_specified_namewords_input:^Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input
΄
Θ
while_cond_5985420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5985420___redundant_placeholder05
1while_while_cond_5985420___redundant_placeholder15
1while_while_cond_5985420___redundant_placeholder25
1while_while_cond_5985420___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
¬	
ν
.__inference_functional_1_layer_call_fn_5984903
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity’StatefulPartitionedCallΤ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_59841172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Γ
Λ
+__inference_lstm_cell_layer_call_fn_5986568

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59826282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/1
Φ
w
1__inference_words_embedding_layer_call_fn_5984919

inputs
unknown
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_words_embedding_layer_call_and_return_conditional_losses_59832302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
°
ͺ
B__inference_dense_layer_call_and_return_conditional_losses_5986263

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

E
)__inference_dropout_layer_call_fn_5986299

inputs
identityΓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
’Α

I__inference_functional_1_layer_call_and_return_conditional_losses_5984581
inputs_0
inputs_1,
(words_embedding_embedding_lookup_59841726
2words_lstm_lstm_cell_split_readvariableop_resource8
4words_lstm_lstm_cell_split_1_readvariableop_resource0
,words_lstm_lstm_cell_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*main_output_matmul_readvariableop_resource/
+main_output_biasadd_readvariableop_resource
identity’words_lstm/while
 words_embedding/embedding_lookupResourceGather(words_embedding_embedding_lookup_5984172inputs_0*
Tindices0*;
_class1
/-loc:@words_embedding/embedding_lookup/5984172*,
_output_shapes
:?????????d*
dtype02"
 words_embedding/embedding_lookup
)words_embedding/embedding_lookup/IdentityIdentity)words_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@words_embedding/embedding_lookup/5984172*,
_output_shapes
:?????????d2+
)words_embedding/embedding_lookup/IdentityΡ
+words_embedding/embedding_lookup/Identity_1Identity2words_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????d2-
+words_embedding/embedding_lookup/Identity_1
words_lstm/ShapeShape4words_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
words_lstm/Shape
words_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
words_lstm/strided_slice/stack
 words_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 words_lstm/strided_slice/stack_1
 words_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 words_lstm/strided_slice/stack_2€
words_lstm/strided_sliceStridedSlicewords_lstm/Shape:output:0'words_lstm/strided_slice/stack:output:0)words_lstm/strided_slice/stack_1:output:0)words_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
words_lstm/strided_slices
words_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros/mul/y
words_lstm/zeros/mulMul!words_lstm/strided_slice:output:0words_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros/mulu
words_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
words_lstm/zeros/Less/y
words_lstm/zeros/LessLesswords_lstm/zeros/mul:z:0 words_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros/Lessy
words_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros/packed/1―
words_lstm/zeros/packedPack!words_lstm/strided_slice:output:0"words_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
words_lstm/zeros/packedu
words_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/zeros/Const’
words_lstm/zerosFill words_lstm/zeros/packed:output:0words_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/zerosw
words_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros_1/mul/y
words_lstm/zeros_1/mulMul!words_lstm/strided_slice:output:0!words_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros_1/muly
words_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
words_lstm/zeros_1/Less/y
words_lstm/zeros_1/LessLesswords_lstm/zeros_1/mul:z:0"words_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros_1/Less}
words_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros_1/packed/1΅
words_lstm/zeros_1/packedPack!words_lstm/strided_slice:output:0$words_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
words_lstm/zeros_1/packedy
words_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/zeros_1/Constͺ
words_lstm/zeros_1Fill"words_lstm/zeros_1/packed:output:0!words_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/zeros_1
words_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
words_lstm/transpose/permΚ
words_lstm/transpose	Transpose4words_embedding/embedding_lookup/Identity_1:output:0"words_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:d?????????2
words_lstm/transposep
words_lstm/Shape_1Shapewords_lstm/transpose:y:0*
T0*
_output_shapes
:2
words_lstm/Shape_1
 words_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 words_lstm/strided_slice_1/stack
"words_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_1/stack_1
"words_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_1/stack_2°
words_lstm/strided_slice_1StridedSlicewords_lstm/Shape_1:output:0)words_lstm/strided_slice_1/stack:output:0+words_lstm/strided_slice_1/stack_1:output:0+words_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
words_lstm/strided_slice_1
&words_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&words_lstm/TensorArrayV2/element_shapeή
words_lstm/TensorArrayV2TensorListReserve/words_lstm/TensorArrayV2/element_shape:output:0#words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
words_lstm/TensorArrayV2Υ
@words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape€
2words_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorwords_lstm/transpose:y:0Iwords_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2words_lstm/TensorArrayUnstack/TensorListFromTensor
 words_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 words_lstm/strided_slice_2/stack
"words_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_2/stack_1
"words_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_2/stack_2Ώ
words_lstm/strided_slice_2StridedSlicewords_lstm/transpose:y:0)words_lstm/strided_slice_2/stack:output:0+words_lstm/strided_slice_2/stack_1:output:0+words_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
words_lstm/strided_slice_2
$words_lstm/lstm_cell/ones_like/ShapeShape#words_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/ones_like/Shape
$words_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$words_lstm/lstm_cell/ones_like/ConstΩ
words_lstm/lstm_cell/ones_likeFill-words_lstm/lstm_cell/ones_like/Shape:output:0-words_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/ones_like
"words_lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"words_lstm/lstm_cell/dropout/ConstΤ
 words_lstm/lstm_cell/dropout/MulMul'words_lstm/lstm_cell/ones_like:output:0+words_lstm/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/lstm_cell/dropout/Mul
"words_lstm/lstm_cell/dropout/ShapeShape'words_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"words_lstm/lstm_cell/dropout/Shape
9words_lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform+words_lstm/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ώΑ«2;
9words_lstm/lstm_cell/dropout/random_uniform/RandomUniform
+words_lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2-
+words_lstm/lstm_cell/dropout/GreaterEqual/y
)words_lstm/lstm_cell/dropout/GreaterEqualGreaterEqualBwords_lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:04words_lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2+
)words_lstm/lstm_cell/dropout/GreaterEqualΏ
!words_lstm/lstm_cell/dropout/CastCast-words_lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2#
!words_lstm/lstm_cell/dropout/CastΟ
"words_lstm/lstm_cell/dropout/Mul_1Mul$words_lstm/lstm_cell/dropout/Mul:z:0%words_lstm/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout/Mul_1
$words_lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_1/ConstΪ
"words_lstm/lstm_cell/dropout_1/MulMul'words_lstm/lstm_cell/ones_like:output:0-words_lstm/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_1/Mul£
$words_lstm/lstm_cell/dropout_1/ShapeShape'words_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_1/Shape
;words_lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Σ₯Ϊ2=
;words_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_1/GreaterEqual/y
+words_lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_1/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_1/CastCast/words_lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_1/CastΧ
$words_lstm/lstm_cell/dropout_1/Mul_1Mul&words_lstm/lstm_cell/dropout_1/Mul:z:0'words_lstm/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_1/Mul_1
$words_lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_2/ConstΪ
"words_lstm/lstm_cell/dropout_2/MulMul'words_lstm/lstm_cell/ones_like:output:0-words_lstm/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_2/Mul£
$words_lstm/lstm_cell/dropout_2/ShapeShape'words_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_2/Shape
;words_lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ξ§ι2=
;words_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_2/GreaterEqual/y
+words_lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_2/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_2/CastCast/words_lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_2/CastΧ
$words_lstm/lstm_cell/dropout_2/Mul_1Mul&words_lstm/lstm_cell/dropout_2/Mul:z:0'words_lstm/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_2/Mul_1
$words_lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_3/ConstΪ
"words_lstm/lstm_cell/dropout_3/MulMul'words_lstm/lstm_cell/ones_like:output:0-words_lstm/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_3/Mul£
$words_lstm/lstm_cell/dropout_3/ShapeShape'words_lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_3/Shape
;words_lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2―2=
;words_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_3/GreaterEqual/y
+words_lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_3/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_3/CastCast/words_lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_3/CastΧ
$words_lstm/lstm_cell/dropout_3/Mul_1Mul&words_lstm/lstm_cell/dropout_3/Mul:z:0'words_lstm/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_3/Mul_1
&words_lstm/lstm_cell/ones_like_1/ShapeShapewords_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&words_lstm/lstm_cell/ones_like_1/Shape
&words_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&words_lstm/lstm_cell/ones_like_1/Constα
 words_lstm/lstm_cell/ones_like_1Fill/words_lstm/lstm_cell/ones_like_1/Shape:output:0/words_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/lstm_cell/ones_like_1
$words_lstm/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_4/Constά
"words_lstm/lstm_cell/dropout_4/MulMul)words_lstm/lstm_cell/ones_like_1:output:0-words_lstm/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_4/Mul₯
$words_lstm/lstm_cell/dropout_4/ShapeShape)words_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_4/Shape
;words_lstm/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Η2=
;words_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_4/GreaterEqual/y
+words_lstm/lstm_cell/dropout_4/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_4/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_4/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_4/CastCast/words_lstm/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_4/CastΧ
$words_lstm/lstm_cell/dropout_4/Mul_1Mul&words_lstm/lstm_cell/dropout_4/Mul:z:0'words_lstm/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_4/Mul_1
$words_lstm/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_5/Constά
"words_lstm/lstm_cell/dropout_5/MulMul)words_lstm/lstm_cell/ones_like_1:output:0-words_lstm/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_5/Mul₯
$words_lstm/lstm_cell/dropout_5/ShapeShape)words_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_5/Shape
;words_lstm/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ύ?­2=
;words_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_5/GreaterEqual/y
+words_lstm/lstm_cell/dropout_5/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_5/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_5/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_5/CastCast/words_lstm/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_5/CastΧ
$words_lstm/lstm_cell/dropout_5/Mul_1Mul&words_lstm/lstm_cell/dropout_5/Mul:z:0'words_lstm/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_5/Mul_1
$words_lstm/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_6/Constά
"words_lstm/lstm_cell/dropout_6/MulMul)words_lstm/lstm_cell/ones_like_1:output:0-words_lstm/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_6/Mul₯
$words_lstm/lstm_cell/dropout_6/ShapeShape)words_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_6/Shape
;words_lstm/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed22=
;words_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_6/GreaterEqual/y
+words_lstm/lstm_cell/dropout_6/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_6/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_6/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_6/CastCast/words_lstm/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_6/CastΧ
$words_lstm/lstm_cell/dropout_6/Mul_1Mul&words_lstm/lstm_cell/dropout_6/Mul:z:0'words_lstm/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_6/Mul_1
$words_lstm/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$words_lstm/lstm_cell/dropout_7/Constά
"words_lstm/lstm_cell/dropout_7/MulMul)words_lstm/lstm_cell/ones_like_1:output:0-words_lstm/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/lstm_cell/dropout_7/Mul₯
$words_lstm/lstm_cell/dropout_7/ShapeShape)words_lstm/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/dropout_7/Shape
;words_lstm/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform-words_lstm/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2΅Ά2=
;words_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform£
-words_lstm/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2/
-words_lstm/lstm_cell/dropout_7/GreaterEqual/y
+words_lstm/lstm_cell/dropout_7/GreaterEqualGreaterEqualDwords_lstm/lstm_cell/dropout_7/random_uniform/RandomUniform:output:06words_lstm/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2-
+words_lstm/lstm_cell/dropout_7/GreaterEqualΕ
#words_lstm/lstm_cell/dropout_7/CastCast/words_lstm/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2%
#words_lstm/lstm_cell/dropout_7/CastΧ
$words_lstm/lstm_cell/dropout_7/Mul_1Mul&words_lstm/lstm_cell/dropout_7/Mul:z:0'words_lstm/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/lstm_cell/dropout_7/Mul_1»
words_lstm/lstm_cell/mulMul#words_lstm/strided_slice_2:output:0&words_lstm/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mulΑ
words_lstm/lstm_cell/mul_1Mul#words_lstm/strided_slice_2:output:0(words_lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_1Α
words_lstm/lstm_cell/mul_2Mul#words_lstm/strided_slice_2:output:0(words_lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_2Α
words_lstm/lstm_cell/mul_3Mul#words_lstm/strided_slice_2:output:0(words_lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_3z
words_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/lstm_cell/Const
$words_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$words_lstm/lstm_cell/split/split_dimΛ
)words_lstm/lstm_cell/split/ReadVariableOpReadVariableOp2words_lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)words_lstm/lstm_cell/split/ReadVariableOp
words_lstm/lstm_cell/splitSplit-words_lstm/lstm_cell/split/split_dim:output:01words_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
words_lstm/lstm_cell/splitΊ
words_lstm/lstm_cell/MatMulMatMulwords_lstm/lstm_cell/mul:z:0#words_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMulΐ
words_lstm/lstm_cell/MatMul_1MatMulwords_lstm/lstm_cell/mul_1:z:0#words_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_1ΐ
words_lstm/lstm_cell/MatMul_2MatMulwords_lstm/lstm_cell/mul_2:z:0#words_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_2ΐ
words_lstm/lstm_cell/MatMul_3MatMulwords_lstm/lstm_cell/mul_3:z:0#words_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_3~
words_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/lstm_cell/Const_1
&words_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&words_lstm/lstm_cell/split_1/split_dimΜ
+words_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp4words_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+words_lstm/lstm_cell/split_1/ReadVariableOpχ
words_lstm/lstm_cell/split_1Split/words_lstm/lstm_cell/split_1/split_dim:output:03words_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
words_lstm/lstm_cell/split_1Θ
words_lstm/lstm_cell/BiasAddBiasAdd%words_lstm/lstm_cell/MatMul:product:0%words_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/BiasAddΞ
words_lstm/lstm_cell/BiasAdd_1BiasAdd'words_lstm/lstm_cell/MatMul_1:product:0%words_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_1Ξ
words_lstm/lstm_cell/BiasAdd_2BiasAdd'words_lstm/lstm_cell/MatMul_2:product:0%words_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_2Ξ
words_lstm/lstm_cell/BiasAdd_3BiasAdd'words_lstm/lstm_cell/MatMul_3:product:0%words_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_3·
words_lstm/lstm_cell/mul_4Mulwords_lstm/zeros:output:0(words_lstm/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_4·
words_lstm/lstm_cell/mul_5Mulwords_lstm/zeros:output:0(words_lstm/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_5·
words_lstm/lstm_cell/mul_6Mulwords_lstm/zeros:output:0(words_lstm/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_6·
words_lstm/lstm_cell/mul_7Mulwords_lstm/zeros:output:0(words_lstm/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_7Ή
#words_lstm/lstm_cell/ReadVariableOpReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#words_lstm/lstm_cell/ReadVariableOp₯
(words_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(words_lstm/lstm_cell/strided_slice/stack©
*words_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice/stack_1©
*words_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*words_lstm/lstm_cell/strided_slice/stack_2ό
"words_lstm/lstm_cell/strided_sliceStridedSlice+words_lstm/lstm_cell/ReadVariableOp:value:01words_lstm/lstm_cell/strided_slice/stack:output:03words_lstm/lstm_cell/strided_slice/stack_1:output:03words_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"words_lstm/lstm_cell/strided_sliceΘ
words_lstm/lstm_cell/MatMul_4MatMulwords_lstm/lstm_cell/mul_4:z:0+words_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_4ΐ
words_lstm/lstm_cell/addAddV2%words_lstm/lstm_cell/BiasAdd:output:0'words_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add
words_lstm/lstm_cell/SigmoidSigmoidwords_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Sigmoid½
%words_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_1©
*words_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_1/stack­
,words_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,words_lstm/lstm_cell/strided_slice_1/stack_1­
,words_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_1/stack_2
$words_lstm/lstm_cell/strided_slice_1StridedSlice-words_lstm/lstm_cell/ReadVariableOp_1:value:03words_lstm/lstm_cell/strided_slice_1/stack:output:05words_lstm/lstm_cell/strided_slice_1/stack_1:output:05words_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_1Κ
words_lstm/lstm_cell/MatMul_5MatMulwords_lstm/lstm_cell/mul_5:z:0-words_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_5Ζ
words_lstm/lstm_cell/add_1AddV2'words_lstm/lstm_cell/BiasAdd_1:output:0'words_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_1
words_lstm/lstm_cell/Sigmoid_1Sigmoidwords_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/Sigmoid_1³
words_lstm/lstm_cell/mul_8Mul"words_lstm/lstm_cell/Sigmoid_1:y:0words_lstm/zeros_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_8½
%words_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_2©
*words_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_2/stack­
,words_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,words_lstm/lstm_cell/strided_slice_2/stack_1­
,words_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_2/stack_2
$words_lstm/lstm_cell/strided_slice_2StridedSlice-words_lstm/lstm_cell/ReadVariableOp_2:value:03words_lstm/lstm_cell/strided_slice_2/stack:output:05words_lstm/lstm_cell/strided_slice_2/stack_1:output:05words_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_2Κ
words_lstm/lstm_cell/MatMul_6MatMulwords_lstm/lstm_cell/mul_6:z:0-words_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_6Ζ
words_lstm/lstm_cell/add_2AddV2'words_lstm/lstm_cell/BiasAdd_2:output:0'words_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_2
words_lstm/lstm_cell/TanhTanhwords_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Tanh³
words_lstm/lstm_cell/mul_9Mul words_lstm/lstm_cell/Sigmoid:y:0words_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_9΄
words_lstm/lstm_cell/add_3AddV2words_lstm/lstm_cell/mul_8:z:0words_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_3½
%words_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_3©
*words_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_3/stack­
,words_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,words_lstm/lstm_cell/strided_slice_3/stack_1­
,words_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_3/stack_2
$words_lstm/lstm_cell/strided_slice_3StridedSlice-words_lstm/lstm_cell/ReadVariableOp_3:value:03words_lstm/lstm_cell/strided_slice_3/stack:output:05words_lstm/lstm_cell/strided_slice_3/stack_1:output:05words_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_3Κ
words_lstm/lstm_cell/MatMul_7MatMulwords_lstm/lstm_cell/mul_7:z:0-words_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_7Ζ
words_lstm/lstm_cell/add_4AddV2'words_lstm/lstm_cell/BiasAdd_3:output:0'words_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_4
words_lstm/lstm_cell/Sigmoid_2Sigmoidwords_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/Sigmoid_2
words_lstm/lstm_cell/Tanh_1Tanhwords_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Tanh_1Ή
words_lstm/lstm_cell/mul_10Mul"words_lstm/lstm_cell/Sigmoid_2:y:0words_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_10₯
(words_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(words_lstm/TensorArrayV2_1/element_shapeδ
words_lstm/TensorArrayV2_1TensorListReserve1words_lstm/TensorArrayV2_1/element_shape:output:0#words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
words_lstm/TensorArrayV2_1d
words_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
words_lstm/time
#words_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#words_lstm/while/maximum_iterations
words_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
words_lstm/while/loop_counter
words_lstm/whileWhile&words_lstm/while/loop_counter:output:0,words_lstm/while/maximum_iterations:output:0words_lstm/time:output:0#words_lstm/TensorArrayV2_1:handle:0words_lstm/zeros:output:0words_lstm/zeros_1:output:0#words_lstm/strided_slice_1:output:0Bwords_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02words_lstm_lstm_cell_split_readvariableop_resource4words_lstm_lstm_cell_split_1_readvariableop_resource,words_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*)
body!R
words_lstm_while_body_5984357*)
cond!R
words_lstm_while_cond_5984356*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
words_lstm/whileΛ
;words_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;words_lstm/TensorArrayV2Stack/TensorListStack/element_shape
-words_lstm/TensorArrayV2Stack/TensorListStackTensorListStackwords_lstm/while:output:3Dwords_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02/
-words_lstm/TensorArrayV2Stack/TensorListStack
 words_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 words_lstm/strided_slice_3/stack
"words_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"words_lstm/strided_slice_3/stack_1
"words_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_3/stack_2έ
words_lstm/strided_slice_3StridedSlice6words_lstm/TensorArrayV2Stack/TensorListStack:tensor:0)words_lstm/strided_slice_3/stack:output:0+words_lstm/strided_slice_3/stack_1:output:0+words_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
words_lstm/strided_slice_3
words_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
words_lstm/transpose_1/perm?
words_lstm/transpose_1	Transpose6words_lstm/TensorArrayV2Stack/TensorListStack:tensor:0$words_lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
words_lstm/transpose_1|
words_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/runtimet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisΑ
concatenate/concatConcatV2#words_lstm/strided_slice_3:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????2
concatenate/concat‘
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeΝ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2 
dropout/dropout/GreaterEqual/yί
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Mul_1²
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!main_output/MatMul/ReadVariableOpͺ
main_output/MatMulMatMuldropout/dropout/Mul_1:z:0)main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/MatMul°
"main_output/BiasAdd/ReadVariableOpReadVariableOp+main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"main_output/BiasAdd/ReadVariableOp±
main_output/BiasAddBiasAddmain_output/MatMul:product:0*main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/BiasAdd
main_output/SoftmaxSoftmaxmain_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
main_output/Softmax
IdentityIdentitymain_output/Softmax:softmax:0^words_lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2$
words_lstm/whilewords_lstm/while:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Α}
Υ
while_body_5986081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1Ί
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΎ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ύ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ύ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3£
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4£
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5£
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6£
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 


,__inference_words_lstm_layer_call_fn_5985568

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59836242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
Έ
°
H__inference_main_output_layer_call_and_return_conditional_losses_5983993

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Γ
Λ
+__inference_lstm_cell_layer_call_fn_5986585

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2’StatefulPartitionedCallΔ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59827122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/1
ά$

while_body_5983006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5983030_0
while_lstm_cell_5983032_0
while_lstm_cell_5983034_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5983030
while_lstm_cell_5983032
while_lstm_cell_5983034’'while/lstm_cell/StatefulPartitionedCallΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΦ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5983030_0while_lstm_cell_5983032_0while_lstm_cell_5983034_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59826282)
'while/lstm_cell/StatefulPartitionedCallτ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ώ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
while/Identity_4Ώ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5983030while_lstm_cell_5983030_0"4
while_lstm_cell_5983032while_lstm_cell_5983032_0"4
while_lstm_cell_5983034while_lstm_cell_5983034_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
I

F__inference_lstm_cell_layer_call_and_return_conditional_losses_5982712

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
ones_like_1`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_2d
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp―
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:?????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:?????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:?????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:?????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:?????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:?????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:?????????2
	BiasAdd_3f
mul_4Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_4f
mul_5Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_5f
mul_6Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_6f
mul_7Mulstatesones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ώ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:?????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:?????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:?????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:?????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:?????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:?????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:?????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????
 
_user_specified_namestates:PL
(
_output_shapes
:?????????
 
_user_specified_namestates
΄
Θ
while_cond_5986080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5986080___redundant_placeholder05
1while_while_cond_5986080___redundant_placeholder15
1while_while_cond_5986080___redundant_placeholder25
1while_while_cond_5986080___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
­
	
words_lstm_while_body_59843572
.words_lstm_while_words_lstm_while_loop_counter8
4words_lstm_while_words_lstm_while_maximum_iterations 
words_lstm_while_placeholder"
words_lstm_while_placeholder_1"
words_lstm_while_placeholder_2"
words_lstm_while_placeholder_31
-words_lstm_while_words_lstm_strided_slice_1_0m
iwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:words_lstm_while_lstm_cell_split_readvariableop_resource_0@
<words_lstm_while_lstm_cell_split_1_readvariableop_resource_08
4words_lstm_while_lstm_cell_readvariableop_resource_0
words_lstm_while_identity
words_lstm_while_identity_1
words_lstm_while_identity_2
words_lstm_while_identity_3
words_lstm_while_identity_4
words_lstm_while_identity_5/
+words_lstm_while_words_lstm_strided_slice_1k
gwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor<
8words_lstm_while_lstm_cell_split_readvariableop_resource>
:words_lstm_while_lstm_cell_split_1_readvariableop_resource6
2words_lstm_while_lstm_cell_readvariableop_resourceΩ
Bwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape
4words_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0words_lstm_while_placeholderKwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype026
4words_lstm/while/TensorArrayV2Read/TensorListGetItemΓ
*words_lstm/while/lstm_cell/ones_like/ShapeShape;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/ones_like/Shape
*words_lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*words_lstm/while/lstm_cell/ones_like/Constρ
$words_lstm/while/lstm_cell/ones_likeFill3words_lstm/while/lstm_cell/ones_like/Shape:output:03words_lstm/while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/ones_like
(words_lstm/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(words_lstm/while/lstm_cell/dropout/Constμ
&words_lstm/while/lstm_cell/dropout/MulMul-words_lstm/while/lstm_cell/ones_like:output:01words_lstm/while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2(
&words_lstm/while/lstm_cell/dropout/Mul±
(words_lstm/while/lstm_cell/dropout/ShapeShape-words_lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2*
(words_lstm/while/lstm_cell/dropout/Shape₯
?words_lstm/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform1words_lstm/while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Δκ2A
?words_lstm/while/lstm_cell/dropout/random_uniform/RandomUniform«
1words_lstm/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>23
1words_lstm/while/lstm_cell/dropout/GreaterEqual/y«
/words_lstm/while/lstm_cell/dropout/GreaterEqualGreaterEqualHwords_lstm/while/lstm_cell/dropout/random_uniform/RandomUniform:output:0:words_lstm/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????21
/words_lstm/while/lstm_cell/dropout/GreaterEqualΡ
'words_lstm/while/lstm_cell/dropout/CastCast3words_lstm/while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2)
'words_lstm/while/lstm_cell/dropout/Castη
(words_lstm/while/lstm_cell/dropout/Mul_1Mul*words_lstm/while/lstm_cell/dropout/Mul:z:0+words_lstm/while/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout/Mul_1
*words_lstm/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_1/Constς
(words_lstm/while/lstm_cell/dropout_1/MulMul-words_lstm/while/lstm_cell/ones_like:output:03words_lstm/while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_1/Mul΅
*words_lstm/while/lstm_cell/dropout_1/ShapeShape-words_lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_1/Shape«
Awords_lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed22C
Awords_lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_1/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_1/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_1/CastCast5words_lstm/while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_1/Castο
*words_lstm/while/lstm_cell/dropout_1/Mul_1Mul,words_lstm/while/lstm_cell/dropout_1/Mul:z:0-words_lstm/while/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_1/Mul_1
*words_lstm/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_2/Constς
(words_lstm/while/lstm_cell/dropout_2/MulMul-words_lstm/while/lstm_cell/ones_like:output:03words_lstm/while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_2/Mul΅
*words_lstm/while/lstm_cell/dropout_2/ShapeShape-words_lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_2/Shape«
Awords_lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ίγ2C
Awords_lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_2/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_2/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_2/CastCast5words_lstm/while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_2/Castο
*words_lstm/while/lstm_cell/dropout_2/Mul_1Mul,words_lstm/while/lstm_cell/dropout_2/Mul:z:0-words_lstm/while/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_2/Mul_1
*words_lstm/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_3/Constς
(words_lstm/while/lstm_cell/dropout_3/MulMul-words_lstm/while/lstm_cell/ones_like:output:03words_lstm/while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_3/Mul΅
*words_lstm/while/lstm_cell/dropout_3/ShapeShape-words_lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_3/Shape«
Awords_lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Χ2C
Awords_lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_3/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_3/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_3/CastCast5words_lstm/while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_3/Castο
*words_lstm/while/lstm_cell/dropout_3/Mul_1Mul,words_lstm/while/lstm_cell/dropout_3/Mul:z:0-words_lstm/while/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_3/Mul_1ͺ
,words_lstm/while/lstm_cell/ones_like_1/ShapeShapewords_lstm_while_placeholder_2*
T0*
_output_shapes
:2.
,words_lstm/while/lstm_cell/ones_like_1/Shape‘
,words_lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,words_lstm/while/lstm_cell/ones_like_1/Constω
&words_lstm/while/lstm_cell/ones_like_1Fill5words_lstm/while/lstm_cell/ones_like_1/Shape:output:05words_lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2(
&words_lstm/while/lstm_cell/ones_like_1
*words_lstm/while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_4/Constτ
(words_lstm/while/lstm_cell/dropout_4/MulMul/words_lstm/while/lstm_cell/ones_like_1:output:03words_lstm/while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_4/Mul·
*words_lstm/while/lstm_cell/dropout_4/ShapeShape/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_4/Shape«
Awords_lstm/while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ή2C
Awords_lstm/while/lstm_cell/dropout_4/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_4/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_4/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_4/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_4/CastCast5words_lstm/while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_4/Castο
*words_lstm/while/lstm_cell/dropout_4/Mul_1Mul,words_lstm/while/lstm_cell/dropout_4/Mul:z:0-words_lstm/while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_4/Mul_1
*words_lstm/while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_5/Constτ
(words_lstm/while/lstm_cell/dropout_5/MulMul/words_lstm/while/lstm_cell/ones_like_1:output:03words_lstm/while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_5/Mul·
*words_lstm/while/lstm_cell/dropout_5/ShapeShape/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_5/Shape«
Awords_lstm/while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ζψ?2C
Awords_lstm/while/lstm_cell/dropout_5/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_5/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_5/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_5/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_5/CastCast5words_lstm/while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_5/Castο
*words_lstm/while/lstm_cell/dropout_5/Mul_1Mul,words_lstm/while/lstm_cell/dropout_5/Mul:z:0-words_lstm/while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_5/Mul_1
*words_lstm/while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_6/Constτ
(words_lstm/while/lstm_cell/dropout_6/MulMul/words_lstm/while/lstm_cell/ones_like_1:output:03words_lstm/while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_6/Mul·
*words_lstm/while/lstm_cell/dropout_6/ShapeShape/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_6/Shapeͺ
Awords_lstm/while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Υ2C
Awords_lstm/while/lstm_cell/dropout_6/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_6/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_6/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_6/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_6/CastCast5words_lstm/while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_6/Castο
*words_lstm/while/lstm_cell/dropout_6/Mul_1Mul,words_lstm/while/lstm_cell/dropout_6/Mul:z:0-words_lstm/while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_6/Mul_1
*words_lstm/while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*words_lstm/while/lstm_cell/dropout_7/Constτ
(words_lstm/while/lstm_cell/dropout_7/MulMul/words_lstm/while/lstm_cell/ones_like_1:output:03words_lstm/while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2*
(words_lstm/while/lstm_cell/dropout_7/Mul·
*words_lstm/while/lstm_cell/dropout_7/ShapeShape/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/dropout_7/Shape«
Awords_lstm/while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform3words_lstm/while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ή2C
Awords_lstm/while/lstm_cell/dropout_7/random_uniform/RandomUniform―
3words_lstm/while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>25
3words_lstm/while/lstm_cell/dropout_7/GreaterEqual/y³
1words_lstm/while/lstm_cell/dropout_7/GreaterEqualGreaterEqualJwords_lstm/while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:0<words_lstm/while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????23
1words_lstm/while/lstm_cell/dropout_7/GreaterEqualΧ
)words_lstm/while/lstm_cell/dropout_7/CastCast5words_lstm/while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2+
)words_lstm/while/lstm_cell/dropout_7/Castο
*words_lstm/while/lstm_cell/dropout_7/Mul_1Mul,words_lstm/while/lstm_cell/dropout_7/Mul:z:0-words_lstm/while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2,
*words_lstm/while/lstm_cell/dropout_7/Mul_1ε
words_lstm/while/lstm_cell/mulMul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0,words_lstm/while/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2 
words_lstm/while/lstm_cell/mulλ
 words_lstm/while/lstm_cell/mul_1Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.words_lstm/while/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_1λ
 words_lstm/while/lstm_cell/mul_2Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.words_lstm/while/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_2λ
 words_lstm/while/lstm_cell/mul_3Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.words_lstm/while/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_3
 words_lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 words_lstm/while/lstm_cell/Const
*words_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*words_lstm/while/lstm_cell/split/split_dimί
/words_lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp:words_lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/words_lstm/while/lstm_cell/split/ReadVariableOp
 words_lstm/while/lstm_cell/splitSplit3words_lstm/while/lstm_cell/split/split_dim:output:07words_lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2"
 words_lstm/while/lstm_cell/split?
!words_lstm/while/lstm_cell/MatMulMatMul"words_lstm/while/lstm_cell/mul:z:0)words_lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/MatMulΨ
#words_lstm/while/lstm_cell/MatMul_1MatMul$words_lstm/while/lstm_cell/mul_1:z:0)words_lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_1Ψ
#words_lstm/while/lstm_cell/MatMul_2MatMul$words_lstm/while/lstm_cell/mul_2:z:0)words_lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_2Ψ
#words_lstm/while/lstm_cell/MatMul_3MatMul$words_lstm/while/lstm_cell/mul_3:z:0)words_lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_3
"words_lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"words_lstm/while/lstm_cell/Const_1
,words_lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,words_lstm/while/lstm_cell/split_1/split_dimΰ
1words_lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp<words_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1words_lstm/while/lstm_cell/split_1/ReadVariableOp
"words_lstm/while/lstm_cell/split_1Split5words_lstm/while/lstm_cell/split_1/split_dim:output:09words_lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2$
"words_lstm/while/lstm_cell/split_1ΰ
"words_lstm/while/lstm_cell/BiasAddBiasAdd+words_lstm/while/lstm_cell/MatMul:product:0+words_lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/while/lstm_cell/BiasAddζ
$words_lstm/while/lstm_cell/BiasAdd_1BiasAdd-words_lstm/while/lstm_cell/MatMul_1:product:0+words_lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_1ζ
$words_lstm/while/lstm_cell/BiasAdd_2BiasAdd-words_lstm/while/lstm_cell/MatMul_2:product:0+words_lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_2ζ
$words_lstm/while/lstm_cell/BiasAdd_3BiasAdd-words_lstm/while/lstm_cell/MatMul_3:product:0+words_lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_3Ξ
 words_lstm/while/lstm_cell/mul_4Mulwords_lstm_while_placeholder_2.words_lstm/while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_4Ξ
 words_lstm/while/lstm_cell/mul_5Mulwords_lstm_while_placeholder_2.words_lstm/while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_5Ξ
 words_lstm/while/lstm_cell/mul_6Mulwords_lstm_while_placeholder_2.words_lstm/while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_6Ξ
 words_lstm/while/lstm_cell/mul_7Mulwords_lstm_while_placeholder_2.words_lstm/while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_7Ν
)words_lstm/while/lstm_cell/ReadVariableOpReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)words_lstm/while/lstm_cell/ReadVariableOp±
.words_lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.words_lstm/while/lstm_cell/strided_slice/stack΅
0words_lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice/stack_1΅
0words_lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0words_lstm/while/lstm_cell/strided_slice/stack_2 
(words_lstm/while/lstm_cell/strided_sliceStridedSlice1words_lstm/while/lstm_cell/ReadVariableOp:value:07words_lstm/while/lstm_cell/strided_slice/stack:output:09words_lstm/while/lstm_cell/strided_slice/stack_1:output:09words_lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(words_lstm/while/lstm_cell/strided_sliceΰ
#words_lstm/while/lstm_cell/MatMul_4MatMul$words_lstm/while/lstm_cell/mul_4:z:01words_lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_4Ψ
words_lstm/while/lstm_cell/addAddV2+words_lstm/while/lstm_cell/BiasAdd:output:0-words_lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2 
words_lstm/while/lstm_cell/addͺ
"words_lstm/while/lstm_cell/SigmoidSigmoid"words_lstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/while/lstm_cell/SigmoidΡ
+words_lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_1΅
0words_lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_1/stackΉ
2words_lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2words_lstm/while/lstm_cell/strided_slice_1/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_1/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_1StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_1:value:09words_lstm/while/lstm_cell/strided_slice_1/stack:output:0;words_lstm/while/lstm_cell/strided_slice_1/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_1β
#words_lstm/while/lstm_cell/MatMul_5MatMul$words_lstm/while/lstm_cell/mul_5:z:03words_lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_5ή
 words_lstm/while/lstm_cell/add_1AddV2-words_lstm/while/lstm_cell/BiasAdd_1:output:0-words_lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_1°
$words_lstm/while/lstm_cell/Sigmoid_1Sigmoid$words_lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/Sigmoid_1Θ
 words_lstm/while/lstm_cell/mul_8Mul(words_lstm/while/lstm_cell/Sigmoid_1:y:0words_lstm_while_placeholder_3*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_8Ρ
+words_lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_2΅
0words_lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_2/stackΉ
2words_lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2words_lstm/while/lstm_cell/strided_slice_2/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_2/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_2StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_2:value:09words_lstm/while/lstm_cell/strided_slice_2/stack:output:0;words_lstm/while/lstm_cell/strided_slice_2/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_2β
#words_lstm/while/lstm_cell/MatMul_6MatMul$words_lstm/while/lstm_cell/mul_6:z:03words_lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_6ή
 words_lstm/while/lstm_cell/add_2AddV2-words_lstm/while/lstm_cell/BiasAdd_2:output:0-words_lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_2£
words_lstm/while/lstm_cell/TanhTanh$words_lstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2!
words_lstm/while/lstm_cell/TanhΛ
 words_lstm/while/lstm_cell/mul_9Mul&words_lstm/while/lstm_cell/Sigmoid:y:0#words_lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_9Μ
 words_lstm/while/lstm_cell/add_3AddV2$words_lstm/while/lstm_cell/mul_8:z:0$words_lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_3Ρ
+words_lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_3΅
0words_lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_3/stackΉ
2words_lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2words_lstm/while/lstm_cell/strided_slice_3/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_3/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_3StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_3:value:09words_lstm/while/lstm_cell/strided_slice_3/stack:output:0;words_lstm/while/lstm_cell/strided_slice_3/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_3β
#words_lstm/while/lstm_cell/MatMul_7MatMul$words_lstm/while/lstm_cell/mul_7:z:03words_lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_7ή
 words_lstm/while/lstm_cell/add_4AddV2-words_lstm/while/lstm_cell/BiasAdd_3:output:0-words_lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_4°
$words_lstm/while/lstm_cell/Sigmoid_2Sigmoid$words_lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/Sigmoid_2§
!words_lstm/while/lstm_cell/Tanh_1Tanh$words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/Tanh_1Ρ
!words_lstm/while/lstm_cell/mul_10Mul(words_lstm/while/lstm_cell/Sigmoid_2:y:0%words_lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/mul_10
5words_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwords_lstm_while_placeholder_1words_lstm_while_placeholder%words_lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype027
5words_lstm/while/TensorArrayV2Write/TensorListSetItemr
words_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/while/add/y
words_lstm/while/addAddV2words_lstm_while_placeholderwords_lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
words_lstm/while/addv
words_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/while/add_1/y­
words_lstm/while/add_1AddV2.words_lstm_while_words_lstm_while_loop_counter!words_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
words_lstm/while/add_1
words_lstm/while/IdentityIdentitywords_lstm/while/add_1:z:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity
words_lstm/while/Identity_1Identity4words_lstm_while_words_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
words_lstm/while/Identity_1
words_lstm/while/Identity_2Identitywords_lstm/while/add:z:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity_2?
words_lstm/while/Identity_3IdentityEwords_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity_3 
words_lstm/while/Identity_4Identity%words_lstm/while/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/while/Identity_4
words_lstm/while/Identity_5Identity$words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/while/Identity_5"?
words_lstm_while_identity"words_lstm/while/Identity:output:0"C
words_lstm_while_identity_1$words_lstm/while/Identity_1:output:0"C
words_lstm_while_identity_2$words_lstm/while/Identity_2:output:0"C
words_lstm_while_identity_3$words_lstm/while/Identity_3:output:0"C
words_lstm_while_identity_4$words_lstm/while/Identity_4:output:0"C
words_lstm_while_identity_5$words_lstm/while/Identity_5:output:0"j
2words_lstm_while_lstm_cell_readvariableop_resource4words_lstm_while_lstm_cell_readvariableop_resource_0"z
:words_lstm_while_lstm_cell_split_1_readvariableop_resource<words_lstm_while_lstm_cell_split_1_readvariableop_resource_0"v
8words_lstm_while_lstm_cell_split_readvariableop_resource:words_lstm_while_lstm_cell_split_readvariableop_resource_0"Τ
gwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensoriwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0"\
+words_lstm_while_words_lstm_strided_slice_1-words_lstm_while_words_lstm_strided_slice_1_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
 
ό
I__inference_functional_1_layer_call_and_return_conditional_losses_5984010
words_input
layout_features_input
words_embedding_5983239
words_lstm_5983902
words_lstm_5983904
words_lstm_5983906
dense_5983947
dense_5983949
main_output_5984004
main_output_5984006
identity’dense/StatefulPartitionedCall’dropout/StatefulPartitionedCall’#main_output/StatefulPartitionedCall’'words_embedding/StatefulPartitionedCall’"words_lstm/StatefulPartitionedCall©
'words_embedding/StatefulPartitionedCallStatefulPartitionedCallwords_inputwords_embedding_5983239*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_words_embedding_layer_call_and_return_conditional_losses_59832302)
'words_embedding/StatefulPartitionedCallβ
"words_lstm/StatefulPartitionedCallStatefulPartitionedCall0words_embedding/StatefulPartitionedCall:output:0words_lstm_5983902words_lstm_5983904words_lstm_5983906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59836242$
"words_lstm/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCall+words_lstm/StatefulPartitionedCall:output:0layout_features_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_59839162
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_5983947dense_5983949*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_59839362
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839642!
dropout/StatefulPartitionedCallΘ
#main_output/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0main_output_5984004main_output_5984006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_59839932%
#main_output/StatefulPartitionedCall·
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall$^main_output/StatefulPartitionedCall(^words_embedding/StatefulPartitionedCall#^words_lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2R
'words_embedding/StatefulPartitionedCall'words_embedding/StatefulPartitionedCall2H
"words_lstm/StatefulPartitionedCall"words_lstm/StatefulPartitionedCall:T P
'
_output_shapes
:?????????d
%
_user_specified_namewords_input:^Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input
Ή
r
H__inference_concatenate_layer_call_and_return_conditional_losses_5983916

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs

¨
*functional_1_words_lstm_while_cond_5982286L
Hfunctional_1_words_lstm_while_functional_1_words_lstm_while_loop_counterR
Nfunctional_1_words_lstm_while_functional_1_words_lstm_while_maximum_iterations-
)functional_1_words_lstm_while_placeholder/
+functional_1_words_lstm_while_placeholder_1/
+functional_1_words_lstm_while_placeholder_2/
+functional_1_words_lstm_while_placeholder_3N
Jfunctional_1_words_lstm_while_less_functional_1_words_lstm_strided_slice_1e
afunctional_1_words_lstm_while_functional_1_words_lstm_while_cond_5982286___redundant_placeholder0e
afunctional_1_words_lstm_while_functional_1_words_lstm_while_cond_5982286___redundant_placeholder1e
afunctional_1_words_lstm_while_functional_1_words_lstm_while_cond_5982286___redundant_placeholder2e
afunctional_1_words_lstm_while_functional_1_words_lstm_while_cond_5982286___redundant_placeholder3*
&functional_1_words_lstm_while_identity
θ
"functional_1/words_lstm/while/LessLess)functional_1_words_lstm_while_placeholderJfunctional_1_words_lstm_while_less_functional_1_words_lstm_strided_slice_1*
T0*
_output_shapes
: 2$
"functional_1/words_lstm/while/Less₯
&functional_1/words_lstm/while/IdentityIdentity&functional_1/words_lstm/while/Less:z:0*
T0
*
_output_shapes
: 2(
&functional_1/words_lstm/while/Identity"Y
&functional_1_words_lstm_while_identity/functional_1/words_lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ψΞ

I__inference_functional_1_layer_call_and_return_conditional_losses_5984859
inputs_0
inputs_1,
(words_embedding_embedding_lookup_59845856
2words_lstm_lstm_cell_split_readvariableop_resource8
4words_lstm_lstm_cell_split_1_readvariableop_resource0
,words_lstm_lstm_cell_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*main_output_matmul_readvariableop_resource/
+main_output_biasadd_readvariableop_resource
identity’words_lstm/while
 words_embedding/embedding_lookupResourceGather(words_embedding_embedding_lookup_5984585inputs_0*
Tindices0*;
_class1
/-loc:@words_embedding/embedding_lookup/5984585*,
_output_shapes
:?????????d*
dtype02"
 words_embedding/embedding_lookup
)words_embedding/embedding_lookup/IdentityIdentity)words_embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@words_embedding/embedding_lookup/5984585*,
_output_shapes
:?????????d2+
)words_embedding/embedding_lookup/IdentityΡ
+words_embedding/embedding_lookup/Identity_1Identity2words_embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????d2-
+words_embedding/embedding_lookup/Identity_1
words_lstm/ShapeShape4words_embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
words_lstm/Shape
words_lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
words_lstm/strided_slice/stack
 words_lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 words_lstm/strided_slice/stack_1
 words_lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 words_lstm/strided_slice/stack_2€
words_lstm/strided_sliceStridedSlicewords_lstm/Shape:output:0'words_lstm/strided_slice/stack:output:0)words_lstm/strided_slice/stack_1:output:0)words_lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
words_lstm/strided_slices
words_lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros/mul/y
words_lstm/zeros/mulMul!words_lstm/strided_slice:output:0words_lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros/mulu
words_lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
words_lstm/zeros/Less/y
words_lstm/zeros/LessLesswords_lstm/zeros/mul:z:0 words_lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros/Lessy
words_lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros/packed/1―
words_lstm/zeros/packedPack!words_lstm/strided_slice:output:0"words_lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
words_lstm/zeros/packedu
words_lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/zeros/Const’
words_lstm/zerosFill words_lstm/zeros/packed:output:0words_lstm/zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/zerosw
words_lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros_1/mul/y
words_lstm/zeros_1/mulMul!words_lstm/strided_slice:output:0!words_lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros_1/muly
words_lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
words_lstm/zeros_1/Less/y
words_lstm/zeros_1/LessLesswords_lstm/zeros_1/mul:z:0"words_lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
words_lstm/zeros_1/Less}
words_lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
words_lstm/zeros_1/packed/1΅
words_lstm/zeros_1/packedPack!words_lstm/strided_slice:output:0$words_lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
words_lstm/zeros_1/packedy
words_lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/zeros_1/Constͺ
words_lstm/zeros_1Fill"words_lstm/zeros_1/packed:output:0!words_lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/zeros_1
words_lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
words_lstm/transpose/permΚ
words_lstm/transpose	Transpose4words_embedding/embedding_lookup/Identity_1:output:0"words_lstm/transpose/perm:output:0*
T0*,
_output_shapes
:d?????????2
words_lstm/transposep
words_lstm/Shape_1Shapewords_lstm/transpose:y:0*
T0*
_output_shapes
:2
words_lstm/Shape_1
 words_lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 words_lstm/strided_slice_1/stack
"words_lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_1/stack_1
"words_lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_1/stack_2°
words_lstm/strided_slice_1StridedSlicewords_lstm/Shape_1:output:0)words_lstm/strided_slice_1/stack:output:0+words_lstm/strided_slice_1/stack_1:output:0+words_lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
words_lstm/strided_slice_1
&words_lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&words_lstm/TensorArrayV2/element_shapeή
words_lstm/TensorArrayV2TensorListReserve/words_lstm/TensorArrayV2/element_shape:output:0#words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
words_lstm/TensorArrayV2Υ
@words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@words_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape€
2words_lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorwords_lstm/transpose:y:0Iwords_lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2words_lstm/TensorArrayUnstack/TensorListFromTensor
 words_lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 words_lstm/strided_slice_2/stack
"words_lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_2/stack_1
"words_lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_2/stack_2Ώ
words_lstm/strided_slice_2StridedSlicewords_lstm/transpose:y:0)words_lstm/strided_slice_2/stack:output:0+words_lstm/strided_slice_2/stack_1:output:0+words_lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
words_lstm/strided_slice_2
$words_lstm/lstm_cell/ones_like/ShapeShape#words_lstm/strided_slice_2:output:0*
T0*
_output_shapes
:2&
$words_lstm/lstm_cell/ones_like/Shape
$words_lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$words_lstm/lstm_cell/ones_like/ConstΩ
words_lstm/lstm_cell/ones_likeFill-words_lstm/lstm_cell/ones_like/Shape:output:0-words_lstm/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/ones_like
&words_lstm/lstm_cell/ones_like_1/ShapeShapewords_lstm/zeros:output:0*
T0*
_output_shapes
:2(
&words_lstm/lstm_cell/ones_like_1/Shape
&words_lstm/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&words_lstm/lstm_cell/ones_like_1/Constα
 words_lstm/lstm_cell/ones_like_1Fill/words_lstm/lstm_cell/ones_like_1/Shape:output:0/words_lstm/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/lstm_cell/ones_like_1Ό
words_lstm/lstm_cell/mulMul#words_lstm/strided_slice_2:output:0'words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mulΐ
words_lstm/lstm_cell/mul_1Mul#words_lstm/strided_slice_2:output:0'words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_1ΐ
words_lstm/lstm_cell/mul_2Mul#words_lstm/strided_slice_2:output:0'words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_2ΐ
words_lstm/lstm_cell/mul_3Mul#words_lstm/strided_slice_2:output:0'words_lstm/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_3z
words_lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/lstm_cell/Const
$words_lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$words_lstm/lstm_cell/split/split_dimΛ
)words_lstm/lstm_cell/split/ReadVariableOpReadVariableOp2words_lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)words_lstm/lstm_cell/split/ReadVariableOp
words_lstm/lstm_cell/splitSplit-words_lstm/lstm_cell/split/split_dim:output:01words_lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
words_lstm/lstm_cell/splitΊ
words_lstm/lstm_cell/MatMulMatMulwords_lstm/lstm_cell/mul:z:0#words_lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMulΐ
words_lstm/lstm_cell/MatMul_1MatMulwords_lstm/lstm_cell/mul_1:z:0#words_lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_1ΐ
words_lstm/lstm_cell/MatMul_2MatMulwords_lstm/lstm_cell/mul_2:z:0#words_lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_2ΐ
words_lstm/lstm_cell/MatMul_3MatMulwords_lstm/lstm_cell/mul_3:z:0#words_lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_3~
words_lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/lstm_cell/Const_1
&words_lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&words_lstm/lstm_cell/split_1/split_dimΜ
+words_lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp4words_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02-
+words_lstm/lstm_cell/split_1/ReadVariableOpχ
words_lstm/lstm_cell/split_1Split/words_lstm/lstm_cell/split_1/split_dim:output:03words_lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
words_lstm/lstm_cell/split_1Θ
words_lstm/lstm_cell/BiasAddBiasAdd%words_lstm/lstm_cell/MatMul:product:0%words_lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/BiasAddΞ
words_lstm/lstm_cell/BiasAdd_1BiasAdd'words_lstm/lstm_cell/MatMul_1:product:0%words_lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_1Ξ
words_lstm/lstm_cell/BiasAdd_2BiasAdd'words_lstm/lstm_cell/MatMul_2:product:0%words_lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_2Ξ
words_lstm/lstm_cell/BiasAdd_3BiasAdd'words_lstm/lstm_cell/MatMul_3:product:0%words_lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/BiasAdd_3Έ
words_lstm/lstm_cell/mul_4Mulwords_lstm/zeros:output:0)words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_4Έ
words_lstm/lstm_cell/mul_5Mulwords_lstm/zeros:output:0)words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_5Έ
words_lstm/lstm_cell/mul_6Mulwords_lstm/zeros:output:0)words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_6Έ
words_lstm/lstm_cell/mul_7Mulwords_lstm/zeros:output:0)words_lstm/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_7Ή
#words_lstm/lstm_cell/ReadVariableOpReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#words_lstm/lstm_cell/ReadVariableOp₯
(words_lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(words_lstm/lstm_cell/strided_slice/stack©
*words_lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice/stack_1©
*words_lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*words_lstm/lstm_cell/strided_slice/stack_2ό
"words_lstm/lstm_cell/strided_sliceStridedSlice+words_lstm/lstm_cell/ReadVariableOp:value:01words_lstm/lstm_cell/strided_slice/stack:output:03words_lstm/lstm_cell/strided_slice/stack_1:output:03words_lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2$
"words_lstm/lstm_cell/strided_sliceΘ
words_lstm/lstm_cell/MatMul_4MatMulwords_lstm/lstm_cell/mul_4:z:0+words_lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_4ΐ
words_lstm/lstm_cell/addAddV2%words_lstm/lstm_cell/BiasAdd:output:0'words_lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add
words_lstm/lstm_cell/SigmoidSigmoidwords_lstm/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Sigmoid½
%words_lstm/lstm_cell/ReadVariableOp_1ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_1©
*words_lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_1/stack­
,words_lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,words_lstm/lstm_cell/strided_slice_1/stack_1­
,words_lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_1/stack_2
$words_lstm/lstm_cell/strided_slice_1StridedSlice-words_lstm/lstm_cell/ReadVariableOp_1:value:03words_lstm/lstm_cell/strided_slice_1/stack:output:05words_lstm/lstm_cell/strided_slice_1/stack_1:output:05words_lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_1Κ
words_lstm/lstm_cell/MatMul_5MatMulwords_lstm/lstm_cell/mul_5:z:0-words_lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_5Ζ
words_lstm/lstm_cell/add_1AddV2'words_lstm/lstm_cell/BiasAdd_1:output:0'words_lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_1
words_lstm/lstm_cell/Sigmoid_1Sigmoidwords_lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/Sigmoid_1³
words_lstm/lstm_cell/mul_8Mul"words_lstm/lstm_cell/Sigmoid_1:y:0words_lstm/zeros_1:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_8½
%words_lstm/lstm_cell/ReadVariableOp_2ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_2©
*words_lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_2/stack­
,words_lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,words_lstm/lstm_cell/strided_slice_2/stack_1­
,words_lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_2/stack_2
$words_lstm/lstm_cell/strided_slice_2StridedSlice-words_lstm/lstm_cell/ReadVariableOp_2:value:03words_lstm/lstm_cell/strided_slice_2/stack:output:05words_lstm/lstm_cell/strided_slice_2/stack_1:output:05words_lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_2Κ
words_lstm/lstm_cell/MatMul_6MatMulwords_lstm/lstm_cell/mul_6:z:0-words_lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_6Ζ
words_lstm/lstm_cell/add_2AddV2'words_lstm/lstm_cell/BiasAdd_2:output:0'words_lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_2
words_lstm/lstm_cell/TanhTanhwords_lstm/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Tanh³
words_lstm/lstm_cell/mul_9Mul words_lstm/lstm_cell/Sigmoid:y:0words_lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_9΄
words_lstm/lstm_cell/add_3AddV2words_lstm/lstm_cell/mul_8:z:0words_lstm/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_3½
%words_lstm/lstm_cell/ReadVariableOp_3ReadVariableOp,words_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02'
%words_lstm/lstm_cell/ReadVariableOp_3©
*words_lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*words_lstm/lstm_cell/strided_slice_3/stack­
,words_lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,words_lstm/lstm_cell/strided_slice_3/stack_1­
,words_lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,words_lstm/lstm_cell/strided_slice_3/stack_2
$words_lstm/lstm_cell/strided_slice_3StridedSlice-words_lstm/lstm_cell/ReadVariableOp_3:value:03words_lstm/lstm_cell/strided_slice_3/stack:output:05words_lstm/lstm_cell/strided_slice_3/stack_1:output:05words_lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2&
$words_lstm/lstm_cell/strided_slice_3Κ
words_lstm/lstm_cell/MatMul_7MatMulwords_lstm/lstm_cell/mul_7:z:0-words_lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/MatMul_7Ζ
words_lstm/lstm_cell/add_4AddV2'words_lstm/lstm_cell/BiasAdd_3:output:0'words_lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/add_4
words_lstm/lstm_cell/Sigmoid_2Sigmoidwords_lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2 
words_lstm/lstm_cell/Sigmoid_2
words_lstm/lstm_cell/Tanh_1Tanhwords_lstm/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/Tanh_1Ή
words_lstm/lstm_cell/mul_10Mul"words_lstm/lstm_cell/Sigmoid_2:y:0words_lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
words_lstm/lstm_cell/mul_10₯
(words_lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2*
(words_lstm/TensorArrayV2_1/element_shapeδ
words_lstm/TensorArrayV2_1TensorListReserve1words_lstm/TensorArrayV2_1/element_shape:output:0#words_lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
words_lstm/TensorArrayV2_1d
words_lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
words_lstm/time
#words_lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#words_lstm/while/maximum_iterations
words_lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
words_lstm/while/loop_counter
words_lstm/whileWhile&words_lstm/while/loop_counter:output:0,words_lstm/while/maximum_iterations:output:0words_lstm/time:output:0#words_lstm/TensorArrayV2_1:handle:0words_lstm/zeros:output:0words_lstm/zeros_1:output:0#words_lstm/strided_slice_1:output:0Bwords_lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02words_lstm_lstm_cell_split_readvariableop_resource4words_lstm_lstm_cell_split_1_readvariableop_resource,words_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*)
body!R
words_lstm_while_body_5984706*)
cond!R
words_lstm_while_cond_5984705*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
words_lstm/whileΛ
;words_lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;words_lstm/TensorArrayV2Stack/TensorListStack/element_shape
-words_lstm/TensorArrayV2Stack/TensorListStackTensorListStackwords_lstm/while:output:3Dwords_lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02/
-words_lstm/TensorArrayV2Stack/TensorListStack
 words_lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 words_lstm/strided_slice_3/stack
"words_lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"words_lstm/strided_slice_3/stack_1
"words_lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"words_lstm/strided_slice_3/stack_2έ
words_lstm/strided_slice_3StridedSlice6words_lstm/TensorArrayV2Stack/TensorListStack:tensor:0)words_lstm/strided_slice_3/stack:output:0+words_lstm/strided_slice_3/stack_1:output:0+words_lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
words_lstm/strided_slice_3
words_lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
words_lstm/transpose_1/perm?
words_lstm/transpose_1	Transpose6words_lstm/TensorArrayV2Stack/TensorListStack:tensor:0$words_lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
words_lstm/transpose_1|
words_lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
words_lstm/runtimet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisΑ
concatenate/concatConcatV2#words_lstm/strided_slice_3:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????2
concatenate/concat‘
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:?????????2
dropout/Identity²
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!main_output/MatMul/ReadVariableOpͺ
main_output/MatMulMatMuldropout/Identity:output:0)main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/MatMul°
"main_output/BiasAdd/ReadVariableOpReadVariableOp+main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"main_output/BiasAdd/ReadVariableOp±
main_output/BiasAddBiasAddmain_output/MatMul:product:0*main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/BiasAdd
main_output/SoftmaxSoftmaxmain_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
main_output/Softmax
IdentityIdentitymain_output/Softmax:softmax:0^words_lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2$
words_lstm/whilewords_lstm/while:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1

c
D__inference_dropout_layer_call_and_return_conditional_losses_5983964

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
β

L__inference_words_embedding_layer_call_and_return_conditional_losses_5984912

inputs
embedding_lookup_5984906
identityΠ
embedding_lookupResourceGatherembedding_lookup_5984906inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5984906*,
_output_shapes
:?????????d*
dtype02
embedding_lookupΑ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5984906*,
_output_shapes
:?????????d2
embedding_lookup/Identity‘
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????d2
embedding_lookup/Identity_1}
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Λ
b
D__inference_dropout_layer_call_and_return_conditional_losses_5986289

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
?

€
words_lstm_while_cond_59847052
.words_lstm_while_words_lstm_while_loop_counter8
4words_lstm_while_words_lstm_while_maximum_iterations 
words_lstm_while_placeholder"
words_lstm_while_placeholder_1"
words_lstm_while_placeholder_2"
words_lstm_while_placeholder_34
0words_lstm_while_less_words_lstm_strided_slice_1K
Gwords_lstm_while_words_lstm_while_cond_5984705___redundant_placeholder0K
Gwords_lstm_while_words_lstm_while_cond_5984705___redundant_placeholder1K
Gwords_lstm_while_words_lstm_while_cond_5984705___redundant_placeholder2K
Gwords_lstm_while_words_lstm_while_cond_5984705___redundant_placeholder3
words_lstm_while_identity
§
words_lstm/while/LessLesswords_lstm_while_placeholder0words_lstm_while_less_words_lstm_strided_slice_1*
T0*
_output_shapes
: 2
words_lstm/while/Less~
words_lstm/while/IdentityIdentitywords_lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
words_lstm/while/Identity"?
words_lstm_while_identity"words_lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ά
|
'__inference_dense_layer_call_fn_5986272

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallσ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_59839362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ζ~
α
 __inference__traced_save_5986816
file_prefix9
5savev2_words_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop1
-savev2_main_output_kernel_read_readvariableop/
+savev2_main_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_words_lstm_lstm_cell_kernel_read_readvariableopD
@savev2_words_lstm_lstm_cell_recurrent_kernel_read_readvariableop8
4savev2_words_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableop'
#savev2_total_11_read_readvariableop'
#savev2_count_11_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop'
#savev2_total_13_read_readvariableop'
#savev2_count_13_read_readvariableop'
#savev2_total_14_read_readvariableop'
#savev2_count_14_read_readvariableop'
#savev2_total_15_read_readvariableop'
#savev2_count_15_read_readvariableop@
<savev2_adam_words_embedding_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop8
4savev2_adam_main_output_kernel_m_read_readvariableop6
2savev2_adam_main_output_bias_m_read_readvariableopA
=savev2_adam_words_lstm_lstm_cell_kernel_m_read_readvariableopK
Gsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop?
;savev2_adam_words_lstm_lstm_cell_bias_m_read_readvariableop@
<savev2_adam_words_embedding_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop8
4savev2_adam_main_output_kernel_v_read_readvariableop6
2savev2_adam_main_output_bias_v_read_readvariableopA
=savev2_adam_words_lstm_lstm_cell_kernel_v_read_readvariableopK
Gsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop?
;savev2_adam_words_lstm_lstm_cell_bias_v_read_readvariableopC
?savev2_adam_words_embedding_embeddings_vhat_read_readvariableop5
1savev2_adam_dense_kernel_vhat_read_readvariableop3
/savev2_adam_dense_bias_vhat_read_readvariableop;
7savev2_adam_main_output_kernel_vhat_read_readvariableop9
5savev2_adam_main_output_bias_vhat_read_readvariableopD
@savev2_adam_words_lstm_lstm_cell_kernel_vhat_read_readvariableopN
Jsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_vhat_read_readvariableopB
>savev2_adam_words_lstm_lstm_cell_bias_vhat_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2d6c1f70dfa0436a82889423a08916e6/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameΚ"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*ά!
value?!BΟ!FB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*‘
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesί
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_words_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop-savev2_main_output_kernel_read_readvariableop+savev2_main_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_words_lstm_lstm_cell_kernel_read_readvariableop@savev2_words_lstm_lstm_cell_recurrent_kernel_read_readvariableop4savev2_words_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableop#savev2_total_11_read_readvariableop#savev2_count_11_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop#savev2_total_13_read_readvariableop#savev2_count_13_read_readvariableop#savev2_total_14_read_readvariableop#savev2_count_14_read_readvariableop#savev2_total_15_read_readvariableop#savev2_count_15_read_readvariableop<savev2_adam_words_embedding_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop4savev2_adam_main_output_kernel_m_read_readvariableop2savev2_adam_main_output_bias_m_read_readvariableop=savev2_adam_words_lstm_lstm_cell_kernel_m_read_readvariableopGsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop;savev2_adam_words_lstm_lstm_cell_bias_m_read_readvariableop<savev2_adam_words_embedding_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop4savev2_adam_main_output_kernel_v_read_readvariableop2savev2_adam_main_output_bias_v_read_readvariableop=savev2_adam_words_lstm_lstm_cell_kernel_v_read_readvariableopGsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop;savev2_adam_words_lstm_lstm_cell_bias_v_read_readvariableop?savev2_adam_words_embedding_embeddings_vhat_read_readvariableop1savev2_adam_dense_kernel_vhat_read_readvariableop/savev2_adam_dense_bias_vhat_read_readvariableop7savev2_adam_main_output_kernel_vhat_read_readvariableop5savev2_adam_main_output_bias_vhat_read_readvariableop@savev2_adam_words_lstm_lstm_cell_kernel_vhat_read_readvariableopJsavev2_adam_words_lstm_lstm_cell_recurrent_kernel_vhat_read_readvariableop>savev2_adam_words_lstm_lstm_cell_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :
Σ:
::	:: : : : : :
:
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
Σ:
::	::
:
::
Σ:
::	::
:
::
Σ:
::	::
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Σ:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :&."
 
_output_shapes
:
Σ:&/"
 
_output_shapes
:
:!0

_output_shapes	
::%1!

_output_shapes
:	: 2

_output_shapes
::&3"
 
_output_shapes
:
:&4"
 
_output_shapes
:
:!5

_output_shapes	
::&6"
 
_output_shapes
:
Σ:&7"
 
_output_shapes
:
:!8

_output_shapes	
::%9!

_output_shapes
:	: :

_output_shapes
::&;"
 
_output_shapes
:
:&<"
 
_output_shapes
:
:!=

_output_shapes	
::&>"
 
_output_shapes
:
Σ:&?"
 
_output_shapes
:
:!@

_output_shapes	
::%A!

_output_shapes
:	: B

_output_shapes
::&C"
 
_output_shapes
:
:&D"
 
_output_shapes
:
:!E

_output_shapes	
::F

_output_shapes
: 
ώή
Υ
while_body_5985102
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Constΐ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Γ26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2&
$while/lstm_cell/dropout/GreaterEqual°
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Cast»
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstΖ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ϊρ₯28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_1/GreaterEqualΆ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_1/CastΓ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstΖ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2΄λ28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_2/GreaterEqualΆ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_2/CastΓ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstΖ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ψο‘28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_3/GreaterEqualΆ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_3/CastΓ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_3/Mul_1
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_4/ConstΘ
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_4/Mul
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2―d28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_4/GreaterEqual/y
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_4/GreaterEqualΆ
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_4/CastΓ
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_4/Mul_1
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_5/ConstΘ
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_5/Mul
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΚΓ28
6while/lstm_cell/dropout_5/random_uniform/RandomUniform
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_5/GreaterEqual/y
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_5/GreaterEqualΆ
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_5/CastΓ
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_5/Mul_1
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_6/ConstΘ
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_6/Mul
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ςφ28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_6/GreaterEqual/y
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_6/GreaterEqualΆ
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_6/CastΓ
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_6/Mul_1
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_7/ConstΘ
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_7/Mul
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed228
6while/lstm_cell/dropout_7/random_uniform/RandomUniform
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_7/GreaterEqual/y
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_7/GreaterEqualΆ
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_7/CastΓ
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_7/Mul_1Ή
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΏ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ώ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ώ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3’
while/lstm_cell/mul_4Mulwhile_placeholder_2#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4’
while/lstm_cell/mul_5Mulwhile_placeholder_2#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5’
while/lstm_cell/mul_6Mulwhile_placeholder_2#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6’
while/lstm_cell/mul_7Mulwhile_placeholder_2#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
¦I

F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986551

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
ones_like_1`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_2d
mul_3Mulinputsones_like:output:0*
T0*(
_output_shapes
:?????????2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp―
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:?????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:?????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:?????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:?????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:?????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:?????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:?????????2
	BiasAdd_3h
mul_4Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_4h
mul_5Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_5h
mul_6Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_6h
mul_7Mulstates_0ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ώ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:?????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:?????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:?????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:?????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:?????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:?????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:?????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/1
’
b
)__inference_dropout_layer_call_fn_5986294

inputs
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ώλ
σ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985962
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const¨
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeς
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2αώ?20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2"
 lstm_cell/dropout/GreaterEqual/yη
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout/Cast£
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeχ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2α°f22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_1/GreaterEqual/yο
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_1/GreaterEqual€
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Cast«
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeψ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Φ?κ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_2/GreaterEqual/yο
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_2/GreaterEqual€
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Cast«
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeψ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2θΖε22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_3/GreaterEqual/yο
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_3/GreaterEqual€
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Cast«
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_4/Const°
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeχ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2 ά22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_4/GreaterEqual/yο
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_4/GreaterEqual€
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Cast«
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_5/Const°
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeχ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΝκD22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_5/GreaterEqual/yο
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_5/GreaterEqual€
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Cast«
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_6/Const°
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeψ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2¨ ±22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_6/GreaterEqual/yο
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_6/GreaterEqual€
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Cast«
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_7/Const°
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeχ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΜΔG22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_7/GreaterEqual/yο
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_7/GreaterEqual€
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Cast«
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5985762*
condR
while_cond_5985761*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeς
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm―
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
°
ͺ
B__inference_dense_layer_call_and_return_conditional_losses_5983936

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
΄
Θ
while_cond_5985761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5985761___redundant_placeholder05
1while_while_cond_5985761___redundant_placeholder15
1while_while_cond_5985761___redundant_placeholder25
1while_while_cond_5985761___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
§

,__inference_words_lstm_layer_call_fn_5986228
inputs_0
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59830752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
¬	
τ
%__inference_signature_wrapper_5984168
layout_features_input
words_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity’StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallwords_inputlayout_features_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_59824402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????d::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input:TP
'
_output_shapes
:?????????d
%
_user_specified_namewords_input
?Ί

*functional_1_words_lstm_while_body_5982287L
Hfunctional_1_words_lstm_while_functional_1_words_lstm_while_loop_counterR
Nfunctional_1_words_lstm_while_functional_1_words_lstm_while_maximum_iterations-
)functional_1_words_lstm_while_placeholder/
+functional_1_words_lstm_while_placeholder_1/
+functional_1_words_lstm_while_placeholder_2/
+functional_1_words_lstm_while_placeholder_3K
Gfunctional_1_words_lstm_while_functional_1_words_lstm_strided_slice_1_0
functional_1_words_lstm_while_tensorarrayv2read_tensorlistgetitem_functional_1_words_lstm_tensorarrayunstack_tensorlistfromtensor_0K
Gfunctional_1_words_lstm_while_lstm_cell_split_readvariableop_resource_0M
Ifunctional_1_words_lstm_while_lstm_cell_split_1_readvariableop_resource_0E
Afunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0*
&functional_1_words_lstm_while_identity,
(functional_1_words_lstm_while_identity_1,
(functional_1_words_lstm_while_identity_2,
(functional_1_words_lstm_while_identity_3,
(functional_1_words_lstm_while_identity_4,
(functional_1_words_lstm_while_identity_5I
Efunctional_1_words_lstm_while_functional_1_words_lstm_strided_slice_1
functional_1_words_lstm_while_tensorarrayv2read_tensorlistgetitem_functional_1_words_lstm_tensorarrayunstack_tensorlistfromtensorI
Efunctional_1_words_lstm_while_lstm_cell_split_readvariableop_resourceK
Gfunctional_1_words_lstm_while_lstm_cell_split_1_readvariableop_resourceC
?functional_1_words_lstm_while_lstm_cell_readvariableop_resourceσ
Ofunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2Q
Ofunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeε
Afunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemfunctional_1_words_lstm_while_tensorarrayv2read_tensorlistgetitem_functional_1_words_lstm_tensorarrayunstack_tensorlistfromtensor_0)functional_1_words_lstm_while_placeholderXfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02C
Afunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItemκ
7functional_1/words_lstm/while/lstm_cell/ones_like/ShapeShapeHfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:29
7functional_1/words_lstm/while/lstm_cell/ones_like/Shape·
7functional_1/words_lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7functional_1/words_lstm/while/lstm_cell/ones_like/Const₯
1functional_1/words_lstm/while/lstm_cell/ones_likeFill@functional_1/words_lstm/while/lstm_cell/ones_like/Shape:output:0@functional_1/words_lstm/while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/ones_likeΡ
9functional_1/words_lstm/while/lstm_cell/ones_like_1/ShapeShape+functional_1_words_lstm_while_placeholder_2*
T0*
_output_shapes
:2;
9functional_1/words_lstm/while/lstm_cell/ones_like_1/Shape»
9functional_1/words_lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9functional_1/words_lstm/while/lstm_cell/ones_like_1/Const­
3functional_1/words_lstm/while/lstm_cell/ones_like_1FillBfunctional_1/words_lstm/while/lstm_cell/ones_like_1/Shape:output:0Bfunctional_1/words_lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????25
3functional_1/words_lstm/while/lstm_cell/ones_like_1
+functional_1/words_lstm/while/lstm_cell/mulMulHfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0:functional_1/words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/while/lstm_cell/mul
-functional_1/words_lstm/while/lstm_cell/mul_1MulHfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0:functional_1/words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_1
-functional_1/words_lstm/while/lstm_cell/mul_2MulHfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0:functional_1/words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_2
-functional_1/words_lstm/while/lstm_cell/mul_3MulHfunctional_1/words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0:functional_1/words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_3 
-functional_1/words_lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/words_lstm/while/lstm_cell/Const΄
7functional_1/words_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7functional_1/words_lstm/while/lstm_cell/split/split_dim
<functional_1/words_lstm/while/lstm_cell/split/ReadVariableOpReadVariableOpGfunctional_1_words_lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02>
<functional_1/words_lstm/while/lstm_cell/split/ReadVariableOpΟ
-functional_1/words_lstm/while/lstm_cell/splitSplit@functional_1/words_lstm/while/lstm_cell/split/split_dim:output:0Dfunctional_1/words_lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2/
-functional_1/words_lstm/while/lstm_cell/split
.functional_1/words_lstm/while/lstm_cell/MatMulMatMul/functional_1/words_lstm/while/lstm_cell/mul:z:06functional_1/words_lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????20
.functional_1/words_lstm/while/lstm_cell/MatMul
0functional_1/words_lstm/while/lstm_cell/MatMul_1MatMul1functional_1/words_lstm/while/lstm_cell/mul_1:z:06functional_1/words_lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_1
0functional_1/words_lstm/while/lstm_cell/MatMul_2MatMul1functional_1/words_lstm/while/lstm_cell/mul_2:z:06functional_1/words_lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_2
0functional_1/words_lstm/while/lstm_cell/MatMul_3MatMul1functional_1/words_lstm/while/lstm_cell/mul_3:z:06functional_1/words_lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_3€
/functional_1/words_lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/words_lstm/while/lstm_cell/Const_1Έ
9functional_1/words_lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9functional_1/words_lstm/while/lstm_cell/split_1/split_dim
>functional_1/words_lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpIfunctional_1_words_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>functional_1/words_lstm/while/lstm_cell/split_1/ReadVariableOpΓ
/functional_1/words_lstm/while/lstm_cell/split_1SplitBfunctional_1/words_lstm/while/lstm_cell/split_1/split_dim:output:0Ffunctional_1/words_lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split21
/functional_1/words_lstm/while/lstm_cell/split_1
/functional_1/words_lstm/while/lstm_cell/BiasAddBiasAdd8functional_1/words_lstm/while/lstm_cell/MatMul:product:08functional_1/words_lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????21
/functional_1/words_lstm/while/lstm_cell/BiasAdd
1functional_1/words_lstm/while/lstm_cell/BiasAdd_1BiasAdd:functional_1/words_lstm/while/lstm_cell/MatMul_1:product:08functional_1/words_lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/BiasAdd_1
1functional_1/words_lstm/while/lstm_cell/BiasAdd_2BiasAdd:functional_1/words_lstm/while/lstm_cell/MatMul_2:product:08functional_1/words_lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/BiasAdd_2
1functional_1/words_lstm/while/lstm_cell/BiasAdd_3BiasAdd:functional_1/words_lstm/while/lstm_cell/MatMul_3:product:08functional_1/words_lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/BiasAdd_3
-functional_1/words_lstm/while/lstm_cell/mul_4Mul+functional_1_words_lstm_while_placeholder_2<functional_1/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_4
-functional_1/words_lstm/while/lstm_cell/mul_5Mul+functional_1_words_lstm_while_placeholder_2<functional_1/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_5
-functional_1/words_lstm/while/lstm_cell/mul_6Mul+functional_1_words_lstm_while_placeholder_2<functional_1/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_6
-functional_1/words_lstm/while/lstm_cell/mul_7Mul+functional_1_words_lstm_while_placeholder_2<functional_1/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_7τ
6functional_1/words_lstm/while/lstm_cell/ReadVariableOpReadVariableOpAfunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype028
6functional_1/words_lstm/while/lstm_cell/ReadVariableOpΛ
;functional_1/words_lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2=
;functional_1/words_lstm/while/lstm_cell/strided_slice/stackΟ
=functional_1/words_lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2?
=functional_1/words_lstm/while/lstm_cell/strided_slice/stack_1Ο
=functional_1/words_lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=functional_1/words_lstm/while/lstm_cell/strided_slice/stack_2ξ
5functional_1/words_lstm/while/lstm_cell/strided_sliceStridedSlice>functional_1/words_lstm/while/lstm_cell/ReadVariableOp:value:0Dfunctional_1/words_lstm/while/lstm_cell/strided_slice/stack:output:0Ffunctional_1/words_lstm/while/lstm_cell/strided_slice/stack_1:output:0Ffunctional_1/words_lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask27
5functional_1/words_lstm/while/lstm_cell/strided_slice
0functional_1/words_lstm/while/lstm_cell/MatMul_4MatMul1functional_1/words_lstm/while/lstm_cell/mul_4:z:0>functional_1/words_lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_4
+functional_1/words_lstm/while/lstm_cell/addAddV28functional_1/words_lstm/while/lstm_cell/BiasAdd:output:0:functional_1/words_lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2-
+functional_1/words_lstm/while/lstm_cell/addΡ
/functional_1/words_lstm/while/lstm_cell/SigmoidSigmoid/functional_1/words_lstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????21
/functional_1/words_lstm/while/lstm_cell/Sigmoidψ
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOpAfunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_1Ο
=functional_1/words_lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2?
=functional_1/words_lstm/while/lstm_cell/strided_slice_1/stackΣ
?functional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_1Σ
?functional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_2ϊ
7functional_1/words_lstm/while/lstm_cell/strided_slice_1StridedSlice@functional_1/words_lstm/while/lstm_cell/ReadVariableOp_1:value:0Ffunctional_1/words_lstm/while/lstm_cell/strided_slice_1/stack:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_1:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask29
7functional_1/words_lstm/while/lstm_cell/strided_slice_1
0functional_1/words_lstm/while/lstm_cell/MatMul_5MatMul1functional_1/words_lstm/while/lstm_cell/mul_5:z:0@functional_1/words_lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_5
-functional_1/words_lstm/while/lstm_cell/add_1AddV2:functional_1/words_lstm/while/lstm_cell/BiasAdd_1:output:0:functional_1/words_lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/add_1Χ
1functional_1/words_lstm/while/lstm_cell/Sigmoid_1Sigmoid1functional_1/words_lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/Sigmoid_1ό
-functional_1/words_lstm/while/lstm_cell/mul_8Mul5functional_1/words_lstm/while/lstm_cell/Sigmoid_1:y:0+functional_1_words_lstm_while_placeholder_3*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_8ψ
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOpAfunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_2Ο
=functional_1/words_lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2?
=functional_1/words_lstm/while/lstm_cell/strided_slice_2/stackΣ
?functional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_1Σ
?functional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_2ϊ
7functional_1/words_lstm/while/lstm_cell/strided_slice_2StridedSlice@functional_1/words_lstm/while/lstm_cell/ReadVariableOp_2:value:0Ffunctional_1/words_lstm/while/lstm_cell/strided_slice_2/stack:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_1:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask29
7functional_1/words_lstm/while/lstm_cell/strided_slice_2
0functional_1/words_lstm/while/lstm_cell/MatMul_6MatMul1functional_1/words_lstm/while/lstm_cell/mul_6:z:0@functional_1/words_lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_6
-functional_1/words_lstm/while/lstm_cell/add_2AddV2:functional_1/words_lstm/while/lstm_cell/BiasAdd_2:output:0:functional_1/words_lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/add_2Κ
,functional_1/words_lstm/while/lstm_cell/TanhTanh1functional_1/words_lstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2.
,functional_1/words_lstm/while/lstm_cell/Tanh?
-functional_1/words_lstm/while/lstm_cell/mul_9Mul3functional_1/words_lstm/while/lstm_cell/Sigmoid:y:00functional_1/words_lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/mul_9
-functional_1/words_lstm/while/lstm_cell/add_3AddV21functional_1/words_lstm/while/lstm_cell/mul_8:z:01functional_1/words_lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/add_3ψ
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOpAfunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02:
8functional_1/words_lstm/while/lstm_cell/ReadVariableOp_3Ο
=functional_1/words_lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2?
=functional_1/words_lstm/while/lstm_cell/strided_slice_3/stackΣ
?functional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_1Σ
?functional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?functional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_2ϊ
7functional_1/words_lstm/while/lstm_cell/strided_slice_3StridedSlice@functional_1/words_lstm/while/lstm_cell/ReadVariableOp_3:value:0Ffunctional_1/words_lstm/while/lstm_cell/strided_slice_3/stack:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_1:output:0Hfunctional_1/words_lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask29
7functional_1/words_lstm/while/lstm_cell/strided_slice_3
0functional_1/words_lstm/while/lstm_cell/MatMul_7MatMul1functional_1/words_lstm/while/lstm_cell/mul_7:z:0@functional_1/words_lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????22
0functional_1/words_lstm/while/lstm_cell/MatMul_7
-functional_1/words_lstm/while/lstm_cell/add_4AddV2:functional_1/words_lstm/while/lstm_cell/BiasAdd_3:output:0:functional_1/words_lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2/
-functional_1/words_lstm/while/lstm_cell/add_4Χ
1functional_1/words_lstm/while/lstm_cell/Sigmoid_2Sigmoid1functional_1/words_lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????23
1functional_1/words_lstm/while/lstm_cell/Sigmoid_2Ξ
.functional_1/words_lstm/while/lstm_cell/Tanh_1Tanh1functional_1/words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????20
.functional_1/words_lstm/while/lstm_cell/Tanh_1
.functional_1/words_lstm/while/lstm_cell/mul_10Mul5functional_1/words_lstm/while/lstm_cell/Sigmoid_2:y:02functional_1/words_lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????20
.functional_1/words_lstm/while/lstm_cell/mul_10Φ
Bfunctional_1/words_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+functional_1_words_lstm_while_placeholder_1)functional_1_words_lstm_while_placeholder2functional_1/words_lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02D
Bfunctional_1/words_lstm/while/TensorArrayV2Write/TensorListSetItem
#functional_1/words_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/words_lstm/while/add/yΙ
!functional_1/words_lstm/while/addAddV2)functional_1_words_lstm_while_placeholder,functional_1/words_lstm/while/add/y:output:0*
T0*
_output_shapes
: 2#
!functional_1/words_lstm/while/add
%functional_1/words_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/words_lstm/while/add_1/yξ
#functional_1/words_lstm/while/add_1AddV2Hfunctional_1_words_lstm_while_functional_1_words_lstm_while_loop_counter.functional_1/words_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/words_lstm/while/add_1¦
&functional_1/words_lstm/while/IdentityIdentity'functional_1/words_lstm/while/add_1:z:0*
T0*
_output_shapes
: 2(
&functional_1/words_lstm/while/IdentityΡ
(functional_1/words_lstm/while/Identity_1IdentityNfunctional_1_words_lstm_while_functional_1_words_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2*
(functional_1/words_lstm/while/Identity_1¨
(functional_1/words_lstm/while/Identity_2Identity%functional_1/words_lstm/while/add:z:0*
T0*
_output_shapes
: 2*
(functional_1/words_lstm/while/Identity_2Υ
(functional_1/words_lstm/while/Identity_3IdentityRfunctional_1/words_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2*
(functional_1/words_lstm/while/Identity_3Η
(functional_1/words_lstm/while/Identity_4Identity2functional_1/words_lstm/while/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2*
(functional_1/words_lstm/while/Identity_4Ζ
(functional_1/words_lstm/while/Identity_5Identity1functional_1/words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2*
(functional_1/words_lstm/while/Identity_5"
Efunctional_1_words_lstm_while_functional_1_words_lstm_strided_slice_1Gfunctional_1_words_lstm_while_functional_1_words_lstm_strided_slice_1_0"Y
&functional_1_words_lstm_while_identity/functional_1/words_lstm/while/Identity:output:0"]
(functional_1_words_lstm_while_identity_11functional_1/words_lstm/while/Identity_1:output:0"]
(functional_1_words_lstm_while_identity_21functional_1/words_lstm/while/Identity_2:output:0"]
(functional_1_words_lstm_while_identity_31functional_1/words_lstm/while/Identity_3:output:0"]
(functional_1_words_lstm_while_identity_41functional_1/words_lstm/while/Identity_4:output:0"]
(functional_1_words_lstm_while_identity_51functional_1/words_lstm/while/Identity_5:output:0"
?functional_1_words_lstm_while_lstm_cell_readvariableop_resourceAfunctional_1_words_lstm_while_lstm_cell_readvariableop_resource_0"
Gfunctional_1_words_lstm_while_lstm_cell_split_1_readvariableop_resourceIfunctional_1_words_lstm_while_lstm_cell_split_1_readvariableop_resource_0"
Efunctional_1_words_lstm_while_lstm_cell_split_readvariableop_resourceGfunctional_1_words_lstm_while_lstm_cell_split_readvariableop_resource_0"
functional_1_words_lstm_while_tensorarrayv2read_tensorlistgetitem_functional_1_words_lstm_tensorarrayunstack_tensorlistfromtensorfunctional_1_words_lstm_while_tensorarrayv2read_tensorlistgetitem_functional_1_words_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
ΠD
Χ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5983207

inputs
lstm_cell_5983125
lstm_cell_5983127
lstm_cell_5983129
identity’!lstm_cell/StatefulPartitionedCall’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5983125lstm_cell_5983127lstm_cell_5983129*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59827122#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5983125lstm_cell_5983127lstm_cell_5983129*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5983138*
condR
while_cond_5983137*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeς
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm―
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
ά	
ύ
.__inference_functional_1_layer_call_fn_5984136
words_input
layout_features_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallwords_inputlayout_features_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_59841172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????d
%
_user_specified_namewords_input:^Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input
?
κ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984068

inputs
inputs_1
words_embedding_5984045
words_lstm_5984048
words_lstm_5984050
words_lstm_5984052
dense_5984056
dense_5984058
main_output_5984062
main_output_5984064
identity’dense/StatefulPartitionedCall’dropout/StatefulPartitionedCall’#main_output/StatefulPartitionedCall’'words_embedding/StatefulPartitionedCall’"words_lstm/StatefulPartitionedCall€
'words_embedding/StatefulPartitionedCallStatefulPartitionedCallinputswords_embedding_5984045*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_words_embedding_layer_call_and_return_conditional_losses_59832302)
'words_embedding/StatefulPartitionedCallβ
"words_lstm/StatefulPartitionedCallStatefulPartitionedCall0words_embedding/StatefulPartitionedCall:output:0words_lstm_5984048words_lstm_5984050words_lstm_5984052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59836242$
"words_lstm/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCall+words_lstm/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_59839162
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_5984056dense_5984058*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_59839362
dense/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839642!
dropout/StatefulPartitionedCallΘ
#main_output/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0main_output_5984062main_output_5984064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_59839932%
#main_output/StatefulPartitionedCall·
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall$^main_output/StatefulPartitionedCall(^words_embedding/StatefulPartitionedCall#^words_lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2R
'words_embedding/StatefulPartitionedCall'words_embedding/StatefulPartitionedCall2H
"words_lstm/StatefulPartitionedCall"words_lstm/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ϋ
σ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5986217
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5986081*
condR
while_cond_5986080*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeς
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm―
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2
whilewhile:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
Λ
b
D__inference_dropout_layer_call_and_return_conditional_losses_5983969

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ά$

while_body_5983138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5983162_0
while_lstm_cell_5983164_0
while_lstm_cell_5983166_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5983162
while_lstm_cell_5983164
while_lstm_cell_5983166’'while/lstm_cell/StatefulPartitionedCallΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemΦ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5983162_0while_lstm_cell_5983164_0while_lstm_cell_5983166_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59827122)
'while/lstm_cell/StatefulPartitionedCallτ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ώ
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
while/Identity_4Ώ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5983162while_lstm_cell_5983162_0"4
while_lstm_cell_5983164while_lstm_cell_5983164_0"4
while_lstm_cell_5983166while_lstm_cell_5983166_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
Ε
ρ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985557

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:d?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5985421*
condR
while_cond_5985420*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::2
whilewhile:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
£
Y
-__inference_concatenate_layer_call_fn_5986252
inputs_0
inputs_1
identityΤ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_59839162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:?????????:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
΄
Θ
while_cond_5983423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5983423___redundant_placeholder05
1while_while_cond_5983423___redundant_placeholder15
1while_while_cond_5983423___redundant_placeholder25
1while_while_cond_5983423___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
ΠD
Χ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5983075

inputs
lstm_cell_5982993
lstm_cell_5982995
lstm_cell_5982997
identity’!lstm_cell/StatefulPartitionedCall’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5982993lstm_cell_5982995lstm_cell_5982997*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_59826282#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5982993lstm_cell_5982995lstm_cell_5982997*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5983006*
condR
while_cond_5983005*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeς
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm―
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
η

-__inference_main_output_layer_call_fn_5986319

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallψ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_59839932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ό

F__inference_lstm_cell_layer_call_and_return_conditional_losses_5982628

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeΤ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2δ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeΪ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Α₯2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_1/GreaterEqual/yΗ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeΪ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ψΰ2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_2/GreaterEqual/yΗ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeΪ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2αχΦ2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_3/GreaterEqual/yΗ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_3/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/ShapeΪ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2?γ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_4/GreaterEqual/yΗ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/ShapeΪ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Χά§2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_5/GreaterEqual/yΗ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/ShapeΪ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2³ή¦2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_6/GreaterEqual/yΗ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/ShapeΩ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Π2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_7/GreaterEqual/yΗ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_7/Mul_1_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_2e
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp―
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:?????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:?????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:?????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:?????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:?????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:?????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:?????????2
	BiasAdd_3e
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_4e
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_5e
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_6e
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ώ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:?????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:?????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:?????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:?????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:?????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:?????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:?????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????
 
_user_specified_namestates:PL
(
_output_shapes
:?????????
 
_user_specified_namestates
Α}
Υ
while_body_5985421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1Ί
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΎ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ύ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ύ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3£
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4£
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5£
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6£
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
΄
Θ
while_cond_5983137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5983137___redundant_placeholder05
1while_while_cond_5983137___redundant_placeholder15
1while_while_cond_5983137___redundant_placeholder25
1while_while_cond_5983137___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
΄
Θ
while_cond_5985101
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5985101___redundant_placeholder05
1while_while_cond_5985101___redundant_placeholder15
1while_while_cond_5985101___redundant_placeholder25
1while_while_cond_5985101___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
¬	
ν
.__inference_functional_1_layer_call_fn_5984881
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity’StatefulPartitionedCallΤ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_59840682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Μλ
ρ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5983624

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:d?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const¨
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeς
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ο’20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2"
 lstm_cell/dropout/GreaterEqual/yη
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout/Cast£
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeψ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2πβͺ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_1/GreaterEqual/yο
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_1/GreaterEqual€
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Cast«
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeψ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Δ?τ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_2/GreaterEqual/yο
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_2/GreaterEqual€
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Cast«
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeψ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2’·Π22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_3/GreaterEqual/yο
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_3/GreaterEqual€
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Cast«
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_4/Const°
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeψ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ώΉ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_4/GreaterEqual/yο
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_4/GreaterEqual€
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Cast«
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_5/Const°
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeψ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ν22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_5/GreaterEqual/yο
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_5/GreaterEqual€
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Cast«
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_6/Const°
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeψ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΑΑε22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_6/GreaterEqual/yο
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_6/GreaterEqual€
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Cast«
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_7/Const°
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeψ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ωΒΕ22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_7/GreaterEqual/yο
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_7/GreaterEqual€
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Cast«
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5983424*
condR
while_cond_5983423*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::2
whilewhile:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
§

,__inference_words_lstm_layer_call_fn_5986239
inputs_0
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59832072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:??????????????????
"
_user_specified_name
inputs/0
β
Ϊ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984037
words_input
layout_features_input
words_embedding_5984014
words_lstm_5984017
words_lstm_5984019
words_lstm_5984021
dense_5984025
dense_5984027
main_output_5984031
main_output_5984033
identity’dense/StatefulPartitionedCall’#main_output/StatefulPartitionedCall’'words_embedding/StatefulPartitionedCall’"words_lstm/StatefulPartitionedCall©
'words_embedding/StatefulPartitionedCallStatefulPartitionedCallwords_inputwords_embedding_5984014*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_words_embedding_layer_call_and_return_conditional_losses_59832302)
'words_embedding/StatefulPartitionedCallβ
"words_lstm/StatefulPartitionedCallStatefulPartitionedCall0words_embedding/StatefulPartitionedCall:output:0words_lstm_5984017words_lstm_5984019words_lstm_5984021*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59838792$
"words_lstm/StatefulPartitionedCall
concatenate/PartitionedCallPartitionedCall+words_lstm/StatefulPartitionedCall:output:0layout_features_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_59839162
concatenate/PartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_5984025dense_5984027*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_59839362
dense/StatefulPartitionedCallσ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_59839692
dropout/PartitionedCallΐ
#main_output/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0main_output_5984031main_output_5984033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_59839932%
#main_output/StatefulPartitionedCall
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^main_output/StatefulPartitionedCall(^words_embedding/StatefulPartitionedCall#^words_lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2R
'words_embedding/StatefulPartitionedCall'words_embedding/StatefulPartitionedCall2H
"words_lstm/StatefulPartitionedCall"words_lstm/StatefulPartitionedCall:T P
'
_output_shapes
:?????????d
%
_user_specified_namewords_input:^Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input
ώή
Υ
while_body_5985762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Constΐ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2΄Ϋ·26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2&
$while/lstm_cell/dropout/GreaterEqual°
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Cast»
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstΖ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ψ128
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_1/GreaterEqualΆ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_1/CastΓ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstΖ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ξ28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_2/GreaterEqualΆ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_2/CastΓ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstΖ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ύμg28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_3/GreaterEqualΆ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_3/CastΓ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_3/Mul_1
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_4/ConstΘ
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_4/Mul
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Οτ28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_4/GreaterEqual/y
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_4/GreaterEqualΆ
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_4/CastΓ
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_4/Mul_1
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_5/ConstΘ
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_5/Mul
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΐυ'28
6while/lstm_cell/dropout_5/random_uniform/RandomUniform
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_5/GreaterEqual/y
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_5/GreaterEqualΆ
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_5/CastΓ
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_5/Mul_1
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_6/ConstΘ
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_6/Mul
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Υα28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_6/GreaterEqual/y
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_6/GreaterEqualΆ
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_6/CastΓ
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_6/Mul_1
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_7/ConstΘ
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_7/Mul
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2©Λ28
6while/lstm_cell/dropout_7/random_uniform/RandomUniform
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_7/GreaterEqual/y
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_7/GreaterEqualΆ
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_7/CastΓ
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_7/Mul_1Ή
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΏ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ώ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ώ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3’
while/lstm_cell/mul_4Mulwhile_placeholder_2#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4’
while/lstm_cell/mul_5Mulwhile_placeholder_2#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5’
while/lstm_cell/mul_6Mulwhile_placeholder_2#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6’
while/lstm_cell/mul_7Mulwhile_placeholder_2#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
Μ

F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986467

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like/Const
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/ShapeΤ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2’Ή¨2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_1/Const
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeΩ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2¨L2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_1/GreaterEqual/yΗ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_1/GreaterEqual
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_1/Cast
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_2/Const
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeΪ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2»2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_2/GreaterEqual/yΗ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_2/GreaterEqual
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_2/Cast
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_2/Mul_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_3/Const
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeΪ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2?ό2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_3/GreaterEqual/yΗ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_3/GreaterEqual
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_3/Cast
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_3/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ones_like_1/Const
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
ones_like_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_4/Const
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/ShapeΪ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2?Μ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_4/GreaterEqual/yΗ
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_4/GreaterEqual
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_4/Cast
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_5/Const
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/ShapeΪ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2κρ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_5/GreaterEqual/yΗ
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_5/GreaterEqual
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_5/Cast
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_5/Mul_1g
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_6/Const
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_6/Mulf
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_6/ShapeΪ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Ύ‘2(
&dropout_6/random_uniform/RandomUniformy
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_6/GreaterEqual/yΗ
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_6/GreaterEqual
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_6/Cast
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_6/Mul_1g
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_7/Const
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_7/Mulf
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_7/ShapeΪ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2»2(
&dropout_7/random_uniform/RandomUniformy
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout_7/GreaterEqual/yΗ
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout_7/GreaterEqual
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_7/Cast
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_7/Mul_1_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_2e
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_3P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype02
split/ReadVariableOp―
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
splitf
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:?????????2
MatMull
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:?????????2

MatMul_1l
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:?????????2

MatMul_2l
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:?????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:?????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:?????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:?????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:?????????2
	BiasAdd_3g
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_4g
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_5g
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_6g
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
mul_7z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ώ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slicet
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:?????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:?????????2	
Sigmoid~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_1v
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2

MatMul_5r
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_1a
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:?????????2
mul_8~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_2v
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2

MatMul_6r
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:?????????2
Tanh_
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:?????????2
mul_9`
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:?????????2
add_3~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
strided_slice_3v
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2

MatMul_7r
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
add_4_
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:?????????2
	Sigmoid_2V
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:?????????2
Tanh_1e
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
mul_10_
IdentityIdentity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identityc

Identity_1Identity
mul_10:z:0*
T0*(
_output_shapes
:?????????2

Identity_1b

Identity_2Identity	add_3:z:0*
T0*(
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:?????????:?????????:?????????::::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
states/1

c
D__inference_dropout_layer_call_and_return_conditional_losses_5986284

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ
	
words_lstm_while_body_59847062
.words_lstm_while_words_lstm_while_loop_counter8
4words_lstm_while_words_lstm_while_maximum_iterations 
words_lstm_while_placeholder"
words_lstm_while_placeholder_1"
words_lstm_while_placeholder_2"
words_lstm_while_placeholder_31
-words_lstm_while_words_lstm_strided_slice_1_0m
iwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:words_lstm_while_lstm_cell_split_readvariableop_resource_0@
<words_lstm_while_lstm_cell_split_1_readvariableop_resource_08
4words_lstm_while_lstm_cell_readvariableop_resource_0
words_lstm_while_identity
words_lstm_while_identity_1
words_lstm_while_identity_2
words_lstm_while_identity_3
words_lstm_while_identity_4
words_lstm_while_identity_5/
+words_lstm_while_words_lstm_strided_slice_1k
gwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor<
8words_lstm_while_lstm_cell_split_readvariableop_resource>
:words_lstm_while_lstm_cell_split_1_readvariableop_resource6
2words_lstm_while_lstm_cell_readvariableop_resourceΩ
Bwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape
4words_lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0words_lstm_while_placeholderKwords_lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype026
4words_lstm/while/TensorArrayV2Read/TensorListGetItemΓ
*words_lstm/while/lstm_cell/ones_like/ShapeShape;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2,
*words_lstm/while/lstm_cell/ones_like/Shape
*words_lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*words_lstm/while/lstm_cell/ones_like/Constρ
$words_lstm/while/lstm_cell/ones_likeFill3words_lstm/while/lstm_cell/ones_like/Shape:output:03words_lstm/while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/ones_likeͺ
,words_lstm/while/lstm_cell/ones_like_1/ShapeShapewords_lstm_while_placeholder_2*
T0*
_output_shapes
:2.
,words_lstm/while/lstm_cell/ones_like_1/Shape‘
,words_lstm/while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,words_lstm/while/lstm_cell/ones_like_1/Constω
&words_lstm/while/lstm_cell/ones_like_1Fill5words_lstm/while/lstm_cell/ones_like_1/Shape:output:05words_lstm/while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2(
&words_lstm/while/lstm_cell/ones_like_1ζ
words_lstm/while/lstm_cell/mulMul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2 
words_lstm/while/lstm_cell/mulκ
 words_lstm/while/lstm_cell/mul_1Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_1κ
 words_lstm/while/lstm_cell/mul_2Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_2κ
 words_lstm/while/lstm_cell/mul_3Mul;words_lstm/while/TensorArrayV2Read/TensorListGetItem:item:0-words_lstm/while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_3
 words_lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 words_lstm/while/lstm_cell/Const
*words_lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*words_lstm/while/lstm_cell/split/split_dimί
/words_lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp:words_lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype021
/words_lstm/while/lstm_cell/split/ReadVariableOp
 words_lstm/while/lstm_cell/splitSplit3words_lstm/while/lstm_cell/split/split_dim:output:07words_lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2"
 words_lstm/while/lstm_cell/split?
!words_lstm/while/lstm_cell/MatMulMatMul"words_lstm/while/lstm_cell/mul:z:0)words_lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/MatMulΨ
#words_lstm/while/lstm_cell/MatMul_1MatMul$words_lstm/while/lstm_cell/mul_1:z:0)words_lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_1Ψ
#words_lstm/while/lstm_cell/MatMul_2MatMul$words_lstm/while/lstm_cell/mul_2:z:0)words_lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_2Ψ
#words_lstm/while/lstm_cell/MatMul_3MatMul$words_lstm/while/lstm_cell/mul_3:z:0)words_lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_3
"words_lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"words_lstm/while/lstm_cell/Const_1
,words_lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,words_lstm/while/lstm_cell/split_1/split_dimΰ
1words_lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp<words_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1words_lstm/while/lstm_cell/split_1/ReadVariableOp
"words_lstm/while/lstm_cell/split_1Split5words_lstm/while/lstm_cell/split_1/split_dim:output:09words_lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2$
"words_lstm/while/lstm_cell/split_1ΰ
"words_lstm/while/lstm_cell/BiasAddBiasAdd+words_lstm/while/lstm_cell/MatMul:product:0+words_lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/while/lstm_cell/BiasAddζ
$words_lstm/while/lstm_cell/BiasAdd_1BiasAdd-words_lstm/while/lstm_cell/MatMul_1:product:0+words_lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_1ζ
$words_lstm/while/lstm_cell/BiasAdd_2BiasAdd-words_lstm/while/lstm_cell/MatMul_2:product:0+words_lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_2ζ
$words_lstm/while/lstm_cell/BiasAdd_3BiasAdd-words_lstm/while/lstm_cell/MatMul_3:product:0+words_lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/BiasAdd_3Ο
 words_lstm/while/lstm_cell/mul_4Mulwords_lstm_while_placeholder_2/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_4Ο
 words_lstm/while/lstm_cell/mul_5Mulwords_lstm_while_placeholder_2/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_5Ο
 words_lstm/while/lstm_cell/mul_6Mulwords_lstm_while_placeholder_2/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_6Ο
 words_lstm/while/lstm_cell/mul_7Mulwords_lstm_while_placeholder_2/words_lstm/while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_7Ν
)words_lstm/while/lstm_cell/ReadVariableOpReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02+
)words_lstm/while/lstm_cell/ReadVariableOp±
.words_lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.words_lstm/while/lstm_cell/strided_slice/stack΅
0words_lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice/stack_1΅
0words_lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0words_lstm/while/lstm_cell/strided_slice/stack_2 
(words_lstm/while/lstm_cell/strided_sliceStridedSlice1words_lstm/while/lstm_cell/ReadVariableOp:value:07words_lstm/while/lstm_cell/strided_slice/stack:output:09words_lstm/while/lstm_cell/strided_slice/stack_1:output:09words_lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2*
(words_lstm/while/lstm_cell/strided_sliceΰ
#words_lstm/while/lstm_cell/MatMul_4MatMul$words_lstm/while/lstm_cell/mul_4:z:01words_lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_4Ψ
words_lstm/while/lstm_cell/addAddV2+words_lstm/while/lstm_cell/BiasAdd:output:0-words_lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2 
words_lstm/while/lstm_cell/addͺ
"words_lstm/while/lstm_cell/SigmoidSigmoid"words_lstm/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2$
"words_lstm/while/lstm_cell/SigmoidΡ
+words_lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_1΅
0words_lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_1/stackΉ
2words_lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2words_lstm/while/lstm_cell/strided_slice_1/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_1/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_1StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_1:value:09words_lstm/while/lstm_cell/strided_slice_1/stack:output:0;words_lstm/while/lstm_cell/strided_slice_1/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_1β
#words_lstm/while/lstm_cell/MatMul_5MatMul$words_lstm/while/lstm_cell/mul_5:z:03words_lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_5ή
 words_lstm/while/lstm_cell/add_1AddV2-words_lstm/while/lstm_cell/BiasAdd_1:output:0-words_lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_1°
$words_lstm/while/lstm_cell/Sigmoid_1Sigmoid$words_lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/Sigmoid_1Θ
 words_lstm/while/lstm_cell/mul_8Mul(words_lstm/while/lstm_cell/Sigmoid_1:y:0words_lstm_while_placeholder_3*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_8Ρ
+words_lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_2΅
0words_lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_2/stackΉ
2words_lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2words_lstm/while/lstm_cell/strided_slice_2/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_2/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_2StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_2:value:09words_lstm/while/lstm_cell/strided_slice_2/stack:output:0;words_lstm/while/lstm_cell/strided_slice_2/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_2β
#words_lstm/while/lstm_cell/MatMul_6MatMul$words_lstm/while/lstm_cell/mul_6:z:03words_lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_6ή
 words_lstm/while/lstm_cell/add_2AddV2-words_lstm/while/lstm_cell/BiasAdd_2:output:0-words_lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_2£
words_lstm/while/lstm_cell/TanhTanh$words_lstm/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2!
words_lstm/while/lstm_cell/TanhΛ
 words_lstm/while/lstm_cell/mul_9Mul&words_lstm/while/lstm_cell/Sigmoid:y:0#words_lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/mul_9Μ
 words_lstm/while/lstm_cell/add_3AddV2$words_lstm/while/lstm_cell/mul_8:z:0$words_lstm/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_3Ρ
+words_lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp4words_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02-
+words_lstm/while/lstm_cell/ReadVariableOp_3΅
0words_lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0words_lstm/while/lstm_cell/strided_slice_3/stackΉ
2words_lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2words_lstm/while/lstm_cell/strided_slice_3/stack_1Ή
2words_lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2words_lstm/while/lstm_cell/strided_slice_3/stack_2¬
*words_lstm/while/lstm_cell/strided_slice_3StridedSlice3words_lstm/while/lstm_cell/ReadVariableOp_3:value:09words_lstm/while/lstm_cell/strided_slice_3/stack:output:0;words_lstm/while/lstm_cell/strided_slice_3/stack_1:output:0;words_lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2,
*words_lstm/while/lstm_cell/strided_slice_3β
#words_lstm/while/lstm_cell/MatMul_7MatMul$words_lstm/while/lstm_cell/mul_7:z:03words_lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2%
#words_lstm/while/lstm_cell/MatMul_7ή
 words_lstm/while/lstm_cell/add_4AddV2-words_lstm/while/lstm_cell/BiasAdd_3:output:0-words_lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2"
 words_lstm/while/lstm_cell/add_4°
$words_lstm/while/lstm_cell/Sigmoid_2Sigmoid$words_lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2&
$words_lstm/while/lstm_cell/Sigmoid_2§
!words_lstm/while/lstm_cell/Tanh_1Tanh$words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/Tanh_1Ρ
!words_lstm/while/lstm_cell/mul_10Mul(words_lstm/while/lstm_cell/Sigmoid_2:y:0%words_lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2#
!words_lstm/while/lstm_cell/mul_10
5words_lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwords_lstm_while_placeholder_1words_lstm_while_placeholder%words_lstm/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype027
5words_lstm/while/TensorArrayV2Write/TensorListSetItemr
words_lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/while/add/y
words_lstm/while/addAddV2words_lstm_while_placeholderwords_lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
words_lstm/while/addv
words_lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
words_lstm/while/add_1/y­
words_lstm/while/add_1AddV2.words_lstm_while_words_lstm_while_loop_counter!words_lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
words_lstm/while/add_1
words_lstm/while/IdentityIdentitywords_lstm/while/add_1:z:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity
words_lstm/while/Identity_1Identity4words_lstm_while_words_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
words_lstm/while/Identity_1
words_lstm/while/Identity_2Identitywords_lstm/while/add:z:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity_2?
words_lstm/while/Identity_3IdentityEwords_lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
words_lstm/while/Identity_3 
words_lstm/while/Identity_4Identity%words_lstm/while/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/while/Identity_4
words_lstm/while/Identity_5Identity$words_lstm/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
words_lstm/while/Identity_5"?
words_lstm_while_identity"words_lstm/while/Identity:output:0"C
words_lstm_while_identity_1$words_lstm/while/Identity_1:output:0"C
words_lstm_while_identity_2$words_lstm/while/Identity_2:output:0"C
words_lstm_while_identity_3$words_lstm/while/Identity_3:output:0"C
words_lstm_while_identity_4$words_lstm/while/Identity_4:output:0"C
words_lstm_while_identity_5$words_lstm/while/Identity_5:output:0"j
2words_lstm_while_lstm_cell_readvariableop_resource4words_lstm_while_lstm_cell_readvariableop_resource_0"z
:words_lstm_while_lstm_cell_split_1_readvariableop_resource<words_lstm_while_lstm_cell_split_1_readvariableop_resource_0"v
8words_lstm_while_lstm_cell_split_readvariableop_resource:words_lstm_while_lstm_cell_split_readvariableop_resource_0"Τ
gwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensoriwords_lstm_while_tensorarrayv2read_tensorlistgetitem_words_lstm_tensorarrayunstack_tensorlistfromtensor_0"\
+words_lstm_while_words_lstm_strided_slice_1-words_lstm_while_words_lstm_strided_slice_1_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
Α
t
H__inference_concatenate_layer_call_and_return_conditional_losses_5986246
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????:?????????:R N
(
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Ε
ρ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5983879

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:d?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likex
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5983743*
condR
while_cond_5983742*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::2
whilewhile:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
΄
Θ
while_cond_5983742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5983742___redundant_placeholder05
1while_while_cond_5983742___redundant_placeholder15
1while_while_cond_5983742___redundant_placeholder25
1while_while_cond_5983742___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
Μλ
ρ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985302

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity’whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:?????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:d?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ξ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ώ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeψ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ύ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2~
lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout/Const¨
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeς
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2π€20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2"
 lstm_cell/dropout/GreaterEqual/yη
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout/Cast£
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_1/Const?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeψ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2λΔΒ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_1/GreaterEqual/yο
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_1/GreaterEqual€
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Cast«
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_2/Const?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeψ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2½ΰ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_2/GreaterEqual/yο
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_2/GreaterEqual€
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Cast«
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_3/Const?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeψ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Π22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_3/GreaterEqual/yο
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_3/GreaterEqual€
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Cast«
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_3/Mul_1x
lstm_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
lstm_cell/ones_like_1/Shape
lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like_1/Const΅
lstm_cell/ones_like_1Fill$lstm_cell/ones_like_1/Shape:output:0$lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/ones_like_1{
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_4/Const°
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_4/Shapeψ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2όΈ22
0lstm_cell/dropout_4/random_uniform/RandomUniform
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_4/GreaterEqual/yο
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_4/GreaterEqual€
lstm_cell/dropout_4/CastCast$lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Cast«
lstm_cell/dropout_4/Mul_1Mullstm_cell/dropout_4/Mul:z:0lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_4/Mul_1{
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_5/Const°
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_5/Shapeψ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΤΉ22
0lstm_cell/dropout_5/random_uniform/RandomUniform
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_5/GreaterEqual/yο
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_5/GreaterEqual€
lstm_cell/dropout_5/CastCast$lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Cast«
lstm_cell/dropout_5/Mul_1Mullstm_cell/dropout_5/Mul:z:0lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_5/Mul_1{
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_6/Const°
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_6/Shapeψ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΉΦ22
0lstm_cell/dropout_6/random_uniform/RandomUniform
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_6/GreaterEqual/yο
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_6/GreaterEqual€
lstm_cell/dropout_6/CastCast$lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Cast«
lstm_cell/dropout_6/Mul_1Mullstm_cell/dropout_6/Mul:z:0lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_6/Mul_1{
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/dropout_7/Const°
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:output:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_7/Shapeψ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2λάΫ22
0lstm_cell/dropout_7/random_uniform/RandomUniform
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2$
"lstm_cell/dropout_7/GreaterEqual/yο
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2"
 lstm_cell/dropout_7/GreaterEqual€
lstm_cell/dropout_7/CastCast$lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Cast«
lstm_cell/dropout_7/Mul_1Mullstm_cell/dropout_7/Mul:z:0lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/dropout_7/Mul_1
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_1
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_2
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_3d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimͺ
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype02 
lstm_cell/split/ReadVariableOpΧ
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/split_1/ReadVariableOpΛ
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd’
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_1’
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_2’
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
lstm_cell/BiasAdd_3
lstm_cell/mul_4Mulzeros:output:0lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_4
lstm_cell/mul_5Mulzeros:output:0lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_5
lstm_cell/mul_6Mulzeros:output:0lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_6
lstm_cell/mul_7Mulzeros:output:0lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_7
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2Ί
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/addw
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Ζ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_5
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_1}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_1
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_8
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Ζ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_6
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_2p
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_9
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_3
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Ζ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
lstm_cell/MatMul_7
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
lstm_cell/add_4}
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
lstm_cell/Tanh_1
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
lstm_cell/mul_10
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shapeΈ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterα
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5985102*
condR
while_cond_5985101*M
output_shapes<
:: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while΅
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shapeι
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:?????????d2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeu
IdentityIdentitystrided_slice_3:output:0^while*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::2
whilewhile:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
¦
±#
#__inference__traced_restore_5987033
file_prefix/
+assignvariableop_words_embedding_embeddings#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias)
%assignvariableop_3_main_output_kernel'
#assignvariableop_4_main_output_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate3
/assignvariableop_10_words_lstm_lstm_cell_kernel=
9assignvariableop_11_words_lstm_lstm_cell_recurrent_kernel1
-assignvariableop_12_words_lstm_lstm_cell_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1
assignvariableop_17_total_2
assignvariableop_18_count_2
assignvariableop_19_total_3
assignvariableop_20_count_3
assignvariableop_21_total_4
assignvariableop_22_count_4
assignvariableop_23_total_5
assignvariableop_24_count_5
assignvariableop_25_total_6
assignvariableop_26_count_6
assignvariableop_27_total_7
assignvariableop_28_count_7
assignvariableop_29_total_8
assignvariableop_30_count_8
assignvariableop_31_total_9
assignvariableop_32_count_9 
assignvariableop_33_total_10 
assignvariableop_34_count_10 
assignvariableop_35_total_11 
assignvariableop_36_count_11 
assignvariableop_37_total_12 
assignvariableop_38_count_12 
assignvariableop_39_total_13 
assignvariableop_40_count_13 
assignvariableop_41_total_14 
assignvariableop_42_count_14 
assignvariableop_43_total_15 
assignvariableop_44_count_159
5assignvariableop_45_adam_words_embedding_embeddings_m+
'assignvariableop_46_adam_dense_kernel_m)
%assignvariableop_47_adam_dense_bias_m1
-assignvariableop_48_adam_main_output_kernel_m/
+assignvariableop_49_adam_main_output_bias_m:
6assignvariableop_50_adam_words_lstm_lstm_cell_kernel_mD
@assignvariableop_51_adam_words_lstm_lstm_cell_recurrent_kernel_m8
4assignvariableop_52_adam_words_lstm_lstm_cell_bias_m9
5assignvariableop_53_adam_words_embedding_embeddings_v+
'assignvariableop_54_adam_dense_kernel_v)
%assignvariableop_55_adam_dense_bias_v1
-assignvariableop_56_adam_main_output_kernel_v/
+assignvariableop_57_adam_main_output_bias_v:
6assignvariableop_58_adam_words_lstm_lstm_cell_kernel_vD
@assignvariableop_59_adam_words_lstm_lstm_cell_recurrent_kernel_v8
4assignvariableop_60_adam_words_lstm_lstm_cell_bias_v<
8assignvariableop_61_adam_words_embedding_embeddings_vhat.
*assignvariableop_62_adam_dense_kernel_vhat,
(assignvariableop_63_adam_dense_bias_vhat4
0assignvariableop_64_adam_main_output_kernel_vhat2
.assignvariableop_65_adam_main_output_bias_vhat=
9assignvariableop_66_adam_words_lstm_lstm_cell_kernel_vhatG
Cassignvariableop_67_adam_words_lstm_lstm_cell_recurrent_kernel_vhat;
7assignvariableop_68_adam_words_lstm_lstm_cell_bias_vhat
identity_70’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Π"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*ά!
value?!BΟ!FB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/1/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/2/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBEvariables/3/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*‘
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityͺ
AssignVariableOpAssignVariableOp+assignvariableop_words_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1€
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2’
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ͺ
AssignVariableOp_3AssignVariableOp%assignvariableop_3_main_output_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_main_output_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5‘
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8’
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ͺ
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_words_lstm_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Α
AssignVariableOp_11AssignVariableOp9assignvariableop_11_words_lstm_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12΅
AssignVariableOp_12AssignVariableOp-assignvariableop_12_words_lstm_lstm_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13‘
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14‘
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_3Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_3Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_4Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_4Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23£
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_5Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24£
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_5Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_6Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_6Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_7Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28£
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_7Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_8Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30£
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_8Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31£
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_9Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32£
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_9Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33€
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_10Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34€
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_10Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35€
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_11Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36€
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_11Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37€
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_12Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38€
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_12Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39€
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_13Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40€
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_13Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41€
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_14Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42€
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_14Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43€
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_15Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44€
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_15Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45½
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adam_words_embedding_embeddings_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46―
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47­
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_dense_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48΅
AssignVariableOp_48AssignVariableOp-assignvariableop_48_adam_main_output_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_main_output_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ύ
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_words_lstm_lstm_cell_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Θ
AssignVariableOp_51AssignVariableOp@assignvariableop_51_adam_words_lstm_lstm_cell_recurrent_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ό
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_words_lstm_lstm_cell_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53½
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_words_embedding_embeddings_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54―
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55­
AssignVariableOp_55AssignVariableOp%assignvariableop_55_adam_dense_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56΅
AssignVariableOp_56AssignVariableOp-assignvariableop_56_adam_main_output_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_main_output_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ύ
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_words_lstm_lstm_cell_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Θ
AssignVariableOp_59AssignVariableOp@assignvariableop_59_adam_words_lstm_lstm_cell_recurrent_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ό
AssignVariableOp_60AssignVariableOp4assignvariableop_60_adam_words_lstm_lstm_cell_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61ΐ
AssignVariableOp_61AssignVariableOp8assignvariableop_61_adam_words_embedding_embeddings_vhatIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62²
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_kernel_vhatIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63°
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_bias_vhatIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Έ
AssignVariableOp_64AssignVariableOp0assignvariableop_64_adam_main_output_kernel_vhatIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ά
AssignVariableOp_65AssignVariableOp.assignvariableop_65_adam_main_output_bias_vhatIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Α
AssignVariableOp_66AssignVariableOp9assignvariableop_66_adam_words_lstm_lstm_cell_kernel_vhatIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Λ
AssignVariableOp_67AssignVariableOpCassignvariableop_67_adam_words_lstm_lstm_cell_recurrent_kernel_vhatIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ώ
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_words_lstm_lstm_cell_bias_vhatIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΜ
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69Ώ
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_70"#
identity_70Identity_70:output:0*«
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
΄
Θ
while_cond_5983005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5983005___redundant_placeholder05
1while_while_cond_5983005___redundant_placeholder15
1while_while_cond_5983005___redundant_placeholder25
1while_while_cond_5983005___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?ή
Υ
while_body_5983424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/dropout/Constΐ
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/lstm_cell/dropout/Shape
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2»·26
4while/lstm_cell/dropout/random_uniform/RandomUniform
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2(
&while/lstm_cell/dropout/GreaterEqual/y?
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2&
$while/lstm_cell/dropout/GreaterEqual°
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Cast»
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout/Mul_1
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_1/ConstΖ
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_1/Mul
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_1/Shape
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2δ΅?28
6while/lstm_cell/dropout_1/random_uniform/RandomUniform
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_1/GreaterEqual/y
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_1/GreaterEqualΆ
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_1/CastΓ
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_1/Mul_1
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_2/ConstΖ
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_2/Mul
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_2/Shape
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ό‘28
6while/lstm_cell/dropout_2/random_uniform/RandomUniform
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_2/GreaterEqual/y
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_2/GreaterEqualΆ
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_2/CastΓ
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_2/Mul_1
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_3/ConstΖ
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_3/Mul
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_3/Shape
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2Β28
6while/lstm_cell/dropout_3/random_uniform/RandomUniform
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_3/GreaterEqual/y
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_3/GreaterEqualΆ
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_3/CastΓ
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_3/Mul_1
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_4/ConstΘ
while/lstm_cell/dropout_4/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_4/Mul
while/lstm_cell/dropout_4/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_4/Shape
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΔΡ₯28
6while/lstm_cell/dropout_4/random_uniform/RandomUniform
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_4/GreaterEqual/y
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_4/GreaterEqualΆ
while/lstm_cell/dropout_4/CastCast*while/lstm_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_4/CastΓ
while/lstm_cell/dropout_4/Mul_1Mul!while/lstm_cell/dropout_4/Mul:z:0"while/lstm_cell/dropout_4/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_4/Mul_1
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_5/ConstΘ
while/lstm_cell/dropout_5/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_5/Mul
while/lstm_cell/dropout_5/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_5/Shape
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2πϋ828
6while/lstm_cell/dropout_5/random_uniform/RandomUniform
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_5/GreaterEqual/y
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_5/GreaterEqualΆ
while/lstm_cell/dropout_5/CastCast*while/lstm_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_5/CastΓ
while/lstm_cell/dropout_5/Mul_1Mul!while/lstm_cell/dropout_5/Mul:z:0"while/lstm_cell/dropout_5/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_5/Mul_1
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_6/ConstΘ
while/lstm_cell/dropout_6/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_6/Mul
while/lstm_cell/dropout_6/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_6/Shape
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2λΝΛ28
6while/lstm_cell/dropout_6/random_uniform/RandomUniform
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_6/GreaterEqual/y
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_6/GreaterEqualΆ
while/lstm_cell/dropout_6/CastCast*while/lstm_cell/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_6/CastΓ
while/lstm_cell/dropout_6/Mul_1Mul!while/lstm_cell/dropout_6/Mul:z:0"while/lstm_cell/dropout_6/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_6/Mul_1
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
while/lstm_cell/dropout_7/ConstΘ
while/lstm_cell/dropout_7/MulMul$while/lstm_cell/ones_like_1:output:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/dropout_7/Mul
while/lstm_cell/dropout_7/ShapeShape$while/lstm_cell/ones_like_1:output:0*
T0*
_output_shapes
:2!
while/lstm_cell/dropout_7/Shape
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0*
seed±?ε)*
seed2ΗΎq28
6while/lstm_cell/dropout_7/random_uniform/RandomUniform
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2*
(while/lstm_cell/dropout_7/GreaterEqual/y
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2(
&while/lstm_cell/dropout_7/GreaterEqualΆ
while/lstm_cell/dropout_7/CastCast*while/lstm_cell/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2 
while/lstm_cell/dropout_7/CastΓ
while/lstm_cell/dropout_7/Mul_1Mul!while/lstm_cell/dropout_7/Mul:z:0"while/lstm_cell/dropout_7/Cast:y:0*
T0*(
_output_shapes
:?????????2!
while/lstm_cell/dropout_7/Mul_1Ή
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΏ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ώ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ώ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3’
while/lstm_cell/mul_4Mulwhile_placeholder_2#while/lstm_cell/dropout_4/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4’
while/lstm_cell/mul_5Mulwhile_placeholder_2#while/lstm_cell/dropout_5/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5’
while/lstm_cell/mul_6Mulwhile_placeholder_2#while/lstm_cell/dropout_6/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6’
while/lstm_cell/mul_7Mulwhile_placeholder_2#while/lstm_cell/dropout_7/Mul_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?

€
words_lstm_while_cond_59843562
.words_lstm_while_words_lstm_while_loop_counter8
4words_lstm_while_words_lstm_while_maximum_iterations 
words_lstm_while_placeholder"
words_lstm_while_placeholder_1"
words_lstm_while_placeholder_2"
words_lstm_while_placeholder_34
0words_lstm_while_less_words_lstm_strided_slice_1K
Gwords_lstm_while_words_lstm_while_cond_5984356___redundant_placeholder0K
Gwords_lstm_while_words_lstm_while_cond_5984356___redundant_placeholder1K
Gwords_lstm_while_words_lstm_while_cond_5984356___redundant_placeholder2K
Gwords_lstm_while_words_lstm_while_cond_5984356___redundant_placeholder3
words_lstm_while_identity
§
words_lstm/while/LessLesswords_lstm_while_placeholder0words_lstm_while_less_words_lstm_strided_slice_1*
T0*
_output_shapes
: 2
words_lstm/while/Less~
words_lstm/while/IdentityIdentitywords_lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
words_lstm/while/Identity"?
words_lstm_while_identity"words_lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
Έ
°
H__inference_main_output_layer_call_and_return_conditional_losses_5986310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


,__inference_words_lstm_layer_call_fn_5985579

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_words_lstm_layer_call_and_return_conditional_losses_59838792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????d
 
_user_specified_nameinputs
β

L__inference_words_embedding_layer_call_and_return_conditional_losses_5983230

inputs
embedding_lookup_5983224
identityΠ
embedding_lookupResourceGatherembedding_lookup_5983224inputs*
Tindices0*+
_class!
loc:@embedding_lookup/5983224*,
_output_shapes
:?????????d*
dtype02
embedding_lookupΑ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/5983224*,
_output_shapes
:?????????d2
embedding_lookup/Identity‘
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:?????????d2
embedding_lookup/Identity_1}
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
ά	
ύ
.__inference_functional_1_layer_call_fn_5984087
words_input
layout_features_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallwords_inputlayout_features_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_functional_1_layer_call_and_return_conditional_losses_59840682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????d:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????d
%
_user_specified_namewords_input:^Z
'
_output_shapes
:?????????
/
_user_specified_namelayout_features_input
Α}
Υ
while_body_5983743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceΓ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeΤ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem’
while/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2!
while/lstm_cell/ones_like/Shape
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/lstm_cell/ones_like/ConstΕ
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like
!while/lstm_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2#
!while/lstm_cell/ones_like_1/Shape
!while/lstm_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!while/lstm_cell/ones_like_1/ConstΝ
while/lstm_cell/ones_like_1Fill*while/lstm_cell/ones_like_1/Shape:output:0*while/lstm_cell/ones_like_1/Const:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/ones_like_1Ί
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mulΎ
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_1Ύ
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_2Ύ
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_3p
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dimΎ
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype02&
$while/lstm_cell/split/ReadVariableOpο
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split2
while/lstm_cell/split¦
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul¬
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_1¬
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_2¬
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dimΏ
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype02(
&while/lstm_cell/split_1/ReadVariableOpγ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split2
while/lstm_cell/split_1΄
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAddΊ
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_1Ί
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_2Ί
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/BiasAdd_3£
while/lstm_cell/mul_4Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_4£
while/lstm_cell/mul_5Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_5£
while/lstm_cell/mul_6Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_6£
while/lstm_cell/mul_7Mulwhile_placeholder_2$while/lstm_cell/ones_like_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_7¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2ή
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2
while/lstm_cell/strided_slice΄
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2κ
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1Ά
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_1
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_8°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2κ
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2Ά
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_2
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_9 
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_3°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2κ
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3Ά
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/add_4
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/Tanh_1₯
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:?????????2
while/lstm_cell/mul_10ή
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :?????????:?????????: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:?????????:.*
(
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: "ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultϋ
W
layout_features_input>
'serving_default_layout_features_input:0?????????
C
words_input4
serving_default_words_input:0?????????d?
main_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ΆΗ
«B
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
Ϊ_default_save_signature
+Ϋ&call_and_return_all_conditional_losses
ά__call__"?
_tf_keras_networkρ>{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "words_input"}, "name": "words_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "words_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 2259, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}, "name": "words_embedding", "inbound_nodes": [[["words_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "words_lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}, "name": "words_lstm", "inbound_nodes": [[["words_embedding", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layout_features_input"}, "name": "layout_features_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["words_lstm", 0, 0, {}], ["layout_features_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "main_output", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "main_output", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["words_input", 0, 0], ["layout_features_input", 0, 0]], "output_layers": [["main_output", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100]}, {"class_name": "TensorShape", "items": [null, 15]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "words_input"}, "name": "words_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "words_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 2259, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}, "name": "words_embedding", "inbound_nodes": [[["words_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "words_lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}, "name": "words_lstm", "inbound_nodes": [[["words_embedding", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layout_features_input"}, "name": "layout_features_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["words_lstm", 0, 0, {}], ["layout_features_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "main_output", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "main_output", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["words_input", 0, 0], ["layout_features_input", 0, 0]], "output_layers": [["main_output", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["acc_PARAGRAPH", "acc_REFERENCE", "acc_MARGINAL", "acc_FOOTNOTE", "acc_HEADING", "acc_FORMULA", "acc_TITLE", "acc_AUTHOR_INFO", "acc_ABSTRACT", "acc_DATE", "acc_CAPTION", "acc_TABLE", "acc_OTHER", "acc_TABLE_OF_CONTENTS", "accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": true}}}}
ρ"ξ
_tf_keras_input_layerΞ{"class_name": "InputLayer", "name": "words_input", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "words_input"}}
·

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
+έ&call_and_return_all_conditional_losses
ή__call__"
_tf_keras_layerό{"class_name": "Embedding", "name": "words_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "words_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 2259, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Λ
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
+ί&call_and_return_all_conditional_losses
ΰ__call__" 

_tf_keras_rnn_layer
{"class_name": "LSTM", "name": "words_lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "words_lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 256]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 256]}}
"
_tf_keras_input_layerδ{"class_name": "InputLayer", "name": "layout_features_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layout_features_input"}}
Μ
regularization_losses
trainable_variables
	variables
	keras_api
+α&call_and_return_all_conditional_losses
β__call__"»
_tf_keras_layer‘{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256]}, {"class_name": "TensorShape", "items": [null, 15]}]}
ρ

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
+γ&call_and_return_all_conditional_losses
δ__call__"Κ
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 271}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 271]}}
γ
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+ε&call_and_return_all_conditional_losses
ζ__call__"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+η&call_and_return_all_conditional_losses
θ__call__"Ψ
_tf_keras_layerΎ{"class_name": "Dense", "name": "main_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "main_output", "trainable": true, "dtype": "float32", "units": 14, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ϋ
.iter

/beta_1

0beta_2
	1decay
2learning_ratemΒmΓmΔ(mΕ)mΖ3mΗ4mΘ5mΙvΚvΛvΜ(vΝ)vΞ3vΟ4vΠ5vΡvhat?vhatΣvhatΤ(vhatΥ)vhatΦ3vhatΧ4vhatΨ5vhatΩ"
	optimizer
 "
trackable_list_wrapper
X
0
31
42
53
4
5
(6
)7"
trackable_list_wrapper
X
0
31
42
53
4
5
(6
)7"
trackable_list_wrapper
Ξ
6metrics

7layers

regularization_losses
	variables
trainable_variables
8non_trainable_variables
9layer_metrics
:layer_regularization_losses
ά__call__
Ϊ_default_save_signature
+Ϋ&call_and_return_all_conditional_losses
'Ϋ"call_and_return_conditional_losses"
_generic_user_object
-
ιserving_default"
signature_map
.:,
Σ2words_embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
°
;metrics

<layers
regularization_losses
trainable_variables
	variables
=non_trainable_variables
>layer_metrics
?layer_regularization_losses
ή__call__
+έ&call_and_return_all_conditional_losses
'έ"call_and_return_conditional_losses"
_generic_user_object
§

3kernel
4recurrent_kernel
5bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+κ&call_and_return_all_conditional_losses
λ__call__"κ
_tf_keras_layerΠ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.2, "recurrent_dropout": 0.2, "implementation": 1}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
Ό

Dstates
Emetrics

Flayers
regularization_losses
trainable_variables
	variables
Gnon_trainable_variables
Hlayer_metrics
Ilayer_regularization_losses
ΰ__call__
+ί&call_and_return_all_conditional_losses
'ί"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Jmetrics

Klayers
regularization_losses
trainable_variables
	variables
Lnon_trainable_variables
Mlayer_metrics
Nlayer_regularization_losses
β__call__
+α&call_and_return_all_conditional_losses
'α"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
Ometrics

Players
 regularization_losses
!trainable_variables
"	variables
Qnon_trainable_variables
Rlayer_metrics
Slayer_regularization_losses
δ__call__
+γ&call_and_return_all_conditional_losses
'γ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Tmetrics

Ulayers
$regularization_losses
%trainable_variables
&	variables
Vnon_trainable_variables
Wlayer_metrics
Xlayer_regularization_losses
ζ__call__
+ε&call_and_return_all_conditional_losses
'ε"call_and_return_conditional_losses"
_generic_user_object
%:#	2main_output/kernel
:2main_output/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°
Ymetrics

Zlayers
*regularization_losses
+trainable_variables
,	variables
[non_trainable_variables
\layer_metrics
]layer_regularization_losses
θ__call__
+η&call_and_return_all_conditional_losses
'η"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-
2words_lstm/lstm_cell/kernel
9:7
2%words_lstm/lstm_cell/recurrent_kernel
(:&2words_lstm/lstm_cell/bias

^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
l14
m15"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
5
30
41
52"
trackable_list_wrapper
°
nmetrics

olayers
@regularization_losses
Atrainable_variables
B	variables
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
λ__call__
+κ&call_and_return_all_conditional_losses
'κ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
»
	stotal
	tcount
u	variables
v	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	wtotal
	xcount
y
_fn_kwargs
z	variables
{	keras_api"»
_tf_keras_metric {"class_name": "MeanMetricWrapper", "name": "acc_PARAGRAPH", "dtype": "float32", "config": {"name": "acc_PARAGRAPH", "dtype": "float32", "fn": "acc_PARAGRAPH"}}

	|total
	}count
~
_fn_kwargs
	variables
	keras_api"»
_tf_keras_metric {"class_name": "MeanMetricWrapper", "name": "acc_REFERENCE", "dtype": "float32", "config": {"name": "acc_REFERENCE", "dtype": "float32", "fn": "acc_REFERENCE"}}


total

count

_fn_kwargs
	variables
	keras_api"Έ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_MARGINAL", "dtype": "float32", "config": {"name": "acc_MARGINAL", "dtype": "float32", "fn": "acc_MARGINAL"}}


total

count

_fn_kwargs
	variables
	keras_api"Έ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_FOOTNOTE", "dtype": "float32", "config": {"name": "acc_FOOTNOTE", "dtype": "float32", "fn": "acc_FOOTNOTE"}}


total

count

_fn_kwargs
	variables
	keras_api"΅
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_HEADING", "dtype": "float32", "config": {"name": "acc_HEADING", "dtype": "float32", "fn": "acc_HEADING"}}


total

count

_fn_kwargs
	variables
	keras_api"΅
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_FORMULA", "dtype": "float32", "config": {"name": "acc_FORMULA", "dtype": "float32", "fn": "acc_FORMULA"}}
ϋ

total

count

_fn_kwargs
	variables
	keras_api"―
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_TITLE", "dtype": "float32", "config": {"name": "acc_TITLE", "dtype": "float32", "fn": "acc_TITLE"}}


total

count

_fn_kwargs
	variables
	keras_api"Α
_tf_keras_metric¦{"class_name": "MeanMetricWrapper", "name": "acc_AUTHOR_INFO", "dtype": "float32", "config": {"name": "acc_AUTHOR_INFO", "dtype": "float32", "fn": "acc_AUTHOR_INFO"}}


total

 count
‘
_fn_kwargs
’	variables
£	keras_api"Έ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_ABSTRACT", "dtype": "float32", "config": {"name": "acc_ABSTRACT", "dtype": "float32", "fn": "acc_ABSTRACT"}}
ψ

€total

₯count
¦
_fn_kwargs
§	variables
¨	keras_api"¬
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_DATE", "dtype": "float32", "config": {"name": "acc_DATE", "dtype": "float32", "fn": "acc_DATE"}}


©total

ͺcount
«
_fn_kwargs
¬	variables
­	keras_api"΅
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_CAPTION", "dtype": "float32", "config": {"name": "acc_CAPTION", "dtype": "float32", "fn": "acc_CAPTION"}}
ϋ

?total

―count
°
_fn_kwargs
±	variables
²	keras_api"―
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_TABLE", "dtype": "float32", "config": {"name": "acc_TABLE", "dtype": "float32", "fn": "acc_TABLE"}}
ϋ

³total

΄count
΅
_fn_kwargs
Ά	variables
·	keras_api"―
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "acc_OTHER", "dtype": "float32", "config": {"name": "acc_OTHER", "dtype": "float32", "fn": "acc_OTHER"}}


Έtotal

Ήcount
Ί
_fn_kwargs
»	variables
Ό	keras_api"Σ
_tf_keras_metricΈ{"class_name": "MeanMetricWrapper", "name": "acc_TABLE_OF_CONTENTS", "dtype": "float32", "config": {"name": "acc_TABLE_OF_CONTENTS", "dtype": "float32", "fn": "acc_TABLE_OF_CONTENTS"}}


½total

Ύcount
Ώ
_fn_kwargs
ΐ	variables
Α	keras_api"Έ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
s0
t1"
trackable_list_wrapper
-
u	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
 1"
trackable_list_wrapper
.
’	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
€0
₯1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
©0
ͺ1"
trackable_list_wrapper
.
¬	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
―1"
trackable_list_wrapper
.
±	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
³0
΄1"
trackable_list_wrapper
.
Ά	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Έ0
Ή1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
½0
Ύ1"
trackable_list_wrapper
.
ΐ	variables"
_generic_user_object
3:1
Σ2!Adam/words_embedding/embeddings/m
%:#
2Adam/dense/kernel/m
:2Adam/dense/bias/m
*:(	2Adam/main_output/kernel/m
#:!2Adam/main_output/bias/m
4:2
2"Adam/words_lstm/lstm_cell/kernel/m
>:<
2,Adam/words_lstm/lstm_cell/recurrent_kernel/m
-:+2 Adam/words_lstm/lstm_cell/bias/m
3:1
Σ2!Adam/words_embedding/embeddings/v
%:#
2Adam/dense/kernel/v
:2Adam/dense/bias/v
*:(	2Adam/main_output/kernel/v
#:!2Adam/main_output/bias/v
4:2
2"Adam/words_lstm/lstm_cell/kernel/v
>:<
2,Adam/words_lstm/lstm_cell/recurrent_kernel/v
-:+2 Adam/words_lstm/lstm_cell/bias/v
6:4
Σ2$Adam/words_embedding/embeddings/vhat
(:&
2Adam/dense/kernel/vhat
!:2Adam/dense/bias/vhat
-:+	2Adam/main_output/kernel/vhat
&:$2Adam/main_output/bias/vhat
7:5
2%Adam/words_lstm/lstm_cell/kernel/vhat
A:?
2/Adam/words_lstm/lstm_cell/recurrent_kernel/vhat
0:.2#Adam/words_lstm/lstm_cell/bias/vhat
2
"__inference__wrapped_model_5982440π
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *`’]
[X
%"
words_input?????????d
/,
layout_features_input?????????
ς2ο
I__inference_functional_1_layer_call_and_return_conditional_losses_5984859
I__inference_functional_1_layer_call_and_return_conditional_losses_5984010
I__inference_functional_1_layer_call_and_return_conditional_losses_5984581
I__inference_functional_1_layer_call_and_return_conditional_losses_5984037ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_functional_1_layer_call_fn_5984136
.__inference_functional_1_layer_call_fn_5984881
.__inference_functional_1_layer_call_fn_5984087
.__inference_functional_1_layer_call_fn_5984903ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
φ2σ
L__inference_words_embedding_layer_call_and_return_conditional_losses_5984912’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ϋ2Ψ
1__inference_words_embedding_layer_call_fn_5984919’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2ό
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985557
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985962
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985302
G__inference_words_lstm_layer_call_and_return_conditional_losses_5986217Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
,__inference_words_lstm_layer_call_fn_5986239
,__inference_words_lstm_layer_call_fn_5985568
,__inference_words_lstm_layer_call_fn_5986228
,__inference_words_lstm_layer_call_fn_5985579Υ
Μ²Θ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
H__inference_concatenate_layer_call_and_return_conditional_losses_5986246’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Χ2Τ
-__inference_concatenate_layer_call_fn_5986252’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
μ2ι
B__inference_dense_layer_call_and_return_conditional_losses_5986263’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_dense_layer_call_fn_5986272’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ2Γ
D__inference_dropout_layer_call_and_return_conditional_losses_5986284
D__inference_dropout_layer_call_and_return_conditional_losses_5986289΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
)__inference_dropout_layer_call_fn_5986299
)__inference_dropout_layer_call_fn_5986294΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
H__inference_main_output_layer_call_and_return_conditional_losses_5986310’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Χ2Τ
-__inference_main_output_layer_call_fn_5986319’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
MBK
%__inference_signature_wrapper_5984168layout_features_inputwords_input
Τ2Ρ
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986551
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986467Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_lstm_cell_layer_call_fn_5986568
+__inference_lstm_cell_layer_call_fn_5986585Ύ
΅²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 Ψ
"__inference__wrapped_model_5982440±354()j’g
`’]
[X
%"
words_input?????????d
/,
layout_features_input?????????
ͺ "9ͺ6
4
main_output%"
main_output??????????
H__inference_concatenate_layer_call_and_return_conditional_losses_5986246[’X
Q’N
LI
# 
inputs/0?????????
"
inputs/1?????????
ͺ "&’#

0?????????
 ©
-__inference_concatenate_layer_call_fn_5986252x[’X
Q’N
LI
# 
inputs/0?????????
"
inputs/1?????????
ͺ "?????????€
B__inference_dense_layer_call_and_return_conditional_losses_5986263^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 |
'__inference_dense_layer_call_fn_5986272Q0’-
&’#
!
inputs?????????
ͺ "?????????¦
D__inference_dropout_layer_call_and_return_conditional_losses_5986284^4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 ¦
D__inference_dropout_layer_call_and_return_conditional_losses_5986289^4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 ~
)__inference_dropout_layer_call_fn_5986294Q4’1
*’'
!
inputs?????????
p
ͺ "?????????~
)__inference_dropout_layer_call_fn_5986299Q4’1
*’'
!
inputs?????????
p 
ͺ "?????????σ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984010₯354()r’o
h’e
[X
%"
words_input?????????d
/,
layout_features_input?????????
p

 
ͺ "%’"

0?????????
 σ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984037₯354()r’o
h’e
[X
%"
words_input?????????d
/,
layout_features_input?????????
p 

 
ͺ "%’"

0?????????
 γ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984581354()b’_
X’U
KH
"
inputs/0?????????d
"
inputs/1?????????
p

 
ͺ "%’"

0?????????
 γ
I__inference_functional_1_layer_call_and_return_conditional_losses_5984859354()b’_
X’U
KH
"
inputs/0?????????d
"
inputs/1?????????
p 

 
ͺ "%’"

0?????????
 Λ
.__inference_functional_1_layer_call_fn_5984087354()r’o
h’e
[X
%"
words_input?????????d
/,
layout_features_input?????????
p

 
ͺ "?????????Λ
.__inference_functional_1_layer_call_fn_5984136354()r’o
h’e
[X
%"
words_input?????????d
/,
layout_features_input?????????
p 

 
ͺ "?????????»
.__inference_functional_1_layer_call_fn_5984881354()b’_
X’U
KH
"
inputs/0?????????d
"
inputs/1?????????
p

 
ͺ "?????????»
.__inference_functional_1_layer_call_fn_5984903354()b’_
X’U
KH
"
inputs/0?????????d
"
inputs/1?????????
p 

 
ͺ "?????????Ο
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986467354’
y’v
!
inputs?????????
M’J
# 
states/0?????????
# 
states/1?????????
p
ͺ "v’s
l’i

0/0?????????
GD
 
0/1/0?????????
 
0/1/1?????????
 Ο
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5986551354’
y’v
!
inputs?????????
M’J
# 
states/0?????????
# 
states/1?????????
p 
ͺ "v’s
l’i

0/0?????????
GD
 
0/1/0?????????
 
0/1/1?????????
 €
+__inference_lstm_cell_layer_call_fn_5986568τ354’
y’v
!
inputs?????????
M’J
# 
states/0?????????
# 
states/1?????????
p
ͺ "f’c

0?????????
C@

1/0?????????

1/1?????????€
+__inference_lstm_cell_layer_call_fn_5986585τ354’
y’v
!
inputs?????????
M’J
# 
states/0?????????
# 
states/1?????????
p 
ͺ "f’c

0?????????
C@

1/0?????????

1/1?????????©
H__inference_main_output_layer_call_and_return_conditional_losses_5986310]()0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
-__inference_main_output_layer_call_fn_5986319P()0’-
&’#
!
inputs?????????
ͺ "?????????
%__inference_signature_wrapper_5984168Ψ354()’
’ 
ͺ
H
layout_features_input/,
layout_features_input?????????
4
words_input%"
words_input?????????d"9ͺ6
4
main_output%"
main_output?????????°
L__inference_words_embedding_layer_call_and_return_conditional_losses_5984912`/’,
%’"
 
inputs?????????d
ͺ "*’'
 
0?????????d
 
1__inference_words_embedding_layer_call_fn_5984919S/’,
%’"
 
inputs?????????d
ͺ "?????????dΊ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985302o354@’=
6’3
%"
inputs?????????d

 
p

 
ͺ "&’#

0?????????
 Ί
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985557o354@’=
6’3
%"
inputs?????????d

 
p 

 
ͺ "&’#

0?????????
 Κ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5985962354P’M
F’C
52
0-
inputs/0??????????????????

 
p

 
ͺ "&’#

0?????????
 Κ
G__inference_words_lstm_layer_call_and_return_conditional_losses_5986217354P’M
F’C
52
0-
inputs/0??????????????????

 
p 

 
ͺ "&’#

0?????????
 
,__inference_words_lstm_layer_call_fn_5985568b354@’=
6’3
%"
inputs?????????d

 
p

 
ͺ "?????????
,__inference_words_lstm_layer_call_fn_5985579b354@’=
6’3
%"
inputs?????????d

 
p 

 
ͺ "?????????’
,__inference_words_lstm_layer_call_fn_5986228r354P’M
F’C
52
0-
inputs/0??????????????????

 
p

 
ͺ "?????????’
,__inference_words_lstm_layer_call_fn_5986239r354P’M
F’C
52
0-
inputs/0??????????????????

 
p 

 
ͺ "?????????