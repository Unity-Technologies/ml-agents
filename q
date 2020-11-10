digraph {
	graph [size="12,12"]
	node [align=left fontsize=12 height=0.2 ranksep=0.1 shape=box style=filled]
	5599426656 [label=MeanBackward1 fillcolor=darkolivegreen1]
	5613756312 -> 5599426656
	5613756312 [label=StackBackward]
	5613977952 -> 5613756312
	5613977952 [label=SumBackward1]
	5613978120 -> 5613977952
	5613978120 [label=MulBackward0]
	5613978176 -> 5613978120
	5613978176 [label=SliceBackward]
	5613978288 -> 5613978176
	5613978288 [label=SqueezeBackward1]
	5613978400 -> 5613978288
	5613978400 [label=AddmmBackward]
	5613978512 -> 5613978400
	5613978512 [label="q1_network.q_heads.value_heads.extrinsic.bias
 (3)" fillcolor=lightblue]
	5613978568 -> 5613978400
	5613978568 [label=SumBackward1]
	5613978904 -> 5613978568
	5613978904 [label=StackBackward]
	5613979016 -> 5613978904
	5613979016 [label=ReluBackward0]
	5613979128 -> 5613979016
	5613979128 [label=AddmmBackward]
	5613979240 -> 5613979128
	5613979240 [label="q1_network.vector_encoders.0.seq_layers.1.bias
 (20)" fillcolor=lightblue]
	5613979296 -> 5613979128
	5613979296 [label=AddmmBackward]
	5613978736 -> 5613979296
	5613978736 [label="q1_network.vector_encoders.0.seq_layers.0.bias
 (20)" fillcolor=lightblue]
	5613979520 -> 5613979296
	5613979520 [label=TBackward]
	5613979464 -> 5613979520
	5613979464 [label="q1_network.vector_encoders.0.seq_layers.0.weight
 (20, 20)" fillcolor=lightblue]
	5613979352 -> 5613979128
	5613979352 [label=TBackward]
	5613979576 -> 5613979352
	5613979576 [label="q1_network.vector_encoders.0.seq_layers.1.weight
 (20, 20)" fillcolor=lightblue]
	5613978624 -> 5613978400
	5613978624 [label=TBackward]
	5613978960 -> 5613978624
	5613978960 [label="q1_network.q_heads.value_heads.extrinsic.weight
 (3, 20)" fillcolor=lightblue]
}
