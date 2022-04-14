# BuildDag

This application is a utility to print out computation DAGs for consumption
by the [From Dag](https://github.com/dcbdan/bbts/tree/partitioning/applications/from_dag),
a [Tensor Operating System](https://github.com/dimitrijejankov/bbts) application.

The DAGs contain nodes that are either Input, Join, Aggregation or Reblock nodes.
For example, a matrix multiply (45,56->46):

```
matmul 4 5 6
I[i0i0f-1.0f1.0]|4,5
I[i0i0f-1.0f1.0]|5,6
R[i1]0|4,5
R[i1]1|5,6
J[i2i2i2i2i0i1i1i2i0i2f1.0]2,0,1$3,1,2:1|4,5,6
A[i7i0]4|4,6
```

Visually:

```
I,0 ----> R,2 ----> J,4 ---> A,5
I,1 ----> R,3 -----^

```

Each node specifies

1. The node type (I,R,J,A)
2. The kernel (i0 is an initialization kernel, i2 is a contraction kernel)
3, The kernel options
4. Which inputs. If it is also a join, the particular join operations as well. (Node 4 has inputs node 2 and node 3 with a particular ordering that implies that block i,j,k of the join depends on block i,j of node 2 and block j,k of node 3. Also, dimension 1 is aggregated out.
5. The actual dimensions of the whole relation.

Note that all kernels and options are specific to this
[TOS-cuTENSOR](https://github.com/dcbdan/bbts/tree/partitioning/applications/from_dag/cutensor).
TOS-cuTENSOR is a CPU and GPU kernel library for the TOS.
The GPU kernels dispatch to [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/index.html).
The CPU kernels reduce to MKL calls.

The From Dag application

1. Parses the DAG.
2. Gives each node a relational partition. That is, a matrix is not represented as a single contiguous chunk of memory but as rows and columns of smaller contiguous "block" matrices--all blocks represnting a larget tensor are called relations. From Dag makes decisions as to how big or small the blocks should be.
3. One a partitioning is assigned, TOS commands are generated
4. Then the TOS executes the commands.
