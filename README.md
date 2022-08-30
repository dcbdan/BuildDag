# BuildDag

This application is a utility to print out computation DAGs for consumption
by [From Dag](https://github.com/dcbdan/bbts/tree/partitioning/applications/from_dag),
a [Tensor Operating System](https://github.com/dimitrijejankov/bbts) application.

The DAGs contain nodes that are either Input, Join, Aggregation or Reblock nodes.
For example, a matrix multiply (45,56->46):

```
matmul 4 5 6
I[i0f-1.0f1.0]|4,5
I[i0f-1.0f1.0]|5,6
R[]0|4,5
R[]1|5,6
J[i0i2i2i2i0i1i1i2i0i2f1.0]2,0,2$3,2,1:2|4,6,5
A[i0]4|4,6
```

Visually:

```
I,0 ----> R,2 ----> J,4 ---> A,5
I,1 ----> R,3 -----^

```

The From Dag application

1. Parses the DAG.
2. Gives each node a relational partition. That is, a matrix is not represented as a single contiguous chunk of memory but as rows and columns of smaller contiguous "block" matrices--all blocks represnting a larget tensor are called relations. From Dag makes decisions as to how big or small the blocks should be.
3. Once a partitioning is assigned, TOS commands are generated
4. Then the TOS executes the commands.
