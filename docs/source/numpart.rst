Number Partitioning Problem
===========================

Introduction
------------

This problem consists on dividing a given set :math:`{S = (a_1 , ... , a_N)}` of integer numbers into 2 subsets S\ :sub:`1`\  and S\ :sub:`2`\  so
that the sum of the elements of both subsets have the same total value. Depending on the initial data
this may not always be possible, but it can be optimized so that the difference between the sums of both subsets
is minimized.

Formally, the problem can be formulated with the following decision variables:

.. math::

   x_i= \begin{cases}
   -1 & \text{if }x_i \text{ belongs to }S_1 \\
   1 & \text{if }x_i \text{ belongs to }S_2
   \end{cases}

And in this case the objective function to be minimized is:

.. math::
    f = (a_1*x_1 + a_2*x_2 + ... + a_N*x_N)^2

This is implemented by the :py:func:`optimization_problems.number_partition.number_partition` function.

It can be simply called as it is:

>>> number_partition(data="dummy", sampler="sim")

Functions
---------

.. autofunction:: optimization_problems.number_partition.number_partition

.. autofunction:: optimization_problems.number_partition.print_sample