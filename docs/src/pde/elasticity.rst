弹性方程
-----------------

张量运算
^^^^^^^^^^^^^^

二阶张量是 :math:`3\times 3` 矩阵，四阶张量是 :math:`9\times 9` 矩阵，我们定义两个常用的张量乘积，一个是 :math:`\mathbf a\otimes\mathbf b` ，叫做Kronecker product，一个是 :math:`\mathbf A:\mathbf B` ，叫做Frobenius inner product。

.. math::
   
   \mathbf A \otimes \mathbf B =
   \left(\begin{array}{cc}
   a_{11}\mathbf B & a_{12}\mathbf B  \\
   a_{21}\mathbf B & a_{22}\mathbf B  
   \end{array}\right)

.. math::

   \mathbf A : \mathbf B = a_{ij}b_{ij}

.. math::
   
   \mathbb A : \mathbf B = a_{ijkl} b_{kl}\mathbf e_i\otimes\mathbf e_j
