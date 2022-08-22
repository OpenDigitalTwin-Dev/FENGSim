算例和C++/Python接口
==========================

几何方程
---------------------------

B-Splines
^^^^^^^^^^^^^^^^^^^^^^

定义
"""""""""""""""

给定 n + 1 个控制点 $P_0, P_1, \cdots , P_n$  和一个节点向量 $U = { u_0, u_1, \cdots , u_m }$ ,  p 次B-样条曲线由这些控制点和节点向量 $U$  定义

$$
C(u) = \sum_{i=0}^n N_{i,p}(u)P_i
$$

其中  $N_{i,p}(u)$ 是 p次B-样条基函数。  B-样条曲线包含更多信息，一系列的 n+1 个控制点， m+1个节点的节点向量，次数 p。 注意n, m 和 p 必须满足m = n + p + 1。更准确地，如果我们想要定义一个有 n + 1控制点的p次B-样条曲线，我们必须提供n + p + 2 个节点  $u_0, u_1, \cdots , u_{n+p+1}$ 。另一方面，如果给出了一个m + 1 个节点的节点向量和n + 1 控制点，B-样条曲线的次数是p = m - n - 1。对应于一个节点 $u_i$ 的曲线上的点，  $C(u_i)$ ，被称为节点（knot point）。


B-样条基函数
"""""""""""""""

B-样条基函数特性，(1)定义域被节点细分（subdivided）； (2) 基函数不是在整个区间非零。

设 U 是m + 1个非递减数的集合，$u_0 <= u_2 <= u_3 <= \cdots <= u_m$。$u_i$ 称为节点（knots）, 集合 U 称为节点向量（knot vector）, 半开区间 $[u_i, u_{i+1})$ 是第i个节点区间（knot span）。注意某些 $u_i$ 可能相等，某些节点区间会不存在。如果一个节点 $u_i$ 出现 k 次 (即，`$u_i = u_{i+1} = \cdots = u_{i+k-1}$`), 其中 k > 1, $u_i$ 是一个重复度（multiplicity）为k 的多重节点，写为 $u_i(k)$。 否则，如果 $u_i$ 只出现一次，它是一个简单节点。如果节点等间距(即， $u_{i+1} - u_i$ 是一个常数，对 $0 <= i <= m - 1$)，节点向量或节点序列称为均匀的；否则它是非均匀的。

节点可认为是分隔点，将区间 $[u_0, u_m]$ 细分为节点区间。所有B-样条基函数被假设定义域在 $[u_0, u_m]$ 上。在本文中使用 $u_0 = 0$ 和 $u_m = 1$，所以定义域是闭区间[0,1]。

为了定义B-样条基函数，还需要一个参数，基函数的次数（degree）p，第i个p次B-样条基函数，写为 $N_{i,p}(u)$，递归定义如下：


Cox-de Boor递归公式

$$
& N_{i, 0} (u) = \begin{Bmatrix} 1 &  if \quad  u_{i} \leq u < u_{i + 1} \\ 0 &  otherwise \end{Bmatrix} \\
& N_{i, p} (u) = \frac{u - u_{i}}{u_{i + p} - u_{i}} N_{i, p-1}(u) + \frac{u_{i + p + 1} - u}{u_{i + p + 1} - u_{i+1}} N_{i+1, p-1}(u).
$$

如果次数（degree）为零（即， p = 0），这些基函数都是阶梯函数，这也是第一个表达式所表明的。即，如果u是在第i个节点区间 $[u_i, u_{i+1})$ 上基函数 $N_{i,0}(u)$ 是1。 例如，如果我们有四个节点 $u_0 = 0, u_1 = 1, u_2 = 2, u_3 = 3$, 节点区间 0, 1 和2是 $[0,1), [1,2), [2,3)$ ，在 [0,1) 0次基函数是 $N_{0,0}(u) = 1$  ，在其它区间是0；在 [1,2)上$N_{1,0}(u) = 1$ ，在其它区间是0；在[2,3)上 $N_{2,0}(u) = 1$，其它区间是0。



算例
"""""""""""""""

C++ ::

    #include<BSplCLib.hxx>
    #include<math_Matrix.hxx>
    #include<TColStd_Array1OfReal.hxx>
    #include<TColStd_Array1OfInteger.hxx>
    #include<Geom2d_BSplineCurve.hxx>
    #include <iostream>


    int main (int argv, char** argc) {
    	//Knot vector: [0,0,0,1,2,3,4,4,5,5,5]
    	TColStd_Array1OfReal knotSeq(1, 11);
    	knotSeq.Init(0);
    	knotSeq.SetValue(1,0);
    	knotSeq.SetValue(2,0);
    	knotSeq.SetValue(3,0);
    	knotSeq.SetValue(4,1);
    	knotSeq.SetValue(5,2);
    	knotSeq.SetValue(6,3);
    	knotSeq.SetValue(7,4);
    	knotSeq.SetValue(8,4);
    	knotSeq.SetValue(9,5);
    	knotSeq.SetValue(10,5);
    	knotSeq.SetValue(11,5);

    	cout << "Knot Sequence: [";
    	for (Standard_Integer i = 1; i <= knotSeq.Length(); i++) {
    		cout << knotSeq.Value(i) << " ";
    	}
    	cout << "]" << endl;

    	Standard_Integer knotsLen = BSplCLib::KnotsLength(knotSeq);
    	TColStd_Array1OfReal knots(1, knotsLen);
    	TColStd_Array1OfInteger mults(1, knotsLen);

    	BSplCLib::Knots(knotSeq, knots, mults);
    	cout << "Knots: [";
    	for (Standard_Integer i = 1; i <= knots.Length(); i++) {
    		cout << knots.Value(i) << " " ;
    	}
    	cout << "]" << endl;

    	cout << "Multiplicity:[";
    	for (Standard_Integer i = 1; i <= mults.Length(); i++){
    		cout << mults.Value(i) << " ";
    	}
    	cout << "]" << endl;
    	if (BSplCLib::KnotForm(knots, 1, knotsLen) == BSplCLib_Uniform) {
    		cout << "Knots is uniform." << endl;
    	}
    	else {
    		cout << "Knots is non-uniform." << endl;
    	}
    	Standard_Real rValue = 2.5;
    	Standard_Integer iOrder = 2 + 1;
    	Standard_Integer iFirstNonZeroIndex = 0;
    	math_Matrix bSplineBasis(1, 1, 1, iOrder, 0);
    	BSplCLib::EvalBsplineBasis(1, 0, iOrder, knotSeq, rValue, iFirstNonZeroIndex, bSplineBasis);
    	cout << "First Non-Zero Basis index:" << iFirstNonZeroIndex << endl;
    	cout << bSplineBasis << endl;
        return 0;
    }

Output ::

    Knot Sequence: [0 0 0 1 2 3 4 4 5 5 5 ]
    Knots: [0 1 2 3 4 5 ]
    Multiplicity:[3 1 1 1 2 3 ]
    Knots is uniform.
    First Non-Zero Basis index:3
    math_Matrix of RowNumber = 1 and ColNumber = 3
    math_Matrix ( 1, 1 ) = 0.125
    math_Matrix ( 1, 2 ) = 0.75
    math_Matrix ( 1, 3 ) = 0.125


Python ::

    import matplotlib.pyplot as plt
    import numpy as np
    def B(x, k, i, t):
        if k == 0:
           return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
           c1 = 0.0
        else:
           c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
           c2 = 0.0
        else:
           c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        return c1 + c2

    t = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    N11 = B(2.5, 2, 2, t)
    N12 = B(2.5, 2, 3, t)
    N13 = B(2.5, 2, 4, t)
    print("N(1,1) : ", N11)
    print("N(1,2) : ", N12)
    print("N(1,3) : ", N13)


Output ::

    N(1,1) :  0.125
    N(1,2) :  0.75
    N(1,3) :  0.125


