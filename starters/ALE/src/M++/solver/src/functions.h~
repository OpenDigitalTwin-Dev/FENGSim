#ifndef HELPER_FUNCTIONS
#define HELPER_FUNCTIONS


/*
 *
 * Authors: Christian Burkhardt and Johannes Friedlein,
 * 	    FAU Erlangen-Nuremberg, 2019/2020
 *
 */


#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gamma.h>

#include <cmath>
#include <cstdlib>

#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace dealii;
using namespace std;


template<int dim>
Tensor<4,dim> get_tensor_operator_G(const SymmetricTensor<2,dim> &Ma, const SymmetricTensor<2,dim> &Mb)
{
	Tensor<4,dim> tmp; // has minor symmetry of indices k,l

	for(unsigned int i=0; i<dim; ++i)
		for(unsigned int j=0; j<dim; ++j)
			for(unsigned int k=0; k<dim; ++k)
				for(unsigned int l=k; l<dim; ++l)
				{
					double tmp_scalar = Ma[i][k] * Mb[j][l] + Ma[i][l] * Mb[j][k];
					tmp[i][j][k][l] = tmp_scalar;
					tmp[i][j][l][k] = tmp_scalar;
				}

	return tmp;
}


// F right F_{a(bc)}
template<int dim>
Tensor<4,dim> get_tensor_operator_F_right(const SymmetricTensor<2,dim> &Ma,
										  const SymmetricTensor<2,dim> &Mb,
										  const SymmetricTensor<2,dim> &Mc,
										  const SymmetricTensor<2,dim> &T )
{
	Tensor<4,dim> tmp; // has minor symmetry of indices k,l

	Tensor<2,dim> temp_tensor = contract<1,0>((Tensor<2,dim>)T, (Tensor<2,dim>)Mc);
	Tensor<2,dim> MbTMc = contract<1,0>((Tensor<2,dim>)Mb,temp_tensor);

	for(unsigned int i=0; i<dim; ++i)
		for(unsigned int j=0; j<dim; ++j)
			for(unsigned int k=0; k<dim; ++k)
				for(unsigned int l=k; l<dim; ++l)
				{
					double tmp_scalar = Ma[i][k] * MbTMc[j][l] + Ma[i][l] * MbTMc[j][k];
					tmp[i][j][k][l] = tmp_scalar;
					tmp[i][j][l][k] = tmp_scalar;
				}

	return tmp;
}


// F right F_{(ab)c}
template<int dim>
Tensor<4,dim> get_tensor_operator_F_left(const SymmetricTensor<2,dim> &Ma,
										 const SymmetricTensor<2,dim> &Mb,
										 const SymmetricTensor<2,dim> &Mc,
										 const SymmetricTensor<2,dim> &T){
	Tensor<4,dim> tmp;

	Tensor<2,dim> temp_tensor = contract<1,0>((Tensor<2,dim>)T, (Tensor<2,dim>)Mb);
	Tensor<2,dim> MaTMb = contract<1,0>((Tensor<2,dim>)Ma,temp_tensor);

	for(unsigned int i=0; i<dim; ++i)
		for(unsigned int j=0; j<dim; ++j)
			for(unsigned int k=0; k<dim; ++k)
				for(unsigned int l=k; l<dim; ++l)
				{
					double tmp_scalar = MaTMb[i][k] * Mc[j][l] + MaTMb[i][l] * Mc[j][k];
					tmp[i][j][k][l] = tmp_scalar;
					tmp[i][j][l][k] = tmp_scalar;
				}

	return tmp;
}



// ############################################################################################################
// Optimised function to compute outer products of symmetric tensors
// ############################################################################################################
#ifndef outer_product_sym_H
#define outer_product_sym_H

template<int dim, typename Number>
SymmetricTensor<4,dim,Number> outer_product_sym( const SymmetricTensor<2,dim,Number> &A, const SymmetricTensor<2,dim,Number> &B )
{
	SymmetricTensor<4,dim,Number> D;
	// Special nested for-loop to access only non-symmetric entries of 4th order sym. tensor
	// ToDo: still not optimal element 1112 and 1211 are both accessed
	for ( unsigned int i=0; i<dim; ++i )
		for ( unsigned int j=i; j<dim; ++j )
			for ( unsigned int k=i; k<dim; ++k )
				for ( unsigned int l=k; l<dim; ++l )
				{
					double tmp = A[i][j] * B[k][l] + B[i][j] * A[k][l];
					D[i][j][k][l] = tmp;
					D[k][l][i][j] = tmp;
				}
	return D;
}
template<int dim, typename Number>
SymmetricTensor<4,dim,Number> outer_product_sym( const SymmetricTensor<2,dim,Number> &A )
{
	SymmetricTensor<4,dim,Number> D;
	// Special nested for-loop to access only non-symmetric entries of 4th order sym. tensor
	// ToDo: still not optimal element 1112 and 1211 are both accessed
	for ( unsigned int i=0; i<dim; ++i )
    	for ( unsigned int j=i; j<dim; ++j )
        	for ( unsigned int k=i; k<dim; ++k )
            	for ( unsigned int l=k; l<dim; ++l )
            	{
            		double tmp = A[i][j] * A[k][l];
            		D[i][j][k][l] = tmp;
            		D[k][l][i][j] = tmp;
            	}
	return D;
}
template<int dim, typename Number>
SymmetricTensor<2,dim,Number> outer_product_sym( const Tensor<1,dim,Number> &A )
{
	SymmetricTensor<2,dim,Number> D;
	// Special nested for-loop to access only non-symmetric entries of 4th order sym. tensor
	// ToDo: still not optimal element 1112 and 1211 are both accessed
	for ( unsigned int i=0; i<dim; ++i )
    	for ( unsigned int j=i; j<dim; ++j )
    		D[i][j] = A[i] * A[j];

	return D;
}

#endif //outer_product_sym_H
// ############################################################################################################
// ############################################################################################################


// temp symmetry check [JF]
template<int dim>
bool symmetry_check ( Tensor<2,dim> &tensor )
{
	for ( unsigned int i=0; i<dim; i++ )
		for ( unsigned int j=0; j<dim; j++ )
			if ( i!=j && ( std::abs(tensor[i][j] - tensor[j][i])>1e-12 ) )
				return false;

	return true;
}
template<int dim>
bool symmetry_check(const Tensor<4,dim> &temp){
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<dim; ++j){
            for(unsigned int k=0; k<dim; ++k){
                for(unsigned int l=0; l<dim; ++l){
                    // absolute difference check
                    if(    ( fabs(temp[i][j][k][l] - temp[j][i][k][l]) > 1e-6 )
                            ||
                          ( fabs(temp[i][j][k][l] - temp[i][j][l][k]) > 1e-6 ) )
                    {
                        // relative difference check
                        if( (fabs(fabs(temp[i][j][k][l] - temp[j][i][k][l])/temp[j][i][k][l])> 1e-8)
                            ||
                          (fabs(fabs(temp[i][j][k][l] - temp[i][j][l][k])/temp[i][j][l][k]) >1e-8) )
                        {
                            deallog<< std::endl;
                            deallog<< "Abs not matching: "<<fabs(temp[i][j][k][l] - temp[j][i][k][l])
                                << " | Rel not matching: "<<fabs(temp[i][j][k][l] - temp[j][i][k][l])/temp[j][i][k][l]
                                << " | Abs not matching: "<<fabs(temp[i][j][k][l] - temp[i][j][l][k])
                                << " | Rel not matching: "<<fabs(temp[i][j][k][l] - temp[i][j][l][k])/temp[i][j][l][k]
                                << " | value ijkl: "<< std::scientific << temp[i][j][k][l]
                                << " | value jikl: "<< std::scientific << temp[j][i][k][l]
                                << " | value ijlk: "<< std::scientific << temp[i][j][l][k]
                                << std::endl;
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}


template<int dim>
SymmetricTensor<4,dim> symmetrize(const Tensor<4,dim> &tensor, const bool sym_check=false ){
    SymmetricTensor<4,dim> sym_tensor;
    if ( sym_check==true )
    	Assert(symmetry_check(tensor), ExcMessage("Tensor to symmetrize is not symmetric!"));

    Tensor<4,dim> temp_tensor;
    for(unsigned int i=0; i<dim; ++i)
        for(unsigned int j=0; j<dim; ++j)
            for(unsigned int k=0; k<dim; ++k)
                for(unsigned int l=0; l<dim; ++l)
                    temp_tensor[i][j][k][l] = (tensor[i][j][k][l]+tensor[j][i][k][l]+tensor[i][j][l][k]) / 3.0;

    for(unsigned int i=0; i<dim; ++i)
        for(unsigned int j=i; j<dim; ++j)
            for(unsigned int k=0; k<dim; ++k)
                for(unsigned int l=k; l<dim; ++l)
                    sym_tensor[i][j][k][l] = temp_tensor[i][j][k][l];
//                    sym_tensor[i][j][k][l] = tensor[i][j][k][l];

    return sym_tensor;
}


template<int dim>
SymmetricTensor<4,dim> symmetrize_minorSym(const Tensor<4,dim> &tensor, const bool sym_check=false ){
    SymmetricTensor<4,dim> sym_tensor;
//    if ( sym_check==true )
//    	Assert(symmetry_check(tensor), ExcMessage("Tensor to symmetrize is not symmetric!"));

    for(unsigned int i=0; i<dim; ++i)
        for(unsigned int j=i; j<dim; ++j)
            for(unsigned int k=0; k<dim; ++k)
                for(unsigned int l=k; l<dim; ++l)
                    sym_tensor[i][j][k][l] = tensor[i][j][k][l];

    return sym_tensor;
}


#endif 
