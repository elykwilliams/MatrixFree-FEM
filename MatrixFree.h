#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>

using namespace dealii;


class MatrixBase{
public:
	virtual void Vmult(const Vector<double> &src, Vector<double> &dst) const = 0;
};


template<int dim>
class CRSMatrix : public MatrixBase{
public:
	CRSMatrix(const DoFHandler<dim> &dof_handler, const FE_Q<dim> &fe, const ConstraintMatrix &constraints){
		assemble_matrix();
	} 

	void Vmult(const Vector<double> &src, Vector<double> &dst) const override {};

private:
	void assemble_matrix(){};
};


template<int dim>
class MFMatrix : public MatrixBase{
public:
	MFMatrix(const DoFHandler<dim> &dof_handler, const FE_Q<dim> &fe, const ConstraintMatrix &constraints) {}; 
	
	void Vmult(const Vector<double> &src, Vector<double> &dst) const override {};

private:

};
