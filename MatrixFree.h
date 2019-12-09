#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>

using namespace dealii;

template<int dim>
class MatrixBase{
public:
    double coefficient(Point<dim> const &p) const { // diffusion
        return 1. / (.05 + 2. * p.square());
    }
	virtual void Vmult(const Vector<double> &src, Vector<double> &dst) const = 0;
};


template<int dim>
class CRSMatrix : public MatrixBase<dim> {
    SparsityPattern sparsity_pattern;
    ConstraintMatrix constraints;
public:
	SparseMatrix<double> system_matrix;
	CRSMatrix(const DoFHandler<dim> &dof_handler, const FE_Q<dim> &fe, const ConstraintMatrix &constraints) : constraints(constraints) {
        std::cout << "Create sparsity pattern . . .\n";
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        std::cout << "Assemble mtx . . .\n";
        QGauss<dim> quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        for (auto const & cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0.;
            cell_rhs    = 0.;
            fe_values.reinit(cell);
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
                const double current_coefficient = coefficient(fe_values.quadrature_point(q_index));
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) +=
                                (current_coefficient *              // a(x_q)
                                 fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                 fe_values.JxW(q_index));           // dx
            }
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
	}
	}
	void Vmult(const Vector<double> &src, Vector<double> &dst) const override {
        dst = 0.;
        #pragma omp parallel for
	for (unsigned int i = 0; i < system_matrix.m(); ++i) {
            auto val = system_matrix.begin(i);
            for (unsigned int k = 0; k < sparsity_pattern.row_length(i); ++k, ++val) {
                auto j = sparsity_pattern.column_number(i, k);
                dst[i] += val->value() * src[j];
            }
        }
        constraints.distribute(dst);
    }
};


template<int dim>
class MFMatrix : public MatrixBase<dim> {
public:
	MFMatrix(const DoFHandler<dim> &dof_handler, const FE_Q<dim> &fe, const ConstraintMatrix &constraints) {}; 
	
	void Vmult(const Vector<double> &src, Vector<double> &dst) const override {};

private:

};
