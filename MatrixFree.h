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
        // std::cout << "Create sparsity pattern . . .\n";
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        // std::cout << "Assemble mtx . . .\n";
        #pragma omp parallel
		#pragma omp single
		{
			for (auto const & cell : dof_handler.active_cell_iterators()) {
				#pragma omp task firstprivate(cell)
				{
					QGauss<dim> quadrature_formula(fe.degree + 1);
					FEValues<dim> fe_values(fe, quadrature_formula, update_gradients | update_quadrature_points | update_JxW_values);
					const unsigned int dofs_per_cell = fe.dofs_per_cell;
					const unsigned int n_q_points    = quadrature_formula.size();
					FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
					Vector<double>     cell_rhs(dofs_per_cell);
					std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
					
					fe_values.reinit(cell);
					for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
						const double current_coefficient = this->coefficient(fe_values.quadrature_point(q_index));
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
							for (unsigned int j = 0; j < dofs_per_cell; ++j)
								cell_matrix(i, j) +=
										(current_coefficient *              // a(x_q)
										 fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
										 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
										 fe_values.JxW(q_index));           // dx
					}
					cell->get_dof_indices(local_dof_indices);
					#pragma omp critical
					{
						constraints.distribute_local_to_global(cell_matrix, local_dof_indices, system_matrix);
					}

				}// end task
			}// end cell
		#pragma omp taskwait
		}// end parallel
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
	MFMatrix(const DoFHandler<dim> &dof_handler, const FE_Q<dim> &fe, const ConstraintMatrix &constraints) : 
		dofh(dof_handler), 
		fe(fe), 
		constraints(constraints),
		quadrature(fe.degree+1),
		reference_cell(),
		fe_values_ref(fe, quadrature, update_gradients)
		{
			GridGenerator::hyper_cube(reference_cell, 0, 1);
			fe_values_ref.reinit(reference_cell.begin());
			
		}
	
	void Vmult(const Vector<double> &src, Vector<double> &dst) const override {
	#pragma omp parallel
	#pragma omp single
	{
		auto cell = dofh.begin_active();
		decltype(cell) endc = dofh.end();
		for(; cell != endc; cell++){
			#pragma omp task firstprivate(cell)
			{	
				FEValues<dim> fe_values(fe, quadrature, update_inverse_jacobians | update_quadrature_points | update_JxW_values);
				fe_values.reinit(cell);
					
					
			
				unsigned int n_q_points = quadrature.size();
				unsigned int dofs_per_cell = fe.dofs_per_cell;
				
				std::vector<types::global_dof_index> local_to_global(dofs_per_cell);
				cell->get_dof_indices(local_to_global);
			
				std::vector<double> coeff_vals(n_q_points);
				for(unsigned int q_index=0; q_index<n_q_points; ++q_index)
					coeff_vals[q_index] = this -> coefficient(fe_values.quadrature_point(q_index));
	
				Vector<double> cell_vec(dofs_per_cell);
				for (unsigned int i = 0; i<dofs_per_cell; ++i)
					cell_vec(i) = src(local_to_global[i]);
				
				// Multiply by ShapeGrad
				Vector<double> temp_vec(n_q_points*dim);	
				for (unsigned int q=0; q<n_q_points; ++q)
					for (unsigned int d=0; d<dim; ++d)
						for (unsigned int i=0; i<dofs_per_cell; ++i)
							temp_vec(q*dim+d) +=
						  		fe_values_ref.shape_grad(i,q)[d] * cell_vec(i);
				
				// Multiply by J^{-T}DJ^{-1}
				for (unsigned int q = 0; q < n_q_points; ++q){
        
        			  // apply the transposed inverse Jacobian of the mapping
					Vector<double> temp(n_q_points*dim);	
          			for (unsigned int d=0; d<dim; ++d)
            			temp[d] = temp_vec(q*dim+d);
          
					for (unsigned int d=0; d<dim; ++d)
            		{
              			double sum = 0;
              			for (unsigned int e=0; e<dim; ++e)
                			sum += fe_values.inverse_jacobian(q)[e][d] * temp[e];
              		
						temp_vec(q*dim+d) = sum;
            		}


					// multiply by coefficient and integration weight
          			for (unsigned int d=0; d<dim; ++d)
            			temp_vec(q*dim+d) *= fe_values.JxW(q) * coeff_vals[q];
          
					// apply the inverse Jacobian of the mapping
          			for (unsigned int d=0; d<dim; ++d)
            			temp[d] = temp_vec(q*dim+d);
          
					for (unsigned int d=0; d<dim; ++d)
           			{
              			double sum = 0;
              			for (unsigned int e=0; e<dim; ++e)
                			sum += fe_values.inverse_jacobian(q)[d][e] * temp[e];
              			
						temp_vec(q*dim+d) = sum;
            		}
				
				}
				
				// Apply gradshape^{T}
				Vector<double> cell_vec_dest(dofs_per_cell);
      			for (unsigned int i=0; i<dofs_per_cell; ++i){
        			for (unsigned int q=0; q<n_q_points; ++q)
          				for (unsigned int d=0; d<dim; ++d)
            				cell_vec_dest(i) += fe_values_ref.shape_grad(i,q)[d] * temp_vec(q*dim+d);
				}

				for(unsigned int i=0; i<dofs_per_cell; i++)
				{	
					#pragma omp atomic
					dst(local_to_global[i]) += cell_vec_dest(i);	
				}

			}// end task
		}// end cell

		#pragma omp taskwait
		//constraints.distribute(dst); 
	}// end parallel
};

private:
	const DoFHandler<dim>& dofh;
	const FE_Q<dim> fe;
	const ConstraintMatrix constraints;
	const QGauss<dim> quadrature;
	Triangulation<dim> reference_cell;
	FEValues<dim> fe_values_ref;
};
