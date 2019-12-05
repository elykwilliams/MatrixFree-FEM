#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>

#include <omp.h>

#include "MatrixFree.h"

using namespace dealii;


const unsigned int degree_finite_element = 2;
const unsigned int dimension             = 3;
const unsigned int n_mesh_refinements 	 = 4;
const unsigned int num_threads			 = 4;


template<int dim>
void run_test(const MatrixBase &system_matrix, const Vector<double> &vec, Vector<double> &out_vec){
	constexpr unsigned int NTEST = 200;
	
	for(unsigned int i = 0; i < NTEST; ++i){
		system_matrix.Vmult(vec, out_vec);
	}
}


template <int dim>
void run(unsigned int n_refinements, unsigned int fe_degree){

	Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler(triangulation);
	FE_Q<dim>       fe(fe_degree);
	ConstraintMatrix constraints;
	
	for (unsigned int cycle = 0; cycle < n_refinements; ++cycle){


		if (cycle == 0){
			GridGenerator::hyper_cube(triangulation, 0., 1.);
			triangulation.refine_global(1);
		
			GridTools::distort_random(0.25, triangulation);				
		}
	
		triangulation.refine_global(1);
		
	
		dof_handler.distribute_dofs(fe);
		//DoFRenumbering::Cuthill_McKee(dof_handler);

		constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler, constraints);
		VectorTools::interpolate_boundary_values(dof_handler,
												 0,
												 Functions::ZeroFunction<dim>(),
												 constraints);
		constraints.close();

		Vector<double> in_vec(dof_handler.n_dofs()), out_vec(dof_handler.n_dofs());
		
		std::cout << "Cycle = " << cycle << "; " 
			      << "n_dofs = " << dof_handler.n_dofs()
				  << std::endl;
	
		// Set Timer Here
		CRSMatrix<dim> crs_matrix(dof_handler, fe, constraints);
		run_test<dim>(crs_matrix, in_vec, out_vec);
		// End Here
		

		// Set Timer Here
		MFMatrix<dim> mf_matrix(dof_handler, fe, constraints);
		run_test<dim>(mf_matrix, in_vec, out_vec);
		// End Timer Here
				
	};
}



int main(int, char **)
try{
	omp_set_num_threads(num_threads);

	run<dimension>(n_mesh_refinements, degree_finite_element);

	return 0;
}
catch (std::exception &exc){
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
}
catch (...){
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
}


