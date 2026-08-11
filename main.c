#include "domain.h"
#include "physics.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>



int main() {

	double height1 = 1;
	double height2 = 0.5;
	double length = 1;
	int points_func = 50;
	int num_points_vertical = 100;
	int points_total = 100;
	
	
	
	struct wall_def top_wall;
	struct wall_def bottom_wall; //these are the actual data containers. U first need to make them, then tell the functions where they are using &. You cannot define these straight away as pointers because 
								// then you are simply creating a variable that holds the memory address, but u are not defining the actual struct. 


	struct mesh mesh;
	struct flow_boundary inlet;
	struct flow_boundary outlet; 
	struct field field;
	
	bool export_mesh = false;

	top_wall.height1 = height1;
	top_wall.height2 = height2;
	top_wall.length = length;
	top_wall.points_func = points_func;
	top_wall.points_total = points_total;
	top_wall.points_vertical = num_points_vertical;

	bottom_wall.height1 = height1;
	bottom_wall.height2 = height2;
	bottom_wall.length = length;
	bottom_wall.points_func = points_func;
	bottom_wall.points_total = points_total;
	bottom_wall.points_vertical = num_points_vertical;

	double dx = bottom_wall.length / (bottom_wall.points_func - 1);

	mesh.num_points_vertical = num_points_vertical;
	mesh.num_points_horizontal = points_total;
	mesh.total_points = points_total * num_points_vertical;
	mesh.vertical_height1=height1;
	mesh.length=length;
	mesh.dx=dx;
	mesh.wall_points=points_func;

	build_topwall(&top_wall, num_points_vertical);
	build_bottomwall(&bottom_wall, num_points_vertical); // calling functions to do the stuff they do, u dont use data type
	build_mesh(&top_wall, &bottom_wall, &mesh);
	construct_cells(&mesh);
	double initial_p_total=101350.0;
	construct_inlet(&inlet, &mesh, true, initial_p_total);
	
	field.initial_ptotal=initial_p_total;
	double initial_pstat=101300.0;
	construct_outlet(&outlet, &mesh, true, initial_pstat);
	initialise_field(&field, &mesh,  0.2, 0.1, initial_pstat, &inlet);
	field.initial_pstat=initial_pstat;
	

	int iter=0;
	int max_iter=8000;
	// export_field_to_csv(&field, &mesh, "field", iter);
	while ((field.residual_mass > 1e-4 || field.residual_u > 1e-4 || field.residual_v > 1e-4) && iter < max_iter) {
		
		diffusive_gradients(&field, &mesh, &inlet);
		
		solve_x_momentum(&field, &mesh, &inlet);
		
		//printf("0:%f, %d\n", field.vel_u_new[37], iter);
		
		solve_y_momentum(&field, &mesh, &inlet);
		
		//printf("1:%f, %d\n", field.vel_u_new[37], iter);
		
		
		rhie_chow_flux(&mesh, &field);
		
		//printf("2:%f, %d\n", field.vel_u_new[37], iter);
		
		set_pressure_zero(&mesh, &field);

		int inner_iter=0;
		int inner_max=3000;
		
		for (inner_iter=0;inner_iter<inner_max;inner_iter++) {
			determine_pressure_correction(&mesh, &field);
			make_old_new_pressure_prime(&mesh, &field);
		}
	
		

		update_field(&mesh, &field);

		

		
		update_mass_fluxes(&mesh, &field);
		
		// the mass residual is only meaningful once the correction has been applied to the fluxes
		compute_mass_residuals(&mesh, &field);
		
		double max_mass_residual = 0.0;
		int worst_cell = 0;
		for (int c = 0; c < mesh.total_cell_centers; c++) {
			double net_mass = field.flux_east[c] + field.flux_west[c] + field.flux_north[c] + field.flux_south[c];
			if (fabs(net_mass) > max_mass_residual) {
				max_mass_residual = fabs(net_mass);
				worst_cell = c;
			}
		}
		printf("Outer Iter %d | Max Net Mass Imbalance: %e at cell %d\n", iter, max_mass_residual, worst_cell);
		
		compute_momentum_residuals(&mesh, &field);

		
		make_old_new(&mesh, &field);
		
		
		printf("Iter %d | Mass Res: %e | U Res: %e | V Res: %e |\n", 
       	iter, field.residual_mass, field.residual_u, field.residual_v);
		

		// if (iter%50==0) { export_field_to_csv(&field, &mesh, "field", iter); }

		iter=iter+1;

		
	}
	
	export_field_to_csv(&field, &mesh, "field", iter);

	




	if (export_mesh) {
	export_mesh_to_csv(&mesh, "mesh_data.csv");
	export_cells_to_csv(&mesh, "cell_data.csv");
	export_flow_boundary_to_csv(&inlet, &mesh, "inlet_data.csv");
	export_flow_boundary_to_csv(&outlet, &mesh, "outlet_data.csv");
	export_wall_to_csv(&top_wall, &mesh, "top_wall.csv");
	export_wall_to_csv(&bottom_wall, &mesh, "bottom_wall.csv");
	}

	

	free(top_wall.pos_x);
	free(top_wall.pos_y);
	free(top_wall.cell_numbers);
	free(bottom_wall.pos_x);
	free(bottom_wall.pos_y);
	free(bottom_wall.cell_numbers);

	free(mesh.intersections_y);
	free(mesh.intersections_x);
	free(mesh.cell_centers_x);
	free(mesh.cell_centers_y);

	free(mesh.east_length);
	free(mesh.east_nx);
	free(mesh.east_ny);

	free(mesh.west_length);
	free(mesh.west_nx);
	free(mesh.west_ny);

	free(mesh.south_length);
	free(mesh.south_nx);
	free(mesh.south_ny);

	free(mesh.north_length);
	free(mesh.north_nx);
	free(mesh.north_ny);
	free(mesh.cell_volumes);

	free(inlet.cell_numbers);
	free(inlet.total_pressure);
	free(outlet.cell_numbers);
	free(outlet.static_pressure);

	free(field.p_stat_old);
	free(field.vel_u_old);
	free(field.vel_v_old);

	free(field.p_stat_new);
	free(field.vel_u_new);
	free(field.vel_v_new);

	free(field.flux_east);
	free(field.flux_west);
	free(field.flux_north);
	free(field.flux_south);

	free(field.diffusive_dudx);
	free(field.diffusive_dudy);
	free(field.diffusive_dvdx);
	free(field.diffusive_dvdy);

	free(field.df_x);
	free(field.df_y);

	free(field.p_prime_new);
	free(field.p_prime_old);

	
	return 0;
	
}
