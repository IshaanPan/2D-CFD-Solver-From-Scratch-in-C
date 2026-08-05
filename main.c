#include "domain.h"
#include "physics.h"
#include <stdlib.h>
#include <stdbool.h>



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
	bool export_field = false;

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

	mesh.num_points_vertical = num_points_vertical;
	mesh.num_points_horizontal = points_total;
	mesh.total_points = points_total * num_points_vertical;

	build_topwall(&top_wall, num_points_vertical);
	build_bottomwall(&bottom_wall, num_points_vertical); // calling functions to do the stuff they do, u dont use data type
	build_mesh(&top_wall, &bottom_wall, &mesh);
	construct_cells(&mesh);
	construct_inlet(&inlet, &mesh, true, 2e5);
	construct_outlet(&outlet, &mesh, true, 0.5e5);
	initialise_field(&field, &mesh, 0.5, 0.0, 0.5e4);
	





	if (export_mesh) {
	export_mesh_to_csv(&mesh, "mesh_data.csv");
	export_cells_to_csv(&mesh, "cell_data.csv");
	export_flow_boundary_to_csv(&inlet, &mesh, "inlet_data.csv");
	export_flow_boundary_to_csv(&outlet, &mesh, "outlet_data.csv");
	export_wall_to_csv(&top_wall, &mesh, "top_wall.csv");
	export_wall_to_csv(&bottom_wall, &mesh, "bottom_wall.csv");
	}

	if (export_field) {
		export_field_to_csv(&field, &mesh, "field.csv");
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

	free(field.p_stat);
	free(field.vel_u);
	free(field.vel_w);

	
	return 0;
	
}