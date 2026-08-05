#include <stdbool.h>
#pragma once


#define rho 1.225


struct wall_def {
	double length;
	double height1;
	double height2;
	int points_func;
	int points_total;
	double* pos_y;
	double* pos_x;
	int* cell_numbers;
	int points_vertical;



};

struct mesh {

	double* intersections_x;
	double* intersections_y;
	double* cell_centers_x;
	double* cell_centers_y;

	double* east_length;
	double* east_nx;
	double* east_ny;

	double* west_length;
	double* west_nx;
	double* west_ny;

	double* south_length;
	double* south_nx;
	double* south_ny;

	double* north_length;
	double* north_nx;
	double* north_ny;

	double* cell_volumes;








	int num_points_vertical;
	int num_points_horizontal;
	int total_points;
	int total_cell_centers;
	
};



struct flow_boundary {
	int* cell_numbers;
	double* total_pressure;
	double* static_pressure;
	double* vel_u;
	double* vel_w;


};


void build_topwall(struct wall_def* wall, int num_points_vertical);
void build_bottomwall(struct wall_def* bottom_wall, int num_points_vertical);
void build_mesh(struct wall_def* top_wall, struct wall_def* bottom_wall, struct mesh* mesh);
void construct_cells(struct mesh* mesh);
void export_mesh_to_csv(struct mesh* mesh, const char* filename);
void export_cells_to_csv(struct mesh* mesh, const char* filename);
void construct_inlet(struct flow_boundary* inlet, struct mesh* mesh, bool total_pressure, double total_pressure_val);
void construct_outlet(struct flow_boundary* outlet, struct mesh* mesh, bool static_pressure, double static_pressure_val);
void export_flow_boundary_to_csv(struct flow_boundary* boundary, struct mesh* mesh, const char* filename);
void export_wall_to_csv(struct wall_def* wall, struct mesh* mesh, const char* filename);