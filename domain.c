#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "domain.h"
// only si units used


double bottomwall_func(double x, double h1, double h2, double l) {
	return (h1 - h2) * pow((x / l), 2);
}




void build_bottomwall(struct wall_def* bottom_wall, int num_points_vertical) {
	bottom_wall->pos_x = (double*)malloc(sizeof(double) * bottom_wall->points_total);
	bottom_wall->pos_y = (double*)malloc(sizeof(double) * bottom_wall->points_total);

	int num_horizontal_cells = bottom_wall->points_total - 1;
	bottom_wall->cell_numbers = malloc(sizeof(int) * num_horizontal_cells);

	bottom_wall->pos_x[0] = 0;
	bottom_wall->pos_y[0] = 0;
	double dx = bottom_wall->length / (bottom_wall->points_func - 1);

	for (int i = 1; i <= bottom_wall->points_func - 1; i++) {
		bottom_wall->pos_x[i] = bottom_wall->pos_x[i - 1] + dx;
		bottom_wall->pos_y[i] = bottomwall_func(bottom_wall->pos_x[i], bottom_wall->height1, bottom_wall->height2, bottom_wall->length);
	}

	for (int i = bottom_wall->points_func; i < bottom_wall->points_total; i++) {
		bottom_wall->pos_x[i] = bottom_wall->pos_x[i - 1] + dx;
		bottom_wall->pos_y[i] = bottom_wall->height1 - bottom_wall->height2;
	}

	int Ny_cells = num_points_vertical - 1;
	for (int i = 0; i < num_horizontal_cells; i++) {
		bottom_wall->cell_numbers[i] = i * Ny_cells;
	}
}

void build_topwall(struct wall_def* wall, int num_points_vertical) {
	wall->pos_x = (double*)malloc(sizeof(double) * wall->points_total);
	wall->pos_y = (double*)malloc(sizeof(double) * wall->points_total);

	int num_horizontal_cells = wall->points_total - 1;
	wall->cell_numbers = malloc(sizeof(int) * num_horizontal_cells);

	wall->pos_x[0] = 0;
	wall->pos_y[0] = wall->height1;
	double dx = wall->length / (wall->points_func - 1);

	for (int i = 1; i < wall->points_total; i++) {
		wall->pos_x[i] = wall->pos_x[i - 1] + dx;
		wall->pos_y[i] = wall->height1;
	}

	int Ny_cells = num_points_vertical - 1;
	for (int i = 0; i < num_horizontal_cells; i++) {
		wall->cell_numbers[i] = (i * Ny_cells) + (Ny_cells - 1);
	}
}

void vertical_interpolator(double top_wall_y,double bottom_wall_y, double bottom_wall_x, struct mesh* mesh, int counter) {
	double dy = (top_wall_y - bottom_wall_y)/(mesh->num_points_vertical-1);
	for (int i = 0; i < mesh->num_points_vertical; i++) {
		double y;
		int index;
		y = bottom_wall_y + dy * (i);
		
		index = (counter * mesh->num_points_vertical) + i;

		mesh->intersections_x[index] = bottom_wall_x;
		mesh->intersections_y[index] = y;


	}
}


void build_mesh(struct wall_def* top_wall, struct wall_def* bottom_wall, struct mesh* mesh) {
	double* top_wall_xx = top_wall->pos_x;
	double* top_wall_yy = top_wall->pos_y;
	double* bottom_wall_xx = bottom_wall->pos_x;
	double* bottom_wall_yy = bottom_wall->pos_y;
	int index_1;
	int index_2;
	int index_3;
	int index_4;
	int cell_counter=0;
	
	
	


	mesh->intersections_y = malloc(sizeof(double) * mesh->total_points);
	mesh->intersections_x = malloc(sizeof(double) * mesh->total_points);

	int total_cell_centers = (mesh->num_points_horizontal - 1) * (mesh->num_points_vertical - 1);
	mesh->total_cell_centers = total_cell_centers;
	mesh->cell_centers_x= malloc(sizeof(double) * total_cell_centers);
	mesh->cell_centers_y = malloc(sizeof(double) * total_cell_centers);

	for (int i=0; i < bottom_wall->points_total; i++) {
		vertical_interpolator(top_wall_yy[i], bottom_wall_yy[i], bottom_wall_xx[i], mesh, i);

	}

	for (int i = 1; i < bottom_wall->points_total; i++) {

		for (int j = 0; j < mesh->num_points_vertical-1; j++) {
			index_1 = mesh->num_points_vertical * i + j; //br
			index_2 = mesh->num_points_vertical * (i-1)+j; //bl
			index_3 = mesh->num_points_vertical * (i - 1) + j + 1; //tl
			index_4 = mesh->num_points_vertical * i + j + 1; // tr
			
			
			mesh->cell_centers_x[cell_counter] = (mesh->intersections_x[index_1] + mesh->intersections_x[index_2] + mesh->intersections_x[index_3] + mesh->intersections_x[index_4]) / 4;
			mesh->cell_centers_y[cell_counter] = (mesh->intersections_y[index_1] + mesh->intersections_y[index_2] + mesh->intersections_y[index_3] + mesh->intersections_y[index_4]) / 4;

			cell_counter = cell_counter + 1;
		}

	}



}




void determine_normal(double* nx, double* ny, struct mesh* mesh,  int p1, int p2, bool clock) {

	double p1_x = mesh->intersections_x[p1];
	double p1_y = mesh->intersections_y[p1];

	double p2_x = mesh->intersections_x[p2];
	double p2_y = mesh->intersections_y[p2];



	if (clock) {
		*nx = p2_y - p1_y;
		*ny = -(p2_x - p1_x);
	}
	else  {
		*nx = -(p2_y - p1_y);
		*ny = p2_x - p1_x;
	}

}



// C functions cant return multiple things, give inputs as pointers, then u deference and set value at pointer address in function so that u can retrieve it later

void construct_cells(struct mesh* mesh) {


	mesh->east_length = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->east_nx = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->east_ny = malloc(sizeof(double) * mesh->total_cell_centers);


	mesh->west_length = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->west_nx = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->west_ny = malloc(sizeof(double) * mesh->total_cell_centers);


	mesh->north_length = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->north_nx = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->north_ny = malloc(sizeof(double) * mesh->total_cell_centers);


	mesh->south_length = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->south_nx = malloc(sizeof(double) * mesh->total_cell_centers);
	mesh->south_ny = malloc(sizeof(double) * mesh->total_cell_centers);


	mesh->cell_volumes = malloc(sizeof(double) * mesh->total_cell_centers);

	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {

		// 1. Get the vertical sizes
		int Ny_nodes = mesh->num_points_vertical;
		int Ny_cells = Ny_nodes - 1; // The number of cells in a column

		// 2. Find the 2D cell coordinate (Working backwards!)
		// i_cell is the column the cell lives in. j_cell is the row.
		int i_cell = cell_number / Ny_cells;
		int j_cell = cell_number % Ny_cells; // bottom row number

		// 3. Map the 2D cell coordinate to the 1D Node indices
		// Remember: bl and tl are on the left side (i_cell)
		//           br and tr are on the right side (i_cell + 1)
		int bl = (i_cell * Ny_nodes) + j_cell;
		int br = ((i_cell + 1) * Ny_nodes) + j_cell;
		int tl = (i_cell * Ny_nodes) + (j_cell + 1);
		int tr = ((i_cell + 1) * Ny_nodes) + (j_cell + 1);

		mesh->east_length[cell_number] = sqrt(pow((mesh->intersections_y[tr] - mesh->intersections_y[br]), 2) + pow((mesh->intersections_x[tr] - mesh->intersections_x[br]), 2));
		mesh->west_length[cell_number] = sqrt(pow((mesh->intersections_y[tl] - mesh->intersections_y[bl]), 2) + pow((mesh->intersections_x[tl] - mesh->intersections_x[bl]), 2));
		mesh->north_length[cell_number] = sqrt(pow((mesh->intersections_y[tr] - mesh->intersections_y[tl]), 2) + pow((mesh->intersections_x[tr] - mesh->intersections_x[tl]), 2));
		mesh->south_length[cell_number] = sqrt(pow((mesh->intersections_y[br] - mesh->intersections_y[bl]), 2) + pow((mesh->intersections_x[bl] - mesh->intersections_x[br]), 2));

		double east_nx;
		double east_ny;

		double west_nx;
		double west_ny;

		double north_nx;
		double north_ny;

		double south_nx;
		double south_ny;

		determine_normal(&east_nx, &east_ny, mesh, tr, br, false);
		determine_normal(&west_nx, &west_ny, mesh, bl, tl, false);
		determine_normal(&north_nx, &north_ny, mesh, tl, tr, false);
		determine_normal(&south_nx, &south_ny, mesh, br, bl, false);

		mesh->east_nx[cell_number] = east_nx / mesh->east_length[cell_number];
		mesh->east_ny[cell_number] = east_ny/ mesh->east_length[cell_number];

		mesh->west_nx[cell_number] = west_nx / mesh->west_length[cell_number];
		mesh->west_ny[cell_number] = west_ny / mesh->west_length[cell_number];

		mesh->north_nx[cell_number] = north_nx / mesh->north_length[cell_number];
		mesh->north_ny[cell_number] = north_ny / mesh->north_length[cell_number]; // divide by length to get unit normals

		mesh->south_nx[cell_number] = south_nx / mesh->south_length[cell_number];
		mesh->south_ny[cell_number] = south_ny / mesh->south_length[cell_number];

		double dx1 = mesh->intersections_x[tr] - mesh->intersections_x[bl];
		double dy1 = mesh->intersections_y[tr] - mesh->intersections_y[bl];

		double dx2 = mesh->intersections_x[tl] - mesh->intersections_x[br];
		double dy2 = mesh->intersections_y[tl] - mesh->intersections_y[br];

		mesh->cell_volumes[cell_number] = fabs(dx1 * dy2 - dx2 * dy1) * 0.5;

	}








}




void construct_inlet(struct flow_boundary* inlet, struct mesh* mesh, bool total_pressure, double total_pressure_val) {
	inlet->cell_numbers = malloc(sizeof(int) * (mesh->num_points_vertical - 1));


	if (total_pressure) {
		inlet->total_pressure= malloc(sizeof(double) * (mesh->num_points_vertical - 1));

		for (int cell_number = 0; cell_number < mesh->num_points_vertical - 1; cell_number++) {
			inlet->total_pressure[cell_number] = total_pressure_val;
			inlet->cell_numbers[cell_number] = cell_number;
		}
	}


}

void construct_outlet(struct flow_boundary* outlet, struct mesh* mesh, bool static_pressure, double static_pressure_val) {
	outlet->cell_numbers= malloc(sizeof(int) * (mesh->num_points_vertical - 1));
	if (static_pressure) {
		outlet->static_pressure = malloc(sizeof(double) * (mesh->num_points_vertical - 1));
		int counter = 0;

		for (int cell_number = mesh->total_cell_centers - (mesh->num_points_vertical - 1); cell_number <= mesh->total_cell_centers - 1; cell_number++) {
			outlet->static_pressure[counter] = static_pressure_val;
			outlet->cell_numbers[counter] = cell_number;
			counter = counter + 1;
		}

	}


}



void export_mesh_to_csv(struct mesh* mesh, const char* filename) {
	// 1. Open a file pointer in "w" (write) mode
	FILE* file = fopen(filename, "w");

	// Safety check: Make sure the file actually opened
	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	// 2. Write the CSV header (column names)
	fprintf(file, "x,y\n");

	// 3. Loop through your entire flattened 1D array
	for (int i = 0; i < mesh->total_points; i++) {
		fprintf(file, "%f,%f\n", mesh->intersections_x[i], mesh->intersections_y[i]);
	}

	// 4. Close the file to save it and free it from memory
	fclose(file);

	printf("Successfully exported mesh to %s\n", filename);
}

void export_cells_to_csv(struct mesh* mesh, const char* filename) {
	// 1. Open a file pointer in "w" (write) mode
	FILE* file = fopen(filename, "w");

	// Safety check: Make sure the file actually opened
	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	// 2. Write the CSV header (column names)
	fprintf(file, "x,y\n");

	// 3. Loop through your entire flattened 1D array
	for (int i = 0; i < mesh->total_cell_centers; i++) {
		fprintf(file, "%f,%f\n", mesh->cell_centers_x[i], mesh->cell_centers_y[i]);
	}

	// 4. Close the file to save it and free it from memory
	fclose(file);

	printf("Successfully exported mesh to %s\n", filename);
}

void export_flow_boundary_to_csv(struct flow_boundary* boundary, struct mesh* mesh, const char* filename) {
	// 1. Open a file pointer in "w" (write) mode
	FILE* file = fopen(filename, "w");

	// Safety check: Make sure the file actually opened
	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	// 2. Write the CSV header (column names)
	fprintf(file, "x,y\n");

	// 3. Loop through your entire flattened 1D array
	for (int i = 0; i < mesh->num_points_vertical-1; i++) {
		fprintf(file, "%f,%f\n", mesh->cell_centers_x[boundary->cell_numbers[i]], mesh->cell_centers_y[boundary->cell_numbers[i]]);
	}

	// 4. Close the file to save it and free it from memory
	fclose(file);

	printf("Successfully exported mesh to %s\n", filename);
}

void export_wall_to_csv(struct wall_def* wall, struct mesh* mesh, const char* filename) {
	FILE* file = fopen(filename, "w");

	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	fprintf(file, "x,y\n");

	int num_wall_cells = wall->points_total - 1;
	for (int i = 0; i < num_wall_cells; i++) {
		int cell_idx = wall->cell_numbers[i];
		fprintf(file, "%f,%f\n", mesh->cell_centers_x[cell_idx], mesh->cell_centers_y[cell_idx]);
	}

	fclose(file);
	printf("Successfully exported wall to %s\n", filename);
}