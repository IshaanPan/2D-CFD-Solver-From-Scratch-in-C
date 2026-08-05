#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "physics.h"
#include "domain.h"


void initialise_field(struct field* field, struct mesh* mesh, double initial_u, double initial_w, double initial_pstat) {
	field->p_stat = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_u = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_w = malloc(sizeof(double) * mesh->total_cell_centers);

	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		field->p_stat[cell_number] = initial_pstat;
		field->vel_u[cell_number] = initial_u;
		field->vel_w[cell_number] = initial_w;

	}


}

void export_field_to_csv(struct field* field, struct mesh* mesh, const char* filename) {
	FILE* file = fopen(filename, "w");

	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", filename);
		return;
	}

	fprintf(file, "x,y,p_stat, vel_u, vel_w\n");

	for (int i = 0; i < mesh->total_cell_centers; i++) {
		fprintf(file, "%f,%f,%f,%f,%f\n", mesh->cell_centers_x[i], mesh->cell_centers_y[i], field->p_stat[i], field->vel_u[i], field->vel_w[i]);
	}

	fclose(file);
	printf("Successfully exported field to %s\n", filename);
}