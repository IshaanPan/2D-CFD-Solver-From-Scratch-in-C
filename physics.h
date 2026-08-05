#pragma once

struct field {
	double* vel_u;
	double* vel_w;
	double* p_stat;
};

void initialise_field(struct field* field, struct mesh* mesh, double initial_u, double initial_w, double initial_pstat);
void export_field_to_csv(struct field* field, struct mesh* mesh, const char* filename);