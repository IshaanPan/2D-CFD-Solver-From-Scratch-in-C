#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include "domain.h"
#include "physics.h"


void diffusive_gradients(struct field* field, struct mesh* mesh, struct flow_boundary* inlet) {
	int vertical_cell=mesh->num_points_vertical-1;
	#pragma omp parallel for schedule(static)
	for (int cell_number=0; cell_number<mesh->total_cell_centers; cell_number++) {
		int i_cell = cell_number / vertical_cell;
		int j_cell = cell_number % vertical_cell;

		if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
		} else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
		} else if (i_cell==0 && j_cell>0 && j_cell<vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double west_face_velocity=sqrt(2*(fmax(field->initial_ptotal-field->p_stat_old[cell_number],0.0))/(rho));

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell>0 && j_cell<vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double east_face_velocity_u=field->vel_u_old[cell_number];
			double east_face_velocity_v=field->vel_v_old[cell_number];

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
		} else if (i_cell==0 && j_cell==0) {
			int right_cell=cell_number+vertical_cell;
			int top_cell=cell_number+1;
			
			double west_face_velocity=sqrt(2*(fmax(field->initial_ptotal-field->p_stat_old[cell_number],0.0))/(rho));
			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
		} else if (i_cell==0 && j_cell==vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int bottom_cell=cell_number-1;
			
			double west_face_velocity=sqrt(2*(fmax(field->initial_ptotal-field->p_stat_old[cell_number],0.0))/(rho));

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(west_face_velocity*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+(0.0*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==0) {
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			
			double east_face_velocity_u=field->vel_u_old[cell_number];
			double east_face_velocity_v=field->vel_v_old[cell_number];

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number]));
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number]));
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
			int bottom_cell=cell_number-1;
			
			double east_face_velocity_u=field->vel_u_old[cell_number];
			double east_face_velocity_v=field->vel_v_old[cell_number];

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_u*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*((east_face_velocity_v*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
		} else {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;

			field->diffusive_dudx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dudy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_u_old[cell_number]+0.5*field->vel_u_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdx[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_nx[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_nx[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_nx[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_nx[cell_number])*mesh->south_length[cell_number]);
			field->diffusive_dvdy[cell_number]=(1.0/mesh->cell_volumes[cell_number])*(((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[right_cell])*mesh->east_ny[cell_number]*mesh->east_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[left_cell])*mesh->west_ny[cell_number]*mesh->west_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[top_cell])*mesh->north_ny[cell_number]*mesh->north_length[cell_number])+((0.5*field->vel_v_old[cell_number]+0.5*field->vel_v_old[bottom_cell])*mesh->south_ny[cell_number])*mesh->south_length[cell_number]);

		}

		 

		
		
	}
}



void initialise_field(struct field* field, struct mesh* mesh, double initial_u, double initial_v, double initial_pstat, struct flow_boundary* inlet) {
	field->p_stat_old = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_u_old = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_v_old = malloc(sizeof(double) * mesh->total_cell_centers);

	field->p_stat_new = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_u_new = malloc(sizeof(double) * mesh->total_cell_centers);
	field->vel_v_new = malloc(sizeof(double) * mesh->total_cell_centers);


	field->flux_east=malloc(sizeof(double) * mesh->total_cell_centers);
	field->flux_west=malloc(sizeof(double) * mesh->total_cell_centers);
	field->flux_north=malloc(sizeof(double) * mesh->total_cell_centers);
	field->flux_south=malloc(sizeof(double) * mesh->total_cell_centers);

	field->diffusive_dudx=malloc(sizeof(double)*mesh->total_cell_centers);
	field->diffusive_dudy=malloc(sizeof(double)*mesh->total_cell_centers);
	field->diffusive_dvdx=malloc(sizeof(double)*mesh->total_cell_centers);
	field->diffusive_dvdy=malloc(sizeof(double)*mesh->total_cell_centers);

	field->df_x=malloc(sizeof(double)*mesh->total_cell_centers);
	field->df_y=malloc(sizeof(double)*mesh->total_cell_centers);

	field->p_prime_new=malloc(sizeof(double)*mesh->total_cell_centers);
	field->p_prime_old=malloc(sizeof(double)*mesh->total_cell_centers);


	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		field->p_stat_old[cell_number] = initial_pstat;
		field->vel_u_old[cell_number] = initial_u;
		field->vel_v_old[cell_number] = initial_v;

		// Initialize the new fields (prevents garbage memory)
        field->p_stat_new[cell_number] = initial_pstat;
        field->vel_u_new[cell_number] = initial_u;
        field->vel_v_new[cell_number] = initial_v;
		field->p_prime_old[cell_number]=0.0;

	}

	for (int cell_number=0; cell_number<mesh->total_cell_centers; cell_number++) {
		int Ny=mesh->num_points_vertical-1;
		int i_cell=cell_number/Ny;
		int j_cell=cell_number%Ny;

		field->flux_east[cell_number]=rho*((mesh->east_nx[cell_number]*initial_u)+(mesh->east_ny[cell_number]*initial_v))*mesh->east_length[cell_number];
		field->flux_west[cell_number]=rho*((mesh->west_nx[cell_number]*initial_u)+(mesh->west_ny[cell_number]*initial_v))*mesh->west_length[cell_number];
		field->flux_north[cell_number]=rho*((mesh->north_nx[cell_number]*initial_u)+(mesh->north_ny[cell_number]*initial_v))*mesh->north_length[cell_number];
		field->flux_south[cell_number]=rho*((mesh->south_nx[cell_number]*initial_u)+(mesh->south_ny[cell_number]*initial_v))*mesh->south_length[cell_number];

		if (j_cell==0) {
			field->flux_south[cell_number]=0;
		}

		if (j_cell==Ny-1){
			field->flux_north[cell_number]=0;
		}



	}

	


	field->residual_mass=10;
	field->residual_u=10;
	field->residual_v=10;

}



double distance_between_cells(int parent_id, int neighbour_id, struct mesh* mesh){
	double parent_xx=mesh->cell_centers_x[parent_id];
	double parent_yy=mesh->cell_centers_y[parent_id];

	double neighbour_xx=mesh->cell_centers_x[neighbour_id];
	double neighbour_yy=mesh->cell_centers_y[neighbour_id];

	double distance=sqrt(pow((parent_xx-neighbour_xx),2)+pow((parent_yy-neighbour_yy),2));
	return distance;
}


// perpendicular distance from a cell centroid to one of its own faces, used for boundary faces
// where there is no neighbouring centroid to measure to
double face_normal_distance(int cell_number, struct mesh* mesh, char face) {
	int Ny_nodes = mesh->num_points_vertical;
	int Ny_cells = Ny_nodes - 1;
	int i_cell = cell_number / Ny_cells;
	int j_cell = cell_number % Ny_cells;

	int bl = (i_cell * Ny_nodes) + j_cell;
	int br = ((i_cell + 1) * Ny_nodes) + j_cell;
	int tl = (i_cell * Ny_nodes) + (j_cell + 1);
	int tr = ((i_cell + 1) * Ny_nodes) + (j_cell + 1);

	int p1, p2;
	double nx, ny;

	if (face == 'e') {
		p1 = br; p2 = tr;
		nx = mesh->east_nx[cell_number];  ny = mesh->east_ny[cell_number];
	} else if (face == 'w') {
		p1 = bl; p2 = tl;
		nx = mesh->west_nx[cell_number];  ny = mesh->west_ny[cell_number];
	} else if (face == 'n') {
		p1 = tl; p2 = tr;
		nx = mesh->north_nx[cell_number]; ny = mesh->north_ny[cell_number];
	} else {
		p1 = bl; p2 = br;
		nx = mesh->south_nx[cell_number]; ny = mesh->south_ny[cell_number];
	}

	double fx = 0.5 * (mesh->intersections_x[p1] + mesh->intersections_x[p2]);
	double fy = 0.5 * (mesh->intersections_y[p1] + mesh->intersections_y[p2]);

	double d = (fx - mesh->cell_centers_x[cell_number]) * nx
	         + (fy - mesh->cell_centers_y[cell_number]) * ny;

	return fmax(fabs(d), 1.0e-12);
}



void retrieve_convective_coefficients(double* e, double* w, double* n, double* s, int cell_number, struct field* field) {
	*e=fmax(field->flux_east[cell_number],0);
	*w=fmax(field->flux_west[cell_number],0);
	*n=fmax(field->flux_north[cell_number],0);
	*s=fmax(field->flux_south[cell_number],0);

	
	
}

double retrieve_convective_neighbour_sum(bool north, bool south, bool east, bool west, int cell_number, struct mesh* mesh, struct field* field, bool x) {
	
	if (x) {
	
	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		sum=sum+fmax(-field->flux_north[cell_number],0)*field->vel_u_old[top_cell];

		
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		sum=sum+fmax(-field->flux_south[cell_number],0)*field->vel_u_old[bottom_cell];
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		sum=sum+fmax(-field->flux_east[cell_number],0)*field->vel_u_old[right_cell];
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		sum=sum+fmax(-field->flux_west[cell_number],0)*field->vel_u_old[left_cell];
	}

	return sum;
	} else {

	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		sum=sum+fmax(-field->flux_north[cell_number],0)*field->vel_v_old[top_cell];

		
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		sum=sum+fmax(-field->flux_south[cell_number],0)*field->vel_v_old[bottom_cell];
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		sum=sum+fmax(-field->flux_east[cell_number],0)*field->vel_v_old[right_cell];
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		sum=sum+fmax(-field->flux_west[cell_number],0)*field->vel_v_old[left_cell];
	}





	return sum;
	}

	
}

double retrieve_diffusive_sum(bool north, bool south, bool east, bool west, int cell_number, struct mesh* mesh, struct field* field, bool x) {
	
	if (x) {
	
	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		double du_dx_face=0.5*field->diffusive_dudx[cell_number]+0.5*field->diffusive_dudx[top_cell];
		double du_dy_face=0.5*field->diffusive_dudy[cell_number]+0.5*field->diffusive_dudy[top_cell];

		double grad_sum=du_dx_face*mesh->north_nx[cell_number]+du_dy_face*mesh->north_ny[cell_number];
		double final_sum=(field->vel_u_old[cell_number]-field->vel_u_old[top_cell])/(distance_between_cells(cell_number, top_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->north_length[cell_number]*mu;
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		double du_dx_face=0.5*field->diffusive_dudx[cell_number]+0.5*field->diffusive_dudx[bottom_cell];
		double du_dy_face=0.5*field->diffusive_dudy[cell_number]+0.5*field->diffusive_dudy[bottom_cell];

		double grad_sum=du_dx_face*mesh->south_nx[cell_number]+du_dy_face*mesh->south_ny[cell_number];
		double final_sum=(field->vel_u_old[cell_number]-field->vel_u_old[bottom_cell])/(distance_between_cells(cell_number, bottom_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->south_length[cell_number]*mu;
		
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		double du_dx_face=0.5*field->diffusive_dudx[cell_number]+0.5*field->diffusive_dudx[right_cell];
		double du_dy_face=0.5*field->diffusive_dudy[cell_number]+0.5*field->diffusive_dudy[right_cell];

		double grad_sum=du_dx_face*mesh->east_nx[cell_number]+du_dy_face*mesh->east_ny[cell_number];
		double final_sum=(field->vel_u_old[cell_number]-field->vel_u_old[right_cell])/(distance_between_cells(cell_number, right_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->east_length[cell_number]*mu;
	
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		double du_dx_face=0.5*field->diffusive_dudx[cell_number]+0.5*field->diffusive_dudx[left_cell];
		double du_dy_face=0.5*field->diffusive_dudy[cell_number]+0.5*field->diffusive_dudy[left_cell];

		double grad_sum=du_dx_face*mesh->west_nx[cell_number]+du_dy_face*mesh->west_ny[cell_number];
		double final_sum=(field->vel_u_old[cell_number]-field->vel_u_old[left_cell])/(distance_between_cells(cell_number, left_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->west_length[cell_number]*mu;
		
	}
	return sum;
	} else {
	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		double dv_dx_face=0.5*field->diffusive_dvdx[cell_number]+0.5*field->diffusive_dvdx[top_cell];
		double dv_dy_face=0.5*field->diffusive_dvdy[cell_number]+0.5*field->diffusive_dvdy[top_cell];

		double grad_sum=dv_dx_face*mesh->north_nx[cell_number]+dv_dy_face*mesh->north_ny[cell_number];
		double final_sum=(field->vel_v_old[cell_number]-field->vel_v_old[top_cell])/(distance_between_cells(cell_number, top_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->north_length[cell_number]*mu;

		
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		double dv_dx_face=0.5*field->diffusive_dvdx[cell_number]+0.5*field->diffusive_dvdx[bottom_cell];
		double dv_dy_face=0.5*field->diffusive_dvdy[cell_number]+0.5*field->diffusive_dvdy[bottom_cell];

		double grad_sum=dv_dx_face*mesh->south_nx[cell_number]+dv_dy_face*mesh->south_ny[cell_number];
		double final_sum=(field->vel_v_old[cell_number]-field->vel_v_old[bottom_cell])/(distance_between_cells(cell_number, bottom_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->south_length[cell_number]*mu;
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		double dv_dx_face=0.5*field->diffusive_dvdx[cell_number]+0.5*field->diffusive_dvdx[right_cell];
		double dv_dy_face=0.5*field->diffusive_dvdy[cell_number]+0.5*field->diffusive_dvdy[right_cell];

		double grad_sum=dv_dx_face*mesh->east_nx[cell_number]+dv_dy_face*mesh->east_ny[cell_number];
		double final_sum=(field->vel_v_old[cell_number]-field->vel_v_old[right_cell])/(distance_between_cells(cell_number, right_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->east_length[cell_number]*mu;
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		double dv_dx_face=0.5*field->diffusive_dvdx[cell_number]+0.5*field->diffusive_dvdx[left_cell];
		double dv_dy_face=0.5*field->diffusive_dvdy[cell_number]+0.5*field->diffusive_dvdy[left_cell];

		double grad_sum=dv_dx_face*mesh->west_nx[cell_number]+dv_dy_face*mesh->west_ny[cell_number];
		double final_sum=(field->vel_v_old[cell_number]-field->vel_v_old[left_cell])/(distance_between_cells(cell_number, left_cell, mesh));

		sum=sum+(grad_sum+final_sum)*mesh->west_length[cell_number]*mu;
	}



	return sum;

	}
}


// Volume-normalised Green-Gauss gradient of the static pressure.
// The four boolean face flags are retained only so the existing call sites keep compiling.
// Which faces are interior and which are boundaries is now derived from the cell index,
// so a caller can no longer mask the wrong face - passing (true,true,true,true) for a
// boundary cell used to read p_stat_old out of bounds.
// Boundary faces use: outlet -> prescribed static pressure, inlet and walls -> zero gradient.
double pressure_grad(bool north, bool south, bool east, bool west, int cell_number, struct mesh* mesh, struct field* field, bool x) {
	(void)north; (void)south; (void)east; (void)west;

	int vertical_cell = mesh->num_points_vertical - 1;
	int horizontal_cell = mesh->num_points_horizontal - 1;
	int i_cell = cell_number / vertical_cell;
	int j_cell = cell_number % vertical_cell;

	double p_p = field->p_stat_old[cell_number];

	double p_east  = (i_cell < horizontal_cell - 1) ? 0.5 * (p_p + field->p_stat_old[cell_number + vertical_cell]) : field->initial_pstat;
	double p_west  = (i_cell > 0)                   ? 0.5 * (p_p + field->p_stat_old[cell_number - vertical_cell]) : p_p;
	double p_north = (j_cell < vertical_cell - 1)   ? 0.5 * (p_p + field->p_stat_old[cell_number + 1])             : p_p;
	double p_south = (j_cell > 0)                   ? 0.5 * (p_p + field->p_stat_old[cell_number - 1])             : p_p;

	const double* n_east  = x ? mesh->east_nx  : mesh->east_ny;
	const double* n_west  = x ? mesh->west_nx  : mesh->west_ny;
	const double* n_north = x ? mesh->north_nx : mesh->north_ny;
	const double* n_south = x ? mesh->south_nx : mesh->south_ny;

	double sum = p_east  * n_east[cell_number]  * mesh->east_length[cell_number]
	           + p_west  * n_west[cell_number]  * mesh->west_length[cell_number]
	           + p_north * n_north[cell_number] * mesh->north_length[cell_number]
	           + p_south * n_south[cell_number] * mesh->south_length[cell_number];

	return sum / mesh->cell_volumes[cell_number];
}


void solve_x_momentum(struct field* field, struct mesh* mesh, struct flow_boundary* inlet){
	int vertical_cell=mesh->num_points_vertical-1;
	#pragma omp parallel for schedule(static)
	for (int cell_number=0; cell_number<mesh->total_cell_centers; cell_number++) {
		int i_cell = cell_number / vertical_cell;
		int j_cell = cell_number % vertical_cell;

		//interior bottom wall
		if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;

			double n,s,e,w;
			retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
			double convective_neighbour_sum=retrieve_convective_neighbour_sum(true, false ,true, true, cell_number, mesh, field, true);
			double diffusive_sum=retrieve_diffusive_sum(true,false,true,true,cell_number, mesh, field, true);

			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh)));
			
			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(face_normal_distance(cell_number, mesh, 's')));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);

			
			// interior top wall
		} else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int bottom_cell=cell_number-1;

			double n,s,e,w;
			retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
			double convective_neighbour_sum=retrieve_convective_neighbour_sum(false, true ,true, true, cell_number, mesh, field, true);
			double diffusive_sum=retrieve_diffusive_sum(false,true,true,true,cell_number, mesh, field, true);
			
			
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
			
			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(face_normal_distance(cell_number, mesh, 'n'))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);


			// interior inlet
		} else if (i_cell==0 && j_cell>0 && j_cell<vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double west_face_velocity=sqrt(fmax(2*(field->initial_ptotal-field->p_stat_old[cell_number]),0.0)/(rho));

			double n,s,e,w;
			retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
			double convective_neighbour_sum=retrieve_convective_neighbour_sum(true, true ,true, false, cell_number, mesh, field, true)+ fmax(-field->flux_west[cell_number], 0) * west_face_velocity;
			double diffusive_sum=retrieve_diffusive_sum(true,true,true,false,cell_number, mesh, field, true);
			
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			
			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(west_face_velocity*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
			
			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);



			// interior outlet
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell>0 && j_cell<vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double east_face_velocity_u=field->vel_u_old[cell_number];

			double n,s,e,w;
			retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
			double convective_neighbour_sum=retrieve_convective_neighbour_sum(true, true ,false, true, cell_number, mesh, field, true)+fmax(-field->flux_east[cell_number], 0) * east_face_velocity_u;
			double diffusive_sum=retrieve_diffusive_sum(true,true,false,true,cell_number, mesh, field, true);
			
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
			
			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);

			

			// bl
		} else if (i_cell==0 && j_cell==0) {
			int right_cell = cell_number + vertical_cell;
			int top_cell = cell_number + 1;

			double west_face_velocity = sqrt(fmax(2 * (field->initial_ptotal - field->p_stat_old[cell_number]), 0.0) / (rho));


			double n, s, e, w;
			retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);


			double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, false, true, false, cell_number, mesh, field, true) + fmax(-field->flux_west[cell_number], 0) * west_face_velocity;


			// This is where it calls retrieve_diffusive_sum next:
			double diffusive_sum = retrieve_diffusive_sum(true, false, true, false, cell_number, mesh, field, true);
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			
			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(west_face_velocity*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh)));
			

			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(face_normal_distance(cell_number, mesh, 's')));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);

			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);


			// tl
		} else if (i_cell==0 && j_cell==vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
            int bottom_cell=cell_number-1;
            
			double west_face_velocity=sqrt(fmax(2*(field->initial_ptotal-field->p_stat_old[cell_number]),0.0)/(rho));

            double n,s,e,w;
            retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
            double convective_neighbour_sum=retrieve_convective_neighbour_sum(false, true, true, false, cell_number, mesh, field, true) + fmax(-field->flux_west[cell_number], 0) * west_face_velocity;
            double diffusive_sum=retrieve_diffusive_sum(false, true, true, false, cell_number, mesh, field, true);
            

			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];

			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(west_face_velocity*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1=n+s+e+w;
            double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(face_normal_distance(cell_number, mesh, 'w'))+(1*mesh->north_length[cell_number])/(face_normal_distance(cell_number, mesh, 'n'))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            //field->vel_u_new[cell_number]=rhs/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);
			
			

			// br
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==0) {
			int left_cell=cell_number-vertical_cell;
            int top_cell=cell_number+1;
            
            double east_face_velocity_u=field->vel_u_old[cell_number];

            double n,s,e,w;
            retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
            double convective_neighbour_sum=retrieve_convective_neighbour_sum(true, false, false, true, cell_number, mesh, field, true) + fmax(-field->flux_east[cell_number], 0) * east_face_velocity_u;
            double diffusive_sum=retrieve_diffusive_sum(true, false, false, true, cell_number, mesh, field, true);
            
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];


            double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh)));
            
            double denom_1=n+s+e+w;
            double denom_2=mu*((1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(face_normal_distance(cell_number, mesh, 's')));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);
			

			//tr
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
            int bottom_cell=cell_number-1;
            
            double east_face_velocity_u=field->vel_u_old[cell_number];

            double n,s,e,w;
            retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
            double convective_neighbour_sum=retrieve_convective_neighbour_sum(false, true, false, true, cell_number, mesh, field, true) + fmax(-field->flux_east[cell_number], 0) * east_face_velocity_u;
            double diffusive_sum=retrieve_diffusive_sum(false, true, false, true, cell_number, mesh, field, true);
            
			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];



            double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1=n+s+e+w;
            double denom_2=mu*((1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(face_normal_distance(cell_number, mesh, 'n'))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));


			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);

			

			//pure interior
		} else {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;

			

			double n,s,e,w;
			retrieve_convective_coefficients(&e,&w,&n,&s, cell_number, field);
			double convective_neighbour_sum=retrieve_convective_neighbour_sum(true, true ,true, true, cell_number, mesh, field, true);
			double diffusive_sum=retrieve_diffusive_sum(true,true,true,true,cell_number, mesh, field, true);
			

			double pressure_gradient=pressure_grad(true, true, true, true, cell_number, mesh, field, true)*mesh->cell_volumes[cell_number];
			
			double rhs=-pressure_gradient+convective_neighbour_sum+diffusive_sum+mu*((field->vel_u_old[right_cell]*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(field->vel_u_old[left_cell]*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(field->vel_u_old[top_cell]*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(field->vel_u_old[bottom_cell]*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));
			
			double denom_1=n+s+e+w;
			double denom_2=mu*((1*mesh->east_length[cell_number])/(distance_between_cells(cell_number, right_cell, mesh))+(1*mesh->west_length[cell_number])/(distance_between_cells(cell_number, left_cell, mesh))+(1*mesh->north_length[cell_number])/(distance_between_cells(cell_number, top_cell, mesh))+(1*mesh->south_length[cell_number])/(distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_x[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_u_new[cell_number] = field->vel_u_old[cell_number] + relaxation_u*(u_computed - field->vel_u_old[cell_number]);



		}

		 

		
		
	}
}

void solve_y_momentum(struct field* field, struct mesh* mesh, struct flow_boundary* inlet){
    int vertical_cell = mesh->num_points_vertical - 1;
    #pragma omp parallel for schedule(static)
    for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
        int i_cell = cell_number / vertical_cell;
        int j_cell = cell_number % vertical_cell;

        // interior bottom wall
        if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, false, true, true, cell_number, mesh, field, false);
            double diffusive_sum = retrieve_diffusive_sum(true, false, true, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (south_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (face_normal_distance(cell_number, mesh, 's')));


			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
          
			double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // interior top wall
        } else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int bottom_cell = cell_number - 1;

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(false, true, true, true, cell_number, mesh, field, false);
            double diffusive_sum = retrieve_diffusive_sum(false, true, true, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (north_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'n')) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
           	double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // interior inlet
        } else if (i_cell == 0 && j_cell > 0 && j_cell < vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
            
            double west_face_velocity_v = 0.0; // V is 0 at a horizontal inlet

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, true, true, false, cell_number, mesh, field, false) + fmax(-field->flux_west[cell_number], 0) * west_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(true, true, true, false, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (west_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (west_face_velocity_v * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // interior outlet
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell > 0 && j_cell < vertical_cell - 1) {
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, true, false, true, cell_number, mesh, field, false) + fmax(-field->flux_east[cell_number], 0) * east_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(true, true, false, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (east_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
          	double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // bl
        } else if (i_cell == 0 && j_cell == 0) {
            int right_cell = cell_number + vertical_cell;
            int top_cell = cell_number + 1;
            
            double west_face_velocity_v = 0.0;

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, false, true, false, cell_number, mesh, field, false) + fmax(-field->flux_west[cell_number], 0) * west_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(true, false, true, false, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (south_ny, west_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (west_face_velocity_v * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (face_normal_distance(cell_number, mesh, 's')));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // tl
        } else if (i_cell == 0 && j_cell == vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int bottom_cell = cell_number - 1;
            
            double west_face_velocity_v = 0.0;

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(false, true, true, false, cell_number, mesh, field, false) + fmax(-field->flux_west[cell_number], 0) * west_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(false, true, true, false, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (north_ny, west_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (west_face_velocity_v * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'w')) + (1 * mesh->north_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'n')) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
           	double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // br
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == 0) {
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, false, false, true, cell_number, mesh, field, false) + fmax(-field->flux_east[cell_number], 0) * east_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(true, false, false, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (south_ny, east_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (face_normal_distance(cell_number, mesh, 's')));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // tr
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            int left_cell = cell_number - vertical_cell;
            int bottom_cell = cell_number - 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(false, true, false, true, cell_number, mesh, field, false) + fmax(-field->flux_east[cell_number], 0) * east_face_velocity_v;
            double diffusive_sum = retrieve_diffusive_sum(false, true, false, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force (north_ny, east_ny)
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];

            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (face_normal_distance(cell_number, mesh, 'n')) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
          	double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

            
        // pure interior
        } else {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;

            double n, s, e, w;
            retrieve_convective_coefficients(&e, &w, &n, &s, cell_number, field);
            double convective_neighbour_sum = retrieve_convective_neighbour_sum(true, true, true, true, cell_number, mesh, field, false);
            double diffusive_sum = retrieve_diffusive_sum(true, true, true, true, cell_number, mesh, field, false);
            
            // Added Y-Pressure Force
            double pressure_gradient = pressure_grad(true, true, true, true, cell_number, mesh, field, false) * mesh->cell_volumes[cell_number];
            
            double rhs = -pressure_gradient + convective_neighbour_sum + diffusive_sum + mu * ((field->vel_v_old[right_cell] * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (field->vel_v_old[left_cell] * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (field->vel_v_old[top_cell] * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (field->vel_v_old[bottom_cell] * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));
            
            double denom_1 = n + s + e + w;
            double denom_2 = mu * ((1 * mesh->east_length[cell_number]) / (distance_between_cells(cell_number, right_cell, mesh)) + (1 * mesh->west_length[cell_number]) / (distance_between_cells(cell_number, left_cell, mesh)) + (1 * mesh->north_length[cell_number]) / (distance_between_cells(cell_number, top_cell, mesh)) + (1 * mesh->south_length[cell_number]) / (distance_between_cells(cell_number, bottom_cell, mesh)));

			field->df_y[cell_number]=relaxation_u*mesh->cell_volumes[cell_number]/(denom_1+denom_2);
            double u_computed=rhs/(denom_1+denom_2);
			field->vel_v_new[cell_number] = field->vel_v_old[cell_number] + relaxation_u*(u_computed - field->vel_v_old[cell_number]);

        }
    }
}

double interp_velocity_u(int parent_cell, int neighbour_cell, struct field* field){
	return 0.5*field->vel_u_new[parent_cell]+0.5*field->vel_u_new[neighbour_cell];

}

double interp_velocity_v(int parent_cell, int neighbour_cell, struct field* field){
	return 0.5*field->vel_v_new[parent_cell]+0.5*field->vel_v_new[neighbour_cell];

}

// double interp_df(int parent_cell, int neighbour_cell, struct field* field, bool x){
// 	if (x) {
// 		return 0.5*field->df_x[parent_cell]+0.5*field->df_x[neighbour_cell];
// 	} else {
// 		return 0.5*field->df_y[parent_cell]+0.5*field->df_y[neighbour_cell];
// 	}
// }

double interp_df(int parent_cell, int neighbour_cell, struct field* field, bool is_x) {
    double val_p = is_x ? field->df_x[parent_cell] : field->df_y[parent_cell];
    double val_n = is_x ? field->df_x[neighbour_cell] : field->df_y[neighbour_cell];

    // Ensure the interpolated coefficient is always physically positive
    return fmax(fabs(0.5 * val_p + 0.5 * val_n), 1.0e-12);
}

double interp_pressure(int parent_cell, int neighbour_cell, struct field* field, struct mesh* mesh){
	return (field->p_stat_old[neighbour_cell]-field->p_stat_old[parent_cell])/(distance_between_cells(parent_cell, neighbour_cell, mesh));
}

void rhie_chow_flux(struct mesh* mesh, struct field* field){
	int vertical_cell=mesh->num_points_vertical-1;
	#pragma omp parallel for schedule(static)
	for (int cell_number=0; cell_number<mesh->total_cell_centers; cell_number++) {
		int i_cell = cell_number / vertical_cell;
		int j_cell = cell_number % vertical_cell;

		//interior bottom wall
		if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;

			double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
			double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
			double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);

			double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];

			double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
			double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);

			double df_x_east=interp_df(cell_number, right_cell, field, true);
			double df_y_east=interp_df(cell_number, right_cell, field, false);

			
			double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
			double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);

			field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

			
            double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);

            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];

            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);

            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);

            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];


            
            double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);

            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];

            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);

            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);

            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];

			field->flux_south[cell_number]=0;
			
			
			// interior top wall
		} else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int bottom_cell=cell_number-1;

			double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
			double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
			double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);

			double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];

			double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
			double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);

			double df_x_east=interp_df(cell_number, right_cell, field, true);
			double df_y_east=interp_df(cell_number, right_cell, field, false);

			
			double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
			double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);

			field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

			
            double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);

            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];

            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);

            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);

            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];

			double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);

            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];

            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);

            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);

            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];
			field->flux_north[cell_number]=0;

			// interior inlet
		} else if (i_cell==0 && j_cell>0 && j_cell<vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;

			double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
			double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
			double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);

			double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];

			double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
			double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);

			double df_x_east=interp_df(cell_number, right_cell, field, true);
			double df_y_east=interp_df(cell_number, right_cell, field, false);

			
			double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
			double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);

			field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

			
			double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);

            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];

            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);

            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);

            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];


         
            double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);

            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];

            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);

            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);

            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];

			// inlet mass flux comes from the total pressure boundary condition, not from the
			// cell centre velocity, otherwise the inflow never responds to the imposed p_total
			double west_face_velocity=sqrt(fmax(2.0*(field->initial_ptotal-field->p_stat_old[cell_number]),0.0)/rho);
			double normal_velocity_west=west_face_velocity*mesh->west_nx[cell_number];
			field->flux_west[cell_number]=normal_velocity_west*rho*mesh->west_length[cell_number];



			// interior outlet
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell>0 && j_cell<vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);

            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];

            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);

            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);

            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];


            
            double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);

            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];

            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);

            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);

            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];


         
            double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);

            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];

            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);

            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);

            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];

			double normal_velocity_east=field->vel_u_new[cell_number]*mesh->east_nx[cell_number]+field->vel_v_new[cell_number]*mesh->east_ny[cell_number];
			field->flux_east[cell_number]=normal_velocity_east*rho*mesh->east_length[cell_number];


			

			// bl
		} else if (i_cell==0 && j_cell==0) {
			int right_cell=cell_number+vertical_cell;
            int top_cell=cell_number+1;
            
            
            double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
            double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
            double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);
            double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];
            double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
            double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);
            double df_x_east=interp_df(cell_number, right_cell, field, true);
            double df_y_east=interp_df(cell_number, right_cell, field, false);
            double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
            double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);
            field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

            
            double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);
            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];
            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);
            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);
            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];

            
            field->flux_south[cell_number]=0;

            
            double west_face_velocity=sqrt(fmax(2.0*(field->initial_ptotal-field->p_stat_old[cell_number]),0.0)/rho);
            double normal_velocity_west=west_face_velocity*mesh->west_nx[cell_number];
            field->flux_west[cell_number]=normal_velocity_west*rho*mesh->west_length[cell_number];
			

			// tl
		} else if (i_cell==0 && j_cell==vertical_cell-1) {
			int right_cell=cell_number+vertical_cell;
            int bottom_cell=cell_number-1;
            
            
            double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
            double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
            double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);
            double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];
            double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
            double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);
            double df_x_east=interp_df(cell_number, right_cell, field, true);
            double df_y_east=interp_df(cell_number, right_cell, field, false);
            double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
            double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);
            field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

            
            double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);
            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];
            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);
            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);
            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];

           
            field->flux_north[cell_number]=0;

            
            double west_face_velocity=sqrt(fmax(2.0*(field->initial_ptotal-field->p_stat_old[cell_number]),0.0)/rho);
            double normal_velocity_west=west_face_velocity*mesh->west_nx[cell_number];
            field->flux_west[cell_number]=normal_velocity_west*rho*mesh->west_length[cell_number];
           
			// br
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==0) {
			int left_cell=cell_number-vertical_cell;
            int top_cell=cell_number+1;
            
            
            double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);
            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];
            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);
            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);
            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];

            
            double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);
            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];
            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);
            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);
            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];

           
            field->flux_south[cell_number]=0;

           
            double normal_velocity_east=field->vel_u_new[cell_number]*mesh->east_nx[cell_number]+field->vel_v_new[cell_number]*mesh->east_ny[cell_number];
            field->flux_east[cell_number]=normal_velocity_east*rho*mesh->east_length[cell_number];
          
			

			//tr
		} else if (i_cell==mesh->num_points_horizontal-2 && j_cell==vertical_cell-1) {
			int left_cell=cell_number-vertical_cell;
            int bottom_cell=cell_number-1;
            
            
            double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);
            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];
            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);
            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);
            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];

            
            double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);
            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];
            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);
            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);
            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];

            
            field->flux_north[cell_number]=0;

            
            double normal_velocity_east=field->vel_u_new[cell_number]*mesh->east_nx[cell_number]+field->vel_v_new[cell_number]*mesh->east_ny[cell_number];
            field->flux_east[cell_number]=normal_velocity_east*rho*mesh->east_length[cell_number];
            
           

			//pure interior
		} else {
			int right_cell=cell_number+vertical_cell;
			int left_cell=cell_number-vertical_cell;
			int top_cell=cell_number+1;
			int bottom_cell=cell_number-1;
			
			double interp_u_east=interp_velocity_u(cell_number, right_cell, field);
			double interp_v_east=interp_velocity_v(cell_number, right_cell, field);
			double interp_pressure_east=interp_pressure(cell_number,right_cell, field, mesh);

			double uncorrected_normal_velocity_east=interp_u_east*mesh->east_nx[cell_number]+interp_v_east*mesh->east_ny[cell_number];

			double fancy_interp_pressure_x_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, true);
			double fancy_interp_pressure_y_east=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, right_cell, mesh, field, false);

			double df_x_east=interp_df(cell_number, right_cell, field, true);
			double df_y_east=interp_df(cell_number, right_cell, field, false);

			
			double normal_fancy_east=fancy_interp_pressure_x_east*mesh->east_nx[cell_number]+fancy_interp_pressure_y_east*mesh->east_ny[cell_number];
			double corrected_normal_velocity_east=uncorrected_normal_velocity_east-(0.5*df_x_east+0.5*df_y_east)*(interp_pressure_east-normal_fancy_east);

			field->flux_east[cell_number]=rho*corrected_normal_velocity_east*mesh->east_length[cell_number];

			
            double interp_u_west=interp_velocity_u(cell_number, left_cell, field);
            double interp_v_west=interp_velocity_v(cell_number, left_cell, field);
            double interp_pressure_west=interp_pressure(cell_number,left_cell, field, mesh);

            double uncorrected_normal_velocity_west=interp_u_west*mesh->west_nx[cell_number]+interp_v_west*mesh->west_ny[cell_number];

            double fancy_interp_pressure_x_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, true);
            double fancy_interp_pressure_y_west=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, left_cell, mesh, field, false);

            double df_x_west=interp_df(cell_number, left_cell, field, true);
            double df_y_west=interp_df(cell_number, left_cell, field, false);
            
            double normal_fancy_west=fancy_interp_pressure_x_west*mesh->west_nx[cell_number]+fancy_interp_pressure_y_west*mesh->west_ny[cell_number];
            double corrected_normal_velocity_west=uncorrected_normal_velocity_west-(0.5*df_x_west+0.5*df_y_west)*(interp_pressure_west-normal_fancy_west);

            field->flux_west[cell_number]=rho*corrected_normal_velocity_west*mesh->west_length[cell_number];


            
            double interp_u_north=interp_velocity_u(cell_number, top_cell, field);
            double interp_v_north=interp_velocity_v(cell_number, top_cell, field);
            double interp_pressure_north=interp_pressure(cell_number,top_cell, field, mesh);

            double uncorrected_normal_velocity_north=interp_u_north*mesh->north_nx[cell_number]+interp_v_north*mesh->north_ny[cell_number];

            double fancy_interp_pressure_x_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, true);
            double fancy_interp_pressure_y_north=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, top_cell, mesh, field, false);

            double df_x_north=interp_df(cell_number, top_cell, field, true);
            double df_y_north=interp_df(cell_number, top_cell, field, false);
            
            double normal_fancy_north=fancy_interp_pressure_x_north*mesh->north_nx[cell_number]+fancy_interp_pressure_y_north*mesh->north_ny[cell_number];
            double corrected_normal_velocity_north=uncorrected_normal_velocity_north-(0.5*df_x_north+0.5*df_y_north)*(interp_pressure_north-normal_fancy_north);

            field->flux_north[cell_number]=rho*corrected_normal_velocity_north*mesh->north_length[cell_number];


         
            double interp_u_south=interp_velocity_u(cell_number, bottom_cell, field);
            double interp_v_south=interp_velocity_v(cell_number, bottom_cell, field);
            double interp_pressure_south=interp_pressure(cell_number,bottom_cell, field, mesh);

            double uncorrected_normal_velocity_south=interp_u_south*mesh->south_nx[cell_number]+interp_v_south*mesh->south_ny[cell_number];

            double fancy_interp_pressure_x_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, true)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, true);
            double fancy_interp_pressure_y_south=0.5*pressure_grad(true, true, true, true, cell_number, mesh, field, false)+0.5*pressure_grad(true, true, true, true, bottom_cell, mesh, field, false);

            double df_x_south=interp_df(cell_number, bottom_cell, field, true);
            double df_y_south=interp_df(cell_number, bottom_cell, field, false);
            
            double normal_fancy_south=fancy_interp_pressure_x_south*mesh->south_nx[cell_number]+fancy_interp_pressure_y_south*mesh->south_ny[cell_number];
            double corrected_normal_velocity_south=uncorrected_normal_velocity_south-(0.5*df_x_south+0.5*df_y_south)*(interp_pressure_south-normal_fancy_south);

            field->flux_south[cell_number]=rho*corrected_normal_velocity_south*mesh->south_length[cell_number];


			


		}

		 

		
		
	}

}

// double compute_df(int parent_cell, int neighbour_cell, struct field* field, struct mesh* mesh, char* face) {
// 	double df_x=interp_df(parent_cell, neighbour_cell, field, true);
// 	double df_y=interp_df(parent_cell, neighbour_cell, field, false);

// 	if (strcmp(face, "east")==0) {
// 		double df=df_x*pow(mesh->east_nx[parent_cell],2)+df_y*pow(mesh->east_ny[parent_cell],2);
// 		return df;
// 	} else if (strcmp(face, "west")==0) {
// 		double df=df_x*pow(mesh->west_nx[parent_cell],2)+df_y*pow(mesh->west_ny[parent_cell],2);
// 		return df;
// 	} else if (strcmp(face, "north")==0) {
// 		double df=df_x*pow(mesh->north_nx[parent_cell],2)+df_y*pow(mesh->north_ny[parent_cell],2);
// 		return df;
// 	} else if (strcmp(face, "south")==0) {
// 		double df=df_x*pow(mesh->south_nx[parent_cell],2)+df_y*pow(mesh->south_ny[parent_cell],2);
// 		return df;
// 	}
// }

double compute_df(int parent_cell, int neighbour_cell, struct field* field, struct mesh* mesh, char* face) {
    double df_x = interp_df(parent_cell, neighbour_cell, field, true);
    double df_y = interp_df(parent_cell, neighbour_cell, field, false);

    double df = 0.0;

    if (strcmp(face, "east") == 0) {
        df = df_x * pow(mesh->east_nx[parent_cell], 2) + df_y * pow(mesh->east_ny[parent_cell], 2);
    } else if (strcmp(face, "west") == 0) {
        df = df_x * pow(mesh->west_nx[parent_cell], 2) + df_y * pow(mesh->west_ny[parent_cell], 2);
    } else if (strcmp(face, "north") == 0) {
        df = df_x * pow(mesh->north_nx[parent_cell], 2) + df_y * pow(mesh->north_ny[parent_cell], 2);
    } else if (strcmp(face, "south") == 0) {
        df = df_x * pow(mesh->south_nx[parent_cell], 2) + df_y * pow(mesh->south_ny[parent_cell], 2);
    }

    // Force df to be strictly positive to guarantee matrix stability
    return fmax(fabs(df), 1.0e-12);
}

void determine_pressure_correction(struct mesh* mesh, struct field* field) {
	 int vertical_cell = mesh->num_points_vertical - 1;
    #pragma omp parallel for schedule(static)
    for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
        int i_cell = cell_number / vertical_cell;
        int j_cell = cell_number % vertical_cell;

        // interior bottom wall
        if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;

			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=0.0;

			rhs=rhs+(a*field->p_prime_old[right_cell]
					+b*field->p_prime_old[left_cell]
					+c*field->p_prime_old[top_cell]);
				
			
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           
            
        // interior top wall
        } else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int bottom_cell = cell_number - 1;

			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=0.0;
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*field->p_prime_old[right_cell]
					+b*field->p_prime_old[left_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           

        // interior inlet
        } else if (i_cell == 0 && j_cell > 0 && j_cell < vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
            
            double west_face_velocity_v = 0.0; 

			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_west = field->df_x[cell_number] * pow(mesh->west_nx[cell_number], 2) + field->df_y[cell_number] * pow(mesh->west_ny[cell_number], 2);
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b = rho * df_west * mesh->west_length[cell_number] / face_normal_distance(cell_number, mesh, 'w');
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*field->p_prime_old[right_cell]
					+c*field->p_prime_old[top_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           

           
            
        // interior outlet
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell > 0 && j_cell < vertical_cell - 1) {
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];

			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=field->df_x[cell_number]*pow(mesh->east_nx[cell_number],2)+field->df_y[cell_number]*pow(mesh->east_ny[cell_number],2);
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/face_normal_distance(cell_number, mesh, 'e');
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*0
					+b*field->p_prime_old[left_cell]
					+c*field->p_prime_old[top_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           

            
            
        // bl
        } else if (i_cell == 0 && j_cell == 0) {
            int right_cell = cell_number + vertical_cell;
            int top_cell = cell_number + 1;
            
            double west_face_velocity_v = 0.0;
					
			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			double df_west = field->df_x[cell_number] * pow(mesh->west_nx[cell_number], 2) + field->df_y[cell_number] * pow(mesh->west_ny[cell_number], 2);

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b = rho * df_west * mesh->west_length[cell_number] / face_normal_distance(cell_number, mesh, 'w');
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=0.0;
			rhs=rhs+(a*field->p_prime_old[right_cell]
					+c*field->p_prime_old[top_cell]);
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           

           
            
        // tl
        } else if (i_cell == 0 && j_cell == vertical_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            int bottom_cell = cell_number - 1;
            
            double west_face_velocity_v = 0.0;
					
			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");
			double df_west = field->df_x[cell_number] * pow(mesh->west_nx[cell_number], 2) + field->df_y[cell_number] * pow(mesh->west_ny[cell_number], 2);

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b = rho * df_west * mesh->west_length[cell_number] / face_normal_distance(cell_number, mesh, 'w');			
			c=0.0;
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*field->p_prime_old[right_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           

            
            
        // br
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == 0) {
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];
					
			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=field->df_x[cell_number]*pow(mesh->east_nx[cell_number],2)+field->df_y[cell_number]*pow(mesh->east_ny[cell_number],2);
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/face_normal_distance(cell_number, mesh, 'e');
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=0.0;

			rhs=rhs+(a*0.0
					+b*field->p_prime_old[left_cell]
					+c*field->p_prime_old[top_cell]);
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           
           
            
        // tr
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            int left_cell = cell_number - vertical_cell;
            int bottom_cell = cell_number - 1;
            
            double east_face_velocity_v = field->vel_v_old[cell_number];

			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=field->df_x[cell_number]*pow(mesh->east_nx[cell_number],2)+field->df_y[cell_number]*pow(mesh->east_ny[cell_number],2);
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/face_normal_distance(cell_number, mesh, 'e');
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=0.0;
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*0.0
					+b*field->p_prime_old[left_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           
			

           
            
        // pure interior
        } else {
            int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
			
			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);

			rhs=rhs+(a*field->p_prime_old[right_cell]
					+b*field->p_prime_old[left_cell]
					+c*field->p_prime_old[top_cell]
					+d*field->p_prime_old[bottom_cell]);
			
			
			double p_prime_computed = rhs / (a + b + c + d);
			field->p_prime_new[cell_number] = field->p_prime_old[cell_number] + relaxation_prime * (p_prime_computed - field->p_prime_old[cell_number]);
           
        }
    }
}

// in case used, but currently not employed
double compute_residual_pressure(struct field* field, struct mesh* mesh) {
	double sum=0.0;
	int vertical_cell = mesh->num_points_vertical - 1;
    for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		int i_cell = cell_number / vertical_cell;
        int j_cell = cell_number % vertical_cell;

		if (i_cell>0 && i_cell<mesh->num_points_horizontal-1 && j_cell>0 && j_cell<vertical_cell) {
			int right_cell = cell_number + vertical_cell;
            int left_cell = cell_number - vertical_cell;
            int top_cell = cell_number + 1;
            int bottom_cell = cell_number - 1;
			
			double rhs=-(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
			
			double df_east=compute_df(cell_number, right_cell, field, mesh, "east");
			double df_west=compute_df(cell_number, left_cell, field, mesh, "west");
			double df_north=compute_df(cell_number, top_cell, field, mesh, "north");
			double df_south=compute_df(cell_number, bottom_cell, field, mesh, "south");

			double a,b,c,d;
			a=rho*df_east*mesh->east_length[cell_number]/distance_between_cells(cell_number, right_cell, mesh);
			b=rho*df_west*mesh->west_length[cell_number]/distance_between_cells(cell_number, left_cell, mesh);
			c=rho*df_north*mesh->north_length[cell_number]/distance_between_cells(cell_number, top_cell, mesh);
			d=rho*df_south*mesh->south_length[cell_number]/distance_between_cells(cell_number,bottom_cell, mesh);
			
			double residual_cell=fabs(rhs-((a+b+c+d)*field->p_prime_new[cell_number]-a*field->p_prime_new[right_cell]
			-b*field->p_prime_new[left_cell]
			-c*field->p_prime_new[top_cell]
			-d*field->p_prime_new[bottom_cell]));

			sum=sum+residual_cell;
		}
	} return sum;
}

void set_pressure_zero (struct mesh* mesh, struct field* field) {
	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++){ 
		field->p_prime_old[cell_number]=0.0;
	}
}

// computes average pressure prime gradient for a cell
double pressure_prime_grad(bool north, bool south, bool east, bool west, int cell_number, struct mesh* mesh, struct field* field, bool x) {
	
	if (x) {
	
	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[top_cell];
		sum=sum+p_face*mesh->north_nx[cell_number]*mesh->north_length[cell_number];
		

		
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[bottom_cell];
		sum=sum+p_face*mesh->south_nx[cell_number]*mesh->south_length[cell_number];

		
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[right_cell];
		sum=sum+p_face*mesh->east_nx[cell_number]*mesh->east_length[cell_number];
		
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[left_cell];
		sum=sum+p_face*mesh->west_nx[cell_number]*mesh->west_length[cell_number];
	
	}
	return sum/mesh->cell_volumes[cell_number];
	} else {
	double sum=0.0;
	int vertical_cell=mesh->num_points_vertical-1;
	if (north) {
		int top_cell=cell_number+1;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[top_cell];
		sum=sum+p_face*mesh->north_ny[cell_number]*mesh->north_length[cell_number];
		

		
	} 

	if (south) {
		int bottom_cell=cell_number-1;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[bottom_cell];
		sum=sum+p_face*mesh->south_ny[cell_number]*mesh->south_length[cell_number];

		
	}

	if (east) {
		int right_cell=cell_number+vertical_cell;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[right_cell];
		sum=sum+p_face*mesh->east_ny[cell_number]*mesh->east_length[cell_number];
		
	}

	if (west) {
		int left_cell=cell_number-vertical_cell;
		double p_face=0.5*field->p_prime_new[cell_number]+0.5*field->p_prime_old[left_cell];
		sum=sum+p_face*mesh->west_ny[cell_number]*mesh->west_length[cell_number];
	
	}


	return sum/mesh->cell_volumes[cell_number];

	}
	
}



void update_field(struct mesh* mesh, struct field* field) {
    int vertical_cell = mesh->num_points_vertical - 1;
	
    #pragma omp parallel for schedule(static)
    for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
        int i_cell = cell_number / vertical_cell;
        int j_cell = cell_number % vertical_cell;
        
        double grad_x = 0.0;
        double grad_y = 0.0;

        // interior bottom wall
        if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == 0) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, false, true, true, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->south_nx[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(true, false, true, true, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->south_ny[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // interior top wall
        } else if (i_cell > 0 && i_cell < mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(false, true, true, true, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->north_nx[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(false, true, true, true, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->north_ny[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);

        // interior inlet
        } else if (i_cell == 0 && j_cell > 0 && j_cell < vertical_cell - 1) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, true, true, false, cell_number, mesh, field, true);
            grad_y = pressure_prime_grad(true, true, true, false, cell_number, mesh, field, false);
            
			
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // interior outlet
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell > 0 && j_cell < vertical_cell - 1) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, true, false, true, cell_number, mesh, field, true);
            grad_y = pressure_prime_grad(true, true, false, true, cell_number, mesh, field, false);
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // bl 
        } else if (i_cell == 0 && j_cell == 0) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, false, true, false, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->south_nx[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(true, false, true, false, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->south_ny[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // tl 
        } else if (i_cell == 0 && j_cell == vertical_cell - 1) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(false, true, true, false, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->north_nx[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(false, true, true, false, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->north_ny[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // br 
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == 0) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, false, false, true, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->south_nx[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(true, false, false, true, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->south_ny[cell_number] * mesh->south_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // tr 
        } else if (i_cell == mesh->num_points_horizontal - 2 && j_cell == vertical_cell - 1) {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(false, true, false, true, cell_number, mesh, field, true) 
                   + (field->p_prime_new[cell_number] * mesh->north_nx[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            grad_y = pressure_prime_grad(false, true, false, true, cell_number, mesh, field, false) 
                   + (field->p_prime_new[cell_number] * mesh->north_ny[cell_number] * mesh->north_length[cell_number]) / mesh->cell_volumes[cell_number];
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
            
        // pure interior
        } else {
            
            field->p_stat_new[cell_number] = field->p_stat_old[cell_number] + relaxation_factor * field->p_prime_new[cell_number];
            
            grad_x = pressure_prime_grad(true, true, true, true, cell_number, mesh, field, true);
            grad_y = pressure_prime_grad(true, true, true, true, cell_number, mesh, field, false);
            
            field->vel_u_new[cell_number] = field->vel_u_new[cell_number] - (field->df_x[cell_number] * grad_x);
            field->vel_v_new[cell_number] = field->vel_v_new[cell_number] - (field->df_y[cell_number] * grad_y);
        }
    }
}

// ideal inlet velocity implied by the total pressure condition, used as the reference
// scale so both residuals are dimensionless and comparable between iterations
double reference_velocity(struct field* field) {
	return sqrt(fmax(2.0 * (field->initial_ptotal - field->initial_pstat), 0.0) / rho);
}

void compute_mass_residuals(struct mesh* mesh, struct field* field) {
	double sum=0.0;
	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		sum=sum+fabs(field->flux_east[cell_number]+field->flux_west[cell_number]+field->flux_north[cell_number]+field->flux_south[cell_number]);
	}
	double mass_flow_ref = fmax(rho * reference_velocity(field) * mesh->vertical_height1, 1.0e-12);
	field->residual_mass=sum/mass_flow_ref;
}

// RMS rather than max-norm so a couple of stubborn cells cannot stall the convergence test
void compute_momentum_residuals(struct mesh* mesh, struct field* field){
	double sum_sq_u=0.0;
	double sum_sq_v=0.0;
	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		double change_u=field->vel_u_new[cell_number]-field->vel_u_old[cell_number];
		double change_v=field->vel_v_new[cell_number]-field->vel_v_old[cell_number];
		sum_sq_u=sum_sq_u+change_u*change_u;
		sum_sq_v=sum_sq_v+change_v*change_v;
	}
	int num_cells = mesh->total_cell_centers > 0 ? mesh->total_cell_centers : 1;
	double velocity_ref = fmax(reference_velocity(field), 1.0e-12);
	field->residual_u=sqrt(sum_sq_u/num_cells)/velocity_ref;
	field->residual_v=sqrt(sum_sq_v/num_cells)/velocity_ref;
}

void make_old_new_pressure_prime(struct mesh* mesh, struct field* field){
	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		field->p_prime_old[cell_number]=field->p_prime_new[cell_number];
	}
}

void make_old_new(struct mesh* mesh, struct field* field){
	for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
		field->vel_u_old[cell_number]=field->vel_u_new[cell_number];
		field->vel_v_old[cell_number]=field->vel_v_new[cell_number];
		field->p_stat_old[cell_number]=field->p_stat_new[cell_number];

	}
}

// Applies the pressure correction to every face of every cell. The coefficients here must be
// identical to the ones assembled in determine_pressure_correction, otherwise the corrected
// fluxes do not satisfy the continuity equation that p_prime was solved from.
// Each interior face is corrected from both of its owners, which keeps flux_east[P] and
// flux_west[E] exact negatives of one another.
void update_mass_fluxes(struct mesh* mesh, struct field* field) {
    int vertical_cell = mesh->num_points_vertical - 1;
    int horizontal_cell = mesh->num_points_horizontal - 1;

    #pragma omp parallel for schedule(static)
    for (int cell_number = 0; cell_number < mesh->total_cell_centers; cell_number++) {
        int i_cell = cell_number / vertical_cell;
        int j_cell = cell_number % vertical_cell;

        double p_prime_p = field->p_prime_new[cell_number];

        // east face: interior neighbour, or the static pressure outlet where p_prime is zero
        if (i_cell < horizontal_cell - 1) {
            int right_cell = cell_number + vertical_cell;
            double a = rho * compute_df(cell_number, right_cell, field, mesh, "east") * mesh->east_length[cell_number]
                     / distance_between_cells(cell_number, right_cell, mesh);
            field->flux_east[cell_number] += a * (p_prime_p - field->p_prime_new[right_cell]);
        } else {
            double df_east = field->df_x[cell_number] * pow(mesh->east_nx[cell_number], 2)
                           + field->df_y[cell_number] * pow(mesh->east_ny[cell_number], 2);
            double a = rho * df_east * mesh->east_length[cell_number] / face_normal_distance(cell_number, mesh, 'e');
            field->flux_east[cell_number] += a * (p_prime_p - 0.0);
        }

        // west face: interior neighbour, or the total pressure inlet where p_prime is zero
        if (i_cell > 0) {
            int left_cell = cell_number - vertical_cell;
            double b = rho * compute_df(cell_number, left_cell, field, mesh, "west") * mesh->west_length[cell_number]
                     / distance_between_cells(cell_number, left_cell, mesh);
            field->flux_west[cell_number] += b * (p_prime_p - field->p_prime_new[left_cell]);
        } else {
            double df_west = field->df_x[cell_number] * pow(mesh->west_nx[cell_number], 2)
                           + field->df_y[cell_number] * pow(mesh->west_ny[cell_number], 2);
            double b = rho * df_west * mesh->west_length[cell_number] / face_normal_distance(cell_number, mesh, 'w');
            field->flux_west[cell_number] += b * (p_prime_p - 0.0);
        }

        // the top and bottom boundaries are solid walls, so they carry no mass and take no correction
        if (j_cell < vertical_cell - 1) {
            int top_cell = cell_number + 1;
            double c = rho * compute_df(cell_number, top_cell, field, mesh, "north") * mesh->north_length[cell_number]
                     / distance_between_cells(cell_number, top_cell, mesh);
            field->flux_north[cell_number] += c * (p_prime_p - field->p_prime_new[top_cell]);
        }

        if (j_cell > 0) {
            int bottom_cell = cell_number - 1;
            double d = rho * compute_df(cell_number, bottom_cell, field, mesh, "south") * mesh->south_length[cell_number]
                     / distance_between_cells(cell_number, bottom_cell, mesh);
            field->flux_south[cell_number] += d * (p_prime_p - field->p_prime_new[bottom_cell]);
        }
    }
}



void export_field_to_csv(struct field* field, struct mesh* mesh, const char* filename, int iter) {
	char new_file[256];
	snprintf(new_file, sizeof(new_file), "%s-%d.csv", filename, iter);
	FILE* file = fopen(new_file, "w");

	if (file == NULL) {
		printf("Error: Could not open file %s for writing.\n", new_file);
		return;
	}

	fprintf(file, "x,y,cell_number,p_stat, vel_u, vel_v\n");

	for (int i = 0; i < mesh->total_cell_centers; i++) {
		fprintf(file, "%f,%f,%d,%f,%f,%f\n", mesh->cell_centers_x[i], mesh->cell_centers_y[i], i,field->p_stat_new[i], field->vel_u_new[i], field->vel_v_new[i]);
	}

	fclose(file);
	printf("Successfully exported field to %s\n", new_file);
}

