#pragma once

struct mesh ;

struct field {
	double* vel_u_new;
	double* vel_v_new;
	double* p_stat_new;

	double* vel_u_old;
	double* vel_v_old;
	double* p_stat_old;	

	double* flux_east; 
	double* flux_west;
	double* flux_north; // this mass flux is what is used in the convective flux and recomputed by rhie and chow
	double* flux_south;

	double* diffusive_dudx; // used to compute gradient in diffusive flux, different from mass flux defined above
	double* diffusive_dudy; // du fluxes used in x momentum and dv fluxes used in y-momentum
	double* diffusive_dvdx;
	double* diffusive_dvdy;


	double* df_x; // this is named incorrectly. These are not the face values 
					//	but rather the values of the cell centroids. Face values use the interpolated values of these
	double* df_y;

	
	double* p_prime_old;
	double* p_prime_new;
	

	double initial_pstat;
	double initial_ptotal;

	double residual_mass;
	double residual_u;
	double residual_v;
};


#define relaxation_u 0.7
// inner Jacobi sweeps must not be under-relaxed; under-relaxation belongs on the pressure update only
#define relaxation_prime 1.0
#define relaxation_factor 0.3

void initialise_field(struct field* field, struct mesh* mesh, double initial_u, double initial_v, double initial_pstat, struct flow_boundary* inlet);
void export_field_to_csv(struct field* field, struct mesh* mesh, const char* filename, int iter);
void diffusive_gradients(struct field* field, struct mesh* mesh, struct flow_boundary* inlet);
void solve_x_momentum(struct field* field, struct mesh* mesh, struct flow_boundary* inlet);
void solve_y_momentum(struct field* field, struct mesh* mesh, struct flow_boundary* inlet);
void rhie_chow_flux(struct mesh* mesh, struct field* field);
void compute_mass_residuals(struct mesh* mesh, struct field* field);
void compute_momentum_residuals(struct mesh* mesh, struct field* field);
void update_field(struct mesh* mesh, struct field* field);
void set_pressure_zero (struct mesh* mesh, struct field* field);
void determine_pressure_correction(struct mesh* mesh, struct field* field);
void make_old_new_pressure_prime(struct mesh* mesh, struct field* field);
void make_old_new(struct mesh* mesh, struct field* field);
void update_mass_fluxes(struct mesh* mesh, struct field* field);