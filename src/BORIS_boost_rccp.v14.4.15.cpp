/* C++ libraries to include */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include <Rcpp.h>

/* Boost includes */
// [[Rcpp::depends(BH)]]
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>

using namespace std;
using namespace Rcpp;

/*----------------------------*/
/*- header -------------------*/
/*----------------------------*/
#define PATH1 "./inputs/"
#define PATH2 "./outputs/"      /*output folder*/
//if debugging, use explicit paths
//#define PATH1 "C:/Data/Visual Studio/Projects/lau.seir.infer.boost/x64/Debug/inputs/"
//#define PATH2 "C:/Data/Visual Studio/Projects/lau.seir.infer.boost/x64/Debug/outputs/"


//set upper limits for to avoid memory allocation issues
#define NLIMIT 500
#define SEQLLIMIT 50000
#define NSEQLIMIT 5
#define MOVELIMIT 10000

using namespace std;
//using namespace boost;
//using namespace Rcpp;


typedef  vector < vector<int> > vec2int;
typedef  vector < vector<double> > vec2d;

// definition of rng for seeding later in main
typedef boost::mt19937 rng_type;
extern rng_type rng;

// type definitions for probability distributions
typedef boost::uniform_real<double> Dunif;
typedef boost::uniform_int<int> Dunif_int;
typedef boost::gamma_distribution<double> Dgamma;
typedef boost::random::weibull_distribution<double> Dweibull;
typedef boost::bernoulli_distribution<double> Dbern;
typedef boost::normal_distribution<double> Dnorm;
//typedef boost::exponential_distribution<double> Dexp;
typedef boost::math::gamma_distribution<double> gamma_mdist; //gamma_mdist(shape, scale)
typedef boost::math::exponential_distribution<double> exp_mdist; //exp_mdist(rate)
typedef boost::math::weibull_distribution<double> weibull_mdist; //weibull_mdist(shape, scale)


// - external parameters ----------------------------------------------------------------------------------------------
extern int debug; //print full debug outputs
extern int opt_latgamma; //1 = latent period from gamma distribution (0 = Lau original, Gaussian)
extern int opt_k80; //1 = reformulated K80 to match paper (0 = Lau original, secondary reference)
extern int opt_betaij; //1 = farm-level covariates incorporated into betaij (0 = Lau original), 2 = normalised
extern int opt_ti_update; //1 = update t_i (0 = Lau original)
extern int opt_mov; //1 = contact-tracing incorporated into betaij (0 = Lau original), 2 = contact-tracing, time-independant

//other required externals, dynamic version, requires declaration of extern here, definition in IO.cpp after IO_parapriorsetc function and then used in inference.cpp
extern int ind_n_base_part;// used in IO_simpara
extern int n_base_part; // used in IO_simpara

// -Structure declarations ----------------------------------------------------------------------------------------------
struct nt_struct {

	vec2int nt; // the sequence corresponds to the  infected-or-infecting event OR sampling of a farm
	vec2d t_nt; // the time corresponds to the sequence above
	vector<int> current_size; // the number of sequences recorded

	vector<double> t_sample; // sampling time

	vec2int infecting_list; // each elements contains the list of subjects being infected by the corresponding source (sorted in order of infection)
	vector<int> infecting_size; // number of subjects infected by the corresponding source (if sampeled, it would be equal to current_size -2 as also excluding the first record corresponds to own infection)

	//constructor to assign size on heap, limited only by virtual memory not RAM
	nt_struct() : nt(NLIMIT, vector<int>(NSEQLIMIT*SEQLLIMIT)), t_nt(NLIMIT, vector<double>(NSEQLIMIT)), current_size(NLIMIT), t_sample(NLIMIT), infecting_list(NLIMIT), infecting_size(NLIMIT) {};
};

//----------------
struct para_key {
	double alpha, beta, a, b, lat_mu, lat_var, c, d, k_1; //key model parameters (a,b are redundant)
	double mu_1, mu_2; // mutation rates
	double p_ber; // varation parameter p (for the master sequence)
	double rho_susc1, rho_susc2; // farm type susceptibility indicator variable(reference i0=cattle, i1 = pigs, i2 = sheep and smallholder)
	double phi_inf1, phi_inf2; // farm type infectivity indicator variable(reference i0=cattle, i1 = pigs, i2 = sheep and smallholder)
	double tau_susc, nu_inf; // farm type susceptibility and infectivity based on herd size
	double beta_m; // the effect on likelihood for each traced movement between a pair of infected premises
};


struct para_key_init {
	double alpha, beta, a, b, lat_mu, lat_var, c, d, k_1;
	double mu_1, mu_2;
	double p_ber;
	double rho_susc1, rho_susc2;
	double phi_inf1, phi_inf2;
	double tau_susc, nu_inf;
	double beta_m;
};

struct para_priors_etc {
	double t_range; // for update of t_i; t_i would range between [t_o - t_range, t_o + range] (see paper SI)
	double t_back; // for update of t_e; 
						//t_up = min(t_sample_arg.at(subject_proposed), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
						//t_low = max(0.0, t_up - para_priors_arg.t_back);
	double t_bound_hi; // t_bound used when proposing initial sources, t_i of infectee must be within t_bound_hi days of proposed t_i of j (essentially the upper bound of the generation interval). For FMD: mean = 6.1; sd=4.6 (Haydon, 2003).
	double rate_exp_prior;// the rate of exp to be used as vague prior
	int ind_n_base_part;// =1 if the seq data is partial
	int n_base_part; // the partial length used if ind_n_base_part =1

	/* lower/upper bounds for parameters */
	double alpha_hi, beta_hi, lat_mu_hi, lat_var_lo, lat_var_hi, c_hi, d_hi;
	double k_1_lo, k_1_hi, mu_1_hi, mu_2_hi, p_ber_hi;
	double rho_susc1_hi, rho_susc2_hi, phi_inf1_hi, phi_inf2_hi;
	double tau_susc_lo, tau_susc_hi, nu_inf_lo, nu_inf_hi;
	double beta_m_hi;
	double trace_window; // twice the maximum incubation period
};

struct para_scaling_factors {
	double alpha_sf, beta_sf, a_sf, b_sf, lat_mu_sf, lat_var_sf, c_sf, d_sf, k_1_sf; //key model parameters (a,b are redundant)
	double mu_1_sf, mu_2_sf; // mutation rates
	double p_ber_sf; // varation parameter p (for the master sequence)
	double rho_susc1_sf, rho_susc2_sf; //  farm type susceptibility indicator variable
	double phi_inf1_sf, phi_inf2_sf; //  farm type infectivity indicator variable
	double tau_susc_sf, nu_inf_sf; //  herd size
	double beta_m_sf; //  probability of infection given contact-traced movement
};

//----------------

struct para_aux { // other model quantities to be fixed/known
	int n, n_seq, n_base; //n, population size (number of farms/sites); n_seq, max number of sequences for a farm (to reserve the capacity of the vector storing sequence data); n_base, number of bases for a sequence
	string kernel_type; // type of spatial kernel used e.g 'exponential/'power_law'
	string coord_type; // type of spatial coordinates ('longlat' or 'cartesian')
	double t_max, unassigned_time; // t_max is the upper limit of observation period; unassigned_time is an abritary extreme value e.g. 9e+10 , to indicate that an event does not happen e.g. no infection
	int np; //number of processes for >1 for cluster computing
	double n_iterations; //number of iterations
	int n_frequ, n_output_source, n_output_gm, n_cout; //frequency of updating infection times, outputing updated sources and console output
	int opt_latgamma; //1 = latent period from gamma distribution (0 = Lau original, Gaussian)
	int opt_k80; //1 = reformulated K80 to match paper (0 = Lau original, secondary reference)
	int opt_betaij; //1 = farm-level covariates incorporated into beta (0 = Lau original)
	int opt_ti_update; //1 = update t_i (0 = Lau original)
	int opt_mov; //1 = contact-tracing incorporated into betaij (0 = Lau original)
	int debug; //1 = print debug outputs and checksums to console
};


//-----------------

struct epi_struct {
	vector<int> k;
	vector<double> q, t_e, t_i, t_r;
	vector<int> status;
	vector<double> coor_x, coor_y;
	vector<double> ftype0; //indicator variable for predominantly cattle farms [reference]
	vector<double> ftype1; //indicator variable for pig farms (not cattle [reference] and not sheep, etc)
	vector<double> ftype2; //indicator variable for sheep and small-holder farms (not cattle [reference] and not pigs)
	vector<double> herdn; //number of animals in each herd
	vector<int> infected_source;

	//constructor to assign size on heap, limited only by virtual memory not RAM
	epi_struct() : k(NLIMIT), q(NLIMIT), t_e(NLIMIT), t_i(NLIMIT), t_r(NLIMIT), status(NLIMIT),
		coor_x(NLIMIT), coor_y(NLIMIT), ftype0(NLIMIT), ftype1(NLIMIT), ftype2(NLIMIT),
		herdn(NLIMIT), infected_source(NLIMIT) {};

};

struct moves_struct { // structure to hold edgelist (contact-traced movement data)

	vector<int> from_k; // source of a contact-traced movement
	vector<int> to_k; // destination of a contact-traced movement
	vector<int> t_m; // day of the movement, from t0 (on same time scale as t_e, etc.)
					  //constructor to assign size on heap, limited only by virtual memory not RAM
	moves_struct() : from_k(MOVELIMIT), to_k(MOVELIMIT), t_m(MOVELIMIT) {}; //limited to 10000 movements (for now)
};

//------------------

struct lh_SQUARE { // the contribution to likelihood function from each individual is subdivided into f_U, f_E, f_I,f_R
	vector<long double> f_U, q_T, kt_sum_U;
	vector<long double> f_E, g_E, k_sum_E, h_E, q_E, kt_sum_E;
	vector<long double> f_I, f_EnI;
	vector<long double> f_R, f_InR;
	vector<long double> log_f_S, log_f_Snull;
	vector<double> movest_sum_U, movest_sum_E, moves_sum_E;
	//constructor to assign size on heap, limited only by virtual memory not RAM
	lh_SQUARE() : f_U(NLIMIT), q_T(NLIMIT), kt_sum_U(NLIMIT), f_E(NLIMIT), g_E(NLIMIT), k_sum_E(NLIMIT),
		h_E(NLIMIT), q_E(NLIMIT), kt_sum_E(NLIMIT), f_I(NLIMIT), f_EnI(NLIMIT), f_R(NLIMIT),
		f_InR(NLIMIT), log_f_S(NLIMIT), log_f_Snull(NLIMIT),
		movest_sum_U(NLIMIT), movest_sum_E(NLIMIT), moves_sum_E(NLIMIT) {};
};


//---Function classes -----------------------

/*
function class
private: available within all functions of this class without including in arguments
public: declaring the parameters
*/

class FUNC {

private:

	double alpha_Clh;
	double beta_Clh;
	double lat_mu_Clh;
	double lat_var_Clh;
	double c_Clh;
	double d_Clh;
	double k_1_Clh;
	double mu_1_Clh;
	double mu_2_Clh;

	double p_ber_Clh;

	double rho_susc1_Clh, rho_susc2_Clh; //unknown parameters for indicator variable effects on infectivity of farm type=1 and =2 (compared to reference =0)
	double phi_inf1_Clh, phi_inf2_Clh; //unknown parameters for indicator variable effects on susceptibility of farm type=1 and =2 (compared to reference =0)

	double tau_susc_Clh, nu_inf_Clh; //unknown parameters for infectivity and susceptibility based on herd size

	double beta_m_Clh; //unknown parameter for contact-traced movt related infection

	int n_Clh, n_seq_Clh, n_base_Clh;
	string kernel_type_Clh;
	string coord_type_Clh;
	double t_max_Clh, unassigned_time_Clh;
	int np_Clh;
	double n_iterations_Clh;
	int n_frequ_Clh, n_output_source_Clh, n_output_gm_Clh, n_cout_Clh;

	vector< vector<double> > coordinate_Clh;

	vector<int> xi_U_Clh, xi_E_Clh, xi_E_minus_Clh, xi_I_Clh, xi_R_Clh, xi_EnI_Clh, xi_InR_Clh;
	vector<double> t_e_Clh, t_i_Clh, t_r_Clh;

	vector<int> index_Clh;

	vector <int> infected_source_Clh;

public:

	void set_para(para_key para_current_arg, para_aux para_other_arg, vector< vector<double> > coordinate_arg, vector<int> xi_U_arg, vector<int> xi_E_arg, vector<int> xi_E_minus_arg, vector<int> xi_I_arg, vector<int> xi_R_arg, vector<int> xi_EnI_arg, vector<int> xi_InR_arg, vector<double> t_e_arg, vector<double> t_i_arg, vector<double> t_r_arg, vector<int> index_arg, vector<int> infected_source_arg) {

		alpha_Clh = para_current_arg.alpha;
		beta_Clh = para_current_arg.beta;
		lat_mu_Clh = para_current_arg.lat_mu;
		lat_var_Clh = para_current_arg.lat_var;
		c_Clh = para_current_arg.c;
		d_Clh = para_current_arg.d;
		k_1_Clh = para_current_arg.k_1;
		mu_1_Clh = para_current_arg.mu_1;
		mu_2_Clh = para_current_arg.mu_2;

		p_ber_Clh = para_current_arg.p_ber;

		rho_susc1_Clh = para_current_arg.rho_susc1;
		rho_susc2_Clh = para_current_arg.rho_susc2;
		phi_inf1_Clh = para_current_arg.phi_inf1;
		phi_inf2_Clh = para_current_arg.phi_inf2;

		tau_susc_Clh = para_current_arg.tau_susc;
		nu_inf_Clh = para_current_arg.nu_inf;

		beta_m_Clh = para_current_arg.beta_m;

		n_Clh = para_other_arg.n;
		kernel_type_Clh = para_other_arg.kernel_type;
		coord_type_Clh = para_other_arg.coord_type;
		t_max_Clh = para_other_arg.t_max;
		unassigned_time_Clh = para_other_arg.unassigned_time;
		np_Clh = para_other_arg.np;

		n_base_Clh = para_other_arg.n_base;
		n_seq_Clh = para_other_arg.n_seq;
		n_iterations_Clh = para_other_arg.n_iterations;
		n_frequ_Clh = para_other_arg.n_frequ;
		n_output_source_Clh = para_other_arg.n_output_source;
		n_output_gm_Clh = para_other_arg.n_output_gm;
		n_cout_Clh = para_other_arg.n_cout;

		coordinate_Clh = coordinate_arg;

		xi_U_Clh = xi_U_arg;
		xi_E_Clh = xi_E_arg;
		xi_E_minus_Clh = xi_E_minus_arg;
		xi_I_Clh = xi_I_arg;
		xi_R_Clh = xi_R_arg;
		xi_EnI_Clh = xi_EnI_arg;
		xi_InR_Clh = xi_InR_arg;


		t_e_Clh = t_e_arg;
		t_i_Clh = t_i_arg;
		t_r_Clh = t_r_arg;

		index_Clh = index_arg;

		infected_source_Clh = infected_source_arg;

	}


	void initialize_kernel_mat(vector< vector<double> >&, vector<double>&); // function prototype for initializing kernel distance  

	void initialize_delta_mat(vector< vector<double> >&); // function prototype for initializing length of exposure time

	void initialize_delta_mat_mov(vector< vector<double> >&, moves_struct& moves_arg); // function prototype for initializing length of exposure time for movements

	void initialize_beta_ij_mat(vector< vector<double> >& beta_ij_mat_arg, vector<double>& herdn_arg, vector<double>& ftype0_arg, vector<double>& ftype1_arg, vector<double>& ftype2_arg);

	void initialize_lh_square(lh_SQUARE& lh_square_arg, vector< vector<double> > kernel_mat_arg, vector< vector<double> > delta_mat_arg, vector<double>& norm_const_arg, nt_struct& nt_data_arg, vector<int>& con_seq, vector< vector<double> > beta_ij_mat_arg, moves_struct& moves_arg, para_priors_etc& para_priors_arg, vector< vector<double> > delta_mat_mov_arg);

};

//-----------------------------

/*
function mcmc update class
private: don’t declare variables that need updating here or they can’t be changed in the mcmc only for constants
*/

class mcmc_UPDATE {

private:

	int n_CUPDATE, n_seq_CUPDATE, n_base_CUPDATE;
	string kernel_type_CUPDATE, coord_type_CUPDATE;
	double t_max_CUPDATE, unassigned_time_CUPDATE;
	int np_CUPDATE;
	double n_iterations_CUPDATE;
	int n_frequ_CUPDATE, n_output_source_CUPDATE, n_output_gm_CUPDATE, n_cout_CUPDATE;

	vector<double> herdn_CUPDATE, ftype0_CUPDATE, ftype1_CUPDATE, ftype2_CUPDATE;

	vector< vector<double> > coordinate_CUPDATE;

	vector<int> index_CUPDATE;

public:

	void set_para(para_aux para_other_arg, vector< vector<double> > coordinate_arg, epi_struct epi_final_arg) {

		n_CUPDATE = para_other_arg.n;
		kernel_type_CUPDATE = para_other_arg.kernel_type;
		coord_type_CUPDATE = para_other_arg.coord_type;
		t_max_CUPDATE = para_other_arg.t_max;
		unassigned_time_CUPDATE = para_other_arg.unassigned_time;
		np_CUPDATE = para_other_arg.np;

		n_base_CUPDATE = para_other_arg.n_base;
		n_seq_CUPDATE = para_other_arg.n_seq;
		n_iterations_CUPDATE = para_other_arg.n_iterations;
		n_frequ_CUPDATE = para_other_arg.n_frequ;
		n_output_source_CUPDATE = para_other_arg.n_output_source;
		n_output_gm_CUPDATE = para_other_arg.n_output_gm;
		n_cout_CUPDATE = para_other_arg.n_cout;

		herdn_CUPDATE = epi_final_arg.herdn;
		ftype0_CUPDATE = epi_final_arg.ftype0;
		ftype1_CUPDATE = epi_final_arg.ftype1;
		ftype2_CUPDATE = epi_final_arg.ftype2;


		coordinate_CUPDATE = coordinate_arg;

		n_base_CUPDATE = para_other_arg.n_base;
		n_seq_CUPDATE = para_other_arg.n_seq;

	}

public:

	void alpha_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_e_arg, const vector<double>& t_i_arg, const vector<double>& t_r_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg);
	void beta_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<double>& t_e_arg, const vector<double>& t_i_arg, const vector<double>& t_r_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg);

	void c_update(lh_SQUARE&, double&, const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&, const vector<int>&, para_key&, para_priors_etc&, para_scaling_factors&, int, rng_type & rng_arg);
	void d_update(lh_SQUARE&, double&, const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&, const vector<int>&, para_key&, para_priors_etc&, para_scaling_factors&, int, rng_type & rng_arg);

	void k_1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);

	void tau_susc_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);
	void nu_inf_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);
	void rho_susc1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);
	void rho_susc2_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);
	void phi_inf1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);
	void phi_inf2_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg);

	void beta_m_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg);

	void mu_1_update(lh_SQUARE&, double&, const vector<int>&, para_key&, nt_struct&, para_priors_etc&, para_scaling_factors&, int iter, rng_type & rng_arg);
	void mu_2_update(lh_SQUARE&, double&, const vector<int>&, para_key&, nt_struct&, para_priors_etc&, para_scaling_factors&, int iter, rng_type & rng_arg);

	void p_ber_update(lh_SQUARE&, double&, const vector<int>&, para_key&, nt_struct&, vector<int>&, vector<int>&, para_priors_etc&, para_scaling_factors&, int iter, rng_type & rng_arg);


	void lat_mu_update(lh_SQUARE&, double&, const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&, const vector<int>&, para_key&, para_priors_etc&, para_scaling_factors&, int, rng_type & rng_arg);
	void lat_var_update(lh_SQUARE&, double&, const vector<int>&, const vector<int>&, const vector<double>&, const vector<double>&, const vector<int>&, para_key&, para_priors_etc&, para_scaling_factors&, int, rng_type & rng_arg);

	void t_e_seq(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg, vec2d& t_nt_current_arg, vec2int& infecting_list_current_arg, const vector<int>& infecting_size_current_arg, vector<int>&  xi_beta_E_arg, vector<int>& con_seq, int& subject_proposed, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg, moves_struct& mov_arg); // jointly update the infection time and sequences

	//void source_update(lh_SQUARE & lh_square_current_arg, double & log_lh_current_arg, const vector<vector<double>>& kernel_mat_current_arg, vector<vector<double>>& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key & para_current_arg, const vector<double>& norm_const_current_arg, vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, vector<int>& current_size_arg, vec2int & nt_current_arg, vec2d & t_nt_current_arg, vec2int & infecting_list_current_arg, vector<int>& infecting_size_current_arg, vector<int>& xi_beta_E_arg, int & subject_proposed, vector<int>& list_update, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, para_priors_etc& para_priors_arg, int iter, vector< vector<double> >& delta_mat_mov_current_arg);

	void source_t_e_update(lh_SQUARE & lh_square_current_arg, double & log_lh_current_arg, const vector<vector<double>>& kernel_mat_current_arg, vector<vector<double>>& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key & para_current_arg, const vector<double>& norm_const_current_arg, vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, vector<int>& current_size_arg, vec2int & nt_current_arg, vec2d & t_nt_current_arg, vec2int & infecting_list_current_arg, vector<int>& infecting_size_current_arg, vector<int>& xi_beta_E_arg, int & subject_proposed, vector<int>& list_update, vector<int>& con_seq, para_priors_etc & para_priors_arg, para_scaling_factors & para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg);

	//void index_first_seq(lh_SQUARE&, double&, const vector< vector<double> >&, vector< vector<double> >&, vector<int>&, vector<int>&, vector<int>& , const vector<int>&, vector<int>&, const vector<double>&, const vector<double>&, vector<double>& , vector<int>&, const para_key&,  const vector<double>& , const vector<double>&, const vector<int>&, const vector<double>&, const vector<int>& , vec2int&, vec2d&,  vec2int&, const vector<int>&, vector<int>&, vector<int>&,int&, int,gsl_rng* &);


	void t_i_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<int>& xi_EnI_arg, const vector<int>& xi_R_arg, const vector<int>& xi_InR_arg, const vector<double>& t_r_arg, vector<double>& t_i_arg, const vector<double>& t_onset_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg, vec2d& t_nt_current_arg, vec2int& infecting_list_current_arg, const vector<int>& infecting_size_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg, moves_struct& mov_arg);
	void seq_n_update(lh_SQUARE & lh_square_current_arg, double & log_lh_current_arg, const vector<vector<double>>& kernel_mat_current_arg, vector<vector<double>>& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key & para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int & nt_current_arg, vec2d & t_nt_current_arg, vector<int>& xi_beta_E_arg, const int & subject_proposed, rng_type & rng_arg); // update the whole seq
	void index_first_seq(lh_SQUARE & lh_square_current_arg, double & log_lh_current_arg, const vector<vector<double>>& kernel_mat_current_arg, vector<vector<double>>& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key & para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int & nt_current_arg, vec2d & t_nt_current_arg, vec2int & infecting_list_current_arg, const vector<int>& infecting_size_current_arg, vector<int>& xi_beta_E_arg, vector<int>& con_seq, int & subject_proposed, int iter, rng_type & rng_arg);

	void con_seq_update(lh_SQUARE&, double&, const vector< vector<double> >&, vector< vector<double> >&, vector<int>&, vector<int>&, vector<int>&, const vector<int>&, vector<int>&, const vector<double>&, const vector<double>&, vector<double>&, vector<int>&, const para_key&, const vector<double>&, const vector<int>&, const vector<double>&, const vector<int>&, vec2int&, vec2d&, vector<int>&, vector<int>&, para_priors_etc&, para_scaling_factors&, int, rng_type & rng_arg); //

	void seq_update(lh_SQUARE & lh_square_current_arg, double & log_lh_current_arg, const vector<vector<double>>& kernel_mat_current_arg, vector<vector<double>>& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key & para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int & nt_current_arg, vec2d & t_nt_current_arg, vector<int>& xi_beta_E_arg, vector<int>& con_seq, const int & subject_proposed, para_priors_etc & para_priors_arg, para_scaling_factors & para_sf_arg, int iter, rng_type & rng_arg);

	//

};

//--Function prototype declarations ---------------------------

void count_type_all(nt_struct &, vector<int>&, int&, int&, int&, int&); // count number of unchanged, transition, transversion (whole dataset)

void count_type_seq(vector<int>&, vector<int>, int&, int&, int&, int&);

//-----------------------------
void IO_parapriorsetc(para_priors_etc&);
void IO_parakeyinit(para_key_init&);
void IO_parascalingfactors(para_scaling_factors&);

void IO_para_aux(para_aux&);
void IO_data(para_aux para_other_arg, vector < vector<double> >& coordinate_arg, epi_struct& epi_final_arg, nt_struct& nt_data_arg, vector<int>& index_arg, vector<int>& con_seq, vector<int>& seed_arg, moves_struct& moves_arg);

void initialize_mcmc(para_key_init& para_init, para_key& para_current, para_aux& para_other, para_priors_etc& para_priors_etc, vector<int>& xi_I_current, vector<int>& xi_U_current, vector<int>& xi_E_current, vector<int>& xi_E_minus_current, vector<int>& xi_R_current, vector<int>& xi_EnI_current, vector<int>& xi_EnIS_current, vector<int>& xi_InR_current, vector<double>& t_e_current, vector<double>& t_i_current, vector<double>& t_r_current, vector<int>& index_current, vector<int>& infected_source_current, vector < vector<double> >& kernel_mat_current, vector <double>& norm_const_current, vec2int& sample_data, vector<double>& t_onset, nt_struct& nt_data_current, vector<int>& con_seq, vector < vector<double> >& beta_ij_mat_current);

void seq_initialize_pair(nt_struct&, int, int, double, int, para_key&); // update sequences of infectious-infected pair

void delete_seq_samples(para_key&, para_aux&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<int>&, vector<double>&, vector<double>&, vector<double>&, vector<int>&, vector<double>&, vector<int>&, vector<int>&, vector < vector<double> >&, vector <double>&, vec2int&, vector<double>&, nt_struct&);

double roundx(double x, int y);

//----------------------------
double log_lh_func(lh_SQUARE, int);
inline void seq_propose_tri(vector<int>&, double&, const vector<int>&, const vector<int>&, const vector<int>&, const double&, const double&, const double&, const double&, const double&, const double&, const  int&, rng_type & rng_arg);
inline void seq_backward_pr_tri(const vector<int>&, double&, const vector<int>&, const vector<int>&, const vector<int>&, const double&, const double&, const double&, const double&, const double&, const double&, const int&);
//--------------------------
double log_lh_base(int&, int&, double, double, double, double);
double log_lh_seq(vector<int>&, vector<int>&, double, double, double, double, int);
//--------------------------
double runif(double x0, double x1, rng_type& rng_arg);
int runif_int(int x0, int x1, rng_type& rng_arg);
double rgamma(double shape, double scale, rng_type& rng_arg);
double rweibull(double, double, rng_type& rng_arg);
int rbern(double p, rng_type& rng_arg);
double rnorm(double mean, double sd, rng_type& rng_arg);
//double rexp(double rate, rng_type& rng_arg);
int edf_sample(vector<double> vec, rng_type& rng_arg);
//--------------------------


//--------------------------
/*- SIM header------------*/
//--------------------------
#define EPSILON 0.0000001   // Define the tolerance when comparing doubles
#define DOUBLE_EQ(x,v) (((v - EPSILON) < x) && (x <( v + EPSILON)))

typedef  vector < vector<int> > vec2int;
typedef  vector < vector<double> > vec2d;

// definition of rng for seeding later in main
typedef boost::mt19937 rng_type;
extern rng_type rng_sim;
// type definitions for probability distributionsrng_sim
typedef boost::uniform_real<double> Dunif;
typedef boost::uniform_int<int> Dunif_int;
typedef boost::gamma_distribution<double> Dgamma;
typedef boost::random::weibull_distribution<double> Dweibull;
typedef boost::bernoulli_distribution<double> Dbern;
typedef boost::normal_distribution<double> Dnorm;
//typedef boost::exponential_distribution<double> Dexp;
typedef boost::math::gamma_distribution<double> gamma_mdist;
typedef boost::math::exponential_distribution<double> exp_mdist;


//sf edit - version control parameter
extern int debug_sim; //print full debug outputs
extern int seed_sim; //universal seed for random number generator
extern int opt_k80_sim; //1 = reformulated K80 to match paper (0 = Lau original, secondary reference)
extern int opt_betaij_sim; //1 = farm-level covariates incorporated into beta (0 = Lau original), 2 = normalised
extern int opt_mov_sim; //1 = contact-tracing incorporated into betaij (0 = Lau original)


struct para_aux_struct {
	int opt_k80, opt_betaij, debug;
	int n, seed, n_base, n_seq;
	double t_max, unassigned_time, sample_range;
	int partial_seq_out, n_base_part, n_index = 1;
	string latent_type, kernel_type, coord_type;
	int opt_mov; //1 = contact-tracing incorporated into betaij (0 = Lau original)
	int n_mov; // total number of contact movements
};


struct para_key_struct {
	double alpha, beta, a, b, c, d, k_1, mu_1, mu_2, p_ber;
	double rho_susc1, rho_susc2, phi_inf1, phi_inf2; //farm type susceptibility and infectivity parameters
	double tau_susc, nu_inf; // farm size non-linear effect on susceptibility and infectivity
	double beta_m; // the effect on likelihood for each traced movement between a pair of infected premises
};



struct epi_struct_sim {
	vector<int> k;  //index
	vector<double> coor_x, coor_y;
	vector<double> t_e, t_i, t_r; //timing of exposure, infectious onset and recovery
	vector<double> ftype0, ftype1, ftype2;  //farm type indicator variable (reference = cattle, ftype1 are pigs, ftype2 are sheep/other mixed)
	vector<double> herdn; //herdn
	vector<int> status;  //1=S, 2=E, 3=I,4=R
	vector<double> q; //Sellke thresholds (see page 21 of Lau Thesis, 2.1.2)
	vector<double> ric, t_next; //remaining infection challenge needed to get infection and time of next event (exposure/infection)
	vector<double> lat_period, inf_period; //latent period, infectious period
};

struct mov_struct { // structure to hold edgelist (contact-traced movement data)

	vector<int> from_k; // source of a contact-traced movement
	vector<int> to_k; // destination of a contact-traced movement
	vector<double> t_m; // day of the movement, from t0 (on same time scale as t_e, etc.)

};

struct nt_struct_sim {

	vec2int  nt; // the sequence corresponds to the  infected-or-infecting event OR sampling of a farm
	vec2d t_nt; // the time corresponds to the sequence above
	vector<int> current_size; // the number of sequences recorded

	vector<double> t_sample; // sampling time
	vector<double> t_last;// time of last event (including infecting; excluding sampling time)

	vector<int> ind_sample; // indicates if it has been actually sampled

	double log_f_S; // the log likelihood contributed by the sequences
	int total_count_1, total_count_2, total_count_3; // total count of unchanged, transition, transverison

};

//

class epi_functions { // a CLASS contains various functions for updating the epidemics

private:

	double alpha_Cepi, beta_Cepi, a_Cepi, b_Cepi, c_Cepi, d_Cepi; // Cepi means within CLASS epi_update
	double k_1_Cepi;
	double mu_1_Cepi, mu_2_Cepi;
	double rho_susc1_Cepi, rho_susc2_Cepi, phi_inf1_Cepi, phi_inf2_Cepi;
	double tau_susc_Cepi, nu_inf_Cepi;
	double beta_m_Cepi;

	double sample_range_Cepi;
	string kernel_type_Cepi;
	string coord_type_Cepi;

	double t_max_Cepi;
	double unassigned_time_Cepi;

	int n_Cepi;
	int n_base_Cepi;
	int n_seq_Cepi;
	int seed_Cepi;

	double p_ber_Cepi;

	vector < vector<double> > distance_mat_Cepi;
	vector < vector<double> > beta_ij_mat_Cepi;

public:

	void set_para(para_key_struct para_key_Cepi, para_aux_struct para_other_Cepi, rng_type & rng_arg, vector < vector<double> > distance_mat_arg, vector < vector<double> > beta_ij_mat_arg) { // this voild function pass parameters to the functions in this class

		alpha_Cepi = para_key_Cepi.alpha;
		beta_Cepi = para_key_Cepi.beta;
		a_Cepi = para_key_Cepi.a;
		b_Cepi = para_key_Cepi.b;
		c_Cepi = para_key_Cepi.c;
		d_Cepi = para_key_Cepi.d;
		k_1_Cepi = para_key_Cepi.k_1;
		mu_1_Cepi = para_key_Cepi.mu_1;
		mu_2_Cepi = para_key_Cepi.mu_2;
		p_ber_Cepi = para_key_Cepi.p_ber;
		rho_susc1_Cepi = para_key_Cepi.rho_susc1;
		rho_susc2_Cepi = para_key_Cepi.rho_susc2;
		phi_inf1_Cepi = para_key_Cepi.phi_inf1;
		phi_inf2_Cepi = para_key_Cepi.phi_inf2;
		tau_susc_Cepi = para_key_Cepi.tau_susc;
		nu_inf_Cepi = para_key_Cepi.nu_inf;
		beta_m_Cepi = para_key_Cepi.beta_m;

		t_max_Cepi = para_other_Cepi.t_max;
		unassigned_time_Cepi = para_other_Cepi.unassigned_time;
		sample_range_Cepi = para_other_Cepi.sample_range;


		n_Cepi = para_other_Cepi.n;
		seed_Cepi = para_other_Cepi.seed;
		n_base_Cepi = para_other_Cepi.n_base;
		n_seq_Cepi = para_other_Cepi.n_seq;

		kernel_type_Cepi = para_other_Cepi.kernel_type;
		coord_type_Cepi = para_other_Cepi.coord_type;

		//r_Cepi = r_arg;

		distance_mat_Cepi = distance_mat_arg;

		beta_ij_mat_Cepi = beta_ij_mat_arg;


	}



	void func_ric(mov_struct&, epi_struct_sim&, vector<int>, vector<int>, const vector<double>&, double, vector< vector<double> >&, rng_type & rng_arg); // function prototype for calculating remaining infection challenged needed to get infection

	void func_t_next(mov_struct&, epi_struct_sim&, vector<int>, double, const vector<double>&, int, vector< vector<double> >&, rng_type & rng_arg); //update times of next event for each remaining susceptible

	void func_ric_j(int, mov_struct&, epi_struct_sim&, vector<int>, vector<int>, const vector<double>&, double, vector< vector<double> >&, rng_type & rng_arg); // function prototype for calculating remaining infection challenged needed to get infection

	void func_t_next_j(int, mov_struct&, epi_struct_sim&, vector<int>, double, const vector<double>&, int, vector< vector<double> >&, rng_type & rng_arg); //update times of next event for each remaining susceptible

	void func_status_update(epi_struct_sim&, vector<int>, rng_type & rng_arg); // function prototype for updating status

	void func_time_update(mov_struct&, epi_struct_sim&, vector<int>&, vector<int>&, vector<int>&, vector <int>&, vector<int>, double, const vector<double>&, nt_struct_sim&, vector<int>&, rng_type & rng_arg); // function prototype for assigning the event time to right places & updating the vectors regarding infectious status

	void seq_update_source(nt_struct_sim&, int, rng_type & rng_arg); // sample sequences at sampling time of single subject
	void seq_update_pair(nt_struct_sim&, int, int, double, rng_type & rng_arg); // update sequences for infectious-infected pair

};

// ----------------------------------------------------------------------------------------------------------

void IO_para_aux(para_aux_struct&);
//
void IO_para_key(para_key_struct&);
//
void IO_para_epi(epi_struct_sim&, para_aux_struct&);
//
void IO_para_mov(mov_struct&, para_aux_struct&);
//

double func_kernel_sim(double, double, double, double, double, string, string); // function prototype for calculating kernel distance

double func_latent_ran(rng_type & rng_arg, double par_1, double par_2);

void initialize_beta_ij_mat(vector< vector<double> >&, para_aux_struct&, epi_struct_sim&, para_key_struct&);

class min_max_functions {
public:
	vector<int> min_position_double(vector<double>, double, int); // function prototype for looking for minimum positions
	friend class epi_functions; //allow access to private members of friend class
};

double roundx_sim(double x, int y);

//--------------------------
double runif_sim(double x0, double x1, rng_type& rng_arg);
int runif_int_sim(int x0, int x1, rng_type& rng_arg);
double rgamma_sim(double shape, double scale, rng_type& rng_arg);
double rweibull_sim(double scale, double shape, rng_type& rng_arg);
int rbern_sim(double p, rng_type& rng_arg);
double rnorm_sim(double mean, double sd, rng_type& rng_arg);
//double rexp_sim(double rate, rng_type& rng_arg);
int edf_sample_sim(vector<double> vec, rng_type& rng_arg);
//--------------------------




/*----------------------------*/
/*- IO functions -------------*/
/*----------------------------*/
void IO_para_aux(para_aux& para_other_arg) { // this function is called to
  
  ifstream myfile_in_simpara;
  ofstream myfile_out_simpara;
  
  string line, field;
  int line_count=0, field_count=0;
  
  myfile_in_simpara.open((string(PATH1)+string("parameters_other.csv")).c_str(),ios::in);
  line_count =0;
  
  while (getline(myfile_in_simpara,line)) {
    
    //getline(myfile_in_simdata,line);
    stringstream ss(line);
    //string field;
    field_count=0;
    
    while (getline(ss, field, ',' )) {
      stringstream fs (field);
      if ((line_count==1) & (field_count==0)) fs >> para_other_arg.n;
      if ((line_count==1) & (field_count==1)) fs >> para_other_arg.kernel_type;
      if ((line_count==1) & (field_count==2)) fs >> para_other_arg.coord_type;
      if ((line_count==1) & (field_count==3)) fs >> para_other_arg.t_max;
      if ((line_count==1) & (field_count==4)) fs >> para_other_arg.unassigned_time;
      if ((line_count==1) & (field_count==5)) fs >> para_other_arg.np;
      if ((line_count==1) & (field_count==6)) fs >> para_other_arg.n_seq;
      if ((line_count==1) & (field_count==7)){
        switch(int (ind_n_base_part==1)){
        case 0:{
        fs >> para_other_arg.n_base;
        break;
      }
        case 1:{
          para_other_arg.n_base = n_base_part; // this change the para_other.n_base to be n_base_part
          break;
        }
        }
      }
      
      if ((line_count == 1) & (field_count == 8)) fs >> para_other_arg.n_iterations;
      if ((line_count == 1) & (field_count == 9)) fs >> para_other_arg.n_frequ;
      if ((line_count == 1) & (field_count == 10)) fs >> para_other_arg.n_output_source;
      if ((line_count == 1) & (field_count == 11)) fs >> para_other_arg.n_output_gm;
      if ((line_count == 1) & (field_count == 12)) fs >> para_other_arg.n_cout;
      if ((line_count == 1) & (field_count == 13)) fs >> para_other_arg.opt_latgamma;
      if ((line_count == 1) & (field_count == 14)) fs >> para_other_arg.opt_k80;
      if ((line_count == 1) & (field_count == 15)) fs >> para_other_arg.opt_betaij;
      if ((line_count == 1) & (field_count == 16)) fs >> para_other_arg.opt_ti_update;
      if ((line_count == 1) & (field_count == 17)) fs >> para_other_arg.opt_mov;
      if ((line_count == 1) & (field_count == 18)) fs >> para_other_arg.debug;
      
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1;
    
  }
  
  myfile_in_simpara.close();
  
  
  if (debug == 1) {
    myfile_out_simpara.open((string(PATH2) + string("parameters_other.csv")).c_str(), ios::out);
    myfile_out_simpara << "n" << "," << "kernel_type" << "," "coord_type" << "," << "t_max" << "," << "unassigned_time" << "," << "nprocessors" << "," << "n_seq" << "," << "n_base" << "," << "n_iterations" << "," << "n_freq" << "," << "n_output_source" << "," << "n_output_gm" << "," << "n_cout" << "," << "opt_latgamma" << "," << "opt_k80" << "," << "opt_betaij" << "," << "opt_ti_update" << "," << "opt_mov" << "," << "debug" << endl;
    myfile_out_simpara << para_other_arg.n << "," << para_other_arg.kernel_type << "," << para_other_arg.coord_type << "," << para_other_arg.t_max << "," << para_other_arg.unassigned_time << "," << para_other_arg.np << "," << para_other_arg.n_seq << "," << para_other_arg.n_base << "," << para_other_arg.n_iterations << "," << para_other_arg.n_frequ << "," << para_other_arg.n_output_source << "," << para_other_arg.n_output_gm << "," << para_other_arg.n_cout << "," << para_other_arg.opt_latgamma << "," << para_other_arg.opt_k80 << "," << para_other_arg.opt_betaij << "," << para_other_arg.opt_ti_update << "," << para_other_arg.opt_mov << "," << para_other_arg.debug << endl;
    myfile_out_simpara.close();
  }
  
}
int opt_latgamma, opt_k80, opt_betaij, opt_ti_update, opt_mov, debug;

/*--------------------------------------------*/
//-------------
void IO_data(para_aux para_other_arg, vector < vector<double> >& coordinate_arg, epi_struct& epi_final_arg, nt_struct& nt_data_arg, vector<int>& index_arg, vector<int>& con_seq, vector<int>& seed_arg, moves_struct& moves_arg){ // this function is called to input/output data
  
  ifstream myfile_in_simdata;
  ofstream myfile_out_simdata;
  
  string line, field;
  int line_count=0, field_count=0;
  
  
  myfile_in_simdata.open((string(PATH1)+string("coordinate.csv")).c_str(),ios::in);
  //string line;
  line_count=0;
  
  //coordinate_arg.resize(para_other_arg.n);
  
  while (getline(myfile_in_simdata,line)) {
    
    //getline(myfile_in_simdata,line);
    stringstream ss(line);
    //string field;
    field_count=0;
    
    while (getline(ss, field, ',' )) {
      stringstream fs (field);
      fs >> coordinate_arg[line_count][field_count];
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1;
  }
  
  myfile_in_simdata.close();
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("coordinate.csv")).c_str(), ios::out);
    for (int i = 0; i <= (para_other_arg.n - 1); i++) {
      myfile_out_simdata << coordinate_arg[i][0] << "," << coordinate_arg[i][1] << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*--------------------------------------------*/
  
  epi_final_arg.k.resize(para_other_arg.n);
  epi_final_arg.coor_x.resize(para_other_arg.n);
  epi_final_arg.coor_y.resize(para_other_arg.n);
  epi_final_arg.t_e.resize(para_other_arg.n);
  epi_final_arg.t_i.resize(para_other_arg.n);
  epi_final_arg.t_r.resize(para_other_arg.n);
  epi_final_arg.ftype0.resize(para_other_arg.n);
  epi_final_arg.ftype1.resize(para_other_arg.n);
  epi_final_arg.ftype2.resize(para_other_arg.n);
  epi_final_arg.herdn.resize(para_other_arg.n);
  epi_final_arg.infected_source.resize(para_other_arg.n);
  
  epi_final_arg.q.resize(para_other_arg.n);
  epi_final_arg.status.resize(para_other_arg.n);
  
  
  myfile_in_simdata.open((string(PATH1)+string("epi.csv")).c_str(),ios::in);
  line_count=0;
  
  while (getline(myfile_in_simdata,line)) {
    
    stringstream ss(line);
    field_count=0;
    
    while (getline(ss, field, ',' )) {
      stringstream fs (field);
      if ((line_count >= 1) & (field_count == 0)) fs >> epi_final_arg.k.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 1)) fs >> epi_final_arg.coor_x.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 2)) fs >> epi_final_arg.coor_y.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 3)) fs >> epi_final_arg.t_e.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 4)) fs >> epi_final_arg.t_i.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 5)) fs >> epi_final_arg.t_r.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 6)) fs >> epi_final_arg.ftype0.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 7)) fs >> epi_final_arg.ftype1.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 8)) fs >> epi_final_arg.ftype2.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 9)) fs >> epi_final_arg.herdn.at(line_count - 1);
      if ((line_count >= 1) & (field_count == 10)) fs >> epi_final_arg.infected_source.at(line_count - 1);
      
      
      
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1 ;
  }
  
  myfile_in_simdata.close();
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("epi_final.csv")).c_str(), ios::app);
    myfile_out_simdata << "k" << "," << "coor_x" << "," << "coor_y" << "," << "t_e" << "," << "t_i" << "," << "t_r" << "," << "infected_source" << endl;
    for (int i = 0; i <= (para_other_arg.n - 1); i++) {
      myfile_out_simdata << epi_final_arg.k.at(i) << "," << epi_final_arg.coor_x.at(i) << "," << epi_final_arg.coor_y.at(i) << "," << epi_final_arg.t_e.at(i) << "," << epi_final_arg.t_i.at(i) << "," << epi_final_arg.t_r.at(i) << "," << epi_final_arg.infected_source.at(i) << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*--------------------------------------------*/
  if (opt_mov == 1) {
    
    myfile_in_simdata.open((string(PATH1) + string("moves.csv")).c_str(), ios::in);
    line_count = 0;
    
    while (getline(myfile_in_simdata, line)) {
      
      stringstream ss(line);
      field_count = 0;
      
      while (getline(ss, field, ',')) {
        stringstream fs(field);
        if ((line_count >= 1) & (field_count == 0)) fs >> moves_arg.from_k.at(line_count - 1);
        if ((line_count >= 1) & (field_count == 1)) fs >> moves_arg.to_k.at(line_count - 1);
        if ((line_count >= 1) & (field_count == 2)) fs >> moves_arg.t_m.at(line_count - 1);
        
        field_count = field_count + 1;
      }
      
      line_count = line_count + 1;
      
    }
    
    myfile_in_simdata.close();
    
    myfile_out_simdata.open((string(PATH2) + string("moves.csv")).c_str(), ios::app);
    myfile_out_simdata << "from_k" << "," << "to_k" << "," << "t_m" << endl;
    for (int i = 0; i < (line_count - 1); i++) {
      myfile_out_simdata << moves_arg.from_k.at(i) << "," << moves_arg.to_k.at(i) << "," << moves_arg.t_m.at(i) << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*--------------------------------------------*/
  
  index_arg.reserve(NLIMIT);
  
  myfile_in_simdata.open((string(PATH1)+string("index.csv")).c_str(),ios::in);
  line_count=0;
  
  
  while (getline(myfile_in_simdata,line)) {
    
    stringstream ss(line);
    
    while (getline(ss, field)) {
      stringstream fs (field);
      if (line_count>=1) {
        int ind;
        fs >> ind;
        index_arg.push_back(ind);
        
      }
    }
    
    line_count = line_count + 1 ;
  }
  
  myfile_in_simdata.close();
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("index.csv")).c_str(), ios::app);
    myfile_out_simdata << "k" << endl;
    for (int i = 0; i <= ((int)index_arg.size() - 1); i++) {
      myfile_out_simdata << index_arg.at(i) << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*--------------------------------------------*/
  
  
  //NLIMIT turned off
  /*
   nt_data_arg.nt.resize(para_other_arg.n);
   
   
   for (int i=0; i<=(para_other_arg.n-1); i++){
   nt_data_arg.nt[i].reserve(para_other_arg.n_seq*para_other_arg.n_base);
   }
   
   nt_data_arg.t_nt.resize(para_other_arg.n);
   for (int i=0; i<=(para_other_arg.n-1); i++){
   nt_data_arg.t_nt[i].reserve(para_other_arg.n_seq);
   }
   
   nt_data_arg.current_size.resize(para_other_arg.n);
   nt_data_arg.t_sample.resize(para_other_arg.n);
   */
  
  /*--------------------------------------------*/
  
  myfile_in_simdata.open((string(PATH1)+string("t_sample.csv")).c_str(),ios::in);
  
  
  line_count=0;
  
  while (getline(myfile_in_simdata,line)) {
    
    stringstream ss(line);
    
    while (getline(ss, field)) {
      stringstream fs (field);
      double t;
      fs >>t;
      nt_data_arg.t_sample.at(line_count) = t;
    }
    
    line_count = line_count + 1 ;
  }
  
  myfile_in_simdata.close();
  
  
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("t_sample.csv")).c_str(), ios::app);
    for (int i = 0; i <= (para_other_arg.n - 1); i++) {
      myfile_out_simdata << nt_data_arg.t_sample.at(i) << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*--------------------------------------------*/
  
  myfile_in_simdata.open((string(PATH1)+ string("con_seq_estm.csv")).c_str(),ios::in);
  
  line_count = 0;
  
  while (getline(myfile_in_simdata,line)) {
    
    stringstream ss(line);
    field_count=0;
    
    while (getline(ss, field, ',')) {
      stringstream fs (field);
      int nt;
      fs >> nt;
      con_seq.push_back(nt);
      
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1 ;
  }
  
  myfile_in_simdata.close();
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("con_seq_estm.csv")).c_str(), ios::app);
    for (int j = 0; j <= (para_other_arg.n_base - 1); j++) {
      int rem = (j + 1) % para_other_arg.n_base;
      if ((rem != 0) | (j == 0)) myfile_out_simdata << con_seq.at(j) << ",";
      if ((rem == 0) & (j != 0)) myfile_out_simdata << con_seq.at(j) << " " << endl;
    }
    myfile_out_simdata.close();
  }
  
  /*---------------------------------------------*/
  
  //seed_arg.reserve(para_other_arg.np);
  
  myfile_in_simdata.open((string(PATH1) + string("seeds.csv")).c_str(), ios::in);
  line_count = 0;
  
  
  while (getline(myfile_in_simdata, line)) {
    
    stringstream ss(line);
    
    while (getline(ss, field)) {
      stringstream fs(field);
      int ind;
      fs >> ind;
      seed_arg.push_back(ind);
    }
    
    line_count = line_count + 1;
  }
  
  myfile_in_simdata.close();
  
  if (debug == 1) {
    myfile_out_simdata.open((string(PATH2) + string("seeds.csv")).c_str(), ios::app);
    for (int i = 0; i <= ((int)seed_arg.size() - 1); i++) {
      myfile_out_simdata << seed_arg.at(i) << endl;
    }
    myfile_out_simdata.close();
  }
  
}

//---------------------------------------------//

void IO_parapriorsetc(para_priors_etc& para_priors_etc_arg) { // this function is called to input/output parameters from simulation
  
  
  ifstream myfile_in_parapriors;
  ofstream myfile_out_parapriors;
  
  string line, field;
  int line_count = 0, field_count = 0;
  
  
  myfile_in_parapriors.open((string(PATH1) + string("parameters_priors_etc.csv")).c_str(), ios::in);
  line_count = 0;
  
  while (getline(myfile_in_parapriors, line)) {
    
    //getline(myfile_in_parapriors,line);
    stringstream ss(line);
    //string field;
    field_count = 0;
    
    while (getline(ss, field, ',')) {
      stringstream fs(field);
      if ((line_count == 1) & (field_count == 0)) fs >> para_priors_etc_arg.t_range;
      if ((line_count == 1) & (field_count == 1)) fs >> para_priors_etc_arg.t_back;
      if ((line_count == 1) & (field_count == 2)) fs >> para_priors_etc_arg.t_bound_hi;
      if ((line_count == 1) & (field_count == 3)) fs >> para_priors_etc_arg.rate_exp_prior;
      if ((line_count == 1) & (field_count == 4)) fs >> para_priors_etc_arg.ind_n_base_part;
      if ((line_count == 1) & (field_count == 5)) fs >> para_priors_etc_arg.n_base_part;
      if ((line_count == 1) & (field_count == 6)) fs >> para_priors_etc_arg.alpha_hi;
      if ((line_count == 1) & (field_count == 7)) fs >> para_priors_etc_arg.beta_hi;
      if ((line_count == 1) & (field_count == 8)) fs >> para_priors_etc_arg.lat_mu_hi;
      if ((line_count == 1) & (field_count == 9)) fs >> para_priors_etc_arg.lat_var_lo;
      if ((line_count == 1) & (field_count == 10)) fs >> para_priors_etc_arg.lat_var_hi;
      if ((line_count == 1) & (field_count == 11)) fs >> para_priors_etc_arg.c_hi;
      if ((line_count == 1) & (field_count == 12)) fs >> para_priors_etc_arg.d_hi;
      if ((line_count == 1) & (field_count == 13)) fs >> para_priors_etc_arg.k_1_lo;
      if ((line_count == 1) & (field_count == 14)) fs >> para_priors_etc_arg.k_1_hi;
      if ((line_count == 1) & (field_count == 15)) fs >> para_priors_etc_arg.mu_1_hi;
      if ((line_count == 1) & (field_count == 16)) fs >> para_priors_etc_arg.mu_2_hi;
      if ((line_count == 1) & (field_count == 17)) fs >> para_priors_etc_arg.p_ber_hi;
      if ((line_count == 1) & (field_count == 18)) fs >> para_priors_etc_arg.phi_inf1_hi;
      if ((line_count == 1) & (field_count == 19)) fs >> para_priors_etc_arg.phi_inf2_hi;
      if ((line_count == 1) & (field_count == 20)) fs >> para_priors_etc_arg.rho_susc1_hi;
      if ((line_count == 1) & (field_count == 21)) fs >> para_priors_etc_arg.rho_susc2_hi;
      if ((line_count == 1) & (field_count == 22)) fs >> para_priors_etc_arg.nu_inf_lo;
      if ((line_count == 1) & (field_count == 23)) fs >> para_priors_etc_arg.nu_inf_hi;
      if ((line_count == 1) & (field_count == 24)) fs >> para_priors_etc_arg.tau_susc_lo;
      if ((line_count == 1) & (field_count == 25)) fs >> para_priors_etc_arg.tau_susc_hi;
      if ((line_count == 1) & (field_count == 26)) fs >> para_priors_etc_arg.beta_m_hi;
      if ((line_count == 1) & (field_count == 27)) fs >> para_priors_etc_arg.trace_window;
      
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1;
    
  }
  
  myfile_in_parapriors.close();
  
  if (debug == 1) {
    myfile_out_parapriors.open((string(PATH2) + string("parameters_priors_etc.csv")).c_str(), ios::out);
    myfile_out_parapriors << "t_range" << "," << "t_back" << "," "t_bound_hi" << "," << "rate_exp_prior" << "," << "ind_n_base_part" << "," << "n_base_part" << "," << "alpha_hi" << "," << "beta_hi" << "," << "lat_mu_hi" << "," << "lat_var_lo" << "," << "lat_var_hi" << "," << "c_hi" << "," << "d_hi" << "," << "k_1_lo" << "," << "k_1_hi" << "," << "mu_1_hi" << "," << "mu_2_hi" << "," << "p_ber_hi" << "," << "rho_susc1_hi" << "," << "rho_susc2_hi" << "," << "phi_inf1_hi" << "," << "phi_inf2_hi" << "," << "tau_susc_lo" << "," << "tau_susc_hi" << "," << "nu_inf_lo" << "," << "nu_inf_hi" << "," << "beta_m_hi" << "," << "trace_window" << endl;
    myfile_out_parapriors << para_priors_etc_arg.t_range << "," << para_priors_etc_arg.t_back << "," << para_priors_etc_arg.t_bound_hi << "," << para_priors_etc_arg.rate_exp_prior << "," << para_priors_etc_arg.ind_n_base_part << "," << para_priors_etc_arg.n_base_part << "," << para_priors_etc_arg.alpha_hi << "," << para_priors_etc_arg.beta_hi << "," << para_priors_etc_arg.lat_mu_hi << "," << para_priors_etc_arg.lat_var_lo << "," << para_priors_etc_arg.lat_var_hi << "," << para_priors_etc_arg.c_hi << "," << para_priors_etc_arg.d_hi << "," << para_priors_etc_arg.k_1_lo << "," << para_priors_etc_arg.k_1_hi << "," << para_priors_etc_arg.mu_1_hi << "," << para_priors_etc_arg.mu_2_hi << "," << para_priors_etc_arg.p_ber_hi << "," << para_priors_etc_arg.rho_susc1_hi << "," << para_priors_etc_arg.rho_susc2_hi << "," << para_priors_etc_arg.phi_inf1_hi << "," << para_priors_etc_arg.phi_inf2_hi << "," << para_priors_etc_arg.tau_susc_lo << "," << para_priors_etc_arg.tau_susc_hi << "," << para_priors_etc_arg.nu_inf_lo << "," << para_priors_etc_arg.nu_inf_hi << "," << para_priors_etc_arg.beta_m_hi << "," << para_priors_etc_arg.trace_window << endl;
    myfile_out_parapriors.close();
  }
  
}
int ind_n_base_part, n_base_part;

//---------------------------------------------//

void IO_parakeyinit(para_key_init& para_key_init_arg) { // this function is called to input/output parameters from simulation
  //sf: added upload function
  
  ifstream myfile_in_parainit;
  ofstream myfile_out_parainit;
  
  string line, field;
  int line_count = 0, field_count = 0;
  
  
  myfile_in_parainit.open((string(PATH1) + string("parameters_key_inits.csv")).c_str(), ios::in);
  line_count = 0;
  
  while (getline(myfile_in_parainit, line)) {
    
    //getline(myfile_in_parainit,line);
    stringstream ss(line);
    //string field;
    field_count = 0;
    
    while (getline(ss, field, ',')) {
      stringstream fs(field);
      if ((line_count == 1) & (field_count == 0)) fs >> para_key_init_arg.alpha;
      if ((line_count == 1) & (field_count == 1)) fs >> para_key_init_arg.beta;
      if ((line_count == 1) & (field_count == 2)) fs >> para_key_init_arg.lat_mu;
      if ((line_count == 1) & (field_count == 3)) fs >> para_key_init_arg.lat_var;
      if ((line_count == 1) & (field_count == 4)) fs >> para_key_init_arg.c;
      if ((line_count == 1) & (field_count == 5)) fs >> para_key_init_arg.d;
      if ((line_count == 1) & (field_count == 6)) fs >> para_key_init_arg.k_1;
      if ((line_count == 1) & (field_count == 7)) fs >> para_key_init_arg.mu_1;
      if ((line_count == 1) & (field_count == 8)) fs >> para_key_init_arg.mu_2;
      if ((line_count == 1) & (field_count == 9)) fs >> para_key_init_arg.p_ber;
      if ((line_count == 1) & (field_count == 10)) fs >> para_key_init_arg.phi_inf1;
      if ((line_count == 1) & (field_count == 11)) fs >> para_key_init_arg.phi_inf2;
      if ((line_count == 1) & (field_count == 12)) fs >> para_key_init_arg.rho_susc1;
      if ((line_count == 1) & (field_count == 13)) fs >> para_key_init_arg.rho_susc2;
      if ((line_count == 1) & (field_count == 14)) fs >> para_key_init_arg.nu_inf;
      if ((line_count == 1) & (field_count == 15)) fs >> para_key_init_arg.tau_susc;
      if ((line_count == 1) & (field_count == 16)) fs >> para_key_init_arg.beta_m;
      
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1;
    
  }
  
  myfile_in_parainit.close();
  
  
  if (debug == 1) {
    myfile_out_parainit.open((string(PATH2) + string("parameters_key_inits.csv")).c_str(), ios::out);
    myfile_out_parainit << "alpha" << "," << "beta" << "," << "lat_mu" << "," << "lat_var" << "," << "c" << "," << "d" << "," << "k_1" << "," << "mu_1" << "," << "mu_2" << "," << "p_ber" << "," << "rho_susc1" << "," << "rho_susc2" << "," << "phi_inf1" << "," << "phi_inf2" << "," << "tau_susc" << "," << "nu_inf" << "," << "beta_m" << endl;
    myfile_out_parainit << para_key_init_arg.alpha << "," << para_key_init_arg.beta << "," << para_key_init_arg.lat_mu << "," << para_key_init_arg.lat_var << "," << para_key_init_arg.c << "," << para_key_init_arg.d << "," << para_key_init_arg.k_1 << "," << para_key_init_arg.mu_1 << "," << para_key_init_arg.mu_2 << "," << para_key_init_arg.p_ber << "," << para_key_init_arg.rho_susc1 << "," << para_key_init_arg.rho_susc2 << "," << para_key_init_arg.phi_inf1 << "," << para_key_init_arg.phi_inf2 << "," << para_key_init_arg.tau_susc << "," << para_key_init_arg.nu_inf << "," << para_key_init_arg.beta_m << endl;
    myfile_out_parainit.close();
  }
  
}

//---------------------------------------------//

void IO_parascalingfactors(para_scaling_factors& para_scaling_factors_arg) { // this function is called to input/output parameters from simulation
  
  
  ifstream myfile_in_parascaling;
  ofstream myfile_out_parascaling;
  
  string line, field;
  int line_count = 0, field_count = 0;
  
  
  myfile_in_parascaling.open((string(PATH1) + string("parameters_scaling_factors.csv")).c_str(), ios::in);
  line_count = 0;
  
  while (getline(myfile_in_parascaling, line)) {
    
    //getline(myfile_in_parascaling,line);
    stringstream ss(line);
    //string field;
    field_count = 0;
    
    while (getline(ss, field, ',')) {
      stringstream fs(field);
      if ((line_count == 1) & (field_count == 0)) fs >> para_scaling_factors_arg.alpha_sf;
      if ((line_count == 1) & (field_count == 1)) fs >> para_scaling_factors_arg.beta_sf;
      if ((line_count == 1) & (field_count == 2)) fs >> para_scaling_factors_arg.lat_mu_sf;
      if ((line_count == 1) & (field_count == 3)) fs >> para_scaling_factors_arg.lat_var_sf;
      if ((line_count == 1) & (field_count == 4)) fs >> para_scaling_factors_arg.c_sf;
      if ((line_count == 1) & (field_count == 5)) fs >> para_scaling_factors_arg.d_sf;
      if ((line_count == 1) & (field_count == 6)) fs >> para_scaling_factors_arg.k_1_sf;
      if ((line_count == 1) & (field_count == 7)) fs >> para_scaling_factors_arg.mu_1_sf;
      if ((line_count == 1) & (field_count == 8)) fs >> para_scaling_factors_arg.mu_2_sf;
      if ((line_count == 1) & (field_count == 9)) fs >> para_scaling_factors_arg.p_ber_sf;
      if ((line_count == 1) & (field_count == 10)) fs >> para_scaling_factors_arg.phi_inf1_sf;
      if ((line_count == 1) & (field_count == 11)) fs >> para_scaling_factors_arg.phi_inf2_sf;
      if ((line_count == 1) & (field_count == 12)) fs >> para_scaling_factors_arg.rho_susc1_sf;
      if ((line_count == 1) & (field_count == 13)) fs >> para_scaling_factors_arg.rho_susc2_sf;
      if ((line_count == 1) & (field_count == 14)) fs >> para_scaling_factors_arg.nu_inf_sf;
      if ((line_count == 1) & (field_count == 15)) fs >> para_scaling_factors_arg.tau_susc_sf;
      if ((line_count == 1) & (field_count == 16)) fs >> para_scaling_factors_arg.beta_m_sf;
      field_count = field_count + 1;
    }
    
    line_count = line_count + 1;
    
  }
  
  myfile_in_parascaling.close();
  
  if (debug == 1) {
    myfile_out_parascaling.open((string(PATH2) + string("parameters_scaling_factors.csv")).c_str(), ios::out);
    myfile_out_parascaling << "alpha_sf" << "," << "beta_sf" << "," << "lat_mu_sf" << "," << "lat_var_sf" << "," << "c_sf" << "," << "d_sf" << "," << "k_1_sf" << "," << "mu_1_sf" << "," << "mu_2_sf" << "," << "p_ber_sf" << "," << "rho_susc1_sf" << "," << "rho_susc2_sf" << "," << "phi_inf1_sf" << "," << "phi_inf2_sf" << "," << "tau_susc_sf" << "," << "nu_inf_sf" << "," << "beta_m_sf" << endl;
    myfile_out_parascaling << para_scaling_factors_arg.alpha_sf << "," << para_scaling_factors_arg.beta_sf << "," << para_scaling_factors_arg.lat_mu_sf << "," << para_scaling_factors_arg.lat_var_sf << "," << para_scaling_factors_arg.c_sf << "," << para_scaling_factors_arg.d_sf << "," << para_scaling_factors_arg.k_1_sf << "," << para_scaling_factors_arg.mu_1_sf << "," << para_scaling_factors_arg.mu_2_sf << "," << para_scaling_factors_arg.p_ber_sf << "," << para_scaling_factors_arg.rho_susc1_sf << "," << para_scaling_factors_arg.rho_susc2_sf << "," << para_scaling_factors_arg.phi_inf1_sf << "," << para_scaling_factors_arg.phi_inf2_sf << "," << para_scaling_factors_arg.tau_susc_sf << "," << para_scaling_factors_arg.tau_susc_sf << "," << para_scaling_factors_arg.beta_m_sf << endl;
    myfile_out_parascaling.close();
  }
  
}






/*----------------------------*/
/*- functions ----------------*/
/*----------------------------*/
///// Boost probability distributions
rng_type rng;

double runif(double x0, double x1, rng_type& rng_arg) {
  boost::variate_generator<rng_type &, Dunif > rndm(rng_arg, Dunif(x0, x1));
  return rndm();
}

int runif_int(int x0, int x1, rng_type& rng_arg)
{
  boost::variate_generator<rng_type &, Dunif_int > rndm(rng_arg, Dunif_int(x0, x1));
  return rndm();
}

double rgamma(double shape, double scale, rng_type& rng_arg) {
  boost::variate_generator<rng_type &, Dgamma > rndm(rng_arg, Dgamma(shape, scale));
  return rndm();
}

double rweibull(double scale, double shape, rng_type& rng_arg) { //note reversed scale and shape from boost itself (shape=1.0 for exponential distribution)
  boost::variate_generator<rng_type &, Dweibull > rndm(rng_arg, Dweibull(shape, scale));
  return rndm();
}

int rbern(double p, rng_type& rng_arg) {
  boost::variate_generator<rng_type &, Dbern > rndm(rng_arg, Dbern(p));
  return rndm();
}

double rnorm(double mean, double sd, rng_type& rng_arg) {
  boost::variate_generator<rng_type &, Dnorm > rndm(rng_arg, Dnorm(mean, sd));
  return rndm();
}

//seeded differently through rcpp than VS so not used
/*
 double rexp(double rate, rng_type& rng_arg) {
 boost::variate_generator<rng_type &, Dexp > rndm(rng_arg, Dexp(rate));
 return rndm();
 }
 */

// empirical distribution random sampler (depends on runif function)
int edf_sample(vector<double> vec, rng_type& rng_arg)
{
  double s = 0;
  vector<double> edf; // dynamic vector
  for (int k = 0; k < (int)(vec.size()); k++)
  {
    s += vec[k];
    edf.push_back(s);
  }
  double u = s * runif(0.0, 1.0, rng_arg);
  vector<double>::iterator low;
  low = lower_bound(edf.begin(), edf.end(), u); // fast search for where to locate u in vector
  int i = int(low - edf.begin());
  return i;  // index from 0 of sampled element that u is within range
}

/*-------------------------------------------------------*/


long double func_kernel (double x_1 , double y_1, double x_2, double y_2, double k_1_arg, const string& kernel_type_arg, const string& coord_type_arg){
  
  
  long double eucli_dist;
  
  //cartesian coordinate system
  if (coord_type_arg == "cartesian") {
    eucli_dist = sqrt(pow((x_1 - x_2), 2) + pow((y_1 - y_2), 2)); //coordinates inputted in kilometres
  }
  
  
  //longlat coordinate system
  if (coord_type_arg == "longlat") {
    double pi = 3.1415926535897;
    double rad = pi / 180.0;
    double x_1_rad = x_1 * rad;
    double y_1_rad = y_1 * rad;
    double x_2_rad = x_2 * rad;
    double y_2_rad = y_2 * rad;
    double dlon = x_2_rad - x_1_rad;
    double dlat = y_2_rad - y_1_rad;
    double a = pow((sin(dlat / 2.0)), 2.0) + cos(y_1_rad) * cos(y_2_rad) * pow((sin(dlon / 2.0)), 2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    double R = 6378.145;  //radius of Earth in km, output is scaled in km
    eucli_dist = R * c;
  }
  
  
  long double func_ker = 0.0;
  if (kernel_type_arg == "exponential") { func_ker = exp((-k_1_arg)*eucli_dist); }
  if (kernel_type_arg == "power_law") { func_ker = 1.0 / (1.0 + pow(eucli_dist, k_1_arg));}
  //cauchy distribution (not cauchy kernel)
  if (kernel_type_arg == "cauchy") { func_ker = (1 / (k_1_arg*(1 + pow(eucli_dist / k_1_arg, 2.0))));}
  if (kernel_type_arg == "gaussian") { func_ker = exp(-pow(eucli_dist, k_1_arg));
  }
  
  
  
  return(func_ker);
}


/*-------------------------------------------*/
//functions to limit distributions to smallest possible value rather than zero, to avoid log(0) = -Inf errors
inline double pdf_weibull_limit(double shape, double scale, double q) {
  double pdf_weibull = 0.0;
  pdf_weibull = pdf(weibull_mdist(shape, scale), q);
  
  if (pdf_weibull == 0) {
    double dbl_min = std::numeric_limits< double >::min();
    pdf_weibull = dbl_min;
  }
  
  return(pdf_weibull);
}

inline double surv_weibull_limit(double shape, double scale, double q) {
  double surv_weibull = 0.0;
  surv_weibull = 1 - cdf(weibull_mdist(shape, scale), q);
  
  if (surv_weibull == 0) {
    double dbl_m = std::numeric_limits< double >::min();
    surv_weibull = dbl_m;
  }
  
  return(surv_weibull);
}

inline double pdf_exp_limit(double rate, double q) {
  double pdf_exp = 0.0;
  pdf_exp = pdf(exp_mdist(rate), q);
  
  if (pdf_exp == 0) {
    double dbl_min = std::numeric_limits< double >::min();
    pdf_exp = dbl_min;
  }
  
  return(pdf_exp);
}

inline double surv_exp_limit(double rate, double q) {
  double surv_exp = 0.0;
  surv_exp = 1 - cdf(exp_mdist(rate), q);
  
  if (surv_exp == 0) {
    double dbl_m = std::numeric_limits< double >::min();
    surv_exp = dbl_m;
  }
  
  return(surv_exp);
}
/*-------------------------------------------*/

inline double func_beta_ij(double n_inf, double n_susc, double nu_inf_arg, double tau_susc_arg, double ftype0_inf, double ftype0_susc, double ftype1_inf, double ftype1_susc, double ftype2_inf, double ftype2_susc, double phi_inf1_arg, double phi_inf2_arg, double rho_susc1_arg, double rho_susc2_arg) {
  
  double func_beta_ij;
  
  func_beta_ij = pow(n_inf, nu_inf_arg) *
    ((1.0 * ftype0_inf) +  //cattle (reference)
    (phi_inf1_arg * ftype1_inf) +  //pigs
    (phi_inf2_arg * ftype2_inf)) * //other
    pow(n_susc, tau_susc_arg) *
    ((1.0 * ftype0_susc) +	//cattle (reference)
    (rho_susc1_arg * ftype1_susc) +  //pigs
    (rho_susc2_arg * ftype2_susc));	//other
  
  return(func_beta_ij);
}

/*-------------------------------------------*/
/*
 
 //normalisation version
 
 inline double func_beta_ij_inf(double n_inf, double nu_inf_arg, double ftype0_inf, double ftype1_inf, double ftype2_inf, double phi_inf1_arg, double phi_inf2_arg) {
 
 double func_beta_ij_inf;
 
 func_beta_ij_inf = pow(n_inf, nu_inf_arg) *
 ((1.0 * ftype0_inf) +  //cattle (reference)
 (phi_inf1_arg * ftype1_inf) +  //pigs
 (phi_inf2_arg * ftype2_inf));  //other
 
 return(func_beta_ij_inf);
 }
 */

/*-------------------------------------------*/
/*
 void FUNC::initialize_beta_ij_mat_inf(vector<double>& beta_ij_inf_arg, vector<double>& herdn_arg, vector<double>& ftype0_arg, vector<double>& ftype1_arg, vector<double>& ftype2_arg) {
 
 for (int i = 0; i <= (n_Clh - 1); i++) { //infectives
 beta_ij_inf_arg[i] = func_beta_ij_inf(herdn_arg[i], nu_inf_Clh, ftype0_arg[i], ftype1_arg[i], ftype2_arg[i], phi_inf1_Clh, phi_inf2_Clh);
 }
 
 //normalise by mean infectivity
 double norm_inf = 0;
 for (int i = 0; i <= (n_Clh - 1); i++) { //infectives
 norm_inf = norm_inf + beta_ij_inf_arg[i];
 }
 norm_inf = norm_inf / n_Clh;
 
 for (int i = 0; i <= (n_Clh - 1); i++) { //infectives
 beta_ij_inf_arg[i] = beta_ij_inf_arg[i] / norm_inf;
 }
 
 }
 
 
 inline double func_beta_ij_susc(double n_susc, double tau_susc_arg, double ftype0_susc, double ftype1_susc, double ftype2_susc, double rho_susc1_arg, double rho_susc2_arg) {
 
 double func_beta_ij_susc;
 
 func_beta_ij_susc = pow(n_susc, tau_susc_arg) *
 ((1.0 * ftype0_susc) +	//cattle (reference)
 (rho_susc1_arg * ftype1_susc) +  //pigs
 (rho_susc2_arg * ftype2_susc));	//other
 
 return(func_beta_ij_susc);
 }
 
 void FUNC::initialize_beta_ij_mat_susc(vector<double>& beta_ij_susc_arg, vector<double>& herdn_arg, vector<double>& ftype0_arg, vector<double>& ftype1_arg, vector<double>& ftype2_arg) {
 
 for (int j = 0; j <= (n_Clh - 1); j++) { //susceptibles
 beta_ij_susc_arg[j] = func_beta_ij_susc(herdn_arg[j], tau_susc_Clh, ftype0_arg[j], ftype1_arg[j], ftype2_arg[j], rho_susc1_Clh, rho_susc2_Clh);
 }
 
 //normalise by mean susceptibility
 double norm_susc = 0;
 for (int j = 0; j <= (n_Clh - 1); j++) { //susceptibles
 norm_susc = norm_susc + beta_ij_susc_arg[j];
 }
 norm_susc = norm_susc / n_Clh;
 
 for (int j = 0; j <= (n_Clh - 1); j++) { //susceptibles
 beta_ij_susc_arg[j] = beta_ij_susc_arg[j] / norm_susc;
 }
 
 }
 
 inline double func_beta_ij_norm(int i_arg, int j_arg, vector<double> beta_ij_inf_arg, vector<double> beta_ij_susc_arg) {
 
 double func_beta_ij;
 
 func_beta_ij = beta_ij_inf_arg[i_arg] * beta_ij_susc_arg[j_arg];
 
 return(func_beta_ij);
 }
 */

/*-------------------------------------------*/


inline double func_moves_cnt(int i_arg, int j_arg, moves_struct& moves_arg, vector<double> t_e_arg, vector<double> t_i_arg, vector<double> t_r_arg, para_priors_etc& para_priors_arg) {
  
  double moves_ij_count = 0;
  
  if (opt_mov == 0) {
    moves_ij_count = 0;
  }
  
  if (opt_mov == 1) {
    
    for (int m = 0; m <= (int)(moves_arg.from_k.size() - 1); m++) {
      if ((moves_arg.from_k[m] == i_arg) && (moves_arg.to_k[m] == j_arg)) {
        
        if ((moves_arg.t_m[m] >= t_i_arg.at(i_arg)) &&
            (moves_arg.t_m[m] <= t_r_arg.at(i_arg)) &&
            //(moves_arg.t_m[m] >= (t_e_arg.at(j_arg)-para_priors_arg.trace_window)) &&
            (moves_arg.t_m[m] <= t_e_arg.at(j_arg))) {
          
          moves_ij_count++;
        }
      }
    }
    
  }
  
  //time independant
  if (opt_mov == 2) {
    
    for (int m = 0; m <= (int)(moves_arg.from_k.size() - 1); m++) {
      if ((moves_arg.from_k[m] == i_arg) && (moves_arg.to_k[m] == j_arg)) {
        
        moves_ij_count++;
        
      }
    }
    
  }
  
  return(moves_ij_count);
}

/*------------------------------------------ - */

void FUNC::initialize_delta_mat_mov(vector< vector<double> >& delta_mat_mov_arg, moves_struct& mov_arg) {
  
  
  if (xi_U_Clh.size() >= 1) {
    for (int i = 0; i <= (int)(xi_U_Clh.size() - 1); i++) {
      
      for (int j = 0; j <= (int)(xi_I_Clh.size() - 1); j++) {
        
        switch (int (t_r_Clh.at(xi_I_Clh.at(j))>t_max_Clh)) {
        case 1: { // not yet recovered
  delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = 0.0;
  for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
    if ((mov_arg.from_k[m] == xi_I_Clh.at(j)) && (mov_arg.to_k[m] == xi_U_Clh.at(i))) {
      
      if ((mov_arg.t_m[m] >= t_i_Clh.at(xi_I_Clh.at(j))) &&
          (mov_arg.t_m[m] <= t_max_Clh)) {
        
        delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] + (t_max_Clh - mov_arg.t_m[m]);
        //delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = (t_max_Clh - mov_arg.t_m[m]);
      }
      
    }
    
  }
  break;
}
        case 0: { // recovered
          delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = 0.0;
          for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
            if ((mov_arg.from_k[m] == xi_I_Clh.at(j)) && (mov_arg.to_k[m] == xi_U_Clh.at(i)) && (t_r_Clh.at(xi_I_Clh.at(j)) <= t_max_Clh)) {
              
              if ((mov_arg.t_m[m] >= t_i_Clh.at(xi_I_Clh.at(j))) &&
                  (mov_arg.t_m[m] <= t_r_Clh.at(xi_I_Clh.at(j)))) {
                
                delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] + (t_r_Clh.at(xi_I_Clh.at(j)) - mov_arg.t_m[m]);
                //delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = (t_r_Clh.at(xi_I_Clh.at(j)) - mov_arg.t_m[m]);
              }
              
            }
            
          }
          
          break;
        }
        }
        
      }
    }
  }
  
  //----------//
  
  
  if (xi_E_minus_Clh.size() >= 1) {
    for (int i = 0; i <= (int)(xi_E_minus_Clh.size() - 1); i++) {
      
      for (int j = 0; j <= (int)(xi_I_Clh.size() - 1); j++) {
        
        if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_minus_Clh.at(i))) {
          
          switch (int (t_r_Clh.at(xi_I_Clh.at(j)) >= t_e_Clh.at(xi_E_minus_Clh.at(i)))) {
          case 1: { // not yet recovered at e_i
    delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = 0.0;
    for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
      if ((mov_arg.from_k[m] == xi_I_Clh.at(j)) && (mov_arg.to_k[m] == xi_E_minus_Clh.at(i))) {
        
        if ((mov_arg.t_m[m] >= t_i_Clh.at(xi_I_Clh.at(j))) &&
            (mov_arg.t_m[m] <= t_e_Clh.at(xi_E_minus_Clh.at(i)))) {
          
          delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] + (t_e_Clh.at(xi_E_minus_Clh.at(i)) - mov_arg.t_m[m]);
          
        }
        
      }
      
    }
    break;
  }
          case 0: { // recovered before e_i
            delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = 0.0;
            for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
              if ((mov_arg.from_k[m] == xi_I_Clh.at(j)) && (mov_arg.to_k[m] == xi_E_minus_Clh.at(i))) {
                
                if ((mov_arg.t_m[m] >= t_i_Clh.at(xi_I_Clh.at(j))) &&
                    (mov_arg.t_m[m] <= t_r_Clh.at(xi_I_Clh.at(j)))) {
                  
                  delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] + (t_r_Clh.at(xi_I_Clh.at(j)) - mov_arg.t_m[m]);
                  
                }
                
              }
              
            }
            break;
          }
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
    }
  }
  
  for (int i = 0; i <= (int)(index_Clh.size() - 1); i++) {
    for (int j = 0; j <= (int)(n_Clh - 1); j++) {
      delta_mat_mov_arg[index_Clh.at(i)][j] = 0.0;
    }
  }
  
}

/*------------------------------------------ - */

inline double lh_snull(const vector<int>& con_seq, const vector<int>& seq, const double& p_ber, const int& n_base){ // compute the log pr a seq for background
  
  double lh_snull=0.0;
  
  int m=0;
  
  for (int i=0;i<=(n_base-1);i++){
    switch(int (seq.at(i)!=con_seq.at(i))){
    case 1:{ // a change
    m = m +1;
    break;
  }
    case 0:{ // not a change
      break;
    }
    }
  }
  
  lh_snull = m*log(p_ber) + (n_base-m)*log(1-p_ber) + m*log(1.0/3.0);
  //	lh_snull = m*log(p_ber) + (n_base-m)*log(1-p_ber) ;
  
  return(lh_snull);
}

/*-------------------------------------------------------*/

inline double lh_snull_base(const int& con_base, const int& base, const double& p_ber){ // compute the log pr a base for background
  
  double lh_snull_base=0.0;
  
  int m=0;
  
  switch(int (base!=con_base)){
  case 1:{ // a change
    m = m +1;
    break;
  }
  case 0:{ // not a change
    break;
  }
  }
  
  lh_snull_base = m*log(p_ber) + (1-m)*log(1-p_ber) + m*log(1.0/3.0);
  //lh_snull_base = m*log(p_ber) + (1-m)*log(1-p_ber) ;
  
  return(lh_snull_base);
}


/*-------------------------------------------------------*/

inline void sample_snull (const vector<int>& con_seq, vector<int>& seq_proposed, const double& p_ber, const int& n_base, rng_type & rng_arg){ //sample a seq for background
  
  for (int j=0;j<=(n_base-1);j++){
    
    //int ber_trial =  gsl_ran_bernoulli (r,p_ber); // return 1 if a change is to happen
    int ber_trial = rbern(p_ber, rng_arg); 
    int base_proposed=0;// any output of 0 would indicate a mistake
    
    
    switch(int (ber_trial ==1)){
    case 1:{ // randomly choose one among other 3
      switch(con_seq.at(j)){
    case 1:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        base_proposed = 2;
        break;
      }
      case 1:{
        base_proposed = 3;
        break;
      }
      case 2:{
        base_proposed = 4;
        break;
      }
      }
      break;
    }
    case 2:{
      int type = runif_int(0, 2, rng_arg);
      
      switch(type){
      case 0:{
        base_proposed = 1;
        break;
      }
      case 1:{
        base_proposed = 3;
        break;
      }
      case 2:{
        base_proposed = 4;
        break;
      }
      }
      break;
    }
    case 3:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        base_proposed = 1;
        break;
      }
      case 1:{
        base_proposed = 2;
        break;
      }
      case 2:{
        base_proposed = 4;
        break;
      }
      }
      break;
    }
    case 4:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        base_proposed = 1;
        break;
      }
      case 1:{
        base_proposed = 2;
        break;
      }
      case 2:{
        base_proposed = 3;
        break;
      }
      }
      break;
    }
    }
      
      seq_proposed.at(j) = base_proposed;
      
      break;
    }
    case 0:{
      seq_proposed.at(j) = con_seq.at(j); // same as consensus seq
      break;
    }
    }
  }
  
}

/*-------------------------------------------------------*/


void seq_propose_cond(vector<int>& seq_proposed, double& log_pr_forward, const vector<int>&  nt_past_forward, const vector<int>& nt_future_forward, const double& t_proposed, const double& t_past, const double& t_future, const double& mu_1, const double& mu_2, int n_base, rng_type & rng_arg){
  
  double T = fabs(t_future - t_past);
  double dt = fabs(t_proposed - t_past);
  double p = dt/T; // pr of assigning a base to be same as the corresponding one in the future sequence
  
  
  //double P[2] = {1.0-p, p};
  //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);
  vector<double> P = { 1.0 - p, p };
  
  int m=0; // number of sites from nt_past were different from the corresponding sites from nt_future
  int dn=0; // number of sites from nt_past become the same as the coresponding sites from nt_future among m
  
  for (int i=0; i<=(n_base-1); i++){
    
    switch(int (nt_past_forward.at(i)==nt_future_forward.at(i))){
    
    case 0:{ // not the same
    
    m = m + 1;
    //int bool_c = gsl_ran_discrete (r_c, g); //  1 = assign to be the same as future
    int bool_c = rbern(p, rng); //  1 = assign to be the same as future
    
    
    switch(bool_c){
    case 1:{
      dn =dn +1;
      seq_proposed.at(i)=nt_future_forward.at(i);
      break;
    }
    case 0:{
      seq_proposed.at(i)=nt_past_forward.at(i);
      break;
    }
    }
    
    break;
  }
      
    case 1:{ // same
      seq_proposed.at(i)=nt_past_forward.at(i);
      break;
    }
      
    }
    
  }
  
  //log_pr_forward = log((long double)pow(p, dn)) + log((long double)pow(1.0-p, m-dn));
  log_pr_forward =dn*log(p) + (m-dn)*log(1.0-p);
  
  
  
}

/*-------------------------------------------*/

void seq_propose_uncond(vector<int>& seq_proposed, double& log_pr_forward,  const vector<int>& nt_past_forward, const double& t_proposed, const double& t_past, const double& t_future,  const double& mu_1, const double& mu_2, int n_base, rng_type & rng_arg){
  
  int n_1, n_2, n_3; // number of unchanged, transition, two types of transversion (compare original nt and nt_proposed)
  n_1=n_2=n_3= 0;
  
  double p_1, p_2, p_3;
  p_1 = p_2 = p_3 = 0.0;
  
  double abs_dt = fabs(t_future - t_past);
  //double abs_dt = fabs(t_proposed - t_past);
  
  //sf edit
  if (opt_k80 == 0) {
    p_1 = 0.25 + 0.25*exp(-4.0*mu_2*abs_dt) + 0.5*exp(-2.0*(mu_1 + mu_2)*abs_dt); // pr of a base not changing
    p_2 = 0.25 + 0.25*exp(-4.0*mu_2*abs_dt) - 0.5*exp(-2.0*(mu_1 + mu_2)*abs_dt); // pr of a transition of a base
    p_3 = 2.0*(0.25 - 0.25*exp(-4.0*mu_2*abs_dt));  // pr of a transversion (two possible events)
  }
  if (opt_k80 == 1) {
    //K80: mu1 = alpha; mu2 = beta
    p_2 = 0.25 - 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // P = pts
    p_3 = 0.5 - 0.5*exp(-8.0*mu_2*abs_dt);  // Q = ptv (2 options)
    p_1 = 1 - p_2 - p_3; // R = no change
    //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
  }
  
  
  
  //double P[3] = {p_1, p_2, p_3};
  //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);
  
  vector<double> P = { p_1, p_2, p_3 };
  
  for (int j=0;j<=(n_base-1);j++){
    
    //int type= gsl_ran_discrete (r_c, g) + 1;
    int type = edf_sample(P, rng) + 1;
    
    switch(nt_past_forward.at(j)){
    
    case 1:{ // an A
      
      switch(type){
    case 1:{
      seq_proposed.at(j) = nt_past_forward.at(j);
      n_1 = n_1 + 1;
      break;
    }
    case 2:{
      seq_proposed.at(j) = 2;
      n_2 = n_2 + 1;
      break;
    }
    case 3:{
      n_3 = n_3 + 1;
      
      //int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion
      int type_trans = runif_int(0, 1, rng_arg);//  uniformly drawn from [0,2) to determine the exact type of transversion
      
      switch(type_trans){
      case 0:{
        seq_proposed.at(j) = 3;
        break;
      }
      case 1:{
        seq_proposed.at(j) = 4;
        break;
      }
      }
      
      break;
    }
    }
      break;
    }
      
    case 2:{ // a G
      switch(type){
    case 1:{
      seq_proposed.at(j) = nt_past_forward.at(j);
      n_1 = n_1 + 1;
      break;
    }
    case 2:{
      seq_proposed.at(j) = 1;
      n_2 = n_2 + 1;
      break;
    }
    case 3:{
      n_3 = n_3 + 1;
      
      int type_trans = runif_int(0, 1, rng_arg);//  uniformly drawn from [0,2) to determine the exact type of transversion
      
      switch(type_trans){
      case 0:{
        seq_proposed.at(j) = 3;
        break;
      }
      case 1:{
        seq_proposed.at(j) = 4;
        break;
      }
      }
      
      break;
    }
    }
      break;
    }
      
    case 3:{ // a T
      switch(type){
    case 1:{
      seq_proposed.at(j) = nt_past_forward.at(j);
      n_1 = n_1 + 1;
      break;
    }
    case 2:{
      seq_proposed.at(j) = 4;
      n_2 = n_2 + 1;
      break;
    }
    case 3:{
      n_3 = n_3 + 1;
      
      int type_trans = runif_int(0, 1, rng_arg);//  uniformly drawn from [0,2) to determine the exact type of transversion
      
      switch(type_trans){
      case 0:{
        seq_proposed.at(j) = 1;
        break;
      }
      case 1:{
        seq_proposed.at(j) = 2;
        break;
      }
      }
      
      break;
    }
    }
      break;
    }
      
    case 4:{ // a C
      switch(type){
    case 1:{
      seq_proposed.at(j) = nt_past_forward.at(j);
      n_1 = n_1 + 1;
      break;
    }
    case 2:{
      seq_proposed.at(j) = 3;
      n_2 = n_2 + 1;
      break;
    }
    case 3:{
      n_3 = n_3 + 1;
      
      int type_trans = runif_int(0, 1, rng_arg);//  uniformly drawn from [0,2) to determine the exact type of transversion
      
      switch(type_trans){
      case 0:{
        seq_proposed.at(j) = 1;
        break;
      }
      case 1:{
        seq_proposed.at(j) = 2;
        break;
      }
      }
      
      break;
    }
    }
      break;
    }
      
    }
    
  }
  
  
  //log_pr_forward = log((long double)pow(p_1, n_1)) + log((long double)pow(p_2, n_2)) + log((long double)pow(p_3, n_3));
  log_pr_forward = n_1*log(p_1) +n_2*log(p_2) + n_3*log(p_3);
  
  
}
/*-------------------------------------------*/

void seq_backward_pr_cond(const vector<int>& seq_proposed_backward, double& log_pr_backward, const vector<int>& nt_past_backward, const vector<int>& nt_future_backward, const double& t_proposed_backward, const double& t_past_backward,  const double& t_future_backward, const double& mu_1, const double& mu_2, int n_base){
  
  double dt = fabs(t_proposed_backward - t_past_backward);
  double T =  fabs(t_future_backward - t_past_backward);
  
  double p = dt/T;
  
  int m=0; // number of sites from nt_past were different from the corresponding sites from nt_future
  int dn=0; // number of sites from nt_past become the same as the coresponding sites from nt_future among m
  
  for (int i=0; i<=(n_base-1); i++){
    
    switch((nt_past_backward.at(i)==seq_proposed_backward.at(i)) & (seq_proposed_backward.at(i)==nt_future_backward.at(i))){
    
    case 0:{// not three bases are the same
    m = m +1;
    
    switch(int (seq_proposed_backward.at(i)==nt_future_backward.at(i))){
    case 1:{// it is the same as future base but not the same as past (i.e.,a change)
      dn = dn + 1;
      break;
    }
    case 0:{
      //do nothing
      break;
    }
    }
    
    break;
  }
      
    case 1:{
      // do nothing
      break;
    }
      
    }
  }
  
  //log_pr_backward = log((long double)pow(p, dn)) + log((long double)pow(1.0-p, m-dn));
  log_pr_backward = dn*log(p) + (m-dn)*log(1.0-p);
  
  
}
/*-------------------------------------------*/

void seq_backward_pr_uncond(const vector<int>& seq_proposed_backward,  double& log_pr_backward, const vector<int>& nt_past_backward, const double& t_proposed_backward, const double& t_past_backward, const double& t_future_backward,  const double& mu_1, const double& mu_2, int n_base){
  
  
  int count_1, count_2, count_3;
  
  count_1=count_2=count_3=0;
  
  double p_1, p_2, p_3;
  p_1 = p_2 = p_3 = 0.0;
  
  double abs_dt = fabs(t_future_backward - t_past_backward);
  //double abs_dt = fabs(t_proposed_backward - t_past_backward);
  
  
  if (opt_k80 == 0) {
    p_1 = 0.25 + 0.25*exp(-4.0*mu_2*abs_dt) + 0.5*exp(-2.0*(mu_1 + mu_2)*abs_dt); // pr of a base not changing
    p_2 = 0.25 + 0.25*exp(-4.0*mu_2*abs_dt) - 0.5*exp(-2.0*(mu_1 + mu_2)*abs_dt); // pr of a transition of a base
    p_3 = 2.0*(0.25 - 0.25*exp(-4.0*mu_2*abs_dt));  // pr of a transversion (two possible events)
  }
  if (opt_k80 == 1) {
    //K80: mu1 = alpha; mu2 = beta
    p_2 = 0.25 - 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // P = pts
    p_3 = 0.5 - 0.5*exp(-8.0*mu_2*abs_dt);  // Q = ptv (2 options)
    p_1 = 1 - p_2 - p_3; // R = no change
    //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
  }
  
  
  for ( int i=0;i<=(n_base-1); i++){
    
    switch(abs(nt_past_backward.at(i)-seq_proposed_backward.at(i))){
    case 0:{
    count_1 = count_1 + 1;
    break;
  }
    case 1:{
      switch( ((nt_past_backward.at(i)==2) & (seq_proposed_backward.at(i)==3)) | ((nt_past_backward.at(i)==3) & (seq_proposed_backward.at(i)==2)) ){
    case 1:{
      count_3 = count_3 + 1;
      break;
    }
    case 0:{
      count_2 = count_2 + 1;
      break;
    }
    }
      break;
    }
    case 2:{
      count_3 = count_3 + 1;
      break;
    }
    case 3:{
      count_3 = count_3 + 1;
      break;
    }
    }
    
  }
  
  //log_pr_backward = log((long double)pow(p_1, count_1)) + log((long double)pow(p_2, count_2)) + log((long double)pow(p_3, count_3));
  log_pr_backward = count_1*log(p_1) + count_2*log(p_2) + count_3*log(p_3) ;
  
  
  
}

double log_lh_base (int& base_1, int& base_2, double t_1_arg, double t_2_arg , double mu_1_arg, double mu_2_arg){
  
  double log_lh=-99.0;;
  
  double dt = t_2_arg - t_1_arg;
  
  
  double p_1, p_2, p_3;
  p_1 = p_2 = p_3 = 0.0;
  
  if (opt_k80 == 0) {
    p_1 = 0.25 + 0.25*exp(-4.0*mu_2_arg*dt) + 0.5*exp(-2.0*(mu_1_arg + mu_2_arg)*dt); // pr of a base not changing
    p_2 = 0.25 + 0.25*exp(-4.0*mu_2_arg*dt) - 0.5*exp(-2.0*(mu_1_arg + mu_2_arg)*dt); // pr of a transition of a base
    p_3 = 1.0*(0.25 - 0.25*exp(-4.0*mu_2_arg*dt));  // pr of a transversion (two possible events)
    //double p_1 = 1.0 - p_2 - 2.0*p_3; // pr of a base not changing
  }
  if (opt_k80 == 1) {
    //K80: mu1 = alpha; mu2 = beta
    p_2 = 0.25 - 0.5*exp(-4.0*(mu_1_arg + mu_2_arg)*dt) + 0.25*exp(-8.0*mu_2_arg*dt); // P = pts
    p_3 = 0.5 - 0.5*exp(-8.0*mu_2_arg*dt);  // Q = ptv (2 options)
    p_1 = 1 - p_2 - p_3; // R = no change
    //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
  }
  
  
  switch(abs(base_1-base_2)){
  case 0:{
    log_lh = log(p_1);
    break;
  }
  case 1:{
    switch( ((base_1==2) & (base_2==3)) | ((base_1==3) & (base_2==2)) ){
  case 1:{
    log_lh = log(p_3);
    break;
  }
  case 0:{
    log_lh = log(p_2);
    break;
  }
  }
    break;
  }
  case 2:{
    log_lh = log(p_3);
    break;
  }
  case 3:{
    log_lh = log(p_3);
    break;
  }
  }
  
  
  return(log_lh);
  
}

/*-------------------------------------------*/

double log_lh_seq (vector<int>& seq_1_arg, vector<int>& seq_2_arg, double t_1_arg, double t_2_arg , double mu_1_arg, double mu_2_arg, int n_base_arg){
  
  double dt = t_2_arg - t_1_arg;
  
  
  int count_1, count_2, count_3; // count_1=count of unchanged sites, ..transition.., transversion
  count_1=count_2=count_3=0;
  
  //vector<int> type(n_base_arg);
  
  
  for ( int i=0;i<=(n_base_arg-1); i++){
    
    switch(abs(seq_1_arg.at(i)-seq_2_arg.at(i))){
    case 0:{
    count_1 = count_1 + 1;
    //type.at(i) = 1;
    break;
  }
    case 1:{
      switch( ((seq_1_arg.at(i)==2) & (seq_2_arg.at(i)==3)) | ((seq_1_arg.at(i)==3) & (seq_2_arg.at(i)==2)) ){
    case 1:{
      count_3 = count_3 + 1;
      //type.at(i) = 3;
      break;
    }
    case 0:{
      count_2 = count_2 + 1;
      //type.at(i) = 2;
      break;
    }
    }
      break;
    }
    case 2:{
      count_3 = count_3 + 1;
      //type.at(i) = 3;
      break;
    }
    case 3:{
      count_3 = count_3 + 1;
      //type.at(i) = 3;
      break;
    }
    }
    
  }
  
  
  double p_1, p_2, p_3;
  p_1 = p_2 = p_3 = 0.0;
  
  if (opt_k80 == 0) {
    p_1 = 0.25 + 0.25*exp(-4.0*mu_2_arg*dt) + 0.5*exp(-2.0*(mu_1_arg + mu_2_arg)*dt); // pr of a base not changing
    p_2 = 0.25 + 0.25*exp(-4.0*mu_2_arg*dt) - 0.5*exp(-2.0*(mu_1_arg + mu_2_arg)*dt); // pr of a transition of a base
    p_3 = 1.0*(0.25 - 0.25*exp(-4.0*mu_2_arg*dt));  // pr of a transversion (two possible events)
  }
  if (opt_k80 == 1) {
    //K80: mu1 = alpha; mu2 = beta
    p_2 = 0.25 - 0.5*exp(-4.0*(mu_1_arg + mu_2_arg)*dt) + 0.25*exp(-8.0*mu_2_arg*dt); // P = pts
    p_3 = 0.5 - 0.5*exp(-8.0*mu_2_arg*dt);  // Q = ptv (2 options)
    p_1 = 1 - p_2 - p_3; // R = no change
    //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
  }
  
  
  double log_lh = count_1*log(p_1) + count_2*log(p_2) + count_3*log(p_3);
  
  
  // double P[3] = {p_1, p_2, 2.0*p_3};
  // unsigned int C[3] = {count_1, count_2, count_3};
  // double log_lh = gsl_ran_multinomial_lnpdf (3, P, C);
  
  
  // 		int total = count_1 + count_2 + count_3;
  // 		double p_total = p_1 + p_2 + 2.0*p_3;
  // 		ofstream myfile_out;
  // 		myfile_out.open((string(PATH2)+string("lh_seq.csv")).c_str(),ios::app);
  // 		myfile_out <<log_lh<< ","<< total << "," << p_total << "," << count_1 <<"," << count_2 << "," << count_3 << "," << p_1 <<"," << p_2 << "," << 2.0*p_3 << endl;
  // 		myfile_out.close();
  
  
  //return(lh);
  return(log_lh);
  
}

/*-------------------------------------------*/
/*
 double dtnorm(double x, double mean, double sd, double a){ // pdf of truncated normal with lower bound = a; upper bound =Inf
 
 double num = (1.0/sd)*gsl_ran_ugaussian_pdf((x-mean)/sd);
 
 double denom = 1.0 - gsl_cdf_ugaussian_P((a-mean)/sd);
 
 double d = num/denom;
 
 return(d);
 
 }
 */

/*-------------------------------------------*/
long double func_latent_pdf(double t , double lat_mu, double lat_var){
  
  long double func_lat_pdf = 0.0;
  
  //sf edit
  if (opt_latgamma == 0) {
    //func_lat_pdf = gsl_ran_gamma_pdf(t, lat_mu, lat_var);
    func_lat_pdf = pdf(gamma_mdist(lat_mu, lat_var), t);
    if (func_lat_pdf == 0) {
      double dbl_min = std::numeric_limits< double >::min();
      func_lat_pdf = dbl_min;
    }
  }
  if (opt_latgamma == 1) {
    double a_lat = lat_mu*lat_mu / lat_var; // when use GAMMA latent
    double b_lat = lat_var / lat_mu;
    //func_lat_pdf = gsl_ran_gamma_pdf(t, a_lat, b_lat);
    func_lat_pdf = pdf(gamma_mdist(a_lat, b_lat), t);
    if (func_lat_pdf == 0) {
      double dbl_min = std::numeric_limits< double >::min();
      func_lat_pdf = dbl_min;
    }
  }
  
  // // double a_lat = log(lat_mu*lat_mu/(sqrt(lat_var +lat_mu*lat_mu))); // when use lognormal latent
  // // double b_lat = sqrt(log(lat_var/(lat_mu*lat_mu)+1));
  // // func_lat_pdf = gsl_ran_lognormal_pdf(t, a_lat, b_lat);
  
  // // double a_lat = lat_mu; // when use exponential latent
  // //func_lat_pdf = gsl_ran_exponential_pdf(t, a_lat);
  
  //func_lat_pdf = gsl_ran_exponential_pdf(t, lat_mu);
  
  
  return(func_lat_pdf);
}
/*-------------------------------------------*/

inline long double func_latent_surv(double t , double lat_mu, double lat_var){
  
  long double func_lat_surv = 0.0;
  
  //sf edit
  if (opt_latgamma == 0) {
    //func_lat_cdf = gsl_cdf_gamma_P(t, lat_mu, lat_var);
    func_lat_surv = 1 - cdf(gamma_mdist(lat_mu, lat_var), t);
    
    if (func_lat_surv == 0) {
      double dbl_min = std::numeric_limits< double >::min();
      func_lat_surv = dbl_min;
    }
  }
  if (opt_latgamma == 1) {
    double a_lat = lat_mu * lat_mu / lat_var; // when use GAMMA latent
    double b_lat = lat_var / lat_mu;
    //func_lat_cdf = gsl_cdf_gamma_P(t, a_lat, b_lat);
    func_lat_surv = 1 - cdf(gamma_mdist(a_lat, b_lat), t);
    
    if (func_lat_surv == 0) {
      double dbl_m = std::numeric_limits< double >::min();
      func_lat_surv = dbl_m;
    }
  }
  
  // // double a_lat = log(lat_mu*lat_mu/(sqrt(lat_var +lat_mu*lat_mu))); // when use lognormal latent
  // // double b_lat = sqrt(log(lat_var/(lat_mu*lat_mu)+1));
  // // func_lat_cdf = gsl_cdf_lognormal_P(t, a_lat, b_lat);
  
  // // double a_lat = lat_mu; // when use exponential latent
  // // func_lat_cdf = gsl_cdf_exponential_P(t, a_lat);
  
  //func_lat_cdf = gsl_cdf_exponential_P(t, lat_mu);
  
  
  return(func_lat_surv);
}
/*-------------------------------------------*/

double log_lh_func (lh_SQUARE lh_square_arg, int n_arg) {
  
  double log_lh_value =0.0;
  
  for (int i=0; i<=(n_arg-1);i++){
    log_lh_value = log_lh_value +  lh_square_arg.log_f_Snull.at(i) +  lh_square_arg.log_f_S.at(i) + log(lh_square_arg.f_U.at(i)) +log(lh_square_arg.f_E.at(i)) + log(lh_square_arg.f_I.at(i)) +log(lh_square_arg.f_R.at(i)) + log(lh_square_arg.f_EnI.at(i)) + log(lh_square_arg.f_InR.at(i));
    //if (debug == 1) {
    //cout << log_lh_value << ", " << lh_square_arg.log_f_Snull.at(i) << ", " << lh_square_arg.log_f_S.at(i) << ", " << log(lh_square_arg.f_U.at(i)) << ", " << log(lh_square_arg.f_E.at(i)) << ", " << log(lh_square_arg.f_I.at(i)) << ", " << log(lh_square_arg.f_R.at(i)) << ", " << log(lh_square_arg.f_EnI.at(i)) << ", " << log(lh_square_arg.f_InR.at(i)) << endl;
    //cout << "test n: " << n_arg << ", i+1:" << i + 1 << ", lh:" << log_lh_value << endl;
    //cin.get();
    //}
  }
  
  return(log_lh_value);
  
}


/*------------------------------------------------*/
void FUNC::initialize_kernel_mat (vector< vector<double> >& kernel_mat_arg, vector<double>& norm_const_arg) {
  
  
  for (int i=0;i<=(n_Clh-1);i++) {
    for (int j=0;j<=(n_Clh-1);j++) {
      if (i==j) kernel_mat_arg[i][j]=0.0;
      if (i<j) kernel_mat_arg[i][j] = func_kernel (coordinate_Clh[i][0],coordinate_Clh[i][1],coordinate_Clh[j][0],coordinate_Clh[j][1],k_1_Clh,kernel_type_Clh,coord_type_Clh);
      if (i>j) kernel_mat_arg[i][j]=kernel_mat_arg[j][i];
    }
  }
  
  for (int j=0;j<=(n_Clh-1);j++) {
    norm_const_arg.at(j) = 0.0;
    for (int i=0;(i<=(n_Clh-1)); i++) {
      norm_const_arg.at(j) = norm_const_arg.at(j) +  kernel_mat_arg[i][j];
    }
  }
  
}

/*------------------------------------------------*/
void FUNC::initialize_delta_mat (vector< vector<double> >& delta_mat_arg){
  
  
  if (xi_U_Clh.size()>=1){
    for (int i=0;i<= (int)(xi_U_Clh.size()-1);i++){
      
      for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++){
        
        switch (int (t_r_Clh.at(xi_I_Clh.at(j))>t_max_Clh)) {
        case 1:{ // not yet recovered
  delta_mat_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = t_max_Clh - t_i_Clh.at(xi_I_Clh.at(j));
  break;
}
        case 0:{ // recovered
          delta_mat_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] = t_r_Clh.at(xi_I_Clh.at(j)) - t_i_Clh.at(xi_I_Clh.at(j));
          break;
        }
        }
        
      }
    }
  }
  
  //----------//
  
  
  if (xi_E_minus_Clh.size()>=1){
    for (int i=0;i<= (int)(xi_E_minus_Clh.size()-1);i++){
      
      for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++){
        
        if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_minus_Clh.at(i))) {
          
          switch (int (t_r_Clh.at(xi_I_Clh.at(j))>=t_e_Clh.at(xi_E_minus_Clh.at(i)))) {
          case 1:{ // not yet recovered at e_i
    delta_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = t_e_Clh.at(xi_E_minus_Clh.at(i)) - t_i_Clh.at(xi_I_Clh.at(j));
    break;
  }
          case 0:{ // recovered before e_i
            delta_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] = t_r_Clh.at(xi_I_Clh.at(j)) - t_i_Clh.at(xi_I_Clh.at(j));
            break;
          }
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
    }
  }
  
  for (int i=0;i<= (int)(index_Clh.size()-1);i++){
    for (int j=0;j<= (int) (n_Clh-1);j++){
      delta_mat_arg[index_Clh.at(i)][j] = 0.0;
    }
  }
  
}

/*------------------------------------------------*/

void FUNC::initialize_beta_ij_mat(vector< vector<double> >& beta_ij_mat_arg, vector<double>& herdn_arg, vector<double>& ftype0_arg, vector<double>& ftype1_arg, vector<double>& ftype2_arg) {
  
  
  for (int i = 0; i <= (n_Clh - 1); i++) { //infectives
    for (int j = 0; j <= (n_Clh - 1); j++) { //susceptibles
      if (i == j) beta_ij_mat_arg[i][j] = 0.0;
      if (i != j) beta_ij_mat_arg[i][j] = func_beta_ij(herdn_arg[i], herdn_arg[j], nu_inf_Clh, tau_susc_Clh, ftype0_arg[i], ftype0_arg[j], ftype1_arg[i], ftype1_arg[j], ftype2_arg[i], ftype2_arg[j], phi_inf1_Clh, phi_inf2_Clh, rho_susc1_Clh, rho_susc2_Clh);
    }
  }
  
}

/*------------------------------------------------*/

/*
 void FUNC::initialize_beta_ij_mat_norm(vector< vector<double> >& beta_ij_mat_arg, vector<double>& beta_ij_inf_arg, vector<double>& beta_ij_susc_arg) {
 for (int i = 0; i <= (n_Clh - 1); i++) { //infectives
 for (int j = 0; j <= (n_Clh - 1); j++) { //susceptibles
 if (i == j) beta_ij_mat_arg[i][j] = 0.0;
 if (i != j) beta_ij_mat_arg[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_arg, beta_ij_susc_arg);
 }
 }
 }
 */

/*------------------------------------------------*/

void FUNC::initialize_lh_square (lh_SQUARE& lh_square_arg, vector< vector<double> > kernel_mat_arg, vector< vector<double> > delta_mat_arg, vector<double>& norm_const_arg, nt_struct& nt_data_arg, vector<int>& con_seq, vector< vector<double> > beta_ij_mat_arg, moves_struct& moves_arg, para_priors_etc& para_priors_arg, vector< vector<double> > delta_mat_mov_arg){
  
  
  if (xi_U_Clh.size()>=1){   // loop over all those that remained unexposed at t_max
    for (int i=0;i<= (int)(xi_U_Clh.size()-1);i++){
      
      for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++){
        
        double delta_t = delta_mat_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)];
        double delta_t_mov = delta_mat_mov_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)];
        //     double delta_t = 0.0;
        //     switch (t_r_Clh.at(xi_I_Clh.at(j))>t_max_Clh) {
        //     case 1:{ // not yet recovered
        //     delta_t = t_max_Clh - t_i_Clh.at(xi_I_Clh.at(j));
        //     break;
        //     }
        //     case 0:{ // recovered
        //     delta_t = t_r_Clh.at(xi_I_Clh.at(j)) - t_i_Clh.at(xi_I_Clh.at(j));
        //     break;
        //     }
        //     }
        
        if (opt_betaij == 0) {
          lh_square_arg.kt_sum_U.at(xi_U_Clh.at(i)) = lh_square_arg.kt_sum_U.at(xi_U_Clh.at(i)) + delta_t*kernel_mat_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)]/norm_const_arg.at(xi_I_Clh.at(j));
        }
        if (opt_betaij == 1) {
          lh_square_arg.kt_sum_U.at(xi_U_Clh.at(i)) = lh_square_arg.kt_sum_U.at(xi_U_Clh.at(i)) + delta_t * beta_ij_mat_arg[xi_I_Clh.at(j)][xi_U_Clh.at(i)] * kernel_mat_arg[xi_U_Clh.at(i)][xi_I_Clh.at(j)] / norm_const_arg.at(xi_I_Clh.at(j));
        }
        
        if (opt_mov == 0) {
          lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i)) = 0.0;
        }
        if (opt_mov == 1) {
          lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i)) = lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i)) = lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i)) + func_moves_cnt(xi_I_Clh.at(j), xi_U_Clh.at(i), moves_arg, t_e_Clh, t_i_Clh, t_r_Clh, para_priors_arg);
        }
        
      }
      
      lh_square_arg.q_T.at(xi_U_Clh.at(i)) = alpha_Clh * t_max_Clh + beta_Clh * lh_square_arg.kt_sum_U.at(xi_U_Clh.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_arg.q_T.at(xi_U_Clh.at(i)) = lh_square_arg.q_T.at(xi_U_Clh.at(i)) + beta_m_Clh * lh_square_arg.movest_sum_U.at(xi_U_Clh.at(i));
      }
      
      //lh_square_arg.f_U.at(xi_U_Clh.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_arg.q_T.at(xi_U_Clh.at(i)),1.0);
      lh_square_arg.f_U.at(xi_U_Clh.at(i)) = surv_exp_limit(1.0, lh_square_arg.q_T.at(xi_U_Clh.at(i)));
      
    }
  }
  
  //------------------------------------------//
  
  for (int i=0;i<= (int)(xi_E_Clh.size()-1);i++){ // loop over all that were exposed before t_max
    
    int k_E = xi_E_Clh.at(i);
    
    switch (int (nt_data_arg.current_size.at(k_E)>1)) {
    
    case 1:{
      
      
      // 	vector<int> seq_1(nt_data_arg.nt[k_E].begin(), nt_data_arg.nt[k_E].begin()+n_base_Clh);
      // 	vector<int> seq_2(nt_data_arg.nt[k_E].begin()+n_base_Clh, nt_data_arg.nt[k_E].begin()+2*n_base_Clh);
      
      for (int j=0;j<=(nt_data_arg.current_size.at(k_E)-2);j++){
      
      vector<int> seq_1(nt_data_arg.nt[k_E].begin()+j*(n_base_Clh), nt_data_arg.nt[k_E].begin()+(j+1)*(n_base_Clh));
      vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*(n_base_Clh), nt_data_arg.nt[k_E].begin()+(j+2)*(n_base_Clh));
      lh_square_arg.log_f_S.at(k_E) =lh_square_arg.log_f_S.at(k_E)+ log_lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_Clh, mu_2_Clh, n_base_Clh);
      
    }
      
      break;
    }
      
    case 0:{
      break;
    }
    }
    
    //--
    switch (int (infected_source_Clh.at(k_E)==9999)) {
    case 1:{
      // 			lh_square_arg.log_f_Snull.at(k_E) = n_base_Clh*log(0.25); // assume background infection gives a random sequence from stationary dist of seq
      vector<int> seq(nt_data_arg.nt[k_E].begin(), nt_data_arg.nt[k_E].begin()+n_base_Clh); // the first seq
      lh_square_arg.log_f_Snull.at(k_E) = lh_snull(con_seq, seq, p_ber_Clh, n_base_Clh); // compute the log pr a seq for background
      
      break;
    }
      
    case 0:{
      break;
    }
    }
    
    //--
    
  }
  
  //------------------------------------------//
  
  
  if (xi_E_minus_Clh.size()>=1){    // loop over all that were exposed before t_max (excluding the initial index)
    
    for (int i=0;i<= (int)(xi_E_minus_Clh.size()-1);i++){
      
      for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++){
        
        if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_minus_Clh.at(i))) {
          
          double delta_t = delta_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)];
          double delta_t_mov = delta_mat_mov_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)];
          // 		switch (t_r_Clh.at(xi_I_Clh.at(j))>=t_e_Clh.at(xi_E_minus_Clh.at(i))) {
          // 		case 1:{ // not yet recovered at e_i
          // 		lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) + kernel_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)]/norm_const_arg.at(xi_I_Clh.at(j)); // update k_sum_E
          // 		break;
          // 		}
          // 		case 0:{ // recovered before e_i
          // 		break;
          // 		}
          // 		}
          if (opt_betaij == 0) {
            lh_square_arg.kt_sum_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.kt_sum_E.at(xi_E_minus_Clh.at(i)) + delta_t * kernel_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] / norm_const_arg.at(xi_I_Clh.at(j)); // update kt_sum_E
          }
          if (opt_betaij == 1) {
            lh_square_arg.kt_sum_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.kt_sum_E.at(xi_E_minus_Clh.at(i)) + delta_t * beta_ij_mat_arg[xi_I_Clh.at(j)][xi_E_minus_Clh.at(i)] * kernel_mat_arg[xi_E_minus_Clh.at(i)][xi_I_Clh.at(j)] / norm_const_arg.at(xi_I_Clh.at(j)); // update kt_sum_E
          }
          
          if (opt_mov == 0) {
            lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i)) = 0;
          }
          
          if (opt_mov == 1) {
            lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i)) + delta_t_mov;
          }
          
          if (opt_mov == 2) {
            lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i)) + func_moves_cnt(xi_I_Clh.at(j), xi_E_minus_Clh.at(i), moves_arg, t_e_Clh, t_i_Clh, t_r_Clh, para_priors_arg);
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      // lh_square_arg.g_E.at(xi_E_minus_Clh.at(i)) = alpha_Clh + beta_Clh*lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i));
      
      //Paper equation (4)
      switch(infected_source_Clh.at(xi_E_minus_Clh.at(i))){
      case 9999:{ // by background
        lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) = 0.0; // update k_sum_E
        lh_square_arg.g_E.at(xi_E_minus_Clh.at(i)) = alpha_Clh;
        lh_square_arg.moves_sum_E.at(xi_E_minus_Clh.at(i)) = 0.0;
        break;
      }
        
      default :{ // not by background
        
        if (opt_betaij == 0) {
        lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) = kernel_mat_arg[xi_E_minus_Clh.at(i)][infected_source_Clh.at(xi_E_minus_Clh.at(i))] / norm_const_arg.at(infected_source_Clh.at(xi_E_minus_Clh.at(i))); // update k_sum_E
      }
        if (opt_betaij == 1) {
          lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) = beta_ij_mat_arg[infected_source_Clh.at(xi_E_minus_Clh.at(i))][xi_E_minus_Clh.at(i)] * kernel_mat_arg[xi_E_minus_Clh.at(i)][infected_source_Clh.at(xi_E_minus_Clh.at(i))] / norm_const_arg.at(infected_source_Clh.at(xi_E_minus_Clh.at(i))); // update k_sum_E
        }
        
        if (opt_mov == 0) {
          lh_square_arg.moves_sum_E.at(xi_E_minus_Clh.at(i)) = 0.0;
          lh_square_arg.g_E.at(xi_E_minus_Clh.at(i)) = beta_Clh * lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i));
        }
        if ((opt_mov == 1) | (opt_mov == 2)) {
          double moves_ij_t = func_moves_cnt(infected_source_Clh.at(xi_E_minus_Clh.at(i)), xi_E_minus_Clh.at(i), moves_arg, t_e_Clh, t_i_Clh, t_r_Clh, para_priors_arg);
          lh_square_arg.moves_sum_E.at(xi_E_minus_Clh.at(i)) = moves_ij_t;
          lh_square_arg.g_E.at(xi_E_minus_Clh.at(i)) = beta_Clh * lh_square_arg.k_sum_E.at(xi_E_minus_Clh.at(i)) + beta_m_Clh * lh_square_arg.moves_sum_E.at(xi_E_minus_Clh.at(i));
        }
        
        
        break;
      }
        
      }
      
      
      lh_square_arg.q_E.at(xi_E_minus_Clh.at(i)) = alpha_Clh * t_e_Clh.at(xi_E_minus_Clh.at(i)) + beta_Clh * lh_square_arg.kt_sum_E.at(xi_E_minus_Clh.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_arg.q_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.q_E.at(xi_E_minus_Clh.at(i)) + beta_m_Clh * lh_square_arg.movest_sum_E.at(xi_E_minus_Clh.at(i));
      }
      
      //lh_square_arg.h_E.at(xi_E_minus_Clh.at(i)) = gsl_ran_exponential_pdf(lh_square_arg.q_E.at(xi_E_minus_Clh.at(i)),1.0);
      lh_square_arg.h_E.at(xi_E_minus_Clh.at(i)) = pdf_exp_limit(1.0, lh_square_arg.q_E.at(xi_E_minus_Clh.at(i)));
      
      lh_square_arg.f_E.at(xi_E_minus_Clh.at(i)) = lh_square_arg.g_E.at(xi_E_minus_Clh.at(i))*lh_square_arg.h_E.at(xi_E_minus_Clh.at(i));
      
    }
    
  }
  
  //----------//
  
  if (xi_I_Clh.size()>=1){   // loop over all that were infectious at some time before t_max
    for (int i=0;i<= (int)(xi_I_Clh.size()-1);i++){
      //lh_square_arg.f_I.at(xi_I_Clh.at(i)) = gsl_ran_gamma_pdf(t_i_Clh.at(xi_I_Clh.at(i)) - t_e_Clh.at(xi_I_Clh.at(i)), a_Clh, b_Clh);
      lh_square_arg.f_I.at(xi_I_Clh.at(i)) = func_latent_pdf(t_i_Clh.at(xi_I_Clh.at(i)) - t_e_Clh.at(xi_I_Clh.at(i)), lat_mu_Clh,lat_var_Clh);
      //if (debug == 1) {
      //	cout << ": " << i << ", t_i - t_e = " << t_i_Clh.at(xi_I_Clh.at(i)) - t_e_Clh.at(xi_I_Clh.at(i)) << ", f_I = " << lh_square_arg.f_I.at(xi_I_Clh.at(i)) << endl;
      //}
    }
  }
  
  //--------//
  
  if (xi_EnI_Clh.size() >= 1) {  // loop over all that were exposed but not yet infectious by t_max
    for (int i = 0; i <= (int)(xi_EnI_Clh.size() - 1); i++) {
      //lh_square_arg.f_EnI.at(xi_EnI_Clh.at(i)) = 1.0 -  gsl_cdf_gamma_P(t_max_Clh - t_e_Clh.at(xi_EnI_Clh.at(i)), a_Clh, b_Clh);
      lh_square_arg.f_EnI.at(xi_EnI_Clh.at(i)) = func_latent_surv(t_max_Clh - t_e_Clh.at(xi_EnI_Clh.at(i)), lat_mu_Clh, lat_var_Clh);
    }
  }
  //-------//
  
  if (xi_R_Clh.size()>=1){   // loop over all that had recovered/been removed by t_max
    for (int i=0;i<= (int)(xi_R_Clh.size()-1);i++){
      
      // lh_square_arg.f_R.at(xi_R_Clh.at(i)) = gsl_ran_weibull_pdf(t_r_Clh.at(xi_R_Clh.at(i)) - t_i_Clh.at(xi_R_Clh.at(i)), c_Clh, d_Clh);
      //lh_square_arg.f_R.at(xi_R_Clh.at(i)) = pdf(weibull_mdist(d_Clh, c_Clh), t_r_Clh.at(xi_R_Clh.at(i)) - t_i_Clh.at(xi_R_Clh.at(i)));
      lh_square_arg.f_R.at(xi_R_Clh.at(i)) = pdf_weibull_limit(d_Clh, c_Clh, t_r_Clh.at(xi_R_Clh.at(i)) - t_i_Clh.at(xi_R_Clh.at(i)));
      
      //lh_square_arg.f_R.at(xi_R_Clh.at(i)) = gsl_ran_exponential_pdf(t_r_Clh.at(xi_R_Clh.at(i)) - t_i_Clh.at(xi_R_Clh.at(i)), c_Clh);
      //lh_square_arg.f_R.at(xi_R_Clh.at(i)) = pdf(exp_mdist(1/c_Clh), t_r_Clh.at(xi_R_Clh.at(i)) - t_i_Clh.at(xi_R_Clh.at(i)));
    }
  }
  
  //-------//
  
  if (xi_InR_Clh.size()>=1){   // loop over all that were infectious but not yet recovered by t_max
    for (int i=0;i<= (int)(xi_InR_Clh.size()-1);i++){
      
      //lh_square_arg.f_InR.at(xi_InR_Clh.at(i)) = 1.0 -  gsl_cdf_exponential_P(t_max_Clh - t_i_Clh.at(xi_InR_Clh.at(i)), c_Clh);
      // lh_square_arg.f_InR.at(xi_InR_Clh.at(i)) = 1.0 -  gsl_cdf_weibull_P(t_max_Clh - t_i_Clh.at(xi_InR_Clh.at(i)), c_Clh, d_Clh);
      //lh_square_arg.f_InR.at(xi_InR_Clh.at(i)) = 1.0 - cdf(exp_mdist(1/c_Clh), t_max_Clh - t_i_Clh.at(xi_InR_Clh.at(i)));
      lh_square_arg.f_InR.at(xi_InR_Clh.at(i)) = surv_weibull_limit(d_Clh, c_Clh, t_max_Clh - t_i_Clh.at(xi_R_Clh.at(i)));
    }
  }
  
  
}


/*------------------------------------------------*/

void mcmc_UPDATE::mu_1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg,  const vector<int>& xi_E_arg, para_key& para_current_arg, nt_struct& nt_data_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double mu_1_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  mu_1_proposed = para_current_arg.mu_1 + para_sf_arg.mu_1_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  // switch (mu_1_proposed<=0) {
  // case 1: {
  // mu_1_proposed = -mu_1_proposed; //reflection
  // break;
  // }
  // case 0: {
  // mu_1_proposed = mu_1_proposed;
  // break;
  // }
  // }
  
  
  //switch (mu_1_proposed<=0) {
  switch( (mu_1_proposed<=0) | (mu_1_proposed>= para_priors_arg.mu_1_hi) ){
  
  case 1: {
    mu_1_proposed = para_current_arg.mu_1;
    break;
  }
  case 0: {
    mu_1_proposed = mu_1_proposed;
    break;
  }
  }
  
  
  //----------
  
  // double mu_1_up = 0.005;
  // switch ((mu_1_proposed<=0) | (mu_1_proposed>=mu_1_up)) {
  // case 1: {
  // 	  if (mu_1_proposed<=0) mu_1_proposed = -mu_1_proposed; //reflection
  // 	  if (mu_1_proposed>=mu_1_up) mu_1_proposed = mu_1_up - (mu_1_proposed - mu_1_up); //reflection
  // break;
  // }
  // case 0: {
  // mu_1_proposed = mu_1_proposed;
  // break;
  // }
  // }
  
  //----------
  
  //mu_1_proposed = gsl_ran_flat(r_c, 0.0001, 0.004); // use a uniform prior for proposal
  
  //----------
  
  for (int i=0;i<= (int)(xi_E_arg.size()-1);i++){ // loop over all the infected
    
    int k_E = xi_E_arg.at(i);
    
    switch (int (nt_data_arg.current_size.at(k_E)>1)) {
    
    case 1:{
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(k_E); //subtract part of likelihood that would be updated below
      
      lh_square_modified.log_f_S.at(k_E) = 0.0;
      
      // 	vector<int> seq_1(nt_data_arg.nt[k_E].begin(), nt_data_arg.nt[k_E].begin()+n_base_CUPDATE);
      // 	vector<int> seq_2(nt_data_arg.nt[k_E].begin()+n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+2*n_base_CUPDATE);
      
      for (int j=0;j<=(nt_data_arg.current_size.at(k_E)-2);j++){
        
        // 		switch(j==0){
        //
        // 		case 1:{
        // 		lh_square_modified.f_S.at(k_E) =lh_square_modified.f_S.at(k_E)* lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_proposed, para_current_arg.mu_2, n_base_CUPDATE);
        // 		break;
        // 		}
        //
        // 		case 0:{
        // 		seq_1 = seq_2;
        // 		vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+2)*n_base_CUPDATE);
        // 		lh_square_modified.f_S.at(k_E)  = lh_square_modified.f_S.at(k_E)* lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_proposed, para_current_arg.mu_2, n_base_CUPDATE);
        // 		break;
        // 		}
        //
        // 		}
        
        vector<int> seq_1(nt_data_arg.nt[k_E].begin()+ j*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE);
        vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+2)*n_base_CUPDATE);
        
        lh_square_modified.log_f_S.at(k_E)  = lh_square_modified.log_f_S.at(k_E)+ log_lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_proposed, para_current_arg.mu_2, n_base_CUPDATE);
        
      }
      
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(k_E);
      
      break;
    }
      
    default:{
      break;
    }
      
    }
  }
  
  
  //boost::math::gamma_distribution <double> mu_1_prior(1,0.003);
  // boost::math::normal_distribution <double> mu_1_prior(0.002,0.0005);
  //
  // double prior_ratio = pdf(mu_1_prior, mu_1_proposed)/pdf(mu_1_prior, para_current_arg.mu_1);
  //
  // acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*prior_ratio);
  
  acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch( int(uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.mu_1= mu_1_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.mu_1 = para_current_arg.mu_1;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
  
}
/*------------------------------------------------*/


void mcmc_UPDATE::mu_2_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg,  const vector<int>& xi_E_arg, para_key& para_current_arg, nt_struct& nt_data_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double mu_2_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  mu_2_proposed = para_current_arg.mu_2 + para_sf_arg.mu_2_sf*rnorm(0.0, 1.0, rng_arg);
  
  // gsl_rng_free(r_c);
  
  // switch (mu_2_proposed<=0) {
  // case 1: {
  // mu_2_proposed = -mu_2_proposed; //reflection
  // break;
  // }
  // case 0: {
  // mu_2_proposed = mu_2_proposed;
  // break;
  // }
  // }
  
  //switch (mu_2_proposed<=0) {
  switch( (mu_2_proposed<=0) | (mu_2_proposed>=para_priors_arg.mu_2_hi) ){
  
  case 1: {
    mu_2_proposed = para_current_arg.mu_2;
    break;
  }
  case 0: {
    mu_2_proposed = mu_2_proposed;
    break;
  }
  }
  
  
  //------------------------
  
  // double mu_2_up = 0.005;
  // switch ((mu_2_proposed<=0) | (mu_2_proposed>=mu_2_up)) {
  // case 1: {
  // 	  if (mu_2_proposed<=0) mu_2_proposed = -mu_2_proposed; //reflection
  // 	  if (mu_2_proposed>=mu_2_up) mu_2_proposed = mu_2_up - (mu_2_proposed - mu_2_up); //reflection
  // break;
  // }
  // case 0: {
  // mu_2_proposed = mu_2_proposed;
  // break;
  // }
  // }
  
  //-------------------------
  
  //mu_2_proposed = gsl_ran_flat(r_c, 0.0001, 0.002); // use a uniform prior for proposal
  
  //------------------
  
  for (int i=0;i<= (int)(xi_E_arg.size()-1);i++){ // loop over all the infected
    
    int k_E = xi_E_arg.at(i);
    
    switch ( int(nt_data_arg.current_size.at(k_E)>1)) {
    
    case 1:{
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(k_E); //subtract part of likelihood that would be updated below
      
      lh_square_modified.log_f_S.at(k_E) = 0.0;
      
      // 	vector<int> seq_1(nt_data_arg.nt[k_E].begin(), nt_data_arg.nt[k_E].begin()+n_base_CUPDATE);
      // 	vector<int> seq_2(nt_data_arg.nt[k_E].begin()+n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+2*n_base_CUPDATE);
      
      for (int j=0;j<=(nt_data_arg.current_size.at(k_E)-2);j++){
        
        // 		switch(j==0){
        //
        // 		case 1:{
        // 		lh_square_modified.f_S.at(k_E) =lh_square_modified.f_S.at(k_E)* lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_proposed, para_current_arg.mu_2, n_base_CUPDATE);
        // 		break;
        // 		}
        //
        // 		case 0:{
        // 		seq_1 = seq_2;
        // 		vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+2)*n_base_CUPDATE);
        // 		lh_square_modified.f_S.at(k_E)  = lh_square_modified.f_S.at(k_E)* lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], mu_1_proposed, para_current_arg.mu_2, n_base_CUPDATE);
        // 		break;
        // 		}
        //
        // 		}
        
        vector<int> seq_1(nt_data_arg.nt[k_E].begin()+ j*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE);
        vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*n_base_CUPDATE, nt_data_arg.nt[k_E].begin()+(j+2)*n_base_CUPDATE);
        
        lh_square_modified.log_f_S.at(k_E)  = lh_square_modified.log_f_S.at(k_E)+ log_lh_seq(seq_1, seq_2, nt_data_arg.t_nt[k_E][j], nt_data_arg.t_nt[k_E][j+1], para_current_arg.mu_1, mu_2_proposed,  n_base_CUPDATE);
        
      }
      
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(k_E);
      
      break;
    }
      
    default:{
      break;
    }
      
    }
  }
  
  
  //boost::math::gamma_distribution <double> mu_2_prior(1.0,0.003);
  // boost::math::normal_distribution <double> mu_2_prior(0.0005,0.00005);
  //
  // double prior_ratio = pdf(mu_2_prior, mu_2_proposed)/pdf(mu_2_prior, para_current_arg.mu_2);
  //
  // acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*prior_ratio);
  
  
  acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch( int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.mu_2= mu_2_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.mu_2 = para_current_arg.mu_2;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
  
}
/*------------------------------------------------*/


void mcmc_UPDATE::p_ber_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg,  const vector<int>& xi_E_arg, para_key& para_current_arg, nt_struct& nt_data_arg, vector<int>& infected_source_current_arg, vector<int>& con_seq, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  long double p_ber_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  p_ber_proposed = para_current_arg.p_ber + para_sf_arg.p_ber_sf*rnorm(0.0, 1.0, rng_arg);
  
  //p_ber_proposed =  gsl_ran_beta(r_c, 1.0,10.0);
  //p_ber_proposed = gsl_ran_flat(r_c, 0.0, 1.0);
  
  switch( (p_ber_proposed<=0) | (p_ber_proposed>= para_priors_arg.p_ber_hi) ){
  
  case 1: {
    p_ber_proposed = para_current_arg.p_ber;
    break;
  }
  case 0: {
    p_ber_proposed = p_ber_proposed;
    break;
  }
  }
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  // log_prior_y = log(gsl_ran_beta_pdf(p_ber_proposed, 1.0, 10.0));
  // log_prior_x = log(gsl_ran_beta_pdf(para_current_arg.p_ber, 1.0, 10.0));
  
  
  
  //------------------------
  
  for (int i=0;i<= (int)(xi_E_arg.size()-1);i++){ // loop over all the infected
    
    int k_E = xi_E_arg.at(i);
    
    switch (int (infected_source_current_arg.at(k_E)==9999)) {
    
    case 1:{//bg infection
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_Snull.at(k_E); //subtract part of likelihood that would be updated below
      
      vector<int> seq(nt_data_arg.nt[k_E].begin(), nt_data_arg.nt[k_E].begin()+n_base_CUPDATE); // the first seq
      lh_square_modified.log_f_Snull.at(k_E) = lh_snull(con_seq, seq, p_ber_proposed, n_base_CUPDATE); // compute the log pr a seq for background
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_Snull.at(k_E);
      
      break;
    }
      
    default:{ // 2nd infection
      break;
    }
      
    }
  }
  
  
  
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg) +  (log_prior_y - log_prior_x)));
  
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.p_ber= p_ber_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.p_ber = para_current_arg.p_ber;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
  
}
/*------------------------------------------------*/

void mcmc_UPDATE::alpha_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_e_arg, const vector<double>& t_i_arg, const vector<double>& t_r_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg){
  
  // const gsl_rng_type* T_c= gsl_rng_default;  // T is pointer points to the type of generator
  // gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
  // gsl_rng_set (r_c, 1000*iter); // set a seed
  
  double alpha_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  alpha_proposed = para_current_arg.alpha + para_sf_arg.alpha_sf*rnorm(0.0, 1.0, rng_arg);
  
  // double log_alpha_proposed;
  // log_alpha_proposed = log(para_current_arg.alpha) + 0.008*gsl_ran_gaussian(r_c,1.0);
  // alpha_proposed = exp(log_alpha_proposed );
  
  
  // gsl_rng_free(r_c);
  
  //double up = 0.01;
  
  // switch (alpha_proposed<=0) {
  // 	case 1: {
  // 	alpha_proposed = -alpha_proposed; //reflection
  // 	break;
  // 	}
  // 	case 0: {
  // 	alpha_proposed = alpha_proposed;
  // 	break;
  // 	}
  // }
  
  
  //switch (alpha_proposed<=0) {
  switch ( (alpha_proposed<=0) | (alpha_proposed>= para_priors_arg.alpha_hi)) {
  
  case 1: {
    alpha_proposed = para_current_arg.alpha; //rejection
    break;
  }
  case 0: {
    alpha_proposed = alpha_proposed;
    break;
  }
  }
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(alpha_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.alpha, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, alpha_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.alpha));
  
  
  if (xi_U_arg.empty()==0){
    for (int i=0; i<=(int)(xi_U_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = alpha_proposed * t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      //lh_square_modified.f_U.at(xi_U_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_modified.q_T.at(xi_U_arg.at(i)),1.0);
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_E_minus_arg.empty()==0){
    for (int i=0; i<= (int)(xi_E_minus_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      // lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = alpha_proposed + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
      //
      
      switch(infected_source_arg.at(xi_E_minus_arg.at(i))){
      case 9999:{ // by background
        lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) ; // this unchanged as long as infectious source unchanged
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = alpha_proposed;
        break;
      }
        
        
      default :{ // not by background
        lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) ; // this unchanged as long as infectious source unchanged
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) ;
        break;
      }
        
      }
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = alpha_proposed * t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      
      
      //lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(xi_E_minus_arg.at(i)),1.0);
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))) ;
    }
  }
  
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg) + (log_prior_y-log_prior_x)));
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*(para_current_arg.alpha/alpha_proposed));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.alpha = alpha_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.alpha = para_current_arg.alpha;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::beta_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<double>& t_e_arg, const vector<double>& t_i_arg, const vector<double>& t_r_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg){
  
  double beta_proposed = 0.0;
  double acp_pr = 0.0;
  
  // double log_prior_x =0.0;
  // double log_prior_y =0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  beta_proposed = para_current_arg.beta + para_sf_arg.beta_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  //switch (beta_proposed<=0) {
  switch ((beta_proposed<=0) | (beta_proposed>= para_priors_arg.beta_hi)) {
  
  case 1: {
    beta_proposed =para_current_arg.beta;
    break;
  }
  case 0: {
    beta_proposed = beta_proposed;
    break;
  }
  }
  
  
  
  if (xi_U_arg.empty()==0){
    for (int i=0; i<=(int)(xi_U_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + beta_proposed*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      //lh_square_modified.f_U.at(xi_U_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_modified.q_T.at(xi_U_arg.at(i)),1.0);
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_E_minus_arg.empty()==0){
    for (int i=0; i<= (int)(xi_E_minus_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      // lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + beta_proposed*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
      
      switch(infected_source_arg.at(xi_E_minus_arg.at(i))){
      case 9999:{ // by background
        //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) ; // this unchanged as long as infectious source unchanged
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) ;
        break;
      }
        
      default :{ // not by background
        //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) ; // this unchanged as long as infectious source unchanged
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) =  beta_proposed*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        if ((opt_mov == 1) | (opt_mov == 2)) {
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
        }
        
        break;
      }
        
      }
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + beta_proposed*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      //lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(xi_E_minus_arg.at(i)),1.0);
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))) ;
    }
  }
  
  
  // boost::math::gamma_distribution <double> beta_prior(10,1);
  // double prior_ratio = pdf(beta_prior, beta_proposed)/pdf(beta_prior, para_current_arg.beta);
  // acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg))*prior_ratio);
  
  
  acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.beta = beta_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.beta = para_current_arg.beta;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}


/*------------------------------------------------*/

void mcmc_UPDATE::beta_m_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, const vector<int>& infected_source_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg) {
  
  double beta_m_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  
  beta_m_proposed = para_current_arg.beta_m + para_sf_arg.beta_m_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((beta_m_proposed <= 0) | (beta_m_proposed >= para_priors_arg.beta_m_hi)) {
  
  case 1: {
    beta_m_proposed = para_current_arg.beta_m; //rejection
    break;
  }
  case 0: {
    beta_m_proposed = beta_m_proposed;
    break;
  }
  }
  
  
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  
  //log_prior_y = log(gsl_ran_exponential_pdf(beta_m_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.beta_m, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, beta_m_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.beta_m));
  
  
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_current_arg.kt_sum_U.at(xi_U_arg.at(i)) + beta_m_proposed * lh_square_current_arg.movest_sum_U.at(xi_U_arg.at(i));
      //lh_square_modified.f_U.at(xi_U_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_modified.q_T.at(xi_U_arg.at(i)), 1.0);
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      
      switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
      case 9999: { // by background
        //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
        break;
      }
        
        
      default: { // not by background
        
        lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_current_arg.k_sum_E.at(xi_E_minus_arg.at(i)) + beta_m_proposed * lh_square_current_arg.moves_sum_E.at(xi_E_minus_arg.at(i));
        break;
      }
        
      }
      
      
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_current_arg.kt_sum_E.at(xi_E_minus_arg.at(i)) + beta_m_proposed * lh_square_current_arg.movest_sum_E.at(xi_E_minus_arg.at(i));
      //lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(xi_E_minus_arg.at(i)), 1.0);
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.beta_m = beta_m_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.beta_m = para_current_arg.beta_m;
    break;
    
  }
  
  
  
}

/*------------------------------------------------*/
void mcmc_UPDATE::lat_mu_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_I_arg, const vector<int>& xi_EnI_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double lat_mu_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  lat_mu_proposed = para_current_arg.lat_mu + para_sf_arg.lat_mu_sf*rnorm(0.0, 1.0, rng_arg);
  //lat_mu_proposed = para_current_arg.lat_mu + gsl_ran_flat(r_c,-0.05,0.05);
  
  // double log_lat_mu_proposed;
  // log_lat_mu_proposed = log(para_current_arg.lat_mu) + 0.1*gsl_ran_gaussian(r_c,1.0);
  // lat_mu_proposed = exp(log_lat_mu_proposed );
  
  
  switch ((lat_mu_proposed<=0) | (lat_mu_proposed>= para_priors_arg.lat_mu_hi)) {
  
  case 1: {
    lat_mu_proposed = para_current_arg.lat_mu;
    break;
  }
  case 0: {
    lat_mu_proposed = lat_mu_proposed;
    break;
  }
  }
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(lat_mu_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.lat_mu, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, lat_mu_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.lat_mu));
  
  // double mu_up = 10.0;
  //
  // switch ((lat_mu_proposed<=0) | (lat_mu_proposed>=mu_up)) {
  // 	case 1: {
  //
  // 		switch (lat_mu_proposed<=0) {
  // 		case 1: {
  // 			lat_mu_proposed = -lat_mu_proposed; //reflection
  // 		break;
  // 		}
  // 		case 0: {
  // 			lat_mu_proposed = mu_up - (lat_mu_proposed - mu_up); //reflection
  // 		break;
  // 		}
  // 		}
  // 	break;
  // 	}
  //
  // 	case 0: {
  // 		lat_mu_proposed = lat_mu_proposed;
  // 	break;
  // 	}
  // 	}
  
  
  if (xi_I_arg.empty()==0){
    for (int i=0; i<=(int)(xi_I_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_I.at(xi_I_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.f_I.at(xi_I_arg.at(i)) = func_latent_pdf(t_i_arg.at(xi_I_arg.at(i)) - t_e_arg.at(xi_I_arg.at(i)), lat_mu_proposed, para_current_arg.lat_var);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_I.at(xi_I_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_EnI_arg.empty()==0){
    for (int i=0; i<=(int)(xi_EnI_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_EnI.at(xi_EnI_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.f_EnI.at(xi_EnI_arg.at(i)) = func_latent_surv(t_max_CUPDATE - t_e_arg.at(xi_EnI_arg.at(i)), lat_mu_proposed, para_current_arg.lat_var);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_EnI.at(xi_EnI_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.lat_mu = lat_mu_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.lat_mu = para_current_arg.lat_mu;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::lat_var_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_I_arg, const vector<int>& xi_EnI_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double lat_var_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  lat_var_proposed = para_current_arg.lat_var + para_sf_arg.lat_var_sf*rnorm(0.0, 1.0, rng_arg);
  //lat_var_proposed = para_current_arg.lat_var + gsl_ran_flat(r_c,-0.03,0.03);
  
  
  // double log_lat_var_proposed;
  // log_lat_var_proposed = log(para_current_arg.lat_var) + 0.1*gsl_ran_gaussian(r_c,1.0);
  // lat_var_proposed = exp(log_lat_var_proposed);
  
  // switch (lat_var_proposed<=0.0) {
  // case 1: {
  // lat_var_proposed = -lat_var_proposed; //reflection
  // break;
  // }
  // case 0: {
  // 	lat_var_proposed = lat_var_proposed;
  // break;
  // }
  // }
  
  //switch (lat_var_proposed<=0.1) {
  switch ((lat_var_proposed<= para_priors_arg.lat_var_lo) | (lat_var_proposed>= para_priors_arg.lat_var_hi)) {
  
  case 1: {
    lat_var_proposed = para_current_arg.lat_var ;
    break;
  }
  case 0: {
    lat_var_proposed = lat_var_proposed;
    break;
  }
  }
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(lat_var_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.lat_var, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, lat_var_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.lat_var));
  
  
  // double var_up = 10.0;
  //
  // switch ((lat_var_proposed<=0) | (lat_var_proposed>=var_up)) {
  // 	case 1: {
  //
  // 		switch (lat_var_proposed<=0) {
  // 		case 1: {
  // 			lat_var_proposed = -lat_var_proposed; //reflection
  // 		break;
  // 		}
  // 		case 0: {
  // 			lat_var_proposed = var_up - (lat_var_proposed - var_up); //reflection
  // 		break;
  // 		}
  // 		}
  // 	break;
  // 	}
  //
  // 	case 0: {
  // 		lat_var_proposed = lat_var_proposed;
  // 	break;
  // 	}
  // 	}
  
  
  if (xi_I_arg.empty()==0){
    for (int i=0; i<=(int)(xi_I_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_I.at(xi_I_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.f_I.at(xi_I_arg.at(i)) = func_latent_pdf(t_i_arg.at(xi_I_arg.at(i)) - t_e_arg.at(xi_I_arg.at(i)), para_current_arg.lat_mu, lat_var_proposed);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_I.at(xi_I_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_EnI_arg.empty()==0){
    for (int i=0; i<=(int)(xi_EnI_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_EnI.at(xi_EnI_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.f_EnI.at(xi_EnI_arg.at(i)) = func_latent_surv(t_max_CUPDATE - t_e_arg.at(xi_EnI_arg.at(i)), para_current_arg.lat_mu, lat_var_proposed);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_EnI.at(xi_EnI_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch((uniform_rv<=acp_pr) & (std::isnan(log_lh_modified)==0)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.lat_var = lat_var_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.lat_var = para_current_arg.lat_var;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::c_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_R_arg, const vector<int>& xi_InR_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<int>& index_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double c_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  c_proposed = para_current_arg.c + para_sf_arg.c_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((c_proposed<=0 ) | (c_proposed>= para_priors_arg.c_hi)) {
  
  case 1: {
    c_proposed =para_current_arg.c;
    break;
  }
  case 0: {
    c_proposed = c_proposed;
    break;
  }
  }
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(c_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.c, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, c_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.c));
  
  
  if (xi_R_arg.empty()==0){
    for (int i=0; i<=(int)(xi_R_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_R.at(xi_R_arg.at(i))); //subtract part of likelihood that would be updated below
      
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = gsl_ran_exponential_pdf(t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)), c_proposed);
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = pdf(exp_mdist(1/c_proposed), t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)));
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = gsl_ran_weibull_pdf(t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)), c_proposed, para_current_arg.d);
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = pdf(weibull_mdist(para_current_arg.d, c_proposed), t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)));
      lh_square_modified.f_R.at(xi_R_arg.at(i)) = pdf_weibull_limit(para_current_arg.d, c_proposed, t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_R.at(xi_R_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_InR_arg.empty()==0){
    for (int i=0; i<=(int)(xi_InR_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_InR.at(xi_InR_arg.at(i))); //subtract part of likelihood that would be updated below
      
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)), c_proposed);
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - cdf(exp_mdist(1/c_proposed), t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)));
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - gsl_cdf_weibull_P(t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)), c_proposed, para_current_arg.d);
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - (cdf(weibull_mdist(para_current_arg.d, c_proposed), t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i))));
      lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = surv_weibull_limit(para_current_arg.d, c_proposed, t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_InR.at(xi_InR_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.c = c_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.c = para_current_arg.c;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/
void mcmc_UPDATE::d_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector<int>& xi_R_arg, const vector<int>& xi_InR_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<int>& index_arg, para_key& para_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg) {
  
  double d_proposed = 0.0;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  
  
  d_proposed = para_current_arg.d + para_sf_arg.d_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((d_proposed <= 0) | (d_proposed >= para_priors_arg.d_hi)) {
  
  case 1: {
    d_proposed = para_current_arg.d;
    break;
  }
  case 0: {
    d_proposed = d_proposed;
    break;
  }
  }
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(d_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.d, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, d_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.d));
  
  
  if (xi_R_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_R_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_R.at(xi_R_arg.at(i))); //subtract part of likelihood that would be updated below
      
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = gsl_ran_weibull_pdf(t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)), para_current_arg.c, d_proposed);
      //lh_square_modified.f_R.at(xi_R_arg.at(i)) = pdf(weibull_mdist(d_proposed, para_current_arg.c), t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)));
      lh_square_modified.f_R.at(xi_R_arg.at(i)) = pdf_weibull_limit(d_proposed, para_current_arg.c, t_r_arg.at(xi_R_arg.at(i)) - t_i_arg.at(xi_R_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_R.at(xi_R_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  if (xi_InR_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_InR_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_InR.at(xi_InR_arg.at(i))); //subtract part of likelihood that would be updated below
      
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - gsl_cdf_weibull_P(t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)), c_proposed, para_current_arg.d);
      //lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = 1.0 - (cdf(weibull_mdist(d_proposed, para_current_arg.c), t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i))));
      lh_square_modified.f_InR.at(xi_InR_arg.at(i)) = surv_weibull_limit(d_proposed, para_current_arg.c, t_max_CUPDATE - t_i_arg.at(xi_InR_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_InR.at(xi_InR_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.d = d_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.d = para_current_arg.d;
    break;
    
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::k_1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg,  const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg){
  
  double k_1_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  vector< vector<double> > kernel_mat_modified = kernel_mat_current_arg;
  vector<double> norm_const_modified = norm_const_current_arg;
  
  
  k_1_proposed = para_current_arg.k_1 + para_sf_arg.k_1_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((k_1_proposed<= para_priors_arg.k_1_lo) | (k_1_proposed>= para_priors_arg.k_1_hi)) {
  
  case 1: {
    k_1_proposed = para_current_arg.k_1;
    break;
  }
  case 0: {
    k_1_proposed = k_1_proposed;
    break;
  }
  }
  
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(k_1_proposed, 1.0/ para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.k_1, 1.0/ para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, k_1_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.k_1));
  
  
  //----------
  for (int i=0;i<=(n_CUPDATE-1);i++) {
    for (int j=0;j<=(n_CUPDATE-1);j++) {
      if (i==j) kernel_mat_modified[i][j]=0.0;
      if (i<j) kernel_mat_modified[i][j] = func_kernel (coordinate_CUPDATE[i][0],coordinate_CUPDATE[i][1],coordinate_CUPDATE[j][0],coordinate_CUPDATE[j][1],k_1_proposed,kernel_type_CUPDATE,coord_type_CUPDATE);
      if (i>j) kernel_mat_modified[i][j]=kernel_mat_modified[j][i];
    }
  }
  
  for (int j=0;j<=(n_CUPDATE-1);j++) {
    norm_const_modified.at(j)=0.0;
    for (int i=0;(i<=(n_CUPDATE-1));i++) {
      norm_const_modified.at(j)= norm_const_modified.at(j) + kernel_mat_modified[i][j];
    }
  }
  
  //----------
  
  if (xi_U_arg.empty()==0){
    for (int i=0;i<= (int)(xi_U_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i))  = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j=0;j<= (int) (xi_I_arg.size()-1);j++){
        
        //     double delta_t = 0.0;
        //     switch (t_r_arg.at(xi_I_arg.at(j))>t_max_CUPDATE) {
        //     case 1:{ // not yet recovered
        //     delta_t = t_max_CUPDATE - t_i_arg.at(xi_I_arg.at(j));
        //     break;
        //     }
        //     case 0:{ // recovered
        //     delta_t = t_r_arg.at(xi_I_arg.at(j)) - t_i_arg.at(xi_I_arg.at(j));
        //     break;
        //     }
        //     }
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        if (opt_betaij == 0) {
          lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * kernel_mat_modified[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_modified.at(xi_I_arg.at(j));
        }
        if (opt_betaij == 1) {
          lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_current_arg[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_modified[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_modified.at(xi_I_arg.at(j));
        }
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      //lh_square_modified.f_U.at(xi_U_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_modified.q_T.at(xi_U_arg.at(i)),1.0);
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty()==0){
    for (int i=0;i<= (int)(xi_E_minus_arg.size()-1);i++){
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) =0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j=0;j<= (int) (xi_I_arg.size()-1);j++){
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          //         switch (t_r_arg.at(xi_I_arg.at(j))>=t_e_arg.at(xi_E_minus_arg.at(i))) {
          //         case 1:{ // not yet recovered at e_i
          //         lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) + kernel_mat_modified[xi_E_minus_arg.at(i)][xi_I_arg.at(j)]/norm_const_modified.at(xi_I_arg.at(j)); // update k_sum_E
          //         break;
          //         }
          //         case 0:{ // recovered before e_i
          //         break;
          //         }
          //         }
          //
          if (opt_betaij == 0) {
            lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * kernel_mat_modified[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_modified.at(xi_I_arg.at(j)); // update kt_sum_E
          }
          if (opt_betaij == 1) {
            lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_current_arg[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_modified[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_modified.at(xi_I_arg.at(j)); // update kt_sum_E
          }
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch(infected_source_arg.at(xi_E_minus_arg.at(i))){
        case 9999:{ // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default :{ // not by background
          
          if (opt_betaij == 0) {
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = kernel_mat_modified[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_modified.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
        }
          if (opt_betaij == 1) {
            lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_current_arg[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_modified[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_modified.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          }
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m * lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      //lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(xi_E_minus_arg.at(i)),1.0);
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg) + (log_prior_y - log_prior_x) ));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    kernel_mat_current_arg = kernel_mat_modified;
    norm_const_current_arg = norm_const_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.k_1 = k_1_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.k_1 = para_current_arg.k_1;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::tau_susc_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double tau_susc_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_susc_modified(n_CUPDATE, 0.0);
  
  tau_susc_proposed = para_current_arg.tau_susc + para_sf_arg.tau_susc_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((tau_susc_proposed <= para_priors_arg.tau_susc_lo) | (tau_susc_proposed >= para_priors_arg.tau_susc_hi)) {
  
  case 1: {
    tau_susc_proposed = para_current_arg.tau_susc;
    break;
  }
  case 0: {
    tau_susc_proposed = tau_susc_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(tau_susc_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.tau_susc, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, tau_susc_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.tau_susc));
  
  //----- recalc beta_ij matrix with proposed tau_susc -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], para_current_arg.nu_inf, tau_susc_proposed, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], para_current_arg.phi_inf1, para_current_arg.phi_inf2, para_current_arg.rho_susc1, para_current_arg.rho_susc2);
    }
  }
  /*
   if (opt_betaij == 2) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = func_beta_ij_susc(herdn_CUPDATE[j], tau_susc_proposed, ftype0_CUPDATE[j], ftype1_CUPDATE[j], ftype2_CUPDATE[j], para_current_arg.rho_susc1, para_current_arg.rho_susc2);
   }
   
   //normalise by mean susceptibility
   double norm_susc = 0;
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   norm_susc = norm_susc + beta_ij_susc_modified[j];
   }
   norm_susc = norm_susc / n_CUPDATE;
   
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = beta_ij_susc_modified[j] / norm_susc;
   }
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_current_arg, beta_ij_susc_modified);
   }
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      //lh_square_modified.f_U.at(xi_U_arg.at(i)) = 1.0 - gsl_cdf_exponential_P(lh_square_modified.q_T.at(xi_U_arg.at(i)), 1.0);
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      //lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(xi_E_minus_arg.at(i)), 1.0);
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_susc_current_arg = beta_ij_susc_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.tau_susc = tau_susc_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.tau_susc = para_current_arg.tau_susc;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::nu_inf_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double nu_inf_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_inf_modified(n_CUPDATE, 0.0);
  
  nu_inf_proposed = para_current_arg.nu_inf + para_sf_arg.nu_inf_sf*rnorm(0.0, 1.0, rng_arg);
  
  switch ((nu_inf_proposed <= para_priors_arg.nu_inf_lo) | (nu_inf_proposed >= para_priors_arg.nu_inf_hi)) {
  
  case 1: {
    nu_inf_proposed = para_current_arg.nu_inf;
    break;
  }
  case 0: {
    nu_inf_proposed = nu_inf_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(nu_inf_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.nu_inf, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, nu_inf_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.nu_inf));
  
  
  //----- recalc beta_ij matrix with proposed nu_inf -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], nu_inf_proposed, para_current_arg.tau_susc, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], para_current_arg.phi_inf1, para_current_arg.phi_inf2, para_current_arg.rho_susc1, para_current_arg.rho_susc2);
    }
  }
  
  /*
   if(opt_betaij == 2) {
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   beta_ij_inf_modified[i] = func_beta_ij_inf(herdn_CUPDATE[i], nu_inf_proposed, ftype0_CUPDATE[i], ftype1_CUPDATE[i], ftype2_CUPDATE[i], para_current_arg.phi_inf1, para_current_arg.phi_inf2);
   }
   
   //normalise by mean infectivity
   double norm_inf = 0;
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   norm_inf = norm_inf + beta_ij_inf_modified[i];
   }
   norm_inf = norm_inf / n_CUPDATE;
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //susceptibles
   beta_ij_inf_modified[i] = beta_ij_inf_modified[i] / norm_inf;
   }
   
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_modified, beta_ij_susc_current_arg);
   }
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_inf_current_arg = beta_ij_inf_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.nu_inf = nu_inf_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.nu_inf = para_current_arg.nu_inf;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::rho_susc1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double rho_susc1_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_susc_modified(n_CUPDATE, 1.0);
  
  rho_susc1_proposed = para_current_arg.rho_susc1 + para_sf_arg.rho_susc1_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((rho_susc1_proposed <= 0) | (rho_susc1_proposed >= para_priors_arg.rho_susc1_hi)) {
  
  case 1: {
    rho_susc1_proposed = para_current_arg.rho_susc1;
    break;
  }
  case 0: {
    rho_susc1_proposed = rho_susc1_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(rho_susc1_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.rho_susc1, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, rho_susc1_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.rho_susc1));
  
  //----- recalc beta_ij matrix with proposed rho_susc1 -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], para_current_arg.nu_inf, para_current_arg.tau_susc, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], para_current_arg.phi_inf1, para_current_arg.phi_inf2, rho_susc1_proposed, para_current_arg.rho_susc2);
    }
  }
  /*
   if (opt_betaij == 2) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = func_beta_ij_susc(herdn_CUPDATE[j], para_current_arg.tau_susc, ftype0_CUPDATE[j], ftype1_CUPDATE[j], ftype2_CUPDATE[j], rho_susc1_proposed, para_current_arg.rho_susc2);
   }
   
   //normalise by mean susceptibility
   double norm_susc = 0;
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   norm_susc = norm_susc + beta_ij_susc_modified[j];
   }
   norm_susc = norm_susc / n_CUPDATE;
   
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = beta_ij_susc_modified[j] / norm_susc;
   }
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_current_arg, beta_ij_susc_modified);
   }
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_susc_current_arg = beta_ij_susc_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.rho_susc1 = rho_susc1_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.rho_susc1 = para_current_arg.rho_susc1;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::rho_susc2_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double rho_susc2_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_susc_modified(n_CUPDATE, 1.0);
  
  
  rho_susc2_proposed = para_current_arg.rho_susc2 + para_sf_arg.rho_susc2_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((rho_susc2_proposed <= 0) | (rho_susc2_proposed >= para_priors_arg.rho_susc2_hi)) {
  
  case 1: {
    rho_susc2_proposed = para_current_arg.rho_susc2;
    break;
  }
  case 0: {
    rho_susc2_proposed = rho_susc2_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(rho_susc2_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.rho_susc2, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, rho_susc2_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.rho_susc2));
  
  
  
  //----- recalc beta_ij matrix with proposed rho_susc2 -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], para_current_arg.nu_inf, para_current_arg.tau_susc, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], para_current_arg.phi_inf1, para_current_arg.phi_inf2, para_current_arg.rho_susc1, rho_susc2_proposed);
    }
  }
  /*
   if (opt_betaij == 2) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = func_beta_ij_susc(herdn_CUPDATE[j], para_current_arg.tau_susc, ftype0_CUPDATE[j], ftype1_CUPDATE[j], ftype2_CUPDATE[j], para_current_arg.rho_susc1, rho_susc2_proposed);
   }
   
   //normalise by mean susceptibility
   double norm_susc = 0;
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   norm_susc = norm_susc + beta_ij_susc_modified[j];
   }
   norm_susc = norm_susc / n_CUPDATE;
   
   for (int j = 0; j <= (n_CUPDATE - 1); j++) { //susceptibles
   beta_ij_susc_modified[j] = beta_ij_susc_modified[j] / norm_susc;
   }
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_current_arg, beta_ij_susc_modified);
   }
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_susc_current_arg = beta_ij_susc_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.rho_susc2 = rho_susc2_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.rho_susc2 = para_current_arg.rho_susc2;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::phi_inf1_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double phi_inf1_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_inf_modified(n_CUPDATE, 0.0);
  
  phi_inf1_proposed = para_current_arg.phi_inf1 + para_sf_arg.phi_inf1_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  
  
  switch ((phi_inf1_proposed <= 0) | (phi_inf1_proposed >= para_priors_arg.phi_inf1_hi)) {
  
  case 1: {
    phi_inf1_proposed = para_current_arg.phi_inf1;
    break;
  }
  case 0: {
    phi_inf1_proposed = phi_inf1_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(phi_inf1_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.phi_inf1, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, phi_inf1_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.phi_inf1));
  
  
  //----- recalc beta_ij matrix with proposed phi_inf1 -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], para_current_arg.nu_inf, para_current_arg.tau_susc, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], phi_inf1_proposed, para_current_arg.phi_inf2, para_current_arg.rho_susc1, para_current_arg.rho_susc2);
    }
  }
  /*
   if (opt_betaij == 2) {
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   beta_ij_inf_modified[i] = func_beta_ij_inf(herdn_CUPDATE[i], para_current_arg.nu_inf, ftype0_CUPDATE[i], ftype1_CUPDATE[i], ftype2_CUPDATE[i], phi_inf1_proposed, para_current_arg.phi_inf2);
   }
   
   //normalise by mean infectivity
   double norm_inf = 0;
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   norm_inf = norm_inf + beta_ij_inf_modified[i];
   }
   norm_inf = norm_inf / n_CUPDATE;
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   beta_ij_inf_modified[i] = beta_ij_inf_modified[i] / norm_inf;
   }
   
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_modified, beta_ij_susc_current_arg);
   }
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_inf_current_arg = beta_ij_inf_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.phi_inf1 = phi_inf1_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.phi_inf1 = para_current_arg.phi_inf1;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::phi_inf2_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, vector< vector<double> >& kernel_mat_current_arg, const vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, para_key& para_current_arg, const vector<int>& infected_source_arg, vector<double>& norm_const_current_arg, vector< vector<double> >& beta_ij_mat_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector<double>& beta_ij_inf_current_arg, vector<double>& beta_ij_susc_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, const vector< vector<double> >& delta_mat_mov_current_arg) {
  
  double phi_inf2_proposed = 0.0;
  double acp_pr = 0.0;
  
  
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified = log_lh_current_arg;
  vector< vector<double> > beta_ij_mat_modified = beta_ij_mat_current_arg;
  vector<double> beta_ij_inf_modified(n_CUPDATE, 0.0);
  
  phi_inf2_proposed = para_current_arg.phi_inf2 + para_sf_arg.phi_inf2_sf*rnorm(0.0, 1.0, rng_arg);
  
  
  switch ((phi_inf2_proposed <= 0) | (phi_inf2_proposed >= para_priors_arg.phi_inf2_hi)) {
  
  case 1: {
    phi_inf2_proposed = para_current_arg.phi_inf2;
    break;
  }
  case 0: {
    phi_inf2_proposed = phi_inf2_proposed;
    break;
  }
  }
  
  
  double log_prior_y = 0.0;
  double log_prior_x = 0.0;
  
  //log_prior_y = log(gsl_ran_exponential_pdf(phi_inf2_proposed, 1.0 / para_priors_arg.rate_exp_prior));
  //log_prior_x = log(gsl_ran_exponential_pdf(para_current_arg.phi_inf2, 1.0 / para_priors_arg.rate_exp_prior));
  log_prior_y = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, phi_inf2_proposed));
  log_prior_x = log(pdf_exp_limit(para_priors_arg.rate_exp_prior, para_current_arg.phi_inf2));
  
  
  //----- recalc beta_ij matrix with proposed phi_inf2 -----
  for (int i = 0; i <= (n_CUPDATE - 1); i++) {
    for (int j = 0; j <= (n_CUPDATE - 1); j++) {
      if (i == j) beta_ij_mat_modified[i][j] = 0.0;
      if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij(herdn_CUPDATE[i], herdn_CUPDATE[j], para_current_arg.nu_inf, para_current_arg.tau_susc, ftype0_CUPDATE[i], ftype0_CUPDATE[j], ftype1_CUPDATE[i], ftype1_CUPDATE[j], ftype2_CUPDATE[i], ftype2_CUPDATE[j], para_current_arg.phi_inf1, phi_inf2_proposed, para_current_arg.rho_susc1, para_current_arg.rho_susc2);
    }
  }
  /*
   if(opt_betaij == 2){
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   beta_ij_inf_modified[i] = func_beta_ij_inf(herdn_CUPDATE[i], para_current_arg.nu_inf, ftype0_CUPDATE[i], ftype1_CUPDATE[i], ftype2_CUPDATE[i], para_current_arg.phi_inf1, phi_inf2_proposed);
   }
   
   //normalise by mean infectivity
   double norm_inf = 0;
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //infectives
   norm_inf = norm_inf + beta_ij_inf_modified[i];
   }
   norm_inf = norm_inf / n_CUPDATE;
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) { //susceptibles
   beta_ij_inf_modified[i] = beta_ij_inf_modified[i] / norm_inf;
   }
   
   
   
   for (int i = 0; i <= (n_CUPDATE - 1); i++) {
   for (int j = 0; j <= (n_CUPDATE - 1); j++) {
   if (i == j) beta_ij_mat_modified[i][j] = 0.0;
   if (i != j) beta_ij_mat_modified[i][j] = func_beta_ij_norm(i, j, beta_ij_inf_modified, beta_ij_susc_current_arg);			}
   }
   }
   */
  //----------
  
  if (xi_U_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_U_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
      
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        
        double delta_t = delta_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        double delta_t_mov = delta_mat_mov_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)];
        
        lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_U_arg.at(i)] * kernel_mat_current_arg[xi_U_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = 0.0;
        }
        
        if (opt_mov == 1) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + delta_t_mov;
        }
        if (opt_mov == 2) {
          lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_U_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
        }
        
        
      }
      
      lh_square_modified.q_T.at(xi_U_arg.at(i)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_T.at(xi_U_arg.at(i)) = lh_square_modified.q_T.at(xi_U_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(i));
      }
      
      lh_square_modified.f_U.at(xi_U_arg.at(i)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(i)));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(i))); //add back part of likelihood that updated above
    }
  }
  
  //----------
  
  if (xi_E_minus_arg.empty() == 0) {
    for (int i = 0; i <= (int)(xi_E_minus_arg.size() - 1); i++) {
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
      
      for (int j = 0; j <= (int)(xi_I_arg.size() - 1); j++) {
        
        if (t_i_arg.at(xi_I_arg.at(j))<t_e_arg.at(xi_E_minus_arg.at(i))) {
          
          double delta_t = delta_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          double delta_t_mov = delta_mat_mov_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)];
          
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i)) + delta_t * beta_ij_mat_modified[xi_I_arg.at(j)][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j)); // update kt_sum_E
          
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = 0.0;
          }
          
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + delta_t_mov;
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i)) + func_moves_cnt(xi_I_arg.at(j), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          }
          
          
        } // end of if (t_i_Clh.at(xi_I_Clh.at(j))<t_e_Clh.at(xi_E_Clh.at(i)))
        
        //lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
        
        switch (infected_source_arg.at(xi_E_minus_arg.at(i))) {
        case 9999: { // by background
          //lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)); // update k_sum_E
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i));
          break;
        }
          
          
        default: { // not by background
          lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i)) = beta_ij_mat_modified[infected_source_arg.at(xi_E_minus_arg.at(i))][xi_E_minus_arg.at(i)] * kernel_mat_current_arg[xi_E_minus_arg.at(i)][infected_source_arg.at(xi_E_minus_arg.at(i))] / norm_const_current_arg.at(infected_source_arg.at(xi_E_minus_arg.at(i))); // update k_sum_E
          
          lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = para_current_arg.beta*lh_square_modified.k_sum_E.at(xi_E_minus_arg.at(i));
          if ((opt_mov == 1) | (opt_mov == 2)) {
            lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i)) = func_moves_cnt(infected_source_arg.at(xi_E_minus_arg.at(i)), xi_E_minus_arg.at(i), moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
            lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(xi_E_minus_arg.at(i));
          }
          
          break;
        }
          
        }
        
      } // end of  for (int j=0;j<= (int) (xi_I_Clh.size()-1);j++)
      
      
      lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(i)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(i));
      if ((opt_mov == 1) | (opt_mov == 2)) {
        lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(i)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(i));
      }
      
      lh_square_modified.h_E.at(xi_E_minus_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(i)));
      
      lh_square_modified.f_E.at(xi_E_minus_arg.at(i)) = lh_square_modified.g_E.at(xi_E_minus_arg.at(i))*lh_square_modified.h_E.at(xi_E_minus_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(i))); //subtract part of likelihood that would be updated below
    }
  }
  //----------
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0, exp((log_lh_modified - log_lh_current_arg) + (log_prior_y - log_prior_x)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch (int (uniform_rv <= acp_pr) ){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    beta_ij_mat_current_arg = beta_ij_mat_modified;
    beta_ij_inf_current_arg = beta_ij_inf_modified;
    log_lh_current_arg = log_lh_modified;
    para_current_arg.phi_inf2 = phi_inf2_proposed;
    break;
  }
    
  case 0: {
  }
    lh_square_current_arg = lh_square_current_arg;
    log_lh_current_arg = log_lh_current_arg;
    para_current_arg.phi_inf2 = para_current_arg.phi_inf2;
    break;
  }
  
  //gsl_rng_free(r_c);
  
}


/*------------------------------------------------*/

void mcmc_UPDATE::con_seq_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg, vector<int>&  xi_beta_E_arg, vector<int>& con_seq, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  double acp_pr = 0.0;
  
  int position_proposed, base_proposed;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  position_proposed =iter;
  
  //---
  
  base_proposed=0;
  int base_current = con_seq.at(position_proposed);
  
  switch(base_current){
  case 1:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 2;
      break;
    }
    case 1:{
      base_proposed = 3;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 2:{
    int type = runif_int(0, 2, rng_arg);
    
    
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 3;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 3:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 2;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 4:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 2;
      break;
    }
    case 2:{
      base_proposed = 3;
      break;
    }
    }
    break;
  }
  }
  //---
  
  
  for (int i=0;i<= (int)(xi_E_arg.size()-1);i++){ // loop over all the infected
    
    int k_E = xi_E_arg.at(i);
    
    switch (int (infected_source_current_arg.at(k_E)==9999)) {
    
    case 1:{//bg infection
      
      int base_k_E = nt_current_arg[k_E][position_proposed];
      
      log_lh_modified =  log_lh_modified - lh_square_modified.log_f_Snull.at(k_E);
      
      double log_x =  lh_snull_base(con_seq.at(position_proposed),base_k_E , para_current_arg.p_ber);
      double log_y =  lh_snull_base(base_proposed, base_k_E, para_current_arg.p_ber);
      
      lh_square_modified.log_f_Snull.at(k_E) =  lh_square_modified.log_f_Snull.at(k_E)  - log_x + log_y;
      
      log_lh_modified =  log_lh_modified + lh_square_modified.log_f_Snull.at(k_E);
      
      break;
    }
      
    default:{ // 2nd infection
      break;
    }
      
    }
  }
  
  //----
  
  acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    con_seq.at(position_proposed) = base_proposed;
    
    break;
  }
    
  case 0: {
    break;
  }
  }
  
}


/*------------------------------------------------*/

void mcmc_UPDATE::seq_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg, vector<int>&  xi_beta_E_arg, vector<int>& con_seq, const int& subject_proposed, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, int iter, rng_type & rng_arg){
  
  //int subject_proposed ;
  double acp_pr = 0.0;
  
  double log_part_x_subject=0.0;
  double log_part_y_subject=0.0;
  
  double log_part_x_source=0.0;
  double log_part_y_source =0.0;
  
  
  int position_proposed, base_proposed;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  //vector< vector<double> > delta_mat_modified = delta_mat_current_arg;
  //vector<double> t_e_modified = t_e_arg;
  //vector <int> index_modified = index_arg;
  //vector <int> xi_E_minus_modified = xi_E_minus_arg;
  
  //vector <int> xi_U_modified = xi_U_arg;
  //vector <int> xi_E_modified = xi_E_arg;
  //vector <int> xi_EnI_modified = xi_EnI_arg;
  
  // const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
  // gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
  // gsl_rng_set (r_c,(iter+1)*time(NULL)); // set a seed
  
  
  
  //subject_proposed = xi_E_arg.at(gsl_rng_uniform_int (r_c, xi_E_arg.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
  //subject_proposed = xi_E_arg.at(gsl_rng_uniform_int (r_c, xi_beta_E_arg.size())); // only change the one from secondary infection
  
  // //----- test---------//
  // while((current_size_arg.at(subject_proposed)>1)| (infected_source_current_arg.at(subject_proposed)==9999)){
  // subject_proposed = xi_E_arg.at(gsl_rng_uniform_int (r_c, xi_E_arg.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
  // }
  // //--------------------//
  
  //subject_proposed =3;
  
  //position_proposed = gsl_rng_uniform_int (r_c, n_base_CUPDATE);
  position_proposed =iter;
  
  
  //vector<int> nt_modified_subject = nt_current_arg.at(subject_proposed);
  
  //vector<int> seq_proposed;
  //seq_proposed.assign(nt_modified_subject.begin() , nt_modified_subject.begin()+n_base_CUPDATE );
  
  
  int subject_source = infected_source_current_arg.at(subject_proposed);
  
  
  //---
  
  //base_proposed = gsl_rng_uniform_int (r_c, 4) + 1; // a new base proposed
  
  base_proposed=0;
  int base_current = nt_current_arg[subject_proposed][position_proposed]; // always refers to the first sequence
  
  switch(base_current){
  case 1:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 2;
      break;
    }
    case 1:{
      base_proposed = 3;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 2:{
    int type = runif_int(0, 2, rng_arg);
    
    
    
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 3;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 3:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 2;
      break;
    }
    case 2:{
      base_proposed = 4;
      break;
    }
    }
    break;
  }
  case 4:{
    int type = runif_int(0, 2, rng_arg);
    switch(type){
    case 0:{
      base_proposed = 1;
      break;
    }
    case 1:{
      base_proposed = 2;
      break;
    }
    case 2:{
      base_proposed = 3;
      break;
    }
    }
    break;
  }
  }
  //---
  
  int base_next_subject =0;
  
  switch (int (current_size_arg.at(subject_proposed)>1)) {
  
  case 1:{
    //--
    base_next_subject = nt_current_arg[subject_proposed][position_proposed + n_base_CUPDATE];
    
    log_part_x_subject = log_lh_base(base_current, base_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2);
    
    log_part_y_subject = log_lh_base(base_proposed, base_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2);
    
    //--
    lh_square_modified.log_f_S.at(subject_proposed) =  lh_square_modified.log_f_S.at(subject_proposed) - log_part_x_subject;
    log_lh_modified = log_lh_modified - log_part_x_subject;
    
    lh_square_modified.log_f_S.at(subject_proposed) =  lh_square_modified.log_f_S.at(subject_proposed) + log_part_y_subject;
    log_lh_modified = log_lh_modified + log_part_y_subject;
    
    
    break;
  }
    
  case 0:{
    break;
  }
  }
  //-------
  
  int rank_source =-1;  //count the rank of the original t_e among t_nt_current_arg.at(subject_source)
  int base_before_source =0;
  int base_next_source = 0;
  
  switch(subject_source ){
  
  case 9999:{ // by background
    //t_proposed= gsl_ran_flat(r_c, 0.0, min( t_sample_arg.at(subject_proposed), min( t_i_arg.at(subject_proposed), t_max_CUPDATE)) );
    
    log_lh_modified =  log_lh_modified - lh_square_modified.log_f_Snull.at(subject_proposed);
    
    double log_x =  lh_snull_base(con_seq.at(position_proposed), base_current, para_current_arg.p_ber);
    double log_y =  lh_snull_base(con_seq.at(position_proposed), base_proposed, para_current_arg.p_ber);
    
    lh_square_modified.log_f_Snull.at(subject_proposed) =  lh_square_modified.log_f_Snull.at(subject_proposed)  - log_x + log_y;
    
    log_lh_modified =  log_lh_modified + lh_square_modified.log_f_Snull.at(subject_proposed);
    
    break;
  }
    
  default :{ // not by background
    
    //nt_modified_source = nt_current_arg.at(subject_source);
    rank_source = (int)(distance( t_nt_current_arg.at(subject_source).begin(), find(t_nt_current_arg.at(subject_source).begin(), t_nt_current_arg.at(subject_source).end(), t_e_arg.at(subject_proposed)) ));
    //nt_modified_source.erase(nt_modified_source.begin()+n_base_CUPDATE*rank_source_x , nt_modified_source.begin()+n_base_CUPDATE*(rank_source_x+1) );  //erase the original nt entry for source
    
    base_before_source =  nt_current_arg[subject_source][(rank_source-1)*n_base_CUPDATE + position_proposed];
    
    switch(int (current_size_arg.at(subject_source)>(rank_source+1))){
    case 1:{// there  is a valid base_next_source
      base_next_source =  nt_current_arg[subject_source][(rank_source+1)*n_base_CUPDATE + position_proposed];
      
      
      log_part_x_source = log_lh_base(base_before_source, base_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2) + log_lh_base(base_current, base_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2);
      
      log_part_y_source = log_lh_base(base_before_source, base_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2) + log_lh_base(base_proposed, base_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2);
      
      
      
      break;
    }
      
    case 0:{
      
      log_part_x_source = log_lh_base(base_before_source, base_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2);
      
      log_part_y_source = log_lh_base(base_before_source, base_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2);
      
      break;
    }
    }
    
    lh_square_modified.log_f_S.at(subject_source) =  lh_square_modified.log_f_S.at(subject_source) - log_part_x_source;
    log_lh_modified = log_lh_modified - log_part_x_source;
    
    
    lh_square_modified.log_f_S.at(subject_source) =  lh_square_modified.log_f_S.at(subject_source) + log_part_y_source;
    log_lh_modified = log_lh_modified + log_part_y_source;
    
    break;
  }
    
  }
  
  //------
  
  // double log_part_y = log_part_y_subject + log_part_y_source;
  // double log_part_x = log_part_x_subject + log_part_x_source;
  //acp_pr = min(1.0,exp(log_part_y-log_part_x));
  
  acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    nt_current_arg[subject_proposed][position_proposed]= base_proposed;
    
    switch (subject_source){
    
    case 9999:{ // by background
      break;
    }
      
    default :{ // not by background
      nt_current_arg[subject_source][(rank_source)*n_base_CUPDATE + position_proposed] = base_proposed;
    }
    }
    
    
    break;
  }
    
  case 0: {
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
}


/*------------------------------------------------*/


void mcmc_UPDATE::t_i_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, const vector<int>& xi_U_arg, const vector<int>& xi_E_arg, const vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, const vector<int>& xi_EnI_arg, const vector<int>& xi_R_arg, const vector<int>& xi_InR_arg, const vector<double>& t_r_arg, vector<double>& t_i_arg, const vector<double>& t_onset_arg, const vector<double>& t_e_arg, const vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg,  vec2int& infecting_list_current_arg, const vector<int>& infecting_size_current_arg, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg, moves_struct& mov_arg){
  
  double t_proposed; // new t_i to be proposed
  double t_low, t_up;
  
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  vector< vector<double> > delta_mat_modified = delta_mat_current_arg;
  vector< vector<double> > delta_mat_mov_modified = delta_mat_mov_current_arg;
  vector<double> t_i_modified = t_i_arg;
  
  
  
  //int subject_proposed = xi_I_arg.at(gsl_rng_uniform_int (r_c, xi_I_arg.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
  int subject_proposed = xi_I_arg.at(runif_int(0, (int)(xi_I_arg.size())-1, rng)); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
  double t_o = t_onset_arg.at(subject_proposed); // this is assumed to be given,e.g, the onset time  (in simulation, this may be taken to be the true t_i)
  //double t_range = 1.0; // prior belief; we assume the sampled t_i would range between [t_o - t_range, t_o + range]
  
  
  
  switch(int (infecting_size_current_arg.at(subject_proposed)>=1)){
  case 1:{
    double min_t = t_e_arg.at(infecting_list_current_arg[subject_proposed][0]); // the minimum t_e among all those exposures being infected by the subject; as infecting_list is sorted according to the order of infecting, the first one is the minimum
    
    t_low = max(t_o - para_priors_arg.t_range, t_e_arg.at(subject_proposed));
    t_up = min(t_o + para_priors_arg.t_range, min_t);
    
    // 		t_low = t_e_arg.at(subject_proposed);
    // 		t_up = min_t;
    
    break;
  }
  case 0:{
    
    t_low = max(t_o - para_priors_arg.t_range, t_e_arg.at(subject_proposed));
    t_up = min( t_max_CUPDATE, min(t_o + para_priors_arg.t_range, t_r_arg.at(subject_proposed)));
    
    // 		t_low = t_e_arg.at(subject_proposed);
    // 		t_up =  min(t_max_CUPDATE, t_r_arg.at(subject_proposed));
    
    break;
  }
  }
  
  //t_proposed= gsl_ran_flat(r_c,t_low, t_up );
  t_proposed = runif(t_low, t_up, rng);
  
  
  // t_proposed = t_i_arg.at(subject_proposed)+ 1.0*gsl_ran_gaussian(r_c,1.0); //random walk//
  //
  // switch ((t_proposed<t_low) | (t_proposed>t_up)) {
  //
  // case 1: {
  // t_proposed = t_i_arg.at(subject_proposed);
  // break;
  // }
  // case 0: {
  // t_proposed = t_proposed;
  // break;
  // }
  // }
  
  
  
  t_i_modified.at(subject_proposed) = t_proposed;
  
  
  double log_prior_y =0.0;
  double log_prior_x=0.0;
  
  // log_prior_y = log(gsl_ran_gamma_pdf(t_proposed, t_o/pow(0.5,2.0), pow(0.5,2.0)));
  // log_prior_x = log(gsl_ran_gamma_pdf(t_i_arg.at(subject_proposed), t_o/pow(0.5,2.0), pow(0.5,2.0)));
  
  //----------------------------------------------------------------------------------//
  
  for (int j=0;j<=(int) (xi_U_arg.size()-1);j++){
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_U.at(xi_U_arg.at(j))); //subtract part of likelihood that would be updated below
    
    
    switch (int (t_r_arg.at(subject_proposed)>=t_max_CUPDATE)) {
    case 1:{
      delta_mat_modified[xi_U_arg.at(j)][subject_proposed] = t_max_CUPDATE  - t_proposed;
      
      delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] = 0.0;
      for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
        if ((mov_arg.from_k[m] == subject_proposed) && (mov_arg.to_k[m] == xi_U_arg.at(j))) {
          if ((mov_arg.t_m[m] >= t_proposed) &&
              (mov_arg.t_m[m] <= t_max_CUPDATE)) {
            delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] = delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] + (t_max_CUPDATE - mov_arg.t_m[m]);
          }
        }
      }
      break;
    }
    case 0:{
      delta_mat_modified[xi_U_arg.at(j)][subject_proposed] = t_r_arg.at(subject_proposed) - t_proposed;
      
      delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] = 0.0;
      for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
        if ((mov_arg.from_k[m] == subject_proposed) && (mov_arg.to_k[m] == xi_U_arg.at(j))) {
          if ((mov_arg.t_m[m] >= t_proposed) &&
              (mov_arg.t_m[m] <= t_r_arg.at(subject_proposed))) {
            delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] = delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed] + (t_r_arg.at(subject_proposed) - mov_arg.t_m[m]);
          }
        }
      }
      break;
    }
    }
    
    
    if (opt_betaij == 0) {
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(j)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(j)) 
      - delta_mat_current_arg[xi_U_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_U_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed) 
      + delta_mat_modified[xi_U_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_U_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed); //subtract the infectious challenge  due to the infectious subject chosen THEN add the updated one back
    }
    
    if (opt_betaij == 1) {
      lh_square_modified.kt_sum_U.at(xi_U_arg.at(j)) = lh_square_modified.kt_sum_U.at(xi_U_arg.at(j)) 
      - delta_mat_current_arg[xi_U_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_U_arg.at(j)] * kernel_mat_current_arg[xi_U_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed) 
      + delta_mat_modified[xi_U_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_U_arg.at(j)] * kernel_mat_current_arg[xi_U_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed); //subtract the infectious challenge  due to the infectious subject chosen THEN add the updated one back
    }
    
    
    if (opt_mov == 0) {
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(j)) = 0.0;
    }
    
    if (opt_mov == 1) {
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(j)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(j)) 
      - delta_mat_mov_current_arg[xi_U_arg.at(j)][subject_proposed] 
      + delta_mat_mov_modified[xi_U_arg.at(j)][subject_proposed];
    }
    
    if (opt_mov == 2) {
      lh_square_modified.movest_sum_U.at(xi_U_arg.at(j)) = lh_square_modified.movest_sum_U.at(xi_U_arg.at(j));//no change
    }
    
    lh_square_modified.q_T.at(xi_U_arg.at(j)) = para_current_arg.alpha*t_max_CUPDATE + para_current_arg.beta*lh_square_modified.kt_sum_U.at(xi_U_arg.at(j)) + para_current_arg.beta_m*lh_square_modified.movest_sum_U.at(xi_U_arg.at(j));
    
    lh_square_modified.f_U.at(xi_U_arg.at(j)) = surv_exp_limit(1.0, lh_square_modified.q_T.at(xi_U_arg.at(j)));
    
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_U.at(xi_U_arg.at(j))); // add back the updated part of likelihood
    
  }
  
  //----------------------------------------------------------------------------------//
  
  for (int j=0;j<=(int) (xi_E_minus_arg.size()-1);j++){
    
    if (xi_E_minus_arg.at(j)!=subject_proposed) {
      
      switch (int (t_e_arg.at(xi_E_minus_arg.at(j))>t_proposed) ) {
      
      case 1:{
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(j))); //subtract part of likelihood that would be updated below
    
    switch (int (t_r_arg.at(subject_proposed)>=t_e_arg.at(xi_E_minus_arg.at(j)))) {
    case 1:{
      delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] = t_e_arg.at(xi_E_minus_arg.at(j)) - t_proposed;
      
      delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] = 0.0;
      for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
        if ((mov_arg.from_k[m] == subject_proposed) && (mov_arg.to_k[m] == xi_E_minus_arg.at(j))) {
          if ((mov_arg.t_m[m] >= t_proposed) &&
              (mov_arg.t_m[m] <= t_e_arg.at(xi_E_minus_arg.at(j)))) {
            delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] = delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] + (t_e_arg.at(xi_E_minus_arg.at(j)) - mov_arg.t_m[m]);
          }
        }
      }
      break;
    }
    case 0:{
      delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] = t_r_arg.at(subject_proposed) - t_proposed;
      
      delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] = 0.0;
      for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
        if ((mov_arg.from_k[m] == subject_proposed) && (mov_arg.to_k[m] == xi_E_minus_arg.at(j))) {
          if ((mov_arg.t_m[m] >= t_proposed) &&
              (mov_arg.t_m[m] <= t_r_arg.at(subject_proposed))) {
            delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] = delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] + (t_r_arg.at(subject_proposed) - mov_arg.t_m[m]);
          }
        }
      }
      break;
    }
    }
    
    
    switch(int (t_e_arg.at(xi_E_minus_arg.at(j))>t_i_arg.at(subject_proposed))){
    case 1:{
      
      if (opt_betaij == 0) {
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed) + delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
    }
      if (opt_betaij == 1) {
        lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_E_minus_arg.at(j)] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed) + delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_E_minus_arg.at(j)] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
      }
      
      if (opt_mov == 0) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = 0.0;
      }
      
      if (opt_mov == 1) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_mov_current_arg[xi_E_minus_arg.at(j)][subject_proposed] + delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed];
      }
      if (opt_mov == 2) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) ; //no change
      }
      
      break;
    }
      
    case 0:{
      if (opt_betaij == 0) {
      lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) + delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
    }
      if (opt_betaij == 1) {
        lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) + delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_E_minus_arg.at(j)] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
      }
      
      if (opt_mov == 0) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = 0.0;
      }
      if (opt_mov == 1) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) + delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed];
      }
      if (opt_mov == 2) {
        lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) ; // no change
      }
      
      break;
    }
      
    }
    
    lh_square_modified.q_E.at(xi_E_minus_arg.at(j))= para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(j)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j));
    if ((opt_mov == 1) | (opt_mov == 2)) {
      lh_square_modified.q_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(j)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j));
    }
    
    lh_square_modified.g_E.at(xi_E_minus_arg.at(j))=   lh_square_modified.g_E.at(xi_E_minus_arg.at(j)); // unchanged as source does not change
    
    lh_square_modified.h_E.at(xi_E_minus_arg.at(j)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(j)));
    
    lh_square_modified.f_E.at(xi_E_minus_arg.at(j))= lh_square_modified.g_E.at(xi_E_minus_arg.at(j))*lh_square_modified.h_E.at(xi_E_minus_arg.at(j));
    
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(j))); // add back the updated part of likelihood
    
    break;
    
  }
        
      case 0:{
        
        log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(xi_E_minus_arg.at(j))); //subtract part of likelihood that would be updated below
        
        delta_mat_modified[xi_E_minus_arg.at(j)][subject_proposed] = 0.0;
        delta_mat_mov_modified[xi_E_minus_arg.at(j)][subject_proposed] = 0.0;
        
        switch(int (t_e_arg.at(xi_E_minus_arg.at(j))>t_i_arg.at(subject_proposed))){
        case 1:{
          if (opt_betaij == 0) {
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
        }
          if (opt_betaij == 1) {
            lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] * beta_ij_mat_current_arg[subject_proposed][xi_E_minus_arg.at(j)] * kernel_mat_current_arg[xi_E_minus_arg.at(j)][subject_proposed] / norm_const_current_arg.at(subject_proposed);
          }
          if (opt_mov == 0) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = 0.0;
          }
          if (opt_mov == 1) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) - delta_mat_mov_current_arg[xi_E_minus_arg.at(j)][subject_proposed];
          }
          if (opt_mov == 2) {
            lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) ; //no change
          }
          
          break;
        }
        case 0:{
          lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j));
          lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j));
        }
        }
        
        lh_square_modified.q_E.at(xi_E_minus_arg.at(j))= para_current_arg.alpha*t_e_arg.at(xi_E_minus_arg.at(j)) + para_current_arg.beta*lh_square_modified.kt_sum_E.at(xi_E_minus_arg.at(j));
        if ((opt_mov == 1) | (opt_mov == 2)) {
          lh_square_modified.q_E.at(xi_E_minus_arg.at(j)) = lh_square_modified.q_E.at(xi_E_minus_arg.at(j)) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(xi_E_minus_arg.at(j));
        }
        
        lh_square_modified.g_E.at(xi_E_minus_arg.at(j))=   lh_square_modified.g_E.at(xi_E_minus_arg.at(j));
        
        lh_square_modified.h_E.at(xi_E_minus_arg.at(j)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(xi_E_minus_arg.at(j)));
        
        lh_square_modified.f_E.at(xi_E_minus_arg.at(j))= lh_square_modified.g_E.at(xi_E_minus_arg.at(j))*lh_square_modified.h_E.at(xi_E_minus_arg.at(j));
        
        log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(xi_E_minus_arg.at(j))); // add back the updated part of likelihood
        
        break;
        
      }
        
      }
      
    }
  }
  
  //----------------------------------------------------------------------------------//
  
  log_lh_modified = log_lh_modified - log(lh_square_modified.f_I.at(subject_proposed)); //subtract part of likelihood that would be updated below
  lh_square_modified.f_I.at(subject_proposed) = func_latent_pdf(t_proposed - t_e_arg.at(subject_proposed), para_current_arg.lat_mu, para_current_arg.lat_var);
  log_lh_modified = log_lh_modified + log(lh_square_modified.f_I.at(subject_proposed));
  
  
  //----------
  
  switch ( find(xi_R_arg.begin(), xi_R_arg.end(),subject_proposed) != (xi_R_arg.end()) ) { //return 1 when the subject is also in xi_R
  case 1:{
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_R.at(subject_proposed)); //subtract part of likelihood that would be updated below
    //	lh_square_modified.f_R.at(subject_proposed) = gsl_ran_weibull_pdf(t_r_arg.at(subject_proposed) - t_proposed, para_current_arg.c, para_current_arg.d);
    lh_square_modified.f_R.at(subject_proposed) = pdf_weibull_limit(para_current_arg.d, para_current_arg.c, t_r_arg.at(subject_proposed) - t_proposed);
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_R.at(subject_proposed));
    break;
  }
  case 0:{
    break;
  }
  }
  
  //----------
  switch ( find(xi_InR_arg.begin(), xi_InR_arg.end(),subject_proposed) != (xi_InR_arg.end()) ) { //return 1 when the subject is also in xi_InR
  case 1:{
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_InR.at(subject_proposed)); //subtract part of likelihood that would be updated below
    //lh_square_modified.f_InR.at(subject_proposed) = 1.0 -  gsl_cdf_weibull_P(t_max_CUPDATE - t_proposed, para_current_arg.c, para_current_arg.d);
    lh_square_modified.f_InR.at(subject_proposed) = surv_weibull_limit(para_current_arg.d, para_current_arg.c, t_max_CUPDATE - t_proposed);
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_InR.at(subject_proposed));
    
    break;
  }
  case 0:{
    break;
  }
  }
  
  //---------------
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg) +(log_prior_y-log_prior_x)));
  
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    delta_mat_current_arg = delta_mat_modified;
    delta_mat_mov_current_arg = delta_mat_mov_modified;
    log_lh_current_arg = log_lh_modified;
    t_i_arg= t_i_modified;
    break;
  }
    
  case 0: {
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
}


/*------------------------------------------------*/


void mcmc_UPDATE::t_e_seq(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg,  vec2int& infecting_list_current_arg, const vector<int>& infecting_size_current_arg, vector<int>&  xi_beta_E_arg, vector<int>& con_seq, int& subject_proposed, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg, moves_struct& mov_arg){
  
  //double t_back =10.0;
  
  //int subject_proposed ;
  double t_proposed = 0.0;
  double t_low, t_up;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  vector< vector<double> > delta_mat_modified = delta_mat_current_arg;
  vector< vector<double> > delta_mat_mov_modified = delta_mat_mov_current_arg;
  vector<double> t_e_modified = t_e_arg;
  vector <int> index_modified = index_arg;
  vector <int> xi_E_minus_modified = xi_E_minus_arg;
  
  // vector <int> xi_U_modified = xi_U_arg;
  // vector <int> xi_E_modified = xi_E_arg;
  // vector <int> xi_EnI_modified = xi_EnI_arg;
  
  /*const gsl_rng_type* T_c= gsl_rng_default;  // T is pointer points to the type of generator
   gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
   gsl_rng_set (r_c,iter); // set a see*/
  
  
  
  vector<int> nt_modified_subject = nt_current_arg.at(subject_proposed);
  vector<double> t_nt_modified_subject = t_nt_current_arg.at(subject_proposed);
  
  vector<int> nt_current_seq; // the orginal sequence of the subject which would be updated
  
  int subject_source = infected_source_current_arg.at(subject_proposed);
  
  
  //int rank_subject_x =distance( t_nt_current_arg.at(subject_proposed).begin(), find(t_nt_current_arg.at(subject_proposed).begin(), t_nt_current_arg.at(subject_proposed).end(), t_e_arg.at(subject_proposed)) ); //count the rank (distance from the first element) of the original t_e among t_nt_current_arg.at(subject_proposed)
  int rank_subject_x =0; // it is always zero as we are updating the sequence at its own infection
  
  t_nt_modified_subject.erase(t_nt_modified_subject.begin() + rank_subject_x); // erase the original t_nt entry for subject_proposed
  
  //---
  
  nt_current_seq.assign(nt_modified_subject.begin()+n_base_CUPDATE*rank_subject_x , nt_modified_subject.begin()+n_base_CUPDATE*(rank_subject_x+1) ); //copy the original nt before erasing
  
  nt_modified_subject.erase(nt_modified_subject.begin()+n_base_CUPDATE*rank_subject_x , nt_modified_subject.begin()+n_base_CUPDATE*(rank_subject_x+1) );  //erase the original nt entry for subject_proposed
  
  
  //--
  
  vector<int> nt_modified_source;
  vector<double> t_nt_modified_source;
  
  vector<int> infecting_list_modified_source;
  
  
  int rank_source_x =-1;  //count the rank of the original t_e among t_nt_current_arg.at(subject_source)
  
  switch(subject_source ){
  
  case 9999:{ // by background
    
    t_up =  min( t_sample_arg.at(subject_proposed), min( t_i_arg.at(subject_proposed), t_max_CUPDATE));
    t_low = max(0.0, t_up- para_priors_arg.t_back);
    //t_low = max( t_e_arg.at(index_arg.at(0)), t_up-t_back);
    
    // 	switch( t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
    //
    // 		case 1:{// with valid t_s
    // 			double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
    //
    // 			switch(t_temp< t_sample_arg.at(subject_proposed)){
    // 				case 1:{
    // 					t_low =  max(0.0, t_temp);
    // 				break;
    // 				}
    // 				case 0:{// should be unlikely if t_back is large enough
    // 					double dt = t_temp -  t_sample_arg.at(subject_proposed);
    // 					t_low =  max(0.0, t_sample_arg.at(subject_proposed) -  dt);
    // 				break;
    // 				}
    // 			}
    // 		break;
    // 		}
    //
    // 		case 0:{ // no valid t_s
    // 			t_low = max(0.0, t_up-t_back);
    // 		break;
    // 		}
    //
    //
    // 	}
    
    t_proposed= runif(t_low, t_up, rng);
    
    //---
    
    // 	t_up =  min( t_sample_arg.at(subject_proposed), min( t_i_arg.at(subject_proposed), t_max_CUPDATE));
    // 	t_low = max(0.0, t_up-10.0);
    //
    // 	t_proposed = t_e_arg.at(subject_proposed) + 0.1*gsl_ran_gaussian(r_c,1.0);
    //
    // 	switch((t_proposed<t_low)| (t_proposed>t_up)){
    // 		case 0:{
    // 			//do nothing
    // 		break;
    // 		}
    // 		case 1:{
    // 			switch(t_proposed<t_low){
    // 				case 0:{ // t_proposed>t_up
    //
    // 					switch((t_proposed - t_up)>(t_up - t_low)){
    //
    // 						case 0:{
    // 						t_proposed = t_up -(t_proposed - t_up);
    // 						break;
    // 						}
    //
    // 						case 1:{
    // 						t_proposed = t_up - fmod(t_proposed - t_up, t_up - t_low);
    // 						break;
    // 						}
    // 					}
    //
    // 				break;
    // 				}
    //
    // 				case 1:{//t_proposed<t_low
    //
    // 					switch(fabs(t_proposed - t_low)>(t_up - t_low)){
    //
    // 						case 0:{
    // 						t_proposed = t_low + fabs(t_proposed - t_low);
    // 						break;
    // 						}
    //
    // 						case 1:{
    // 						t_proposed = t_low + fmod(fabs(t_proposed - t_low), t_up - t_low);
    // 						break;
    // 						}
    // 					}
    //
    // 				break;
    // 				}
    // 			}
    // 		break;
    // 		}
    // 	}
    
    //--
    
    
    break;
  }
    
  default :{ // not by background
    
    nt_modified_source = nt_current_arg.at(subject_source);
    t_nt_modified_source = t_nt_current_arg.at(subject_source);
    
    t_up =   min( t_sample_arg.at(subject_proposed), min( min( t_i_arg.at(subject_proposed), t_r_arg.at(subject_source)),t_max_CUPDATE));
    t_low = max(t_i_arg.at(subject_source), t_up - para_priors_arg.t_back );
    
    // 	double t_temp_2 = min(t_sample_arg.at(subject_proposed), t_r_arg.at(subject_source));
    // 	switch( t_temp_2!=unassigned_time_CUPDATE){
    //
    // 		case 1:{// with valid t_s (subject) or t_r(source)
    // 			double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
    // 			switch(t_temp< t_temp_2){
    // 				case 1:{
    // 					t_low =  max(t_i_arg.at(subject_source), t_temp);
    // 				break;
    // 				}
    // 				case 0:{// should be unlikely if t_back is large enough
    // 					double dt = t_temp -  t_temp_2;
    // 					t_low =  max(t_i_arg.at(subject_source), t_temp_2 -  dt);
    // 				break;
    // 				}
    // 			}
    // 		break;
    // 		}
    //
    // 		case 0:{ // no with valid t_s (subject) and t_r(source)
    // 			t_low =  max(t_i_arg.at(subject_source), t_up - t_back);
    // 		break;
    // 		}
    // 	}
    
    
    t_proposed= runif(t_low, t_up, rng);
    
    
    //-----
    
    // 	t_up =   min( t_sample_arg.at(subject_proposed), min( min( t_i_arg.at(subject_proposed), t_r_arg.at(subject_source)),t_max_CUPDATE));
    // 	t_low = max(t_i_arg.at(subject_source), t_up -10.0 );
    //
    // 	t_proposed = t_e_arg.at(subject_proposed) + 0.1*gsl_ran_gaussian(r_c,1.0);
    //
    // 	switch((t_proposed<t_low)| (t_proposed>t_up)){
    // 		case 0:{
    // 			//do nothing
    // 		break;
    // 		}
    // 		case 1:{
    // 			switch(t_proposed<t_low){
    // 				case 0:{ // t_proposed>t_up
    //
    // 					switch((t_proposed - t_up)>(t_up - t_low)){
    //
    // 						case 0:{
    // 						t_proposed = t_up -(t_proposed - t_up);
    // 						break;
    // 						}
    //
    // 						case 1:{
    // 						t_proposed = t_up - fmod(t_proposed - t_up, t_up - t_low);
    // 						break;
    // 						}
    // 					}
    //
    // 				break;
    // 				}
    //
    // 				case 1:{//t_proposed<t_low
    //
    // 					switch(fabs(t_proposed - t_low)>(t_up - t_low)){
    //
    // 						case 0:{
    // 						t_proposed = t_low + fabs(t_proposed - t_low);
    // 						break;
    // 						}
    //
    // 						case 1:{
    // 						t_proposed = t_low + fmod(fabs(t_proposed - t_low), t_up - t_low);
    // 						break;
    // 						}
    // 					}
    //
    // 				break;
    // 				}
    // 			}
    // 		break;
    // 		}
    // 	}
    
    //--
    
    
    rank_source_x = (int)(distance( t_nt_current_arg.at(subject_source).begin(), find(t_nt_current_arg.at(subject_source).begin(), t_nt_current_arg.at(subject_source).end(), t_e_arg.at(subject_proposed)) ));
    
    
    t_nt_modified_source.erase(t_nt_modified_source.begin() + rank_source_x); // erase the original t_nt entry for source
    nt_modified_source.erase(nt_modified_source.begin()+n_base_CUPDATE*rank_source_x , nt_modified_source.begin()+n_base_CUPDATE*(rank_source_x+1) );  //erase the original nt entry for source
    
    break;
  }
    
  }
  
  t_e_modified.at(subject_proposed) = t_proposed;
  
  
  //----------------------------------------------------------------------------------//
  
  t_nt_modified_subject.push_back(t_proposed); // the unsorted t_nt with inserted t_proposed
  sort( t_nt_modified_subject.begin(),  t_nt_modified_subject.end()); // the sorted t_nt with inserted t_proposed
  
  //int rank_subject_y =distance( t_nt_modified_subject.begin(), find(t_nt_modified_subject.begin(), t_nt_modified_subject.end(), t_proposed) ); //count the NEW rank of the t_proposed  among t_nt_modifed.at(subject_proposed)
  int rank_subject_y = 0;
  
  //nt_modified_subject.insert(nt_modified_subject.begin()+(rank_subject_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());  //insert  the new nt
  
  int rank_source_y =-1;  //count the NEW rank of the t_proposed  among t_nt_modifed.at(subject_source)
  
  switch(subject_source ){
  
  case 9999:{ // by background
    break;
  }
    
  default :{ // not by background
    
    
    //vector <double>  t_nt_source (t_nt_modified.at(subject_source)) ;
    
    t_nt_modified_source.push_back(t_proposed);
    
    sort( t_nt_modified_source.begin(),  t_nt_modified_source.end());
    
    rank_source_y = (int)(distance(t_nt_modified_source.begin(), find(t_nt_modified_source.begin(),t_nt_modified_source.end(), t_proposed) ));
    
    //nt_modified_source.insert(nt_modified_source.begin()+(rank_source_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());
    
    //------------
    
    infecting_list_modified_source = infecting_list_current_arg.at(subject_source);
    
    int rank_x = (int)(distance(infecting_list_modified_source.begin(), find(infecting_list_modified_source.begin(), infecting_list_modified_source.end(), subject_proposed)));
    
    infecting_list_modified_source.erase(infecting_list_modified_source.begin()+rank_x);
    
    
    vector<double> t_y(infecting_size_current_arg.at(subject_source));
    for (int i=0;i<=(infecting_size_current_arg.at(subject_source)-1);i++){
      t_y.at(i) = t_e_modified.at(infecting_list_current_arg[subject_source][i]);
    }
    
    sort(t_y.begin(), t_y.end());
    
    int rank_y = (int)(distance(t_y.begin(), find(t_y.begin(), t_y.end(), t_e_modified.at(subject_proposed))));
    infecting_list_modified_source.insert(infecting_list_modified_source.begin()+rank_y, subject_proposed);
    
    //--------------
    
    
    
    
    break;
  }
    
  }
  
  
  
  
  //---------------------------------------- proposing a new sequence & the proposal probability ----------------------------------------------------------//
  
  vector<int> seq_proposed(n_base_CUPDATE);
  
  double dt;
  
  double t_past, t_future;
  
  double log_pr_forward=0.0; // the log of proposal probability
  
  vector<int> nt_past_forward(n_base_CUPDATE); // the sequence at the nearest past (in the direction of time change) compared to the time of the proposed sequence; this might be or might not be the original sequence which gotta be replaced
  vector<int> nt_future_forward(n_base_CUPDATE); // the sequence at the nearest future(in the direction of time change) compared to the time of the proposed sequence
  
  dt = t_proposed - t_e_arg.at(subject_proposed); // the dimension of dt tells the direction of time change
  
  //
  
  double t_proposed_backward;
  
  vector<int> seq_proposed_backward(n_base_CUPDATE);
  
  double t_past_backward, t_future_backward;
  
  double log_pr_backward=0.0; // the log of proposal probability
  
  vector<int> nt_past_backward(n_base_CUPDATE);
  vector<int> nt_future_backward(n_base_CUPDATE);
  //
  
  switch(int (subject_source==9999)){
  case 0:{ // NOT from background
    
    switch(int (current_size_arg.at(subject_proposed)>1)){ // return 1 when the subject has more than one sequence available
    
  case 0:{ //  ONLY one sequence available for the subject
    
    switch(int (dt>=0)){
  case 1:{ // propose the time to the right
    
    switch(int (rank_source_x==rank_source_y)){
    
  case 1:{ // unchanged rank_source
    nt_past_forward = nt_current_seq;
    t_past = t_e_arg.at(subject_proposed);
    
    switch(int (rank_source_y==(current_size_arg.at(subject_source)-1))){ // see if the sequence is the last one
    case 1:{ // it is the last sequence
      t_future = t_proposed;
      seq_propose_uncond(seq_proposed, log_pr_forward, nt_past_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// no defined nt_future_forward
      
      //------------ backward direction------//
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward,  nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // compute the backward proposal pr with future sequence
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // note: the argument takes t_future_backwards with value t_proposed_backward
      
      
      //------------------------------------------------//
      
      break;
    }
    case 0:{ // not the last sequence
      t_future = t_nt_modified_source.at(rank_source_y+1);
      
      nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+2)*n_base_CUPDATE); // the next sequence with rank=rank_source_y +1
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// with defined nt_future_forward
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
      
      //------------------------------------------------//
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    }
    
    break;
  }
    
  case 0:{ //changed rank_source (rank_source_y will be > rank_source_x as the time is proposed to the right)
    nt_past_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE); // note: start point is not rank_source_y - 1 due to change of ranking of the sequence now before the proposed sequence
    
    t_past = t_nt_modified_source.at(rank_source_y-1);
    
    
    switch(int (rank_source_y==(current_size_arg.at(subject_source)-1))){ // see if the sequence is the last one
    case 1:{ // it is the last sequence
      t_future = t_proposed;
      seq_propose_uncond(seq_proposed,  log_pr_forward,nt_past_forward, t_proposed, t_past, t_future,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// no defined nt_future_forward
      
      //------------------------------------------------//
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    case 0:{ // not the last sequence
      t_future = t_nt_modified_source.at(rank_source_y+1);
      
      nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+2)*n_base_CUPDATE); // the next sequence with rank=rank_source_y +1; note: still rank_source_y+1 as the ranking of the sequence that still after proposed sequence deos not change
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// with defined nt_future_forward
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
      
      //------------------------------------------------//
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    }
    
    break;
  }
  }
    
    break;
  }
    
  case 0:{  // propose the time to the left
    
    switch(int (rank_source_x==rank_source_y)){
    
  case 1:{ // unchanged rank_source
    
    nt_past_forward = nt_current_seq;
    t_past = t_e_arg.at(subject_proposed);
    
    t_future = t_nt_modified_source.at(rank_source_y-1);
    
    nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE); // the previous sequence with rank=rank_source_y - 1 (it always exists as there is a least one sequence corresponds to the infection of the source itself
    
    seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE,rng);// with defined nt_future_forward
    //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    
    //------------------------------------------------//
    
    switch(int (rank_source_x==(current_size_arg.at(subject_source)-1))){ // see if the (original) sequence was the last one
    
    case 0:{ // it was not the last sequence
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x+1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
      
    case 1:{ // it was the last sequence
      
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_proposed_backward;
      
      seq_backward_pr_uncond(seq_proposed_backward,  log_pr_backward,nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // no future sequence
      
      break;
    }
    }
    //------------------------------------------------//
    
    break;
  }
    
  case 0:{ // changed rank_source (rank_source_y will be < rank_source_x as the time is proposed to the left)
    
    nt_past_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE); // note: start point is not rank_source_y - 1 due to change of ranking of the sequence now after the proposed sequence
    
    t_past = t_nt_modified_source.at(rank_source_y+1);
    
    t_future = t_nt_modified_source.at(rank_source_y-1);
    
    nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE); // the next sequence with rank=rank_source_y -1; note: still rank_source_y-1 as the ranking of the sequence that still before proposed sequence deos not change
    
    seq_propose_cond(seq_proposed,  log_pr_forward,nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future,para_current_arg.mu_1, para_current_arg.mu_2,  n_base_CUPDATE, rng);// with defined nt_future_forward
    //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    
    //------------------------------------------------//
    
    switch(int(rank_source_x==(current_size_arg.at(subject_source)-1))){ // see if the (original) sequence was the last one
    
    case 0:{ // it was not the last sequence
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x+1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
      
    case 1:{ // it was the last sequence
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_proposed_backward;
      
      seq_backward_pr_uncond(seq_proposed_backward,  log_pr_backward,nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // no future sequence
      
      break;
    }
    }
    //------------------------------------------------//
    
    break;
  }
    
  }
    
    break;
  }
  }
    
    break;
  }
    
    
  case 1:{ //  MORE than one sequence available for the subject
    
    switch(int (dt>=0)){
  case 1:{ // propose the time to the right
    
    switch(int(rank_source_x==rank_source_y)){
    
  case 1:{ // unchanged rank_source
    nt_past_forward = nt_current_seq;
    t_past = t_e_arg.at(subject_proposed);
    
    switch(int(rank_source_y==(current_size_arg.at(subject_source)-1))){ // see if the sequence is the last one (on source)
    case 1:{ // it is the last sequence
      
      //t_future = t_proposed;
      //seq_propose_uncond(seq_proposed, log_pr_forward, nt_past_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);// no defined nt_future_forward
      
      t_future = t_nt_modified_subject.at(1);
      nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE); // the 2nd sequence; it cannot take over 2nd sequence as 2nd sequence only happens after time of becoming infectious where t_proposed cannot exceed
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
      
      
      //------------ backward direction------//
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward,  nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // compute the backward proposal pr with future sequence
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE); // note: the argument takes t_future_backwards with value t_proposed_backward
      
      
      //------------------------------------------------//
      
      break;
    }
    case 0:{ // not the last sequence
      
      switch(int(t_nt_modified_source.at(rank_source_y+1)<t_nt_modified_subject.at(1))){
    case 1:{
      t_future = t_nt_modified_source.at(rank_source_y+1);
      nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+2)*n_base_CUPDATE); // the next sequence with rank=rank_source_y +1
      break;
    }
    case 0:{
      t_future = t_nt_modified_subject.at(1);
      nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
      break;
    }
    }
      
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// with defined nt_future_forward
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
      
      //------------------------------------------------//
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    }
    
    break;
  }
    
  case 0:{ //changed rank_source (rank_source_y will be > rank_source_x as the time is proposed to the right)
    nt_past_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE); // note: start point is not rank_source_y - 1 due to change of ranking of the sequence now before the proposed sequence
    
    t_past = t_nt_modified_source.at(rank_source_y-1);
    
    
    switch(int(rank_source_y==(current_size_arg.at(subject_source)-1))){ // see if the sequence is the last one
    case 1:{ // it is the last sequence
      
      //t_future = t_proposed;
      //seq_propose_uncond(seq_proposed,  log_pr_forward,nt_past_forward, t_proposed, t_past, t_future,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);// no defined nt_future_forward
      
      t_future = t_nt_modified_subject.at(1);
      nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE); // the 2nd sequence; it cannot take over 2nd sequence as 2nd sequence only happens after time of becoming infectious where t_proposed cannot exceed
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
      
      
      //------------------------------------------------//
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    case 0:{ // not the last sequence
      
      switch(int(t_nt_modified_source.at(rank_source_y+1)<t_nt_modified_subject.at(1))){
    case 1:{
      t_future = t_nt_modified_source.at(rank_source_y+1);
      nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+2)*n_base_CUPDATE); // the next sequence with rank=rank_source_y +1
      break;
    }
    case 0:{
      t_future = t_nt_modified_subject.at(1);
      nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
      break;
    }
    }
      
      seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);// with defined nt_future_forward
      //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
      
      //------------------------------------------------//
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_source.at(rank_source_x-1);
      nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //------------------------------------------------//
      
      break;
    }
    }
    
    break;
  }
  }
    
    break;
  }
    
  case 0:{  // propose the time to the left
    
    switch(int(rank_source_x==rank_source_y)){
    
  case 1:{ // unchanged rank_source
    
    nt_past_forward = nt_current_seq;
    t_past = t_e_arg.at(subject_proposed);
    
    t_future = t_nt_modified_source.at(rank_source_y-1);
    
    nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE); // the previous sequence with rank=rank_source_y - 1 (it always exists as there is a least one sequence corresponds to the infection of the source itself
    
    seq_propose_cond(seq_proposed,  log_pr_forward, nt_past_forward, nt_future_forward, t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE,rng);// with defined nt_future_forward
    //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    
    //------------------------------------------------//
    
    switch(int(rank_source_x==(current_size_arg.at(subject_source)-1))){ // see if the (original) sequence was the last one
    
    case 0:{ // it was not the last sequence
      
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      
      switch(int(t_nt_modified_source.at(rank_source_x+1)<t_nt_modified_subject.at(1))){
      case 1:{
        t_future_backward = t_nt_modified_source.at(rank_source_x+1);
        nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
        break;
      }
      case 0:{
        t_future_backward = t_nt_modified_subject.at(1);
        nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
        break;
      }
      }
      
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
      
    case 1:{ // it was the last sequence
      
      t_past_backward = t_proposed;
      nt_past_backward = seq_proposed;
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_subject.at(1);
      nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      
      break;
    }
    }
    //------------------------------------------------//
    
    break;
  }
    
  case 0:{ // changed rank_source (rank_source_y will be < rank_source_x as the time is proposed to the left)
    
    nt_past_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y+1)*n_base_CUPDATE); // note: start point is not rank_source_y - 1 due to change of ranking of the sequence now after the proposed sequence
    
    t_past = t_nt_modified_source.at(rank_source_y+1);
    
    t_future = t_nt_modified_source.at(rank_source_y-1);
    
    nt_future_forward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_y-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_y)*n_base_CUPDATE); // the next sequence with rank=rank_source_y -1; note: still rank_source_y-1 as the ranking of the sequence that still before proposed sequence deos not change
    
    seq_propose_cond(seq_proposed,  log_pr_forward,nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future,para_current_arg.mu_1, para_current_arg.mu_2,  n_base_CUPDATE, rng);// with defined nt_future_forward
    //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_current_seq, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
    
    //------------------------------------------------//
    
    switch(int(rank_source_x==(current_size_arg.at(subject_source)-1))){ // see if the (original) sequence was the last one
    
    case 0:{ // it was not the last sequence
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      switch(int(t_nt_modified_source.at(rank_source_x+1)<t_nt_modified_subject.at(1))){
      case 1:{
        t_future_backward = t_nt_modified_source.at(rank_source_x+1);
        nt_future_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x+2)*n_base_CUPDATE);
        break;
      }
      case 0:{
        t_future_backward = t_nt_modified_subject.at(1);
        nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
        break;
      }
      }
      
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
      
    case 1:{ // it was the last sequence
      
      t_past_backward = t_nt_modified_source.at(rank_source_x);
      nt_past_backward.assign(nt_current_arg.at(subject_source).begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_arg.at(subject_source).begin()+(rank_source_x)*n_base_CUPDATE);
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward = nt_current_seq;
      
      t_future_backward = t_nt_modified_subject.at(1);
      nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
      
      seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, seq_proposed,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      
      break;
    }
    }
    //------------------------------------------------//
    
    break;
  }
    
  }
    
    break;
  }
  }
    
    break;
  }
    
  }
    
    break;
  }
    
  case 1:{ //  from background
    
    // 		sample_snull (con_seq_CUPDATE, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE, r_c); //sample a seq for background
    // 		log_pr_forward = lh_snull(con_seq_CUPDATE, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
    //
    // 		log_pr_backward = lh_snull(con_seq_CUPDATE, nt_current_seq, para_current_arg.p_ber, n_base_CUPDATE);
    
    
    switch(int (current_size_arg.at(subject_proposed)>1)){ // return 1 when the subject has more than one sequence available
    
  case 0:{ //  ONLY one sequence available for the subject
    
    //------------------------------------------------//
    
    // 				t_past = t_e_arg.at(subject_proposed);
    // 				nt_past_forward = nt_current_seq;
    // 				t_future = t_proposed;
    // 				seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_future,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);// no defined nt_future_forward
    // 				//------------------------------------------------//
    //
    // 				t_past_backward = t_proposed;
    // 				nt_past_backward =  seq_proposed;
    //
    // 				t_proposed_backward = t_e_arg.at(subject_proposed);
    // 				seq_proposed_backward = nt_current_seq;
    //
    // 				t_future_backward =t_proposed_backward;
    //
    // 				seq_backward_pr_uncond(seq_proposed_backward,  log_pr_backward,nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    //------------------------------------------------//
    
    sample_snull (con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE, rng); //sample a seq for background
    log_pr_forward = lh_snull(con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
    
    log_pr_backward = lh_snull(con_seq, nt_current_seq, para_current_arg.p_ber, n_base_CUPDATE);
    
    break;
  }
    
  case 1:{ // MORE than one sequence available for the subject
    
    switch(int (dt>=0)){
  case 1:{ // propose the time to the right
    t_past = t_e_arg.at(subject_proposed);
    nt_past_forward = nt_current_seq;
    
    t_future = t_nt_modified_subject.at(1);
    
    
    nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE); // the 2nd sequence; it cannot take over 2nd sequence as 2nd sequence only happens after time of becoming infectious where t_proposed cannot exceed
    
    seq_propose_cond(seq_proposed,  log_pr_forward,nt_past_forward, nt_future_forward, t_proposed,  t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
    //seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_proposed, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    //------------------------------------------------//
    
    t_past_backward = t_proposed;
    nt_past_backward =  seq_proposed;
    
    t_proposed_backward = t_e_arg.at(subject_proposed);
    seq_proposed_backward = nt_current_seq;
    
    t_future_backward =t_proposed_backward;
    
    seq_backward_pr_uncond(seq_proposed_backward,  log_pr_backward,nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    
    //------------------------------------------------//
    
    break;
  }
    
  case 0:{  // propose the time to the left
    t_past = t_e_arg.at(subject_proposed);
    nt_past_forward = nt_current_seq;
    t_future = t_proposed;
    seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_future,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
    
    //------------------------------------------------//
    
    t_past_backward = t_proposed;
    nt_past_backward =  seq_proposed;
    
    t_proposed_backward = t_e_arg.at(subject_proposed);
    seq_proposed_backward = nt_current_seq;
    
    t_future_backward = t_nt_modified_subject.at(1);
    nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
    
    seq_backward_pr_cond(seq_proposed_backward, log_pr_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward, t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2,  n_base_CUPDATE);
    //seq_backward_pr_uncond(seq_proposed_backward, log_pr_backward, nt_past_backward,  t_proposed_backward, t_past_backward, t_proposed_backward,para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    //------------------------------------------------//
    
    break;
  }
  }
    
    break;
  }
    
  }
    
    break;
  }
  }
  
  //-------------------------------------------end of proposing a new sequence ---------------------------------------------------------------------------------//
  
  
  
  nt_modified_subject.insert(nt_modified_subject.begin()+(rank_subject_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());  //insert  the new nt
  
  switch(subject_source ){
  
  case 9999:{ // by background
    break;
  }
    
  default :{ // not by background
    
    nt_modified_source.insert(nt_modified_source.begin()+(rank_source_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());
    
    break;
  }
    
  }
  
  
  //----------------------------------------------------------------------------------//
  
  switch (int (current_size_arg.at(subject_proposed)>1)) {
  
  case 1:{
    
    log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(subject_proposed); //subtract part of likelihood that would be updated below
    
    lh_square_modified.log_f_S.at(subject_proposed) = 0.0;
    
    for (int j=0;j<=(current_size_arg.at(subject_proposed)-2);j++){
      
      
      vector<int> seq_1(nt_modified_subject.begin()+j*(n_base_CUPDATE), nt_modified_subject.begin()+(j+1)*(n_base_CUPDATE));
      vector<int> seq_2(nt_modified_subject.begin()+(j+1)*(n_base_CUPDATE), nt_modified_subject.begin()+(j+2)*(n_base_CUPDATE));
      
      lh_square_modified.log_f_S.at(subject_proposed) =lh_square_modified.log_f_S.at(subject_proposed) + log_lh_seq(seq_1, seq_2, t_nt_modified_subject.at(j), t_nt_modified_subject.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
    }
    
    log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(subject_proposed);
    
    break;
  }
    
  default:{
    break;
  }
    
  }
  
  //----------------------------------------------------------------------------------//
  
  switch (subject_source){
  
  case 9999:{ // by background
    
    log_lh_modified =  log_lh_modified - lh_square_modified.log_f_Snull.at(subject_proposed);
    
    lh_square_modified.log_f_Snull.at(subject_proposed)  = lh_snull(con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
    
    log_lh_modified =  log_lh_modified + lh_square_modified.log_f_Snull.at(subject_proposed);
    
    break;
  }
    
  default :{ // not by background
    
    
    switch (int(current_size_arg.at(subject_source)>1)) {
    
  case 1:{
    
    log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(subject_source); //subtract part of likelihood that would be updated below
    
    lh_square_modified.log_f_S.at(subject_source) = 0.0;
    
    for (int j=0;j<=(current_size_arg.at(subject_source)-2);j++){
      
      
      vector<int> seq_1(nt_modified_source.begin()+j*(n_base_CUPDATE), nt_modified_source.begin()+(j+1)*(n_base_CUPDATE));
      vector<int> seq_2(nt_modified_source.begin()+(j+1)*(n_base_CUPDATE), nt_modified_source.begin()+(j+2)*(n_base_CUPDATE));
      
      lh_square_modified.log_f_S.at(subject_source) =lh_square_modified.log_f_S.at(subject_source)+log_lh_seq(seq_1, seq_2, t_nt_modified_source.at(j), t_nt_modified_source.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
    }
    
    log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(subject_source);
    
    break;
  }
    
  default:{
    break;
  }
    
  }
    break;
  }
    
  }
  
  //----------------------------------------------------------------------------------//
  
  lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
  lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
  lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
  lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
  
  for (int j=0;j<=(int) (xi_I_arg.size()-1);j++){ // update delta_mat; and pre-assign k_sum_E & kt_sum_E which might be changed later again
    
    if (t_i_arg.at(xi_I_arg.at(j))<t_proposed) {
      
      switch (int(t_r_arg.at(xi_I_arg.at(j))>=t_proposed)) {
      case 1:{
    delta_mat_modified[subject_proposed][xi_I_arg.at(j)] = t_proposed - t_i_arg.at(xi_I_arg.at(j));
    
    delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = 0.0;
    for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
      if ((mov_arg.from_k[m] == xi_I_arg.at(j)) && (mov_arg.to_k[m] == subject_proposed)) {
        if ((mov_arg.t_m[m] >= t_i_arg.at(xi_I_arg.at(j))) &&
            (mov_arg.t_m[m] <= t_proposed)) {
          delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] + (t_proposed - mov_arg.t_m[m]);
        }
      }
    }
    
    //lh_square_modified.k_sum_E.at(subject_proposed) =  lh_square_modified.k_sum_E.at(subject_proposed) + kernel_mat_current_arg[subject_proposed][xi_I_arg.at(j)]/norm_const_current_arg.at(xi_I_arg.at(j)) ;
    
    break;
  }
      case 0:{
        delta_mat_modified[subject_proposed][xi_I_arg.at(j)] = t_r_arg.at(xi_I_arg.at(j)) - t_i_arg.at(xi_I_arg.at(j));
        
        delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = 0.0;
        for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
          if ((mov_arg.from_k[m] == xi_I_arg.at(j)) && (mov_arg.to_k[m] == subject_proposed)) {
            if ((mov_arg.t_m[m] >= t_i_arg.at(xi_I_arg.at(j))) &&
                (mov_arg.t_m[m] <= t_r_arg.at(xi_I_arg.at(j)))) {
              delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] + (t_r_arg.at(xi_I_arg.at(j)) - mov_arg.t_m[m]);
            }
          }
        }
        
        break;
      }
      }//end switch
      
      if (opt_betaij == 0) {
        lh_square_modified.kt_sum_E.at(subject_proposed) = lh_square_modified.kt_sum_E.at(subject_proposed) + delta_mat_modified[subject_proposed][xi_I_arg.at(j)] * kernel_mat_current_arg[subject_proposed][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
      }
      if (opt_betaij == 1) {
        lh_square_modified.kt_sum_E.at(subject_proposed) = lh_square_modified.kt_sum_E.at(subject_proposed) + delta_mat_modified[subject_proposed][xi_I_arg.at(j)] * beta_ij_mat_current_arg[xi_I_arg.at(j)][subject_proposed] * kernel_mat_current_arg[subject_proposed][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
      }
      
      if (opt_mov == 0) {
        lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
      }
      
      if (opt_mov == 1) {
        lh_square_modified.movest_sum_E.at(subject_proposed) = lh_square_modified.movest_sum_E.at(subject_proposed) + delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)];
      }
      if (opt_mov == 2) {
        lh_square_modified.movest_sum_E.at(subject_proposed) = lh_square_modified.movest_sum_E.at(subject_proposed); //no change
      }
      
    }
  }
  
  //----------
  switch(infected_source_current_arg.at(subject_proposed)){
  
  case 9999:{ // by background
    //lh_square_modified.k_sum_E.at(subject_proposed) = 0.0; // update k_sum_E
    lh_square_modified.g_E.at(subject_proposed) =  para_current_arg.alpha;
    break;
  }
    
  default :{ // not by background
    
    if (opt_betaij == 0) {
    lh_square_modified.k_sum_E.at(subject_proposed) = kernel_mat_current_arg[subject_proposed][infected_source_current_arg.at(subject_proposed)] / norm_const_current_arg.at(infected_source_current_arg.at(subject_proposed)); // update k_sum_E
  }
    if (opt_betaij == 1) {
      lh_square_modified.k_sum_E.at(subject_proposed) = beta_ij_mat_current_arg[infected_source_current_arg.at(subject_proposed)][subject_proposed] * kernel_mat_current_arg[subject_proposed][infected_source_current_arg.at(subject_proposed)] / norm_const_current_arg.at(infected_source_current_arg.at(subject_proposed)); // update k_sum_E
    }
    
    if (opt_mov == 0) {
      lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
    }
    if ((opt_mov == 1) | (opt_mov == 2)) {
      double moves_ij_t = func_moves_cnt(infected_source_current_arg.at(subject_proposed), subject_proposed, mov_arg, t_e_modified, t_i_arg, t_r_arg, para_priors_arg);
      lh_square_modified.moves_sum_E.at(subject_proposed) = moves_ij_t;
    }
    
    
    lh_square_modified.g_E.at(subject_proposed) = para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(subject_proposed);
    
    
    break;
  }
    
  }
  
  
  //---------------this part is needed when do not update index-----------------
  
  // log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
  //
  // lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed);
  // // 		lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
  // lh_square_modified.h_E.at(subject_proposed) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(subject_proposed),1.0);
  // lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
  //
  // log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
  
  //--------------------------------
  
  switch (find(index_arg.begin(),index_arg.end(),subject_proposed)==index_arg.end()) { // return 1 if proposed subject not one of the original indexes
  
  case 1:{
    
    switch(int(t_proposed<t_e_arg.at(index_arg.at(0)))){ // original indexes would be replace by the chosen subject
    
  case 1:{
    
    
    index_modified.clear();
    index_modified.assign(1,subject_proposed);// replace index
    xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),subject_proposed));
    
    for ( int i =0; i<= (int) (index_arg.size()-1); i++){
      xi_E_minus_modified.push_back(index_arg.at(i)); // the original indexes in xi_E_minus now
    }
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
    
    lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.q_E.at(subject_proposed)=0.0;
    lh_square_modified.g_E.at(subject_proposed)=1.0;
    lh_square_modified.h_E.at(subject_proposed)=1.0;
    lh_square_modified.f_E.at(subject_proposed)=1.0;
    lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
    lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
    
    
    for (int i=0; i<=(int) (index_arg.size()-1);i++){ // the original indexes have to be acocunted in likelihood now
      
      //log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(index_arg.at(i)));
      
      lh_square_modified.g_E.at(index_arg.at(i)) = para_current_arg.alpha; // this is not the subject_proposed
      lh_square_modified.q_E.at(index_arg.at(i)) =para_current_arg.alpha*t_e_arg.at(index_arg.at(i));
      lh_square_modified.h_E.at(index_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(index_arg.at(i)));
      lh_square_modified.f_E.at(index_arg.at(i)) = lh_square_modified.g_E.at(index_arg.at(i))*lh_square_modified.h_E.at(index_arg.at(i));
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(index_arg.at(i)));
    }
    
    break;
  }
    
    
  case 0:{
    
    if (t_proposed==t_e_arg.at(index_arg.at(0))){ // addtion of one more index
    
    
    //if (find(index_arg.begin(),index_arg.end(),subject_proposed)==index_arg.end()){ // return 1 if proposed subject not one of the original indexes
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); // this subject would have to be removed from likelihood function
    index_modified.push_back(subject_proposed); // add index
    xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),subject_proposed)); // removed from xi_E_minus
    lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.q_E.at(subject_proposed)=0.0;
    lh_square_modified.g_E.at(subject_proposed)=1.0;
    lh_square_modified.h_E.at(subject_proposed)=1.0;
    lh_square_modified.f_E.at(subject_proposed)=1.0;
    lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
    lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
    
    //}
    
  }
    
    if (t_proposed>t_e_arg.at(index_arg.at(0))){ // no shift of cases between xi_E and xi_E_minus
      
      
      
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
      
      lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(subject_proposed);
      // 		lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
      lh_square_modified.h_E.at(subject_proposed) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(subject_proposed));
      lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
      
      
    } // end if t_proposs>t_e_arg.at()
    
    break;
  }
    
  }
    
    
    break;
  }
    
    
  case 0: { // when chosen subject is one of the indexes
    
    
    
    index_modified.clear();
    
    int first_min = (int)(distance(t_e_modified.begin(), min_element(t_e_modified.begin(), t_e_modified.end())));
    double min_t = t_e_modified.at(first_min); // the minimum time of exposure
    
    int num_min = (int) count(t_e_modified.begin(), t_e_modified.end(), min_t); // numberof subects with the min exposure time
    
    
    switch (int(num_min>1)) {
    case 1: {
      index_modified.reserve(n_CUPDATE);
      for (int i=0; i<=(n_CUPDATE-1);i++){
        if (t_e_modified.at(i)==min_t ) index_modified.push_back(i);
      }
      break;
    }
    case 0:{
      index_modified.assign(1,first_min);
      break;
    }
      
    }
    
    xi_E_minus_modified = xi_E_arg;
    
    
    for (int i=0;i<= (int) (index_modified.size()-1); i++){
      
      xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),index_modified.at(i)));
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(index_modified.at(i))); // this subject would have to be removed from likelihood function ( new index might be orginally an index, but the the log(lh_square_modified.f_E.at(index_modified.at(i)) will be zero in this case)
      //lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
      //lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
      //lh_square_modified.q_E.at(subject_proposed)=0.0;
      //lh_square_modified.g_E.at(subject_proposed)=1.0;
      //lh_square_modified.h_E.at(subject_proposed)=1.0;
      //lh_square_modified.f_E.at(subject_proposed)=1.0;
      
      lh_square_modified.k_sum_E.at(index_modified.at(i))=0.0;
      lh_square_modified.kt_sum_E.at(index_modified.at(i))=0.0;
      lh_square_modified.q_E.at(index_modified.at(i))=0.0;
      lh_square_modified.g_E.at(index_modified.at(i))=1.0;
      lh_square_modified.h_E.at(index_modified.at(i))=1.0;
      lh_square_modified.f_E.at(index_modified.at(i))=1.0;
      lh_square_modified.moves_sum_E.at(index_modified.at(i)) = 0.0;
      lh_square_modified.movest_sum_E.at(index_modified.at(i)) = 0.0;
      
    }
    
    switch(find(index_modified.begin(),index_modified.end(),subject_proposed) ==index_modified.end() ){ //return 1 when the chosen  subject is NO longer an index
    case 1:{
      
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed));
      
      lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(subject_proposed);
      //lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
      lh_square_modified.h_E.at(subject_proposed) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(subject_proposed));
      lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
      
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
      
      break;
    }
    case 0:{
      
      
      break;
    }
    }
    
    break;
  }
    
  }
  
  //--------------------//
  
  switch ( find(xi_I_arg.begin(), xi_I_arg.end(),subject_proposed) != (xi_I_arg.end()) ) { //return 1 when the subject is also in xi_I
  case 1:{
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_I.at(subject_proposed)); //subtract part of likelihood that would be updated below
    lh_square_modified.f_I.at(subject_proposed) = func_latent_pdf(t_i_arg.at(subject_proposed) - t_proposed, para_current_arg.lat_mu, para_current_arg.lat_var);
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_I.at(subject_proposed));
    
    
    break;
  }
  case 0:{
    break;
  }
  }
  
  //----------
  
  switch ( find(xi_EnI_arg.begin(), xi_EnI_arg.end(),subject_proposed) != (xi_EnI_arg.end()) ) { //return 1 when the subject is also in xi_EnI
  case 1:{
    
    log_lh_modified = log_lh_modified - log(lh_square_modified.f_EnI.at(subject_proposed)); //subtract part of likelihood that would be updated below
    lh_square_modified.f_EnI.at(subject_proposed) = func_latent_surv(t_max_CUPDATE - t_proposed, para_current_arg.lat_mu, para_current_arg.lat_var);
    log_lh_modified = log_lh_modified + log(lh_square_modified.f_EnI.at(subject_proposed));
    
    
    
    break;
  }
  case 0:{
    break;
  }
  }
  
  //----------
  
  // switch(isfinite(exp(log_lh_modified-log_lh_current_arg)*exp(log_pr_backward-log_pr_forward))){
  // 	case 1:{
  // 		acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*exp(log_pr_backward-log_pr_forward));
  // 	break;
  // 	}
  //
  // 	case 0:{
  // 		acp_pr =0.0;
  // 	break;
  // 	}
  // }
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*exp(log_pr_backward-log_pr_forward));
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg)+(log_pr_backward-log_pr_forward)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    delta_mat_current_arg = delta_mat_modified;
    delta_mat_mov_current_arg = delta_mat_mov_modified;
    log_lh_current_arg = log_lh_modified;
    t_e_arg= t_e_modified;
    index_arg = index_modified;
    xi_E_minus_arg = xi_E_minus_modified;
    
    // nt_current_arg = nt_modified;
    // t_nt_current_arg = t_nt_modified;
    
    nt_current_arg.at(subject_proposed) = nt_modified_subject;
    t_nt_current_arg.at(subject_proposed) = t_nt_modified_subject;
    
    
    switch (subject_source){
    
    case 9999:{ // by background
      break;
    }
      
    default :{ // not by background
      nt_current_arg.at(subject_source) = nt_modified_source;
      t_nt_current_arg.at(subject_source) = t_nt_modified_source;
      infecting_list_current_arg.at(subject_source) = infecting_list_modified_source;
    }
    }
    
    
    break;
  }
    
  case 0: {
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
  
}

/*------------------------------------------------*/

void mcmc_UPDATE::source_t_e_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg, vec2int& infecting_list_current_arg, vector<int>& infecting_size_current_arg, vector<int>&  xi_beta_E_arg, int& subject_proposed, vector<int>& list_update,vector<int>& con_seq, para_priors_etc& para_priors_arg, para_scaling_factors& para_sf_arg, vector< vector<double> >& beta_ij_mat_current_arg, moves_struct& moves_arg, int iter, rng_type & rng_arg, vector< vector<double> >& delta_mat_mov_current_arg){
  
  //double t_back =10.0;
  
  double acp_pr = 0.0;
  double t_low, t_up;
  
  double log_pr_forward=0.0;
  double log_pr_backward=0.0;
  
  double log_pr_t_e_forward=0.0;
  double log_pr_t_e_backward=0.0;
  
  double log_pr_seq_forward=0.0;
  double log_pr_seq_backward=0.0;
  
  double log_pr_ds_forward=0.0;
  double log_pr_ds_backward=0.0;
  
  
  double t_proposed;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  vector< vector<double> > delta_mat_modified = delta_mat_current_arg;
  vector< vector<double> > delta_mat_mov_modified = delta_mat_mov_current_arg;
  vector<double> t_e_modified = t_e_arg;
  
  vector <int> index_modified = index_arg;
  vector <int> xi_E_minus_modified = xi_E_minus_arg;
  
  // vector <int> xi_U_modified = xi_U_arg;
  // vector <int> xi_E_modified = xi_E_arg;
  // vector <int> xi_EnI_modified = xi_EnI_arg;
  
  vector<int> current_size_modified = current_size_arg;
  
  vec2int nt_modified = nt_current_arg;
  vec2d t_nt_modified = t_nt_current_arg;
  
  vec2int infecting_list_modified= infecting_list_current_arg;
  vector<int> infecting_size_modified = infecting_size_current_arg;
  
  // const gsl_rng_type* T_c= gsl_rng_default;  // T is pointer points to the type of generator
  // gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
  // gsl_rng_set (r_c,iter); // set a seed
  
  //vector<int> infected_source_modified = infected_source_current_arg;
  
  vector<int> nt_modified_subject = nt_current_arg.at(subject_proposed);
  vector<double> t_nt_modified_subject = t_nt_current_arg.at(subject_proposed);
  
  //vector<int> nt_subject_seq; // the orginal first sequence of the subject
  //nt_subject_seq.assign(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE);
  
  int source_x = infected_source_current_arg.at(subject_proposed);
  
  int rank_source_x; // the rank of removed sequence in old source (note: it has different meaning in function t_e_seq in which the source remains the same)
  vector<int> nt_current_source_x;
  vector<double> t_nt_current_source_x;
  vector<int> nt_modified_source_x;
  vector<double> t_nt_modified_source_x;
  
  
  
  //-------------- propose a new source --------------//
  
  int source_y;
  int rank_source_y;
  vector<int> nt_current_source_y;
  vector<double> t_nt_current_source_y;
  vector<int> nt_modified_source_y;
  vector<double> t_nt_modified_source_y;
  
  vector<int> source_pool; // vector contains the indices of possible source for the subject
  
  double t_bound = min(t_sample_arg.at(subject_proposed), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
  
  for (int i=0;i<=(int)(xi_I_arg.size()-1);i++){
    
    switch(int(t_i_arg.at(xi_I_arg.at(i))<t_bound)){
    //	switch( (t_i_arg.at(xi_I_arg.at(i))<t_bound) & ((t_bound  - t_r_arg.at(xi_I_arg.at(i)))<=t_back) ){
    
    case 1:{
    source_pool.push_back(xi_I_arg.at(i));
    break;
  }
    case 0:{
      break;
    }
    }
  }
  
  
  source_pool.insert(source_pool.begin(),9999);
  
  int num_infectious = (int)source_pool.size();
  
  //-----------------------------propose uniformly-------------------------------------------//
  
  //source_y = source_pool.at(gsl_rng_uniform_int(r_c, num_infectious)); // uniformly choose a new source (including bg)
  
  //-----propose according to infectious challenge--------------------//
  
  vector<double> ic(num_infectious);
  ic.at(0) = para_current_arg.alpha;
  
  switch(int(num_infectious>=2)){
  
  case 1:{ // with 2nd sources from pool
    
    for (int j=1;j<=(num_infectious-1);j++){
    
    
    if (opt_betaij == 0) {
      ic.at(j) = para_current_arg.beta * kernel_mat_current_arg[subject_proposed][source_pool.at(j)] / norm_const_current_arg.at(source_pool.at(j)); // a new source will be proposed according to the infectious challenges
    }
    if (opt_betaij == 1) {
      ic.at(j) = para_current_arg.beta*beta_ij_mat_current_arg[source_pool.at(j)][subject_proposed] * kernel_mat_current_arg[subject_proposed][source_pool.at(j)] / norm_const_current_arg.at(source_pool.at(j)); // a new source will be proposed according to the infectious challenges
    }
    if ((opt_mov == 1) | (opt_mov == 2)) {
      double moves_ij_t = func_moves_cnt(source_pool.at(j), subject_proposed, moves_arg, t_e_modified, t_i_arg, t_r_arg, para_priors_arg);
      ic.at(j) = ic.at(j) + para_current_arg.beta_m*moves_ij_t;
    }
    
  }
    
    //double *P=&ic.at(0); // convert vector to array
    //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc ((int)ic.size(),P);
    //int link= gsl_ran_discrete (r_c, g);
    //gsl_ran_discrete_free (g);
    int link = edf_sample(ic, rng);
    
    
    source_y = source_pool.at(link); // a new source
    log_pr_forward = log(ic.at(link));
    
    switch(int(source_x==9999)){
    case 0:{
      
      double ic_source_x = 0.0;
      if (opt_betaij == 0) {
        ic_source_x = para_current_arg.beta * kernel_mat_current_arg[subject_proposed][source_x] / norm_const_current_arg.at(source_x);
        
      }
      if (opt_betaij == 1) {
        ic_source_x = para_current_arg.beta*beta_ij_mat_current_arg[source_x][subject_proposed] * kernel_mat_current_arg[subject_proposed][source_x] / norm_const_current_arg.at(source_x);
      }
      
      if ((opt_mov == 1) | (opt_mov == 2)) {
        double moves_ij_t = func_moves_cnt(source_x, subject_proposed, moves_arg, t_e_modified, t_i_arg, t_r_arg, para_priors_arg);
        ic_source_x = ic_source_x + para_current_arg.beta_m*moves_ij_t;
      }
      
      log_pr_backward = log(ic_source_x);
      
      
      break;
    }
    case 1:{
      double ic_source_x =  para_current_arg.alpha;
      log_pr_backward =log(ic_source_x);
      break;
    }
    }
    
    
    break;
  }
    
  case 0:{ // only primary source from pool
    
    source_y = 9999;
    log_pr_forward = log(para_current_arg.alpha);
    
    double ic_source_x =  para_current_arg.alpha;
    log_pr_backward =log(ic_source_x);
    
    break;
  }
  }
  
  
  //--------------//
  
  //----------end of proposing a new source -----//
  
  //----------------------------------------------------------------------------------------------------------------//
  
  switch(int(source_y==source_x)){
  
  case 0:{
    
    
    //-------- propose a new t_e---------------//
    switch(source_y){
    
  case 9999:{ // by background
    
    t_up = min(t_sample_arg.at(subject_proposed), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
    t_low = max(0.0, t_up- para_priors_arg.t_back);
    
    // 			switch( t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
    //
    // 				case 1:{// with valid t_s
    // 					double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
    //
    // 					switch(t_temp< t_sample_arg.at(subject_proposed)){
    // 						case 1:{
    // 							t_low =  max(0.0, t_temp);
    // 						break;
    // 						}
    // 						case 0:{// should be unlikely if t_back is large enough
    // 							double dt = t_temp -  t_sample_arg.at(subject_proposed);
    // 							t_low =  max(0.0, t_sample_arg.at(subject_proposed) -  dt);
    // 						break;
    // 						}
    // 					}
    // 				break;
    // 				}
    //
    // 				case 0:{ // no valid t_s
    // 					t_low = max(0.0, t_up-t_back);
    // 				break;
    // 				}
    //
    //
    // 			}
    
    t_proposed= runif(t_low, t_up, rng);
    
    log_pr_t_e_forward = log(1.0/(t_up-t_low));
    
    break;
  }
    
  default :{ // not by background
    
    t_up = min(min(t_sample_arg.at(subject_proposed), t_r_arg.at(source_y)), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
    t_low = max(t_i_arg.at(source_y), t_up - para_priors_arg.t_back );
    
    // 			double t_temp_2 = min(t_sample_arg.at(subject_proposed), t_r_arg.at(source_y));
    // 			switch( t_temp_2!=unassigned_time_CUPDATE){
    //
    // 				case 1:{// with valid t_s (subject) or t_r(source)
    // 					double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
    // 					switch(t_temp< t_temp_2){
    // 						case 1:{
    // 							t_low =  max(t_i_arg.at(source_y), t_temp);
    // 						break;
    // 						}
    // 						case 0:{// should be unlikely if t_back is large enough
    // 							double dt = t_temp -  t_temp_2;
    // 							t_low =  max(t_i_arg.at(source_y), t_temp_2 -  dt);
    // 						break;
    // 						}
    // 					}
    // 				break;
    // 				}
    //
    // 				case 0:{ // no with valid t_s (subject) and t_r(source)
    // 					t_low =  max(t_i_arg.at(source_y), t_up - t_back);
    // 				break;
    // 				}
    // 			}
    
    
    t_proposed= runif(t_low, t_up, rng);
    
    log_pr_t_e_forward = log(1.0/(t_up-t_low));
    
    break;
  }
    
  }
    
    //------------//
    t_e_modified.at(subject_proposed) = t_proposed;
    //------------//
    
    switch(source_x){
    
    case 9999:{ // by background
      
      t_up = min( t_sample_arg.at(subject_proposed), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
      t_low = max(0.0, t_up- para_priors_arg.t_back);
      
      // 			switch( t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
      //
      // 				case 1:{// with valid t_s
      // 					double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
      //
      // 					switch(t_temp< t_sample_arg.at(subject_proposed)){
      // 						case 1:{
      // 							t_low =  max(0.0, t_temp);
      // 						break;
      // 						}
      // 						case 0:{// should be unlikely if t_back is large enough
      // 							double dt = t_temp -  t_sample_arg.at(subject_proposed);
      // 							t_low =  max(0.0, t_sample_arg.at(subject_proposed) -  dt);
      // 						break;
      // 						}
      // 					}
      // 				break;
      // 				}
      //
      // 				case 0:{ // no valid t_s
      // 					t_low = max(0.0, t_up-t_back);
      // 				break;
      // 				}
      //
      //
      // 			}
      
      
      log_pr_t_e_backward = log(1.0/(t_up-t_low));
      
      break;
    }
      
    default :{ // not by background
      
      t_up = min(min(t_sample_arg.at(subject_proposed), t_r_arg.at(source_x)), min(t_i_arg.at(subject_proposed), t_max_CUPDATE));
      t_low = max(t_i_arg.at(source_x), t_up - para_priors_arg.t_back );
      
      // 			double t_temp_2 = min(t_sample_arg.at(subject_proposed), t_r_arg.at(source_x));
      // 			switch( t_temp_2!=unassigned_time_CUPDATE){
      //
      // 				case 1:{// with valid t_s (subject) or t_r(source)
      // 					double t_temp = min( t_i_arg.at(subject_proposed), t_max_CUPDATE) - t_back;
      // 					switch(t_temp< t_temp_2){
      // 						case 1:{
      // 							t_low =  max(t_i_arg.at(source_x), t_temp);
      // 						break;
      // 						}
      // 						case 0:{// should be unlikely if t_back is large enough
      // 							double dt = t_temp -  t_temp_2;
      // 							t_low =  max(t_i_arg.at(source_x), t_temp_2 -  dt);
      // 						break;
      // 						}
      // 					}
      // 				break;
      // 				}
      //
      // 				case 0:{ // no with valid t_s (subject) and t_r(source)
      // 					t_low =  max(t_i_arg.at(source_x), t_up - t_back);
      // 				break;
      // 				}
      // 			}
      
      
      log_pr_t_e_backward = log(1.0/(t_up-t_low));
      
      break;
    }
      
    }
    
    //--------------------------------------//
    
    
    vector<int> seq_proposed(n_base_CUPDATE); // newly proposed sequence when source changes
    vector<int> nt_past_forward(n_base_CUPDATE);
    vector<int> nt_future_forward(n_base_CUPDATE);
    double t_past, t_future;
    
    vector<int> seq_proposed_backward(n_base_CUPDATE);
    vector<int> nt_past_backward(n_base_CUPDATE);
    vector<int> nt_future_backward(n_base_CUPDATE);
    double t_proposed_backward, t_past_backward, t_future_backward;
    
    //------------------------------------------------
    
    switch(int(source_y==9999)){
    
    case 0:{// new 2nd infection
      
      infecting_size_modified.at(source_y) = infecting_size_modified.at(source_y) +1;
      
      vector<double> t_y(infecting_size_current_arg.at(source_y));
      for (int i=0;i<=(infecting_size_current_arg.at(source_y)-1);i++){
        t_y.at(i) = t_e_arg.at(infecting_list_current_arg[source_y][i]);
      }
      t_y.push_back(t_proposed);
      sort(t_y.begin(), t_y.end());
      
      int rank_y = (int)(distance(t_y.begin(), find(t_y.begin(), t_y.end(), t_proposed)));
      infecting_list_modified.at(source_y).insert(infecting_list_modified.at(source_y).begin()+rank_y, subject_proposed);
      
      //----------------------------------------------------//
      
      nt_current_source_y = nt_current_arg.at(source_y);
      t_nt_current_source_y = t_nt_current_arg.at(source_y);
      
      nt_modified_source_y = nt_current_source_y;
      t_nt_modified_source_y = t_nt_current_source_y;
      
      t_nt_modified_source_y.push_back(t_proposed);
      
      sort( t_nt_modified_source_y.begin(),  t_nt_modified_source_y.end());
      
      rank_source_y = (int)(distance(t_nt_modified_source_y.begin(), find(t_nt_modified_source_y.begin(),t_nt_modified_source_y.end(), t_proposed) ));
      
      current_size_modified.at(source_y) = current_size_modified.at(source_y) + 1;
      
      //----------------------------------------------------------------------------------------------------------------//
      t_past = t_nt_modified_source_y.at(rank_source_y-1);
      nt_past_forward.assign(nt_current_source_y.begin()+(rank_source_y-1)*n_base_CUPDATE, nt_current_source_y.begin()+(rank_source_y)*n_base_CUPDATE);
      
      //t_proposed = t_e_subject;
      
      switch(int (current_size_arg.at(subject_proposed)>1)){
      //switch(t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
      
      case 1:{// with 2nd seq in subject
        
        switch(int(current_size_modified.at(source_y)>(rank_source_y+1))){
        
      case 1:{// inserted seq will NOT be last seq at rank_source_y  in source_y (take closer seq as the future seq)
        
        switch(int(t_nt_current_arg[subject_proposed][1]<t_nt_modified_source_y.at(rank_source_y +1))){
        //switch(t_sample_arg.at(subject_proposed)<t_nt_modified_source_y.at(rank_source_y +1)){
        
      case 1:{// take the one from subject as future seq
        
        t_future  = t_nt_current_arg[subject_proposed][1];
        nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE);
        
        seq_propose_cond(seq_proposed, log_pr_seq_forward, nt_past_forward,  nt_future_forward,  t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
        
        break;
      }
        
      case 0:{ // take the one from source_y as future seq
        t_future = t_nt_modified_source_y.at(rank_source_y+1);
        nt_future_forward.assign(nt_current_source_y.begin()+(rank_source_y)*n_base_CUPDATE, nt_current_source_y.begin()+(rank_source_y+1)*n_base_CUPDATE);
        
        seq_propose_cond(seq_proposed, log_pr_seq_forward, nt_past_forward,  nt_future_forward,  t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
        
        break;
      }
      }
        
        break;
      }
        
      case 0:{//  inserted seq will be last seq at rank_source_y  in source_y
        
        t_future  = t_nt_current_arg[subject_proposed][1];
        nt_future_forward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE);
        
        seq_propose_cond(seq_proposed, log_pr_seq_forward, nt_past_forward,  nt_future_forward,  t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
        break;
      }
      }
        
        
        
        break;
      }
        
      case 0:{// with no 2nd seq in subject
        
        switch(int(current_size_modified.at(source_y)>(rank_source_y+1))){
        
      case 1:{// inserted seq will NOT be last seq at rank_source_y  in source_y
        t_future = t_nt_modified_source_y.at(rank_source_y+1);
        nt_future_forward.assign(nt_current_source_y.begin()+(rank_source_y)*n_base_CUPDATE, nt_current_source_y.begin()+(rank_source_y+1)*n_base_CUPDATE);
        
        seq_propose_cond(seq_proposed, log_pr_seq_forward, nt_past_forward,  nt_future_forward,  t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
        break;
      }
        
      case 0:{// inserted seq will be last seq at rank_source_y  in source_y
        t_future = t_proposed;
        seq_propose_uncond( seq_proposed, log_pr_seq_forward,  nt_past_forward, t_proposed, t_past,  t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
        break;
      }
      }
        
        break;
      }
      }
      
      //----------------------------------------------------------------------------------------------------------------//
      
      //nt_modified_source_y.insert(nt_modified_source_y.begin()+(rank_source_y)*n_base_CUPDATE, nt_subject_seq.begin(), nt_subject_seq.end());
      nt_modified_source_y.insert(nt_modified_source_y.begin()+(rank_source_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());
      
      nt_modified.at(source_y) = nt_modified_source_y;
      t_nt_modified.at(source_y) = t_nt_modified_source_y;
      
      break;
    }
      
    case 1:{// new bg infection
      
      // 				sample_snull (con_seq_CUPDATE, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE, r_c); //sample a seq for background
      // 				log_pr_seq_forward = lh_snull(con_seq_CUPDATE, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
      
      
      switch(int (current_size_arg.at(subject_proposed)>1)){
      //switch(t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
      
    case 1:{// with 2nd seq in subject
      
      t_past  = t_nt_current_arg[subject_proposed][1]; // yes, t_past!
      nt_past_forward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE);
      
      t_future = t_proposed;
      
      seq_propose_uncond( seq_proposed, log_pr_seq_forward,  nt_past_forward, t_proposed, t_past,  t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
      
      break;
    }
      
    case 0:{// with no 2nd seq in subject
      
      // 						log_pr_seq_forward = n_base_CUPDATE*log(0.25);
      //
      // 						for (int i=0; i<=(n_base_CUPDATE-1);i++){
      // 						seq_proposed.at(i) = gsl_rng_uniform_int(r_c, 4) +1;
      // 						}
      
      
      sample_snull (con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE, rng); //sample a seq for background
      log_pr_seq_forward = lh_snull(con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
      
      break;
    }
    }
      
      
      break;
    }
      
    }
    
    //--------------------------------------------
    nt_modified_subject.erase(nt_modified_subject.begin() , nt_modified_subject.begin()+n_base_CUPDATE);
    nt_modified_subject.insert(nt_modified_subject.begin(), seq_proposed.begin(), seq_proposed.end());
    
    t_nt_modified_subject.erase(t_nt_modified_subject.begin());
    t_nt_modified_subject.push_back(t_proposed);
    sort( t_nt_modified_subject.begin(),  t_nt_modified_subject.end());
    
    nt_modified.at(subject_proposed) = nt_modified_subject;
    t_nt_modified.at(subject_proposed) = t_nt_modified_subject;
    //---------------------------------------------
    
    switch(int(source_x==9999)){
    
    case 0:{// was 2nd infection
      
      infecting_size_modified.at(source_x) = infecting_size_modified.at(source_x) - 1;
      
      int rank_x = (int)(distance(infecting_list_current_arg.at(source_x).begin(), find(infecting_list_current_arg.at(source_x).begin(), infecting_list_current_arg.at(source_x).end(), subject_proposed)));
      
      infecting_list_modified.at(source_x).erase(infecting_list_modified.at(source_x).begin()+rank_x);
      
      //----------------------------------------------------------------------------------------------------------------//
      
      nt_current_source_x = nt_current_arg.at(source_x);
      t_nt_current_source_x = t_nt_current_arg.at(source_x);
      
      rank_source_x = (int)(distance( t_nt_current_source_x.begin(), find(t_nt_current_source_x.begin(), t_nt_current_source_x.end(), t_e_arg.at(subject_proposed)) ));
      
      //----------------------------------------------------------------------------------------------------------------//
      
      t_past_backward = t_nt_current_source_x.at(rank_source_x-1);
      nt_past_backward.assign(nt_current_source_x.begin()+(rank_source_x-1)*n_base_CUPDATE, nt_current_source_x.begin()+(rank_source_x)*n_base_CUPDATE);
      
      t_proposed_backward =t_e_arg.at(subject_proposed);
      seq_proposed_backward.assign(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE);
      
      switch(int (current_size_arg.at(subject_proposed)>1)){
      //switch(t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
      
      case 1:{// with2nd seq subject
        
        switch(int(current_size_arg.at(source_x)>(rank_source_x+1))){
        
      case 1:{// also NOT last seq at rank_source_x  in source_x (take closer seq as the future seq)
        switch(int(t_nt_current_arg[subject_proposed][1]<t_nt_current_source_x.at(rank_source_x +1))){
        //switch(t_sample_arg.at(subject_proposed)<t_nt_current_source_x.at(rank_source_x +1)){
        
      case 1:{// take the one from subject as future seq
        
        t_future_backward  = t_nt_current_arg[subject_proposed][1];
        nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE );
        
        seq_backward_pr_cond(seq_proposed_backward, log_pr_seq_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward,  t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
        break;
      }
      case 0:{// take the one from source_x as future seq
        t_future_backward  = t_nt_current_source_x.at(rank_source_x +1);
        nt_future_backward.assign(nt_current_source_x.begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_source_x.begin()+(rank_source_x+2)*n_base_CUPDATE);
        
        seq_backward_pr_cond(seq_proposed_backward, log_pr_seq_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward,  t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        break;
      }
      }
        break;
      }
        
      case 0:{ //last seq at rank_source_x  in source_x
        
        t_future_backward  = t_nt_current_arg[subject_proposed][1]; // always takes the 2nd seq in subject as the future sequence (both for forward and backward)
        nt_future_backward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE );
        
        seq_backward_pr_cond(seq_proposed_backward, log_pr_seq_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward,  t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        break;
      }
      }
        
        
        break;
      }
        
      case 0:{ // with no 2nd seq in the subject
        
        switch(int(current_size_arg.at(source_x)>(rank_source_x+1))){
      case 1:{ // not last seq at rank_source_x  in source_x
        t_future_backward  = t_nt_current_source_x.at(rank_source_x +1);
        nt_future_backward.assign(nt_current_source_x.begin()+(rank_source_x+1)*n_base_CUPDATE, nt_current_source_x.begin()+(rank_source_x+2)*n_base_CUPDATE);
        
        seq_backward_pr_cond(seq_proposed_backward, log_pr_seq_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward,  t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
        break;
      }
      case 0:{ // last seq rank_source_x  in source_x
        t_future_backward = t_proposed_backward;
        
        seq_backward_pr_uncond(seq_proposed_backward,  log_pr_seq_backward, nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        break;
      }
      }
        
        break;
      }
      }
      
      //----------------------------------------------------------------------------------------------------------------//
      
      nt_modified_source_x = nt_current_source_x;
      t_nt_modified_source_x = t_nt_current_source_x;
      
      t_nt_modified_source_x.erase(t_nt_modified_source_x.begin() + rank_source_x); // erase the original t_nt entry for source_x
      nt_modified_source_x.erase(nt_modified_source_x.begin()+n_base_CUPDATE*rank_source_x , nt_modified_source_x.begin()+n_base_CUPDATE*(rank_source_x+1) );  //erase the original nt entry for source_x
      
      nt_modified.at(source_x) = nt_modified_source_x;
      t_nt_modified.at(source_x) = t_nt_modified_source_x;
      
      
      current_size_modified.at(source_x) = current_size_modified.at(source_x) - 1;
      
      break;
    }
      
    case 1:{// was bg infection
      
      // 				vector<int> seq(n_base_CUPDATE);
      // 				seq.assign(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE );
      // 				log_pr_seq_backward = lh_snull(con_seq_CUPDATE, seq, para_current_arg.p_ber, n_base_CUPDATE);
      
      
      switch(int (current_size_arg.at(subject_proposed)>1)){
      //switch(t_sample_arg.at(subject_proposed)!=unassigned_time_CUPDATE){
      
    case 1:{// with 2nd seq in subject
      
      t_past_backward = t_nt_current_arg[subject_proposed][1]; // note: it is "past"!
      nt_past_backward.assign(nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+ 2*n_base_CUPDATE );
      
      t_proposed_backward = t_e_arg.at(subject_proposed);
      seq_proposed_backward.assign(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE);
      
      t_future_backward = t_proposed_backward;
      
      seq_backward_pr_uncond(seq_proposed_backward,  log_pr_seq_backward, nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      
      break;
    }
      
    case 0:{// with no 2nd seq in subject
      // 						log_pr_seq_backward = n_base_CUPDATE*log(0.25); // note: this is the proposal density! not refers to the model!
      vector<int> seq(n_base_CUPDATE);
      seq.assign(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+ n_base_CUPDATE );
      log_pr_seq_backward = lh_snull(con_seq, seq, para_current_arg.p_ber, n_base_CUPDATE);
      break;
    }
    }
      
      
      break;
    }
      
    }
    
    
    
    //----------propose the downstream sequences (starting from the newly proposed seq at subject) & change of lilelihood--------------------------------//
    
    // 	int size_current_initial = infecting_size_current_arg.at(subject_proposed);// initital value
    //
    //
    // 	switch(size_current_initial>=1){
    //
    // 		case 1:{
    //
    // 			//vector<int> list_update; // contains the subjects whose FIRST sequence would be updated, with a sequential order (i.e., level-wise and time-wise) of updating (note: as each event needed to be updated corresponds to an infection event, it would be sufficient to update the first sequence of necessary subjects so as to update all downstream seq)
    //
    // 			int source_current = subject_proposed;
    // 			int size_current = size_current_initial;
    // 			int source_pst = -1;
    //
    //
    // 			while( source_pst<=(int)(list_update.size()-1) ){
    //
    // 				source_pst = source_pst  + 1;
    //
    // 				switch(size_current>=1){
    // 					case 1:{
    // 						list_update.insert(list_update.end(), infecting_list_current_arg.at(source_current).begin(), infecting_list_current_arg.at(source_current).end());
    // 					break;
    // 					}
    // 					case 0:{
    // 					break;
    // 					}
    // 				}
    //
    // 				switch(source_pst<=(int)(list_update.size()-1)){
    // 					case 1:{
    // 						source_current = list_update.at(source_pst);
    // 						size_current = infecting_size_current_arg.at(source_current);
    // 					break;
    // 					}
    // 					case 0:{
    // 					break;
    // 					}
    // 				}
    // 			}
    //
    // 			//-----------------------------------------------//
    //
    // 			int subject_ds;
    // 			int source_ds;
    // 			int rank_source_ds;
    //
    // 			double t_past, t_future;
    // 			vector<int> nt_past_forward(n_base_CUPDATE), seq_proposed_ds(n_base_CUPDATE), nt_future_forward(n_base_CUPDATE);
    //
    // 			double t_past_backward, t_proposed_backward, t_future_backward;
    // 			vector<int> nt_past_backward(n_base_CUPDATE), seq_proposed_ds_backward(n_base_CUPDATE), nt_future_backward(n_base_CUPDATE);
    //
    // 			double log_pr_dsi_forward;
    // 			double log_pr_dsi_backward;
    //
    // 			for (int i=0; i<=(int)list_update.size()-1;i++){
    //
    // 				subject_ds = list_update.at(i);
    // 				source_ds = infected_source_current_arg.at(subject_ds);
    //
    // 				rank_source_ds = distance( t_nt_current_arg.at(source_ds).begin(), find(t_nt_current_arg.at(source_ds).begin(), t_nt_current_arg.at(source_ds).end(), t_e_arg.at(subject_ds)) );
    //
    // 				nt_past_forward.assign(nt_modified.at(source_ds).begin()+ (rank_source_ds-1)*n_base_CUPDATE, nt_modified.at(source_ds).begin()+ rank_source_ds*n_base_CUPDATE);
    //
    // 				t_past =  t_nt_current_arg.at(source_ds).at(rank_source_ds - 1);
    // 				t_proposed = t_e_arg.at(subject_ds);
    //
    // 				t_past_backward = t_past;
    // 				t_proposed_backward = t_proposed;
    //
    // //				switch(t_sample_arg.at(subject_ds)!=unassigned_time_CUPDATE){
    //
    // // 					case 1:{// with sample on subject_ds
    // //
    // // 						int rank_sample_subject_ds = distance( t_nt_current_arg.at(subject_ds).begin(), find( t_nt_current_arg.at(subject_ds).begin(), t_nt_current_arg.at(subject_ds).end(), t_sample_arg.at(subject_ds)) );
    // //
    // // 						t_future  = t_sample_arg.at(subject_ds);
    // // 						nt_future_forward.assign(nt_current_arg.at(subject_ds).begin()+ rank_sample_subject_ds*n_base_CUPDATE, nt_current_arg.at(subject_ds).begin()+ (rank_sample_subject_ds+1)*n_base_CUPDATE);
    // //
    // //  						log_pr_dsi_forward=0.0;
    // // 						seq_propose_cond(seq_proposed_ds, log_pr_dsi_forward, nt_past_forward,  nt_future_forward,  t_proposed, t_past, t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    // //
    // // 						log_pr_ds_forward = log_pr_ds_forward + log_pr_dsi_forward;
    // //
    // // 						nt_modified.at(subject_ds).erase(nt_modified.at(subject_ds).begin(), nt_modified.at(subject_ds).begin()+n_base_CUPDATE );
    // // 						nt_modified.at(subject_ds).insert(nt_modified.at(subject_ds).begin(), seq_proposed_ds.begin(), seq_proposed_ds.end());
    // //
    // // 						nt_modified.at(source_ds).erase(nt_modified.at(source_ds).begin()+(rank_source_ds)*n_base_CUPDATE, nt_modified.at(source_ds).begin()+ (rank_source_ds+1)*n_base_CUPDATE );
    // // 						nt_modified.at(source_ds).insert(nt_modified.at(source_ds).begin()+(rank_source_ds)*n_base_CUPDATE, seq_proposed_ds.begin(), seq_proposed_ds.end());
    // //
    // // 						//---------------------------------//
    // // 						t_future_backward = t_future;
    // // 						nt_future_backward = nt_future_forward;
    // //
    // // 						nt_past_backward.assign(nt_current_arg.at(source_ds).begin()+ (rank_source_ds-1)*n_base_CUPDATE, nt_current_arg.at(source_ds).begin()+ rank_source_ds*n_base_CUPDATE);
    // //
    // // 						seq_proposed_ds_backward.assign(nt_current_arg.at(subject_ds).begin(), nt_current_arg.at(subject_ds).begin()+n_base_CUPDATE);
    // //
    // //  						log_pr_dsi_backward=0.0;
    // // 						seq_backward_pr_cond(seq_proposed_ds_backward, log_pr_dsi_backward, nt_past_backward, nt_future_backward, t_proposed_backward, t_past_backward,  t_future_backward, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    // //
    // // 						log_pr_ds_backward = log_pr_ds_backward + log_pr_dsi_backward;
    // //
    // //
    // // 						//---------------------------------//
    // //
    // // 					break;
    // // 					}
    //
    // //					case 0:{ // with no sample in subject_ds
    // 						t_future = t_proposed;
    //
    //  						log_pr_dsi_forward=0.0;
    // 						seq_propose_uncond( seq_proposed_ds, log_pr_dsi_forward,  nt_past_forward, t_proposed, t_past,  t_future, para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, r_c);
    //
    //
    // 						log_pr_ds_forward = log_pr_ds_forward + log_pr_dsi_forward;
    //
    // 						nt_modified.at(subject_ds).erase(nt_modified.at(subject_ds).begin(), nt_modified.at(subject_ds).begin()+n_base_CUPDATE );
    // 						nt_modified.at(subject_ds).insert(nt_modified.at(subject_ds).begin(), seq_proposed_ds.begin(), seq_proposed_ds.end());
    //
    // 						nt_modified.at(source_ds).erase(nt_modified.at(source_ds).begin()+(rank_source_ds)*n_base_CUPDATE, nt_modified.at(source_ds).begin()+ (rank_source_ds+1)*n_base_CUPDATE );
    // 						nt_modified.at(source_ds).insert(nt_modified.at(source_ds).begin()+(rank_source_ds)*n_base_CUPDATE, seq_proposed_ds.begin(), seq_proposed_ds.end());
    //
    // 						//---------------------------------//
    // 						t_future_backward = t_future;
    //
    // 						nt_past_backward.assign(nt_current_arg.at(source_ds).begin()+ (rank_source_ds-1)*n_base_CUPDATE, nt_current_arg.at(source_ds).begin()+ rank_source_ds*n_base_CUPDATE);
    //
    // 						seq_proposed_ds_backward.assign(nt_current_arg.at(subject_ds).begin(), nt_current_arg.at(subject_ds).begin()+n_base_CUPDATE);
    //
    //  						log_pr_dsi_backward=0.0;
    // 						seq_backward_pr_uncond(seq_proposed_ds_backward,  log_pr_dsi_backward, nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    //
    // 						log_pr_ds_backward = log_pr_ds_backward + log_pr_dsi_backward;
    //
    // 						//---------------------------------//
    // // 					break;
    // // 					}
    // //				}
    //
    // 			}// end 1st for loop
    //
    // 			//--- update the likelihood contributed sequences in subject_ds--//
    //
    // 			for (int i=0; i<=(int)list_update.size()-1;i++){
    //
    // 				subject_ds = list_update.at(i);
    // 				source_ds = infected_source_current_arg.at(subject_ds);
    //
    //
    // 				switch (current_size_arg.at(subject_ds)>1) {
    //
    // 				case 1:{
    //
    // 				log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(subject_ds); //subtract part of likelihood that would be updated below
    //
    // 				lh_square_modified.log_f_S.at(subject_ds) = 0.0;
    //
    // 				for (int j=0;j<=(current_size_arg.at(subject_ds)-2);j++){
    //
    //
    // 				vector<int> seq_1(nt_modified.at(subject_ds).begin()+j*(n_base_CUPDATE), nt_modified.at(subject_ds).begin()+(j+1)*(n_base_CUPDATE));
    // 				vector<int> seq_2(nt_modified.at(subject_ds).begin()+(j+1)*(n_base_CUPDATE), nt_modified.at(subject_ds).begin()+(j+2)*(n_base_CUPDATE));
    //
    // 				lh_square_modified.log_f_S.at(subject_ds) =lh_square_modified.log_f_S.at(subject_ds) + log_lh_seq(seq_1, seq_2, t_nt_current_arg[subject_ds][j], t_nt_current_arg[subject_ds][j+1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    //
    // 				}
    //
    // 				log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(subject_ds);
    //
    // 				break;
    // 				}
    //
    // 				default:{
    // 				break;
    // 				}
    //
    // 				}
    //
    // 			}// end 2nd for loop
    //
    // 			//---------------------------------------------------------------//
    //
    //
    // 		break;
    // 		}
    //
    // 		case 0:{//size_current_initial<1
    // 		break;
    // 		}
    //
    //
    // 	}
    
    //---------- end of proposing the downstream sequences (starting from the newly proposed seq at subject) & change of likelihood--------------------------------//
    
    
    //------------- deal with change of likelihood due to change of source and sequences in subject_proposed)---------------//
    
    switch (int (current_size_arg.at(subject_proposed)>1)) {
    
    case 1:{
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(subject_proposed); //subtract part of likelihood that would be updated below
      
      lh_square_modified.log_f_S.at(subject_proposed) = 0.0;
      
      for (int j=0;j<=(current_size_arg.at(subject_proposed)-2);j++){
        
        vector<int> seq_1(nt_modified.at(subject_proposed).begin()+j*(n_base_CUPDATE), nt_modified.at(subject_proposed).begin()+(j+1)*(n_base_CUPDATE));
        vector<int> seq_2(nt_modified.at(subject_proposed).begin()+(j+1)*(n_base_CUPDATE), nt_modified.at(subject_proposed).begin()+(j+2)*(n_base_CUPDATE));
        
        lh_square_modified.log_f_S.at(subject_proposed) =lh_square_modified.log_f_S.at(subject_proposed) + log_lh_seq(seq_1, seq_2,t_nt_modified_subject.at(j), t_nt_modified_subject.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      }
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(subject_proposed);
      
      break;
    }
      
    default:{
      break;
    }
      
    }
    
    
    //-----------------------------------------------------------------------------------------//
    
    lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
    
    for (int j=0;j<=(int) (xi_I_arg.size()-1);j++){ // update delta_mat; and pre-assign k_sum_E & kt_sum_E which might be changed later again
      
      if (t_i_arg.at(xi_I_arg.at(j))<t_proposed) {
        
        switch (int(t_r_arg.at(xi_I_arg.at(j))>=t_proposed)) {
        case 1:{
      delta_mat_modified[subject_proposed][xi_I_arg.at(j)] = t_proposed - t_i_arg.at(xi_I_arg.at(j));
      
      delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = 0.0;
      for (int m = 0; m <= (int)(moves_arg.from_k.size() - 1); m++) {
        if ((moves_arg.from_k[m] == xi_I_arg.at(j)) && (moves_arg.to_k[m] == subject_proposed)) {
          if ((moves_arg.t_m[m] >= t_i_arg.at(xi_I_arg.at(j))) &&
              (moves_arg.t_m[m] <= t_proposed)) {
            
            delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] + (t_proposed - moves_arg.t_m[m]);
          }
        }
      }
      
      break;
    }
        case 0:{
          delta_mat_modified[subject_proposed][xi_I_arg.at(j)] = t_r_arg.at(xi_I_arg.at(j)) - t_i_arg.at(xi_I_arg.at(j));
          
          delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = 0.0;
          for (int m = 0; m <= (int)(moves_arg.from_k.size() - 1); m++) {
            if ((moves_arg.from_k[m] == xi_I_arg.at(j)) && (moves_arg.to_k[m] == subject_proposed)) {
              if ((moves_arg.t_m[m] >= t_i_arg.at(xi_I_arg.at(j))) &&
                  (moves_arg.t_m[m] <= t_r_arg.at(xi_I_arg.at(j)))) {
                delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] = delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)] + (t_r_arg.at(xi_I_arg.at(j)) - moves_arg.t_m[m]);
              }
              
            }
            
          }
          break;
        }
        }
        if (opt_betaij == 0) {
          lh_square_modified.kt_sum_E.at(subject_proposed) = lh_square_modified.kt_sum_E.at(subject_proposed) + delta_mat_modified[subject_proposed][xi_I_arg.at(j)] * kernel_mat_current_arg[subject_proposed][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        }
        if (opt_betaij == 1) {
          lh_square_modified.kt_sum_E.at(subject_proposed) = lh_square_modified.kt_sum_E.at(subject_proposed) + delta_mat_modified[subject_proposed][xi_I_arg.at(j)] * beta_ij_mat_current_arg[xi_I_arg.at(j)][subject_proposed] * kernel_mat_current_arg[subject_proposed][xi_I_arg.at(j)] / norm_const_current_arg.at(xi_I_arg.at(j));
        }
        
        if (opt_mov == 0) {
          lh_square_modified.movest_sum_E.at(subject_proposed) = 0;
        }
        if (opt_mov == 1) {
          //double moves_ij_t = func_moves_cnt(xi_I_arg.at(j), subject_proposed, moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          lh_square_modified.movest_sum_E.at(subject_proposed) = lh_square_modified.movest_sum_E.at(subject_proposed) + delta_mat_mov_modified[subject_proposed][xi_I_arg.at(j)];
        }
        if (opt_mov == 2) {
          //double moves_ij_t = func_moves_cnt(xi_I_arg.at(j), subject_proposed, moves_arg, t_e_arg, t_i_arg, t_r_arg, para_priors_arg);
          lh_square_modified.movest_sum_E.at(subject_proposed) = lh_square_modified.movest_sum_E.at(subject_proposed); //no change
        }
        
      }
    }
    //----------
    lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
    lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
    
    switch(source_y){
    
    case 9999:{ // by background
      lh_square_modified.g_E.at(subject_proposed) =  para_current_arg.alpha;
      break;
    }
      
    default :{ // not by background
      
      if (opt_betaij == 0) {
      lh_square_modified.k_sum_E.at(subject_proposed) = kernel_mat_current_arg[subject_proposed][source_y] / norm_const_current_arg.at(source_y);
    }
      if (opt_betaij == 1) {
        lh_square_modified.k_sum_E.at(subject_proposed) = beta_ij_mat_current_arg[source_y][subject_proposed] * kernel_mat_current_arg[subject_proposed][source_y] / norm_const_current_arg.at(source_y);
      }
      
      if (opt_mov == 0) {
        lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
      }
      if ((opt_mov == 1) | (opt_mov == 2)) {
        double moves_ij_t = func_moves_cnt(source_y, subject_proposed, moves_arg, t_e_modified, t_i_arg, t_r_arg, para_priors_arg);
        lh_square_modified.moves_sum_E.at(subject_proposed) = moves_ij_t;
      }
      
      
      lh_square_modified.g_E.at(subject_proposed) = para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.moves_sum_E.at(subject_proposed);
      
      
      break;
    }
      
    }
    
    //-----this part is needed when do not update index--------
    
    // 		log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
    //
    // 		lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed);
    // // 		lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
    // 		lh_square_modified.h_E.at(subject_proposed) = gsl_ran_exponential_pdf(lh_square_modified.q_E.at(subject_proposed),1.0);
    // 		lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
    //
    // 		log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
    //------------------------------------------------------------------------------------
    
    switch (find(index_arg.begin(),index_arg.end(),subject_proposed)==index_arg.end()) { // return 1 if proposed subject not one of the original indexes
    
    case 1:{
      
      switch(int(t_proposed<t_e_arg.at(index_arg.at(0)))){ // original indexes would be replace by the chosen subject
      
    case 1:{
      
      index_modified.clear();
      index_modified.assign(1,subject_proposed);// replace index
      xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),subject_proposed));
      
      for ( int i =0; i<= (int) (index_arg.size()-1); i++){
        xi_E_minus_modified.push_back(index_arg.at(i)); // the original indexes in xi_E_minus now
      }
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
      
      lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
      lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
      lh_square_modified.q_E.at(subject_proposed)=0.0;
      lh_square_modified.g_E.at(subject_proposed)=1.0;
      lh_square_modified.h_E.at(subject_proposed)=1.0;
      lh_square_modified.f_E.at(subject_proposed)=1.0;
      lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
      lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
      
      
      for (int i=0; i<=(int) (index_arg.size()-1);i++){ // the original indexes have to be acocunted in likelihood now
        
        //log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(index_arg.at(i)));
        
        lh_square_modified.g_E.at(index_arg.at(i)) = para_current_arg.alpha; // this is not the subject_proposed
        lh_square_modified.q_E.at(index_arg.at(i)) =para_current_arg.alpha*t_e_arg.at(index_arg.at(i));
        lh_square_modified.h_E.at(index_arg.at(i)) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(index_arg.at(i)));
        lh_square_modified.f_E.at(index_arg.at(i)) = lh_square_modified.g_E.at(index_arg.at(i))*lh_square_modified.h_E.at(index_arg.at(i));
        
        log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(index_arg.at(i)));
      }
      
      break;
    }
      
      
    case 0:{
      
      if (t_proposed==t_e_arg.at(index_arg.at(0))){ // addtion of one more index
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); // this subject would have to be removed from likelihood function
      index_modified.push_back(subject_proposed); // add index
      xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),subject_proposed)); // removed from xi_E_minus
      lh_square_modified.k_sum_E.at(subject_proposed)=0.0;
      lh_square_modified.kt_sum_E.at(subject_proposed)=0.0;
      lh_square_modified.q_E.at(subject_proposed)=0.0;
      lh_square_modified.g_E.at(subject_proposed)=1.0;
      lh_square_modified.h_E.at(subject_proposed)=1.0;
      lh_square_modified.f_E.at(subject_proposed)=1.0;
      lh_square_modified.moves_sum_E.at(subject_proposed) = 0.0;
      lh_square_modified.movest_sum_E.at(subject_proposed) = 0.0;
      
    }
      
      if (t_proposed>t_e_arg.at(index_arg.at(0))){ // no shift of cases between xi_E and xi_E_minus
        
        log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed)); //subtract part of likelihood that would be updated below
        
        lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(subject_proposed);
        // 		lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
        lh_square_modified.h_E.at(subject_proposed) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(subject_proposed));
        lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
        
        log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
        
        
      } // end if t_proposs>t_e_arg.at()
      
      break;
    }
      
    }
      
      break;
    }
      
      
    case 0: { // when chosen subject is one of the indexes
      
      index_modified.clear();
      
      int first_min = (int)(distance(t_e_modified.begin(), min_element(t_e_modified.begin(), t_e_modified.end())));
      double min_t = t_e_modified.at(first_min); // the minimum time of exposure
      
      int num_min = (int) count(t_e_modified.begin(), t_e_modified.end(), min_t); // numberof subects with the min exposure time
      
      switch (int(num_min>1)) {
      case 1: {
        index_modified.reserve(n_CUPDATE);
        for (int i=0; i<=(n_CUPDATE-1);i++){
          if (t_e_modified.at(i)==min_t ) index_modified.push_back(i);
        }
        break;
      }
      case 0:{
        index_modified.assign(1,first_min);
        break;
      }
      }
      
      xi_E_minus_modified = xi_E_arg;
      
      
      for (int i=0;i<= (int) (index_modified.size()-1); i++){
        
        xi_E_minus_modified.erase(find(xi_E_minus_modified.begin(),xi_E_minus_modified.end(),index_modified.at(i)));
        
        log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(index_modified.at(i))); // this subject would have to be removed from likelihood function ( new index might be orginally an index, but the the log(lh_square_modified.f_E.at(index_modified.at(i)) will be zero in this case)
        
        lh_square_modified.k_sum_E.at(index_modified.at(i))=0.0;
        lh_square_modified.kt_sum_E.at(index_modified.at(i))=0.0;
        lh_square_modified.q_E.at(index_modified.at(i))=0.0;
        lh_square_modified.g_E.at(index_modified.at(i))=1.0;
        lh_square_modified.h_E.at(index_modified.at(i))=1.0;
        lh_square_modified.f_E.at(index_modified.at(i))=1.0;
        lh_square_modified.moves_sum_E.at(index_modified.at(i)) = 0.0;
        lh_square_modified.movest_sum_E.at(index_modified.at(i)) = 0.0;
        
      }
      
      switch(find(index_modified.begin(),index_modified.end(),subject_proposed) ==index_modified.end() ){ //return 1 when the chosen  subject is NO longer an index
      case 1:{
        
        log_lh_modified = log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed));
        
        lh_square_modified.q_E.at(subject_proposed) = para_current_arg.alpha*t_proposed + para_current_arg.beta*lh_square_modified.kt_sum_E.at(subject_proposed) + para_current_arg.beta_m*lh_square_modified.movest_sum_E.at(subject_proposed);
        //lh_square_modified.g_E.at(subject_proposed) = para_current_arg.alpha + para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
        lh_square_modified.h_E.at(subject_proposed) = pdf_exp_limit(1.0, lh_square_modified.q_E.at(subject_proposed));
        lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
        
        log_lh_modified = log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed)); //add back the  part of likelihood
        
        break;
      }
      case 0:{
        break;
      }
      }
      
      break;
    }
      
    }
    
    //--------------------//
    
    
    switch ( find(xi_I_arg.begin(), xi_I_arg.end(),subject_proposed) != (xi_I_arg.end()) ) { //return 1 when the subject is also in xi_I
    case 1:{
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_I.at(subject_proposed)); //subtract part of likelihood that would be updated below
      lh_square_modified.f_I.at(subject_proposed) = func_latent_pdf(t_i_arg.at(subject_proposed) - t_proposed, para_current_arg.lat_mu, para_current_arg.lat_var);
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_I.at(subject_proposed));
      
      break;
    }
      
    case 0:{
      break;
    }
    }
    
    //----------
    
    switch ( find(xi_EnI_arg.begin(), xi_EnI_arg.end(),subject_proposed) != (xi_EnI_arg.end()) ) { //return 1 when the subject is also in xi_EnI
    case 1:{
      
      log_lh_modified = log_lh_modified - log(lh_square_modified.f_EnI.at(subject_proposed)); //subtract part of likelihood that would be updated below
      lh_square_modified.f_EnI.at(subject_proposed) = func_latent_surv(t_max_CUPDATE - t_proposed, para_current_arg.lat_mu, para_current_arg.lat_var);
      log_lh_modified = log_lh_modified + log(lh_square_modified.f_EnI.at(subject_proposed));
      
      break;
    }
    case 0:{
      break;
    }
    }
    
    
    //-----------------------------------------------------------------------------------------//
    
    switch(source_x){
    
    case 9999:{ // was background infection
      
      switch(source_y){
    case 9999:{ // new  bg infection
      break;
    }
      
    default:{ // new secondary infection
      
      //log_lh_modified =  log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed));
      
      log_lh_modified =  log_lh_modified - lh_square_modified.log_f_Snull.at(subject_proposed);
      
      //lh_square_modified.k_sum_E.at(subject_proposed) = kernel_mat_current_arg[subject_proposed][source_y]/norm_const_current_arg.at(source_y); // update k_sum_E
      //lh_square_modified.g_E.at(subject_proposed) = para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
      //lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
      
      lh_square_modified.log_f_Snull.at(subject_proposed) = 0.0;
      
      //log_lh_modified =  log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed));
      
      log_lh_modified =  log_lh_modified + lh_square_modified.log_f_Snull.at(subject_proposed); // in fact redudancy, but here for clarity
      
      //--
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(source_y); // if source_y had one seq only, log_f_s would be zero anyway
      
      lh_square_modified.log_f_S.at(source_y) = 0.0;
      
      for (int j=0;j<=(current_size_modified.at(source_y)-2);j++){
        
        vector<int> seq_1(nt_modified_source_y.begin()+j*(n_base_CUPDATE), nt_modified_source_y.begin()+(j+1)*(n_base_CUPDATE));
        vector<int> seq_2(nt_modified_source_y.begin()+(j+1)*(n_base_CUPDATE), nt_modified_source_y.begin()+(j+2)*(n_base_CUPDATE));
        
        lh_square_modified.log_f_S.at(source_y) =lh_square_modified.log_f_S.at(source_y)+log_lh_seq(seq_1, seq_2, t_nt_modified_source_y.at(j), t_nt_modified_source_y.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
      }
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(source_y);
      
      
      break;
    }
    }
      
      break;
    }
      
    default :{ // was secondary infection
      
      switch(source_y){
    case 9999:{ // new  bg infection
      
      //log_lh_modified =  log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed));
      
      log_lh_modified =  log_lh_modified - lh_square_modified.log_f_Snull.at(subject_proposed); // redudant as second term must be zero
      
      
      //lh_square_modified.k_sum_E.at(subject_proposed) = 0.0; // update k_sum_E
      //lh_square_modified.g_E.at(subject_proposed) =  para_current_arg.alpha;
      //lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
      
      // 						lh_square_modified.log_f_Snull.at(subject_proposed) =  n_base_CUPDATE*log(0.25);
      lh_square_modified.log_f_Snull.at(subject_proposed) = lh_snull(con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE);
      
      
      //log_lh_modified =  log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed));
      
      log_lh_modified =  log_lh_modified + lh_square_modified.log_f_Snull.at(subject_proposed);
      
      //--
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(source_x); // source_x must had more than or equal to 2 sequences
      
      lh_square_modified.log_f_S.at(source_x) = 0.0;
      
      switch(int(current_size_modified.at(source_x)>1)){// only have to count log_f_S if there are more than or equal to 2 seq left in source_x
      case 1:{
        for (int j=0;j<=(current_size_modified.at(source_x)-2);j++){
        
        vector<int> seq_1(nt_modified_source_x.begin()+j*(n_base_CUPDATE), nt_modified_source_x.begin()+(j+1)*(n_base_CUPDATE));
        vector<int> seq_2(nt_modified_source_x.begin()+(j+1)*(n_base_CUPDATE), nt_modified_source_x.begin()+(j+2)*(n_base_CUPDATE));
        
        lh_square_modified.log_f_S.at(source_x) =lh_square_modified.log_f_S.at(source_x)+log_lh_seq(seq_1, seq_2, t_nt_modified_source_x.at(j), t_nt_modified_source_x.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
      }
        break;
      }
        
      case 0:{
        break;
      }
      }
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(source_x); // 2nd term migt be zero depends on if there are more than or equal to 2 seq left in source_x
      
      
      
      break;
    }
      
    default:{ // new secondary infection
      
      //log_lh_modified =  log_lh_modified - log(lh_square_modified.f_E.at(subject_proposed));
      
      //lh_square_modified.k_sum_E.at(subject_proposed) = kernel_mat_current_arg[subject_proposed][source_y]/norm_const_current_arg.at(source_y); // update k_sum_E
      //lh_square_modified.g_E.at(subject_proposed) = para_current_arg.beta*lh_square_modified.k_sum_E.at(subject_proposed);
      //lh_square_modified.f_E.at(subject_proposed) = lh_square_modified.g_E.at(subject_proposed)*lh_square_modified.h_E.at(subject_proposed);
      
      //log_lh_modified =  log_lh_modified + log(lh_square_modified.f_E.at(subject_proposed));
      
      //--
      
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(source_x); // source_x must had more than or equal to 2 sequences
      
      lh_square_modified.log_f_S.at(source_x) = 0.0;
      
      switch(int(current_size_modified.at(source_x)>1)){// only have to count log_f_S if there are more than or equal to 2 seq left in source_x
      case 1:{
        for (int j=0;j<=(current_size_modified.at(source_x)-2);j++){
        
        vector<int> seq_1(nt_modified_source_x.begin()+j*(n_base_CUPDATE), nt_modified_source_x.begin()+(j+1)*(n_base_CUPDATE));
        vector<int> seq_2(nt_modified_source_x.begin()+(j+1)*(n_base_CUPDATE), nt_modified_source_x.begin()+(j+2)*(n_base_CUPDATE));
        
        lh_square_modified.log_f_S.at(source_x) =lh_square_modified.log_f_S.at(source_x)+log_lh_seq(seq_1, seq_2, t_nt_modified_source_x.at(j), t_nt_modified_source_x.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
      }
        break;
      }
        
      case 0:{
        break;
      }
      }
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(source_x); // 2nd term migt be zero depends on if there are more than or equal to 2 seq left in source_x
      
      //---
      
      log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(source_y); // if source_y had one seq only, log_f_s would be zero anyway
      
      lh_square_modified.log_f_S.at(source_y) = 0.0;
      
      for (int j=0;j<=(current_size_modified.at(source_y)-2);j++){
        
        vector<int> seq_1(nt_modified_source_y.begin()+j*(n_base_CUPDATE), nt_modified_source_y.begin()+(j+1)*(n_base_CUPDATE));
        vector<int> seq_2(nt_modified_source_y.begin()+(j+1)*(n_base_CUPDATE), nt_modified_source_y.begin()+(j+2)*(n_base_CUPDATE));
        
        lh_square_modified.log_f_S.at(source_y) =lh_square_modified.log_f_S.at(source_y)+log_lh_seq(seq_1, seq_2, t_nt_modified_source_y.at(j), t_nt_modified_source_y.at(j+1), para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
        
      }
      
      log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(source_y);
      
      
      break;
    }
    }
      
      
      break;
    }
      
    }
    
    
    
    //------------- end of with change of likelihood (due to change of source and sequences in subject_proposed)---------------//
    
    //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*exp(log_pr_backward-log_pr_forward)*exp(log_pr_seq_backward-log_pr_seq_forward)*exp(log_pr_ds_backward-log_pr_ds_forward));
    
    
    
    acp_pr = min(1.0,exp( (log_lh_modified-log_lh_current_arg) + (log_pr_backward-log_pr_forward)+(log_pr_seq_backward-log_pr_seq_forward) +(log_pr_t_e_backward-log_pr_t_e_forward)+ (log_pr_ds_backward-log_pr_ds_forward)) );
    
    
    double uniform_rv = runif(0.0, 1.0, rng_arg);
    
    switch(int (uniform_rv<=acp_pr)){
    case 1: {
      
      lh_square_current_arg = lh_square_modified;
      log_lh_current_arg = log_lh_modified;
      
      //nt_current_arg.at(subject_proposed) = nt_modified_subject;
      nt_current_arg = nt_modified;
      t_nt_current_arg = t_nt_modified;
      
      delta_mat_current_arg = delta_mat_modified;
      delta_mat_mov_current_arg = delta_mat_mov_modified;
      t_e_arg= t_e_modified;
      index_arg = index_modified;
      xi_E_minus_arg = xi_E_minus_modified;
      
      current_size_arg = current_size_modified;
      infected_source_current_arg.at(subject_proposed) =  source_y;
      //infected_source_current_arg = infected_source_modified;
      
      infecting_list_current_arg = infecting_list_modified;
      infecting_size_current_arg = infecting_size_modified;
      
      // 			switch (source_x){
      //
      // 				case 9999:{
      // 				break;
      // 				}
      //
      // 				default :{
      // 				//nt_current_arg.at(source_x) = nt_modified_source_x;
      // 				t_nt_current_arg.at(source_x) = t_nt_modified_source_x;
      // 				break;
      // 				}
      // 			}
      //
      // 			switch (source_y){
      //
      // 				case 9999:{
      // 				break;
      // 				}
      //
      // 				default :{
      // 				//nt_current_arg.at(source_y) = nt_modified_source_y;
      // 				t_nt_current_arg.at(source_y) = t_nt_modified_source_y;
      // 				break;
      // 				}
      // 			}
      
      break;
    }
      
    case 0: {
      break;
    }
    }
    
    break;
  }
    
  case 1:{ // source_y==source_x
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
  //--------------------------------------------------
  
  
}

/*------------------------------------------------*/

void count_type_all(nt_struct & nt_data_arg, vector<int>& xi_E_current, int& n_base_arg, int& total_count_1, int& total_count_2, int& total_count_3){ // count number of unchanged, transition, transversion (whole dataset)
  
  for (int i=0;i<= (int)(xi_E_current.size()-1);i++){ // loop over all the infected
    
    int k_E = xi_E_current.at(i);
    
    switch (int(nt_data_arg.current_size.at(k_E)>1)) {
    
    case 1:{
      
      
      for (int j=0;j<=(nt_data_arg.current_size.at(k_E)-2);j++){
      
      
      vector<int> seq_1(nt_data_arg.nt[k_E].begin()+j*(n_base_arg), nt_data_arg.nt[k_E].begin()+(j+1)*(n_base_arg));
      vector<int> seq_2(nt_data_arg.nt[k_E].begin()+(j+1)*(n_base_arg), nt_data_arg.nt[k_E].begin()+(j+2)*(n_base_arg));
      
      count_type_seq(seq_1, seq_2, n_base_arg, total_count_1, total_count_2, total_count_3); // count between two sequence
      
    }
      
      break;
    }
      
    default:{
      break;
    }
      
    }
  }
  
}
//-----------------------

void count_type_seq (vector<int>& seq_1_arg, vector<int> seq_2_arg, int& n_base_arg, int& total_count_1, int& total_count_2, int& total_count_3){
  
  int count_1, count_2, count_3; // count_1=count of unchanged sites, ..transition.., transversion
  count_1=count_2=count_3=0;
  
  for ( int i=0;i<=(n_base_arg-1); i++){
    
    switch(abs(seq_1_arg.at(i)-seq_2_arg.at(i))){
    case 0:{
    count_1 = count_1 + 1;
    break;
  }
    case 1:{
      switch( ((seq_1_arg.at(i)==2) & (seq_2_arg.at(i)==3)) | ((seq_1_arg.at(i)==3) & (seq_2_arg.at(i)==2)) ){
    case 1:{
      count_3 = count_3 + 1;
      break;
    }
    case 0:{
      count_2 = count_2 + 1;
      break;
    }
    }
      break;
    }
    case 2:{
      count_3 = count_3 + 1;
      break;
    }
    case 3:{
      count_3 = count_3 + 1;
      break;
    }
    }
    
  }
  
  total_count_1 = total_count_1 + count_1;
  total_count_2 = total_count_2 + count_2;
  total_count_3 = total_count_3 + count_3;
  
}
//------------------------


inline void seq_propose_tri (vector<int>& seq_proposed,  double& log_pr_seq_forward, const vector<int>& nt_past_forward, const vector<int>& nt_tri_1_forward, const vector<int>& nt_tri_2_forward, const double& t_past, const double& t_tri_1_forward, const double& t_tri_2_forward, const double& t_proposed,  const double& mu_1, const double& mu_2, const  int& n_base_CUPDATE, rng_type & rng_arg){
  
  double lambda=0.5;
  double total_pr = 0.5;
  
  double dt_1 = fabs(t_proposed - t_past);
  double dt_2 = fabs(t_proposed - t_tri_1_forward);
  double dt_3 = fabs(t_proposed - t_tri_2_forward);
  
  double total_risk = exp(-lambda*dt_1) + exp(-lambda*dt_2) + exp(-lambda*dt_3);
  
  //double P[4] = {total_pr*exp(-lambda*dt_1)/total_risk, total_pr*exp(-lambda*dt_2)/total_risk, total_pr*exp(-lambda*dt_3)/total_risk, 1.0-total_pr};
  //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);
  vector<double> P = {total_pr*exp(-lambda * dt_1) / total_risk, total_pr*exp(-lambda * dt_2) / total_risk, total_pr*exp(-lambda * dt_3) / total_risk, 1.0 - total_pr};
  
  for (int i=0; i<=(n_base_CUPDATE-1); i++){
    
    //int type = gsl_ran_discrete (r_c, g);
    int type = edf_sample(P, rng);
    
    
    switch(type){
    case 0:{
      seq_proposed.at(i) = nt_past_forward.at(i);
      log_pr_seq_forward =  log_pr_seq_forward + log(P[0]);
      break;
    }
    case 1:{
      seq_proposed.at(i) = nt_tri_1_forward.at(i);
      log_pr_seq_forward =  log_pr_seq_forward + log(P[1]);
      break;
    }
    case 2:{
      seq_proposed.at(i) = nt_tri_2_forward.at(i);
      log_pr_seq_forward =  log_pr_seq_forward + log(P[2]);
      break;
    }
    case 3:{
      //seq_proposed.at(i) = gsl_rng_uniform_int(r_c, 4) +1;
      seq_proposed.at(i) = runif_int(1, 4, rng_arg);
      log_pr_seq_forward =  log_pr_seq_forward + log(P[3]) + log(0.25);
      break;
    }
    }
    
    
  }
  
  
  
}

//--------------------------------------------------


inline void seq_backward_pr_tri (const vector<int>& seq_proposed_backward,  double& log_pr_seq_backward, const vector<int>& nt_past_backward, const vector<int>& nt_tri_1_backward, const vector<int>& nt_tri_2_backward, const double& t_past_backward, const double& t_tri_1_backward, const double& t_tri_2_backward, const double& t_proposed_backward, const double& mu_1, const double& mu_2, const int& n_base_CUPDATE){
  
  
  double lambda=0.5;
  double total_pr = 0.5;
  
  double dt_1 = fabs(t_proposed_backward - t_past_backward);
  double dt_2 = fabs(t_proposed_backward - t_tri_1_backward);
  double dt_3 = fabs(t_proposed_backward - t_tri_2_backward);
  
  double total_risk = exp(-lambda*dt_1) + exp(-lambda*dt_2) + exp(-lambda*dt_3);
  
  double P[4] = {total_pr*exp(-lambda*dt_1)/total_risk, total_pr*exp(-lambda*dt_2)/total_risk, total_pr*exp(-lambda*dt_3)/total_risk, 1.0-total_pr};
  
  //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);
  
  
  for (int i=0; i<=(n_base_CUPDATE-1); i++){
    
    double pr_sum=0.0; // an "or" probability
    
    switch(int(seq_proposed_backward.at(i)==nt_past_backward.at(i))){
    case 1:{
      pr_sum  = pr_sum + P[0];
      break;
    }
    case 0:{
      break;
    }
    }
    
    switch(int(seq_proposed_backward.at(i)==nt_tri_1_backward.at(i))){
    case 1:{
      pr_sum  = pr_sum + P[1];
      break;
    }
    case 0:{
      break;
    }
    }
    
    switch( int (seq_proposed_backward.at(i)==nt_tri_2_backward.at(i))){
    case 1:{
      pr_sum  = pr_sum + P[2];
      break;
    }
    case 0:{
      break;
    }
    }
    
    log_pr_seq_backward = log_pr_seq_backward + log(pr_sum+P[3]*0.25);
    
    
  }
  
  
}

//--------------------------------------------------


void mcmc_UPDATE::seq_n_update(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg, vector<int>&  xi_beta_E_arg, const int& subject_proposed, rng_type & rng_arg){
  
  //int subject_proposed ;
  double acp_pr = 0.0;
  
  double log_part_x_subject=0.0;
  double log_part_y_subject=0.0;
  
  double log_part_x_source=0.0;
  double log_part_y_source =0.0;
  
  
  //int position_proposed, base_proposed;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  
  //position_proposed =iter;
  
  int subject_source = infected_source_current_arg.at(subject_proposed);
  
  
  //vector<int> nt_modified_subject = nt_current_arg.at(subject_proposed);
  
  vector<int> seq_proposed (n_base_CUPDATE);
  //seq_proposed.assign(nt_modified_subject.begin() , nt_modified_subject.begin()+n_base_CUPDATE );
  
  vector<int> nt_current (n_base_CUPDATE);
  nt_current.assign( nt_current_arg.at(subject_proposed).begin() , nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE);
  
  vector<int> nt_next_subject (n_base_CUPDATE);
  
  //---
  
  for (int i=0;i<=(n_base_CUPDATE-1);i++){
    
    int base_current = nt_current_arg[subject_proposed][i]; // always refers to the first sequence
    
    switch(base_current){
    case 1:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        seq_proposed.at(i)  = 2;
        break;
      }
      case 1:{
        seq_proposed.at(i) = 3;
        break;
      }
      case 2:{
        seq_proposed.at(i) = 4;
        break;
      }
      }
      break;
    }
    case 2:{
      int type = runif_int(0, 2, rng_arg);
      
      switch(type){
      case 0:{
        seq_proposed.at(i) = 1;
        break;
      }
      case 1:{
        seq_proposed.at(i) = 3;
        break;
      }
      case 2:{
        seq_proposed.at(i) = 4;
        break;
      }
      }
      break;
    }
    case 3:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        seq_proposed.at(i) = 1;
        break;
      }
      case 1:{
        seq_proposed.at(i) = 2;
        break;
      }
      case 2:{
        seq_proposed.at(i) = 4;
        break;
      }
      }
      break;
    }
    case 4:{
      int type = runif_int(0, 2, rng_arg);
      switch(type){
      case 0:{
        seq_proposed.at(i) = 1;
        break;
      }
      case 1:{
        seq_proposed.at(i) = 2;
        break;
      }
      case 2:{
        seq_proposed.at(i) = 3;
        break;
      }
      }
      break;
    }
    }
    
  }
  //---
  
  //nt base_next_subject =0;
  
  switch (int (current_size_arg.at(subject_proposed)>1)) {
  
  case 1:{
    //--
    
    //		base_next_subject = nt_current_arg[subject_proposed][position_proposed + n_base_CUPDATE];
    
    // 		log_part_x_subject = log_lh_base(base_current, base_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2);
    //
    // 		log_part_y_subject = log_lh_base(base_proposed, base_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2);
    
    nt_next_subject.assign( nt_current_arg.at(subject_proposed).begin() + n_base_CUPDATE, nt_current_arg.at(subject_proposed).begin()+2*n_base_CUPDATE);
    
    log_part_x_subject = log_lh_seq (nt_current, nt_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    log_part_y_subject = log_lh_seq (seq_proposed, nt_next_subject, t_nt_current_arg[subject_proposed][0], t_nt_current_arg[subject_proposed][1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    //--
    lh_square_modified.log_f_S.at(subject_proposed) =  lh_square_modified.log_f_S.at(subject_proposed) - log_part_x_subject;
    log_lh_modified = log_lh_modified - log_part_x_subject;
    
    lh_square_modified.log_f_S.at(subject_proposed) =  lh_square_modified.log_f_S.at(subject_proposed) + log_part_y_subject;
    log_lh_modified = log_lh_modified + log_part_y_subject;
    
    
    break;
  }
    
  case 0:{
    break;
  }
  }
  //-------
  
  int rank_source =-1;  //count the rank of the original t_e among t_nt_current_arg.at(subject_source)
  
  // int base_before_source =0;
  // int base_next_source = 0;
  vector<int> nt_before_source(n_base_CUPDATE);
  vector<int> nt_next_source(n_base_CUPDATE);
  
  
  switch(subject_source ){
  
  case 9999:{ // by background
    break;
  }
    
  default :{ // not by background
    
    //nt_modified_source = nt_current_arg.at(subject_source);
    rank_source = (int)(distance( t_nt_current_arg.at(subject_source).begin(), find(t_nt_current_arg.at(subject_source).begin(), t_nt_current_arg.at(subject_source).end(), t_e_arg.at(subject_proposed)) ));
    //nt_modified_source.erase(nt_modified_source.begin()+n_base_CUPDATE*rank_source_x , nt_modified_source.begin()+n_base_CUPDATE*(rank_source_x+1) );  //erase the original nt entry for source
    
    //base_before_source =  nt_current_arg[subject_source][(rank_source-1)*n_base_CUPDATE + position_proposed];
    nt_before_source.assign(nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*(rank_source-1), nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*rank_source );
    
    switch(int (current_size_arg.at(subject_source)>(rank_source+1))){
    case 1:{// there  is a valid base_next_source
      
      // 				base_next_source =  nt_current_arg[subject_source][(rank_source+1)*n_base_CUPDATE + position_proposed];
      
      // 				log_part_x_source = log_lh_base(base_before_source, base_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2) + log_lh_base(base_current, base_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2);
      //
      // 				log_part_y_source = log_lh_base(base_before_source, base_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2) + log_lh_base(base_proposed, base_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2);
      //
      
      nt_next_source.assign(nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*(rank_source+1), nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*(rank_source+2) );
      
      log_part_x_source = log_lh_seq (nt_before_source, nt_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE) + log_lh_seq (nt_current, nt_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      log_part_y_source = log_lh_seq (nt_before_source, seq_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE) +log_lh_seq (seq_proposed, nt_next_source, t_nt_current_arg[subject_source][rank_source], t_nt_current_arg[subject_source][rank_source+1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
      
    case 0:{
      
      // 				log_part_x_source = log_lh_base(base_before_source, base_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2);
      //
      // 				log_part_y_source = log_lh_base(base_before_source, base_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2);
      
      log_part_x_source = log_lh_seq (nt_before_source, nt_current, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE) ;
      
      log_part_y_source = log_lh_seq (nt_before_source, seq_proposed, t_nt_current_arg[subject_source][rank_source-1], t_nt_current_arg[subject_source][rank_source], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
      break;
    }
    }
    
    lh_square_modified.log_f_S.at(subject_source) =  lh_square_modified.log_f_S.at(subject_source) - log_part_x_source;
    log_lh_modified = log_lh_modified - log_part_x_source;
    
    
    lh_square_modified.log_f_S.at(subject_source) =  lh_square_modified.log_f_S.at(subject_source) + log_part_y_source;
    log_lh_modified = log_lh_modified + log_part_y_source;
    
    break;
  }
    
  }
  
  //------
  
  double log_part_y = log_part_y_subject + log_part_y_source;
  double log_part_x = log_part_x_subject + log_part_x_source;
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg));
  acp_pr = min(1.0,exp(log_part_y-log_part_x));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    
    lh_square_current_arg = lh_square_modified;
    log_lh_current_arg = log_lh_modified;
    
    //nt_current_arg[subject_proposed][position_proposed]= base_proposed;
    
    nt_current_arg.at(subject_proposed).erase(nt_current_arg.at(subject_proposed).begin(), nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE);  //erase the original nt entry for source
    
    nt_current_arg.at(subject_proposed).insert(nt_current_arg.at(subject_proposed).begin(), seq_proposed.begin(), seq_proposed.end());  //insert  the new nt
    
    switch (subject_source){
    
    case 9999:{ // by background
      break;
    }
      
    default :{ // not by background
      //			nt_current_arg[subject_source][(rank_source)*n_base_CUPDATE + position_proposed] = base_proposed;
      
      nt_current_arg.at(subject_source).erase(nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*rank_source, nt_current_arg.at(subject_source).begin()+n_base_CUPDATE*(rank_source+1) );  //erase the original nt entry for source
      
      nt_current_arg.at(subject_source).insert(nt_current_arg.at(subject_source).begin()+(rank_source)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());  //insert  the new nt
      break;
    }
    }
    
    
    break;
  }
    
  case 0: {
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
  
}


/*------------------------------------------------*/

void delete_seq_samples(para_key& para_current, para_aux& para_other,  vector<int>& xi_I_current, vector<int>& xi_U_current, vector<int>& xi_E_current, vector<int>& xi_E_minus_current, vector<int>& xi_R_current,  vector<int>& xi_EnI_current,  vector<int>& xi_EnIS_current,  vector<int>& xi_InR_current,  vector<double>& t_e_current, vector<double>& t_i_current, vector<double>& t_r_current, vector<int>& index_current, vector<int>& infected_source_current, vector < vector<double> >& kernel_mat_current, vector <double>& norm_const_current, vec2int& sample_data, vector<double>& t_onset, nt_struct& nt_data_current){ // keep the true config, only delete the sampled sequences
  
  
  //const gsl_rng_type* T= gsl_rng_default;  // T is pointer points to the type of generator
  //gsl_rng *r = gsl_rng_alloc (T); // r is pointer points to an object with Type T
  
  int seed_intial_mcmc = 1;  //1,999,-1000,-10000,123456
  
  //gsl_rng_set (r, seed_intial_mcmc); // set a seed
  rng_type rng_r(seed_intial_mcmc); //set a universal seed
  
  ofstream myfile_out;
  
  
  
  
  para_current.alpha = para_current.alpha*1.0; //initialization of parameter to be estimated
  para_current.beta = para_current.beta *1.0 ; //initialization of parameter to be estimated
  
  // para_current.lat_mu = para_current.a*para_current.b*1; //initialization of parameter to be estimated
  // para_current.lat_var = para_current.a*para_current.b*para_current.b*1; //initialization of parameter to be estimated
  para_current.lat_mu = para_current.a; //initialization of parameter to be estimated
  para_current.lat_var = para_current.b; //initialization of parameter to be estimated
  
  para_current.c = para_current.c*1.0; //initialization of parameter to be estimated
  para_current.d = para_current.d*1.0; //weibull parameter (redundant if exponential for c)
  
  para_current.k_1 = para_current.k_1*1.0; //initialization of parameter to be estimated
  
  para_current.mu_1 = para_current.mu_1*1.0; //initialization of parameter to be estimated
  para_current.mu_2 = para_current.mu_2*1.0; //initialization of parameter to be estimated
  
  para_current.p_ber = para_current.p_ber*1.0; //initialization of parameter to be estimated
  
  para_current.beta_m = para_current.beta_m*1.0; //initialization of parameter to be estimated
  
  
  //-------- for partial deletion----///
  
  double p_sample = 0.12;   // pr a sample included for an exposure
  
  //--- this part is to match the usage of seed and xi_E_current.size() in initializemcmc such that the unsampled individuals match
  xi_E_current =xi_I_current; // individuals gone through I (assumed known here) would be initialized as infected; sampled would also be infected (see below); assume index has gone through I
  
  xi_EnI_current.clear() ;
  
  for (int i=0; i<= (int)(para_other.n-1);i++){
    
    switch((nt_data_current.t_sample.at(i)!=para_other.unassigned_time)&(find(xi_I_current.begin(),xi_I_current.end(),i)==xi_I_current.end())){
    
    case 1:{ // sampled(infected) but not in xi_I_current
    xi_E_current.push_back(i);
    xi_EnI_current.push_back(i);
    break;
  }
    case 0:{
      break;
    }
      
    }
    
  }
  
  xi_EnIS_current.clear() ; // this would be empty as initial value
  
  xi_U_current.clear();
  
  for (int i=0; i<= (int)(para_other.n-1);i++){
    if(find(xi_E_current.begin(),xi_E_current.end(),i)==xi_E_current.end()){
      xi_U_current.push_back(i);
      t_e_current.at(i)=para_other.unassigned_time;
      infected_source_current.at(i) = -99;
    }
  }
  for (int i=0; i<= (int)(xi_I_current.size()-1);i++){
    //gsl_ran_flat(r, 0,1); //? not assigned
    runif(0.0, 1.0, rng_r); //? not assigned
  }
  
  //---
  
  for (int i=0; i<= (int)(xi_E_current.size()-1);i++){// loop over infections
    
    int subject= xi_E_current.at(i);
    
    //double PS[2] = {1.0 - p_sample, p_sample};
    //gsl_ran_discrete_t * gs = gsl_ran_discrete_preproc (sizeof(PS)/sizeof(PS[0]),PS);
    //int sample_ind = gsl_ran_discrete (r, gs); // 1 = would include the sample
    int sample_ind = rbern(p_sample, rng_r);
    
    //gsl_ran_discrete_free(gs);
    
    switch(sample_ind){
    case 0:{// exclude
      
      if ((find( index_current.begin(), index_current.end(), xi_E_current.at(i) )==index_current.end())&(nt_data_current.t_sample.at(xi_E_current.at(i)) !=para_other.unassigned_time)){ // if it is an non-index & it has sample
      
      if((find(xi_EnI_current.begin(),xi_EnI_current.end(),xi_E_current.at(i))!=xi_EnI_current.end())&(nt_data_current.t_sample.at(xi_E_current.at(i)) !=para_other.unassigned_time)){// this infection is in xi_EnI& had a sample(which the sample gotta be deleted)
        xi_EnIS_current.push_back(xi_E_current.at(i));
      }
      
      int rank = (int)(distance( nt_data_current.t_nt.at(subject).begin(), find(nt_data_current.t_nt.at(subject).begin(),nt_data_current.t_nt.at(subject).end(),nt_data_current.t_sample.at(subject) ) ));
      
      nt_data_current.t_nt.at(subject).erase(nt_data_current.t_nt.at(subject).begin() + rank);
      nt_data_current.nt.at(subject).erase(nt_data_current.nt.at(subject).begin()+para_other.n_base*rank , nt_data_current.nt.at(subject).begin()+para_other.n_base*(rank+1) );
      
      nt_data_current.t_sample.at(xi_E_current.at(i)) =para_other.unassigned_time; // exclude the sample
      
      nt_data_current.current_size.at(xi_E_current.at(i)) = nt_data_current.current_size.at(xi_E_current.at(i)) - 1;
    }
      
      //----
      break;
    }
    case 1:{ // keep the original t_sample (could be unassigned_time)
      //nt_data_current.t_sample.at(xi_E_current.at(i)) = nt_data_current.t_sample.at(xi_E_current.at(i));
      break;
    }
    }
  }
  
  myfile_out.open((string(PATH2)+string("seed_delete_seq.csv")).c_str(),ios::out);
  myfile_out << seed_intial_mcmc;
  myfile_out.close();
  
  
  myfile_out.open((string(PATH2)+string("p_sample.csv")).c_str(),ios::out);
  myfile_out << p_sample;
  myfile_out.close();
  
  int total_sample =0;
  for (int i=0; i<= (int)(xi_E_current.size()-1);i++){// loop over infections
    if (nt_data_current.t_sample.at(xi_E_current.at(i))!=para_other.unassigned_time) total_sample = total_sample+1;
  }
  double p_sample_actual = ((double)  total_sample)/ ((double) xi_E_current.size() );
  myfile_out.open((string(PATH2)+string("p_sample_actual.csv")).c_str(),ios::out);
  myfile_out << p_sample_actual;
  myfile_out.close();
  
  myfile_out.open((string(PATH2)+string("xi_EnIS_initial.csv")).c_str(),ios::out);
  myfile_out << "k" << endl;
  if (xi_EnIS_current.empty()!=1){
    for (int i=0; i<=((int)xi_EnIS_current.size()-1);i++){
      myfile_out << xi_EnIS_current.at(i) << endl;
    }
  }
  myfile_out << "size" << endl;
  myfile_out << xi_EnIS_current.size();
  myfile_out.close();
  
  myfile_out.open((string(PATH2)+string("t_sample_initial.csv")).c_str(),ios::app);
  for (int i=0; i<=(para_other.n-1);i++){
    myfile_out << nt_data_current.t_sample.at(i) << endl;
  }
  myfile_out.close();
  
  
  
  //-----------------------
  
  //----- this part works for deleting all samples-----//
  
  /*
   for (int i=0; i<=(int) xi_E_current.size() -1 ; i++){
   
   int subject= xi_E_current.at(i);
   
   switch(nt_data_current.t_sample.at(subject)==para_other.unassigned_time){
   case 0:{// with sample, delete the sample and udpate current_size s correspondingly
   int rank = distance( nt_data_current.t_nt.at(subject).begin(), find(nt_data_current.t_nt.at(subject).begin(),nt_data_current.t_nt.at(subject).end(),nt_data_current.t_sample.at(subject) ) );
   
   nt_data_current.t_nt.at(subject).erase(nt_data_current.t_nt.at(subject).begin() + rank);
   nt_data_current.nt.at(subject).erase(nt_data_current.nt.at(subject).begin()+para_other.n_base*rank , nt_data_current.nt.at(subject).begin()+para_other.n_base*(rank+1) );
   
   nt_data_current.current_size.at(subject) = nt_data_current.current_size.at(subject) - 1;
   
   nt_data_current.t_sample.at(subject)=para_other.unassigned_time;
   
   break;
   }
   case 1:{// without sample, keep it
   break;
   }
   }
   }
   
   xi_EnIS_current.clear() ; // this would be empty as initial value
   
   xi_EnIS_current = xi_EnI_current;
   
   //--------------
   
   myfile_out.open((string(PATH2)+string("t_sample_initial.csv")).c_str(),ios::app);
   for (int i=0; i<=(para_other.n-1);i++){
   myfile_out << nt_data_current.t_sample.at(i) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("xi_EnIS_initial.csv")).c_str(),ios::out);
   myfile_out << "k" << endl;
   if (xi_EnIS_current.empty()!=1){
   for (int i=0; i<=((int)xi_EnIS_current.size()-1);i++){
   myfile_out << xi_EnIS_current.at(i) << endl;
   }
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("current_size_initial.csv")).c_str(),ios::app);
   for (int i=0; i<=(para_other.n-1);i++){
   myfile_out << nt_data_current.current_size.at(i) << endl;
   }
   myfile_out.close();
   */
  
}

/*------------------------------------------------*/


void mcmc_UPDATE::index_first_seq(lh_SQUARE& lh_square_current_arg, double& log_lh_current_arg, const vector< vector<double> >& kernel_mat_current_arg, vector< vector<double> >& delta_mat_current_arg, vector<int>& xi_U_arg, vector<int>& xi_E_arg, vector<int>& xi_E_minus_arg, const vector<int>& xi_I_arg, vector<int>& xi_EnI_arg, const vector<double>& t_r_arg, const vector<double>& t_i_arg, vector<double>& t_e_arg, vector<int>& index_arg, const para_key& para_current_arg, const vector<double>& norm_const_current_arg, const vector<int>& infected_source_current_arg, const vector<double>& t_sample_arg, const vector<int>& current_size_arg, vec2int& nt_current_arg , vec2d& t_nt_current_arg,  vec2int& infecting_list_current_arg, const vector<int>& infecting_size_current_arg, vector<int>&  xi_beta_E_arg, vector<int>& con_seq,int& subject_proposed,int iter, rng_type & rng_arg){
  
  //double t_back =10.0;
  
  //int subject_proposed ;
  
  //double t_low, t_up;
  double acp_pr = 0.0;
  
  lh_SQUARE lh_square_modified = lh_square_current_arg;
  double log_lh_modified =  log_lh_current_arg;
  
  //vector< vector<double> > delta_mat_modified = delta_mat_current_arg;
  //vector<double> t_e_modified = t_e_arg;
  //vector <int> index_modified = index_arg;
  //vector <int> xi_E_minus_modified = xi_E_minus_arg;
  
  // vector <int> xi_U_modified = xi_U_arg;
  // vector <int> xi_E_modified = xi_E_arg;
  // vector <int> xi_EnI_modified = xi_EnI_arg;
  
  /*const gsl_rng_type* T_c= gsl_rng_default;  // T is pointer points to the type of generator
   gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
   gsl_rng_set (r_c,iter); // set a see*/
  
  
  
  vector<int> nt_modified_subject = nt_current_arg.at(subject_proposed);
  //vector<double> t_nt_modified_subject = t_nt_current_arg.at(subject_proposed);
  
  vector<int> nt_current_seq; // the orginal first sequence of the subject which would be updated
  vector<int> nt_second_seq; // the second seq of subject (if available)
  
  //int subject_source = infected_source_current_arg.at(subject_proposed);
  
  
  //int rank_subject_x =distance( t_nt_current_arg.at(subject_proposed).begin(), find(t_nt_current_arg.at(subject_proposed).begin(), t_nt_current_arg.at(subject_proposed).end(), t_e_arg.at(subject_proposed)) ); //count the rank (distance from the first element) of the original t_e among t_nt_current_arg.at(subject_proposed)
  int rank_subject_x =0; // it is always zero as we are updating the sequence at its own infection
  
  //t_nt_modified_subject.erase(t_nt_modified_subject.begin() + rank_subject_x); // erase the original t_nt entry for subject_proposed
  
  //---
  
  nt_current_seq.assign(nt_modified_subject.begin()+n_base_CUPDATE*rank_subject_x , nt_modified_subject.begin()+n_base_CUPDATE*(rank_subject_x+1) ); //copy the original nt before erasing
  
  nt_modified_subject.erase(nt_modified_subject.begin()+n_base_CUPDATE*rank_subject_x , nt_modified_subject.begin()+n_base_CUPDATE*(rank_subject_x+1) );  //erase the original nt entry for subject_proposed
  
  //---------------------------------------- proposing a new sequence & the proposal probability ----------------------------------------------------------//
  
  vector<int> seq_proposed(n_base_CUPDATE);
  
  //double dt;
  
  double t_proposed = 0.0; // not really gonna be used
  
  double t_past, t_future;
  
  double log_pr_forward=0.0; // the log of proposal probability
  
  vector<int> nt_past_forward(n_base_CUPDATE); // the sequence at the nearest past (in the direction of time change) compared to the time of the proposed sequence; this might be or might not be the original sequence which gotta be replaced
  vector<int> nt_future_forward(n_base_CUPDATE); // the sequence at the nearest future(in the direction of time change) compared to the time of the proposed sequence
  
  //dt = t_proposed - t_e_arg.at(subject_proposed); // the dimension of dt tells the direction of time change
  
  //
  
  double t_proposed_backward = 0.0;
  
  vector<int> seq_proposed_backward(n_base_CUPDATE);
  
  double t_past_backward, t_future_backward;
  
  double log_pr_backward=0.0; // the log of proposal probability
  
  vector<int> nt_past_backward(n_base_CUPDATE);
  vector<int> nt_future_backward(n_base_CUPDATE);
  
  //-----------------
  
  switch(int (current_size_arg.at(subject_proposed)>1)){ // return 1 when the subject has more than one sequence available
  
  case 0:{ //  ONLY one sequence available for the subject; XX not relevant as we assume #seq on index>=2 XX
    
    // 		for (int i=0; i<=(n_base_CUPDATE-1);i++){
    // 		seq_proposed.at(i) = gsl_rng_uniform_int(r_c, 4) +1;
    // 		}
    
    break;
  }
    
  case 1:{ // MORE than one sequence available for the subject
    
    nt_second_seq.assign(nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE , nt_current_arg.at(subject_proposed).begin()+n_base_CUPDATE*2 );
    
    t_past =   t_nt_current_arg[subject_proposed][1];
    nt_past_forward =nt_second_seq;
    t_future = t_e_arg.at(subject_proposed);//=0.0
    
    seq_propose_uncond(seq_proposed,  log_pr_forward, nt_past_forward, t_proposed, t_past, t_future,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE, rng);
    
    //---
    t_past_backward =  t_nt_current_arg[subject_proposed][1];
    nt_past_backward =  nt_second_seq;
    t_future_backward = t_e_arg.at(subject_proposed);//=0.0
    
    seq_proposed_backward = nt_current_seq;
    
    seq_backward_pr_uncond(seq_proposed_backward,  log_pr_backward,nt_past_backward, t_proposed_backward, t_past_backward, t_future_backward,  para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
    
    break;
  }
    
  }
  
  
  //-------------------------------------------end of proposing a new sequence ---------------------------------------------------------------------------------//
  
  int rank_subject_y = 0;
  
  nt_modified_subject.insert(nt_modified_subject.begin()+(rank_subject_y)*n_base_CUPDATE, seq_proposed.begin(), seq_proposed.end());  //insert  the new nt
  
  //----------------------------------------------------------------------------------//
  
  switch (int (current_size_arg.at(subject_proposed)>1)) {
  
  case 1:{
    
    log_lh_modified = log_lh_modified - lh_square_modified.log_f_S.at(subject_proposed); //subtract part of likelihood that would be updated below
    
    lh_square_modified.log_f_S.at(subject_proposed) = 0.0;
    
    for (int j=0;j<=(current_size_arg.at(subject_proposed)-2);j++){
      
      
      vector<int> seq_1(nt_modified_subject.begin()+j*(n_base_CUPDATE), nt_modified_subject.begin()+(j+1)*(n_base_CUPDATE));
      vector<int> seq_2(nt_modified_subject.begin()+(j+1)*(n_base_CUPDATE), nt_modified_subject.begin()+(j+2)*(n_base_CUPDATE));
      
      lh_square_modified.log_f_S.at(subject_proposed) =lh_square_modified.log_f_S.at(subject_proposed) + log_lh_seq(seq_1, seq_2, t_nt_current_arg[subject_proposed][j],t_nt_current_arg[subject_proposed][j+1], para_current_arg.mu_1, para_current_arg.mu_2, n_base_CUPDATE);
      
    }
    
    log_lh_modified = log_lh_modified + lh_square_modified.log_f_S.at(subject_proposed);
    
    break;
  }
    
  default:{
    break;
  }
    
  }
  
  
  //----------------------------------------------
  
  log_lh_modified = log_lh_modified - lh_square_modified.log_f_Snull.at(subject_proposed); //subtract part of likelihood that would be updated below
  
  lh_square_modified.log_f_Snull.at(subject_proposed) = lh_snull(con_seq, seq_proposed, para_current_arg.p_ber, n_base_CUPDATE); // compute the log pr a seq for background
  
  log_lh_modified = log_lh_modified + lh_square_modified.log_f_Snull.at(subject_proposed);
  
  //----------------------------------------------------------------------------------//
  
  //acp_pr = min(1.0,exp(log_lh_modified-log_lh_current_arg)*exp(log_pr_backward-log_pr_forward));
  acp_pr = min(1.0,exp((log_lh_modified-log_lh_current_arg)+(log_pr_backward-log_pr_forward)));
  
  
  double uniform_rv = runif(0.0, 1.0, rng_arg);
  
  
  switch(int (uniform_rv<=acp_pr)){
  case 1: {
    lh_square_current_arg = lh_square_modified;
    // delta_mat_current_arg = delta_mat_modified;
    log_lh_current_arg = log_lh_modified;
    // t_e_arg= t_e_modified;
    //index_arg = index_modified;
    //xi_E_minus_arg = xi_E_minus_modified;
    
    // nt_current_arg = nt_modified;
    // t_nt_current_arg = t_nt_modified;
    
    nt_current_arg.at(subject_proposed) = nt_modified_subject;
    //t_nt_current_arg.at(subject_proposed) = t_nt_modified_subject;
    
    
    /*	switch (subject_source){
     
    case 9999:{ // by background
     break;
    }
     
    default :{ // not by background
     nt_current_arg.at(subject_source) = nt_modified_source;
     t_nt_current_arg.at(subject_source) = t_nt_modified_source;
     infecting_list_current_arg.at(subject_source) = infecting_list_modified_source;
    }
    }
     */
    
    break;
  }
    
  case 0: {
    break;
  }
  }
  
  //gsl_rng_free(r_c);
  
  
}

/*------------------------------------------------*/

//------------------------ initialization of parameters/unobserved data for McMC ---------------------------------//

/* Assumptions: t_r known; t_i is known within a range */

void initialize_mcmc(para_key_init& para_init, para_key& para_current, para_aux& para_other, para_priors_etc& para_priors_etc, vector<int>& xi_I_current, vector<int>& xi_U_current, vector<int>& xi_E_current, vector<int>& xi_E_minus_current, vector<int>& xi_R_current,  vector<int>& xi_EnI_current,  vector<int>& xi_EnIS_current,  vector<int>& xi_InR_current,  vector<double>& t_e_current, vector<double>& t_i_current, vector<double>& t_r_current, vector<int>& index_current, vector<int>& infected_source_current, vector < vector<double> >& kernel_mat_current, vector <double>& norm_const_current, vec2int& sample_data, vector<double>& t_onset, nt_struct& nt_data_current, vector<int>& con_seq, vector < vector<double> >& beta_ij_mat_current){
  
  
  //const gsl_rng_type* T= gsl_rng_default;  // T is pointer points to the type of generator
  //gsl_rng *r = gsl_rng_alloc (T); // r is pointer points to an object with Type T
  
  int seed_intial_mcmc = 1;  //1,999,-1000,-10000,123456
  
  //gsl_rng_set (r, seed_intial_mcmc); // set a seed
  rng_type rng_m(seed_intial_mcmc); //set a seed
  
  ofstream myfile_out;
  
  if (debug == 1) {
    myfile_out.open((string(PATH2) + string("seed_initial_mcmc.csv")).c_str(), ios::out);
    myfile_out << seed_intial_mcmc;
    myfile_out.close();
  }
  para_current.alpha = para_init.alpha; //initialization of parameter to be estimated
  para_current.beta = para_init.beta; //initialization of parameter to be estimated
  
  para_current.lat_mu = para_init.lat_mu; //initialization of parameter to be estimated
  para_current.lat_var = para_init.lat_var; //initialization of parameter to be estimated
  
  para_current.c = para_init.c; //initialization of parameter to be estimated
  para_current.d = para_init.d; // weibull parameter (redundant if exponential)
  
  para_current.k_1 = para_init.k_1; //initialization of parameter to be estimated
  
  para_current.mu_1 = para_init.mu_1; //initialization of parameter to be estimated
  para_current.mu_2 = para_init.mu_2; //initialization of parameter to be estimated
  
  para_current.p_ber = para_init.p_ber;
  
  para_current.rho_susc1 = para_init.rho_susc1; //initialization of parameter to be estimated
  para_current.rho_susc2 = para_init.rho_susc2; //initialization of parameter to be estimated
  para_current.phi_inf1 = para_init.phi_inf1; //initialization of parameter to be estimated
  para_current.phi_inf2 = para_init.phi_inf2; //initialization of parameter to be estimated
  
  para_current.tau_susc = para_init.tau_susc; //initialization of parameter to be estimated
  para_current.nu_inf = para_init.nu_inf; //initialization of parameter to be estimated
  
  para_current.beta_m = para_init.beta_m; //initialization of parameter to be estimated
  
  //--------
  
  nt_data_current.t_nt.resize(para_other.n);
  nt_data_current.nt.resize(para_other.n);
  
  nt_data_current.current_size.resize(para_other.n);
  nt_data_current.infecting_size.resize(para_other.n);
  nt_data_current.infecting_list.resize(para_other.n);
  
  for (int i=0; i<= (int)(para_other.n-1);i++){
    nt_data_current.t_nt.at(i).clear();
    nt_data_current.nt.at(i).clear();
    
    nt_data_current.current_size.at(i)  = 0;
    
    nt_data_current.infecting_size.at(i) = 0; // initially infecting no one
    nt_data_current.infecting_list.at(i).clear();
    
  }
  
  
  
  //---------------------------------------------------------------------------------------//*/
  
  for (int i=0; i<= (int)(xi_E_current.size()-1);i++){// loop over infections
    
    switch( int(nt_data_current.t_sample.at(xi_E_current.at(i))!=para_other.unassigned_time)){
    
    case 1:{ // with sample, must be infected
    
    nt_data_current.current_size.at(xi_E_current.at(i)) = 2;//  (initial size) infection + the sample
    
    nt_data_current.t_nt.at(xi_E_current.at(i)).push_back(nt_data_current.t_sample.at(xi_E_current.at(i)));
    //nt_data_current.nt.at(xi_E_current.at(i)).insert(nt_data_current.nt.at(xi_E_current.at(i)).begin(), sample_data.at(xi_E_current.at(i)).begin(), sample_data.at(xi_E_current.at(i)).end());
    break;
  }
      
    case 0:{
      nt_data_current.current_size.at(xi_E_current.at(i)) = 1;//  (initial size) infection
      break;
    }
      
    }
  }
  
  //--
  
  for (int i=0; i<= (int)(xi_E_current.size()-1);i++){// loop over infections
    
    int source;
    
    //--- when propose t_e before deciding source_pool---//
    // 	double t_up =   min( nt_data_current.t_sample.at(xi_E_current.at(i)), min( t_i_current.at(xi_E_current.at(i)),para_other.t_max));
    // 	double t_low = max(0.0, t_up - 2.0 );
    //
    // 	//t_e_current.at(xi_E_current.at(i)) = gsl_ran_flat(r,t_low, t_up);
    // 	t_e_current.at(xi_E_current.at(i)) =  max(0.0, t_up - gsl_ran_gamma(r, 10.0, 0.5) );
    //
    // 	//t_e_current.at(xi_E_current.at(i)) =t_e_current.at(xi_E_current.at(i));
    //----------------------------------------------------------//
    
    vector<int> source_pool;
    
    for (int j=0;j<=(int)(xi_I_current.size()-1);j++){
      
      //double t_low = min(nt_data_current.t_sample.at(xi_E_current.at(i)),t_i_current.at(xi_E_current.at(i)));
      double t_bound = min(nt_data_current.t_sample.at(xi_E_current.at(i)), min(t_i_current.at(xi_E_current.at(i)), para_other.t_max)); //
      
      switch( (t_i_current.at(xi_I_current.at(j))<t_bound) & (t_i_current.at(xi_I_current.at(j))>=(t_bound-para_priors_etc.t_bound_hi)) ){ // when propose t_e after deciding source_pool; 2nd condition only allows a subset of xi_I to be possible sources (avoid f_EnI=0)
      //		switch( (t_i_current.at(xi_I_current.at(j))<t_e_current.at(xi_E_current.at(i))) &  (t_r_current.at(xi_I_current.at(j))>=t_e_current.at(xi_E_current.at(i)))  ){ //when propose t_e before deciding source_pool
      
      
      case 1:{
        source_pool.push_back(xi_I_current.at(j));
        break;
      }
      case 0:{
        break;
      }
      }
    }
    
    source_pool.insert(source_pool.begin(),9999);
    
    int num_infectious = (int)source_pool.size();
    
    //-----------propose uniformly ----------------//
    
    //source = source_pool.at(gsl_rng_uniform_int(r, num_infectious)); // uniformly choose a new source (including bg)
    source = source_pool.at(runif_int(0, num_infectious-1, rng_m)); // uniformly choose a new source (including bg)
    
    
    //-------------------------------------------------------------------------------------------------//
    
    if ( find( index_current.begin(), index_current.end(), xi_E_current.at(i) )!=index_current.end()) source = 9999; // index is known, source = 9999
    
    //-------------------------------------------------------------------------------------------------//
    
    infected_source_current.at(xi_E_current.at(i)) = source;
    
    switch(int (source==9999)){
    case 0:{// 2nd infection
      
      //--------when propose t_e after deciding source_pool//
      
      double t_up = min(min(nt_data_current.t_sample.at(xi_E_current.at(i)), t_r_current.at(source)), min(t_i_current.at(xi_E_current.at(i)), para_other.t_max));
      double t_low = max(t_i_current.at(source), t_up - para_priors_etc.t_bound_hi);
      //t_e_current.at(xi_E_current.at(i)) = gsl_ran_flat(r,t_low, t_up);
      t_e_current.at(xi_E_current.at(i)) = runif(t_low, t_up, rng_m);
      
      //-------------------------------------------------------------------------//
      
      nt_data_current.t_nt.at(xi_E_current.at(i)).push_back(t_e_current.at(xi_E_current.at(i)));
      sort(nt_data_current.t_nt.at(xi_E_current.at(i)).begin(), nt_data_current.t_nt.at(xi_E_current.at(i)).end() );
      
      nt_data_current.t_nt.at(source).push_back(t_e_current.at(xi_E_current.at(i)));
      sort(nt_data_current.t_nt.at(source).begin(), nt_data_current.t_nt.at(source).end() );
      
      
      switch(int (nt_data_current.infecting_size.at(source)>=1)){
      case 1:{
        vector<double> t_y(nt_data_current.infecting_size.at(source));
        for (int k=0;k<=(nt_data_current.infecting_size.at(source)-1);k++){
          t_y.at(k) = t_e_current.at(nt_data_current.infecting_list[source][k]);
        }
        t_y.push_back(t_e_current.at(xi_E_current.at(i)));
        sort(t_y.begin(), t_y.end());
        
        int rank_source = (int)(distance(t_y.begin(), find(t_y.begin(), t_y.end(),t_e_current.at(xi_E_current.at(i)))));
        
        nt_data_current.infecting_list.at(source).insert( nt_data_current.infecting_list.at(source).begin() + rank_source, xi_E_current.at(i));
        
        break;
      }
        
      case 0:{
        nt_data_current.infecting_list.at(source).push_back(xi_E_current.at(i));
        break;
      }
      }
      
      
      nt_data_current.infecting_size.at(source) = nt_data_current.infecting_size.at(source) + 1;
      nt_data_current.current_size.at(source) = nt_data_current.current_size.at(source) + 1;
      
      break;
    }
    case 1:{// bg infection
      
      //--------when propose t_e after deciding source_pool//
      // 			double t_up = min(nt_data_current.t_sample.at(xi_E_current.at(i)), min(t_i_current.at(xi_E_current.at(i)), para_other.t_max));
      // 			double t_low = max(0.0, t_up -10.0 );
      // 			t_e_current.at(xi_E_current.at(i)) = gsl_ran_flat(r,t_low, t_up);
      //
      switch( find( index_current.begin(), index_current.end(), xi_E_current.at(i) )!=index_current.end()){// return 1 when it is an index
    case 0:{
      double t_up = min(nt_data_current.t_sample.at(xi_E_current.at(i)), min(t_i_current.at(xi_E_current.at(i)), para_other.t_max));
      double t_low = max(0.0, t_up -10.0 );
      //t_e_current.at(xi_E_current.at(i)) = gsl_ran_flat(r,t_low, t_up);
      t_e_current.at(xi_E_current.at(i)) = runif(t_low, t_up, rng_m);
      break;
    }
    case 1:{
      t_e_current.at(xi_E_current.at(i)) = 0.0;
      // t_e_current.at(xi_E_current.at(i)) = t_e_current.at(xi_E_current.at(i));
      
      break;
    }
    }
      //-------------------------------------------------------------------------//
      
      nt_data_current.t_nt.at(xi_E_current.at(i)).push_back(t_e_current.at(xi_E_current.at(i)));
      sort(nt_data_current.t_nt.at(xi_E_current.at(i)).begin(), nt_data_current.t_nt.at(xi_E_current.at(i)).end());
      
      break;
    }
    }
    
  }// end of loop over infections
  
  
  ////----initialization of index_current and xi_E_minus --///
  
  index_current.clear();
  xi_E_minus_current = xi_E_current;
  
  double min_t = *min_element(t_e_current.begin(),t_e_current.end());
  for (int i=0; i<= (int)(xi_E_current.size()-1);i++){
    if (t_e_current.at(xi_E_current.at(i))==min_t) {
      index_current.push_back(xi_E_current.at(i));
    }
  }
  for (int i=0; i<= (int)(index_current.size()-1);i++){
    xi_E_minus_current.erase(find(xi_E_minus_current.begin(),xi_E_minus_current.end(),index_current.at(i)));
  }
  
  /*
   myfile_out.open((string(PATH2)+string("initial_index.csv")).c_str(),ios::app);
   for (int i=0; i<= (int) (index_current.size()-1);i++){
   myfile_out << index_current.at(i) << "," << t_e_current.at(index_current.at(i)) << "," << infected_source_current.at(index_current.at(i)) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("initial_current_size.csv")).c_str(),ios::app);
   for (int i=0; i<= (int) (para_other.n-1);i++){
   myfile_out << nt_data_current.current_size.at(i) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("initial_t_e.csv")).c_str(),ios::app);
   for (int i=0; i<= (int) (para_other.n-1);i++){
   myfile_out << t_e_current.at(i) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("initial_t_i.csv")).c_str(),ios::app);
   for (int i=0; i<= (int) (para_other.n-1);i++){
   myfile_out << t_i_current.at(i) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("initial_infected_source.csv")).c_str(),ios::app);
   for (int i=0; i<= (int) (para_other.n-1);i++){
   myfile_out << infected_source_current.at(i) << endl;
   }
   myfile_out.close();
   
   myfile_out.open((string(PATH2)+string("initial_infecting_size.csv")).c_str(),ios::app);
   int sum_infected=0;
   for (int i=0; i<= (int) (para_other.n-1);i++){
   sum_infected = sum_infected +  nt_data_current.infecting_size.at(i) ;
   myfile_out << nt_data_current.infecting_size.at(i) << endl;
   }
   myfile_out <<endl;
   myfile_out <<"total" <<endl;
   myfile_out << sum_infected;
   myfile_out.close();
   */
  
  ////---- intialization of nt_data_current.nt ////
  
  vector<double> t_e_sort; // sorted t_e excluding susceptible
  
  for (int i=0; i<=(int) xi_E_current.size() -1 ; i++){
    if (t_e_current.at(xi_E_current.at(i))!=para_other.unassigned_time)  t_e_sort.push_back(t_e_current.at(xi_E_current.at(i)));
  }
  
  sort( t_e_sort.begin(),  t_e_sort.end());
  
  vector<int> xi_E_sort((int) xi_E_current.size());
  
  for (int i=0; i<=(int) xi_E_current.size() -1 ; i++){
    
    int rank_t = (int)(distance(t_e_sort.begin(), find(t_e_sort.begin(),t_e_sort.end(), t_e_current.at(xi_E_current.at(i))) ));
    
    xi_E_sort.at(rank_t) = xi_E_current.at(i);
  }
  
  /*
   myfile_out.open((string(PATH2)+string("xi_E_sort_initial.csv")).c_str(),ios::out);
   for (int i=0; i<=((int)xi_E_sort.size()-1);i++){
   myfile_out << xi_E_sort.at(i) << endl;
   }
   myfile_out.close();
   
   
   myfile_out.open((string(PATH2)+string("t_e_sort_initial.csv")).c_str(),ios::out);
   for (int i=0; i<=((int)xi_E_sort.size()-1);i++){
   myfile_out << t_e_current.at(xi_E_sort.at(i)) << endl;
   }
   myfile_out.close();
   */
  
  vector<int> ind_sample (NLIMIT, 0); // indicate if the sample a sampled case  has been included
  
  
  for (int i=0; i<=(int) xi_E_sort.size() -1 ; i++){
    
    int subject = xi_E_sort.at(i);
    int source = infected_source_current.at(subject);
    
    // 	nt_data_current.nt.at(subject).erase(nt_data_current.nt.at(subject).begin(), nt_data_current.nt.at(subject).end()); // erase the whole sequences record
    
    nt_data_current.nt.at(subject).clear(); // erase the whole sequences record
    
    // 	nt_data_current.nt.at(subject).resize(nt_data_current.current_size.at(subject) *para_other.n_base);
    
    
    switch(int (source==9999)) {
    case 0:{// secondary
      switch(int (nt_data_current.t_sample.at(source)==para_other.unassigned_time)){
    case 1:{// source has no sample
      seq_initialize_pair(nt_data_current, source, subject, t_e_current.at(subject), para_other.n_base, para_current);
      
      
      break;
    }
    case 0:{// source has sample
      switch((nt_data_current.t_sample.at(source)<= t_e_current.at(subject)) & (ind_sample.at(source)==0)){
    case 1:{ // has to insert sample first
      nt_data_current.nt.at(source).insert(nt_data_current.nt.at(source).end(),sample_data.at(source).begin(), sample_data.at(source).end());
      ind_sample.at(source) = 1;
      seq_initialize_pair(nt_data_current, source, subject, t_e_current.at(subject), para_other.n_base, para_current);
      
      
      
      break;
    }
    case 0:{
      seq_initialize_pair(nt_data_current, source, subject, t_e_current.at(subject), para_other.n_base, para_current);
      
      break;
    }
    }
      break;
    }
    }
      break;
    }
    case 1:{// bg
      
      vector<int> seq_new(para_other.n_base);
      sample_snull(con_seq, seq_new, para_current.p_ber, para_other.n_base, rng);
      nt_data_current.nt.at(subject).insert(nt_data_current.nt.at(subject).begin(),seq_new.begin(), seq_new.begin()+para_other.n_base);
      
      
      break;
    }
    }
  }
  
  
  for (int i=0; i<=(int) xi_E_sort.size() -1 ; i++){
    
    int subject = xi_E_sort.at(i);
    
    switch((nt_data_current.t_sample.at(subject)!=para_other.unassigned_time) &(ind_sample.at(subject)==0)){
    case 1:{ // has sample but not yet inclued
      nt_data_current.nt.at(subject).insert(nt_data_current.nt.at(subject).end(),sample_data.at(subject).begin(), sample_data.at(subject).end());
      ind_sample.at(subject)=1;
      
      
      break;
    }
    case 0:{
      break;
    }
      
    }
    
  }
  
  
  //---------------------end of  intialization of nt_data_current.nt-------------------------//
  
  
}


//---------------------------------//

void seq_initialize_pair(nt_struct& nt_data_arg, int  k_source_arg, int k_arg, double t_now_arg, int n_base, para_key& para_current){ // update sequences of infectious-infected pair
  
  // int rank_source= nt_data_arg.current_size.at(k_source_arg);
  // double t_null = nt_data_arg.t_nt[k_source_arg][rank_source-1];
  
  vector<int> seq_new(n_base);
  
  int rank_source = (int)(distance(nt_data_arg.t_nt.at(k_source_arg).begin(), find(nt_data_arg.t_nt.at(k_source_arg).begin(), nt_data_arg.t_nt.at(k_source_arg).end(), t_now_arg)));
  double t_null = nt_data_arg.t_nt[k_source_arg][rank_source-1];
  
  double dt = t_now_arg - t_null;
  
  double p_1, p_2, p_3, p_4;
  p_1 = p_2 = p_3 = p_4 = 0.0;
  
  if (opt_k80 == 0) {
    p_1 = 0.25 + 0.25*exp(-4.0*para_current.mu_2*dt) + 0.5*exp(-2.0*(para_current.mu_1 + para_current.mu_2)*dt); // pr of a base not changing
    p_2 = 0.25 + 0.25*exp(-4.0*para_current.mu_2*dt) - 0.5*exp(-2.0*(para_current.mu_1 + para_current.mu_2)*dt); // pr of a transition of a base
    p_3 = 1.0*(0.25 - 0.25*exp(-4.0*para_current.mu_2*dt));  // pr of a transversion (two possible events)
    p_4 = p_3;
    //double p_1 = 1.0- p_2 - 2.0*p_3;
  }
  if (opt_k80 == 1) {
    //K80: mu1 = alpha; mu2 = beta
    p_2 = 0.25 - 0.5*exp(-4.0*(para_current.mu_1 + para_current.mu_2)*dt) + 0.25*exp(-8.0*para_current.mu_2*dt); // P = pts
    p_3 = 0.5 - 0.5*exp(-8.0*para_current.mu_2*dt);  // Q = ptv (2 options)
    p_3 = p_3 / 2.0;
    p_4 = p_3;
    p_1 = 1 - p_2 - p_3 - p_4;
  }
  
  
  
  //double P[4] = {p_1, p_2, p_3, p_4};
  vector<double> P = { p_1, p_2, p_3, p_4 };
  
  //const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
  //gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
  //gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);
  
  //gsl_rng_set (r_c,-1000*k_arg*k_source_arg*(rank_source+1)); // set a seed
  rng_type rng_s(-1000 * k_arg*k_source_arg*(rank_source + 1)); // set a seed
  
  for (int j=0;j<=(n_base-1);j++){
    
    //int type= gsl_ran_discrete (r_c, g) + 1;
    int type = edf_sample(P, rng_s) + 1;
    
    switch(nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j]){
    
    case 1:{ // an A
      switch(type){
    case 1:{
      //		nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j]);
      seq_new.at(j) = nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j];
      break;
    }
    case 2:{
      // 		nt_data_arg.nt[k_source_arg].push_back(2);
      seq_new.at(j) = 2;
      break;
    }
    case 3:{
      // 		nt_data_arg.nt[k_source_arg].push_back(3);
      seq_new.at(j) = 3;
      break;
    }
    case 4:{
      // 		nt_data_arg.nt[k_source_arg].push_back(4);
      seq_new.at(j) = 4;
      break;
    }
    }
      break;
    }
      
    case 2:{ // a G
      switch(type){
    case 1:{
      // 		nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j]);
      seq_new.at(j) = nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j];
      break;
    }
    case 2:{
      // 		nt_data_arg.nt[k_source_arg].push_back(1);
      seq_new.at(j) = 1;
      break;
    }
    case 3:{
      // 		nt_data_arg.nt[k_source_arg].push_back(3);
      seq_new.at(j) = 3;
      break;
    }
    case 4:{
      // 		nt_data_arg.nt[k_source_arg].push_back(4);
      seq_new.at(j) = 4;
      break;
    }
    }
      break;
    }
      
    case 3:{ // a T
      switch(type){
    case 1:{
      // 		nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j]);
      seq_new.at(j) = nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j];
      break;
    }
    case 2:{
      // 		nt_data_arg.nt[k_source_arg].push_back(4);
      seq_new.at(j) = 4;
      break;
    }
    case 3:{
      // 		nt_data_arg.nt[k_source_arg].push_back(1);
      seq_new.at(j) = 1;
      break;
    }
    case 4:{
      // 		nt_data_arg.nt[k_source_arg].push_back(2);
      seq_new.at(j) = 2;
      break;
    }
    }
      break;
    }
      
    case 4:{ // a C
      switch(type){
    case 1:{
      // 		nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j]);
      seq_new.at(j) = nt_data_arg.nt[k_source_arg][(rank_source-1)*n_base+j];
      break;
    }
    case 2:{
      // 		nt_data_arg.nt[k_source_arg].push_back(3);
      seq_new.at(j) = 3;
      break;
    }
    case 3:{
      // 		nt_data_arg.nt[k_source_arg].push_back(1);
      seq_new.at(j) = 1;
      break;
    }
    case 4:{
      // 		nt_data_arg.nt[k_source_arg].push_back(2);
      seq_new.at(j) = 2;
      break;
    }
    }
      break;
    }
      
    }
  }
  
  //gsl_ran_discrete_free (g);
  
  nt_data_arg.nt.at(k_source_arg).insert(nt_data_arg.nt.at(k_source_arg).end(), seq_new.begin(), seq_new.end());
  nt_data_arg.nt.at(k_arg).insert(nt_data_arg.nt.at(k_arg).end(), seq_new.begin(), seq_new.end());
  
  
}


/*-------------------------------------------------------*/
///// round a double to y decimal places
double roundx(double x, int y) {
  x = round(x * pow(10, y)) / pow(10, y);
  return x;
}

/*-------------------------------------------------------*/
///// find index of first instance of an element in a vector<int>
inline int find_in_vec_int(int key_arg, vector<int> vec) {
  
  int rint = -99;
  for (int i = 0; i < int(vec.size()); i++)
    if (vec[i] == key_arg) {
      rint = i;
      break;
    }
    return(rint);
}
/*-------------------------------------------------------*/



/*----------------------------*/
/*- SIM functions ------------*/
/*----------------------------*/
///// Boost probability distributions
rng_type rng_sim;
int seed_sim;

double runif_sim(double x0, double x1, rng_type& rng_arg) {
	boost::variate_generator<rng_type &, Dunif > rndm(rng_arg, Dunif(x0, x1));
	return rndm();
}

int runif_int_sim(int x0, int x1, rng_type& rng_arg)
{
	boost::variate_generator<rng_type &, Dunif_int > rndm(rng_arg, Dunif_int(x0, x1));
	return rndm();
}

double rgamma_sim(double shape, double scale, rng_type& rng_arg) {
	boost::variate_generator<rng_type &, Dgamma > rndm(rng_arg, Dgamma(shape, scale));
	return rndm();
}

double rweibull_sim(double scale, double shape, rng_type& rng_arg) { //note reversed scale and shape from boost itself (shape=1.0 for exponential distribution)
	boost::variate_generator<rng_type &, Dweibull > rndm(rng_arg, Dweibull(shape, scale));
	return rndm();
}

int rbern_sim(double p, rng_type& rng_arg) {
	boost::variate_generator<rng_type &, Dbern > rndm(rng_arg, Dbern(p));
	return rndm();
}

double rnorm_sim(double mean, double sd, rng_type& rng_arg) {
	boost::variate_generator<rng_type &, Dnorm > rndm(rng_arg, Dnorm(mean, sd));
	return rndm();
}

//seeded differently through rcpp than VS so not used
/*
 double rexp_sim(double rate, rng_type& rng_arg) {
boost::variate_generator<rng_type &, Dexp > rndm(rng_arg, Dexp(rate));
return rndm();
}
*/

// empirical distribution random sampler (depends on runif function)
int edf_sample_sim(vector<double> vec, rng_type& rng_arg)
{
	double s = 0;
	vector<double> edf; // dynamic vector
	for (int k = 0; k < (int)(vec.size()); k++)
	{
		s += vec[k];
		edf.push_back(s);
	}
	double u = s * runif_sim(0.0, 1.0, rng_arg);
	vector<double>::iterator low;
	low = lower_bound(edf.begin(), edf.end(), u); // fast search for where to locate u in vector
	int i = int(low - edf.begin());
	return i;  // index from 0 of sampled element that u is within range
}

/*-------------------------------------------------------*/

void IO_para_aux(para_aux_struct& para_other_arg) {

	ifstream myfile_in_para_aux;
	ofstream myfile_out_para_aux;

	string line, field;
	int line_count = 0, field_count = 0;

	myfile_in_para_aux.open((string(PATH1) + string("parameters_other.csv")).c_str(), ios::in);
	line_count = 0;

	while (getline(myfile_in_para_aux, line)) {

		//getline(myfile_in_simdata,line);
		stringstream ss(line);
		//string field;
		field_count = 0;

		while (getline(ss, field, ',')) {
			stringstream fs(field);
			if ((line_count == 1) & (field_count == 0)) fs >> para_other_arg.n;
			if ((line_count == 1) & (field_count == 1)) fs >> para_other_arg.seed;
			if ((line_count == 1) & (field_count == 2)) fs >> para_other_arg.n_base;
			if ((line_count == 1) & (field_count == 3)) fs >> para_other_arg.n_seq;
			if ((line_count == 1) & (field_count == 4)) fs >> para_other_arg.t_max;
			if ((line_count == 1) & (field_count == 5)) fs >> para_other_arg.unassigned_time;
			if ((line_count == 1) & (field_count == 6)) fs >> para_other_arg.sample_range;
			if ((line_count == 1) & (field_count == 7)) fs >> para_other_arg.partial_seq_out;
			if ((line_count == 1) & (field_count == 8)) fs >> para_other_arg.n_base_part;
			if ((line_count == 1) & (field_count == 9)) fs >> para_other_arg.n_index;
			if ((line_count == 1) & (field_count == 10)) fs >> para_other_arg.coord_type;
			if ((line_count == 1) & (field_count == 11)) fs >> para_other_arg.kernel_type;
			if ((line_count == 1) & (field_count == 12)) fs >> para_other_arg.latent_type;
			if ((line_count == 1) & (field_count == 13)) fs >> para_other_arg.opt_k80;
			if ((line_count == 1) & (field_count == 14)) fs >> para_other_arg.opt_betaij;
			if ((line_count == 1) & (field_count == 15)) fs >> para_other_arg.opt_mov;
			if ((line_count == 1) & (field_count == 16)) fs >> para_other_arg.n_mov;
			if ((line_count == 1) & (field_count == 17)) fs >> para_other_arg.debug;

			field_count = field_count + 1;
		}

		line_count = line_count + 1;

	}

	myfile_in_para_aux.close();


	/*
	myfile_out_para_aux.open((string(PATH2) + string("parameters_other.csv")).c_str(), ios::out);
	myfile_out_para_aux << "n" << "," << "seed" << "," << "n_base" << "," << "n_seq" << "," << "t_max" << "," << "unassigned_time" << "," << "sample_range" << "," << "partial_seq_out" << "," << "n_base_part" << "," << "n_index" << "," << "coord_type" << "," << "kernel_type" << "," << "latent_type" << "," << "opt_k80" << "," << "opt_betaij" <<  "," << "opt_mov" << "," << "n_mov" << "," << "debug" << endl;
	myfile_out_para_aux << para_other_arg.n << "," << para_other_arg.seed << "," << para_other_arg.n_base << "," << para_other_arg.n_seq << "," << para_other_arg.t_max << "," << para_other_arg.unassigned_time << "," << para_other_arg.sample_range << "," << para_other_arg.partial_seq_out << "," << para_other_arg.n_base_part << "," << para_other_arg.n_index << "," << para_other_arg.coord_type << "," << para_other_arg.kernel_type << "," << para_other_arg.latent_type << "," << para_other_arg.opt_k80 << "," << para_other_arg.opt_betaij << "," << para_other_arg.opt_mov << "," << para_other_arg.n_mov << "," << para_other_arg.debug << endl;
	myfile_out_para_aux.close();
	*/

}
int opt_k80_sim, opt_betaij_sim, opt_mov_sim, debug_sim;

//---------


void IO_para_key(para_key_struct& para_key_arg) {

	ifstream myfile_in_para_key;
	ofstream myfile_out_para_key;

	string line, field;
	int line_count = 0, field_count = 0;

	myfile_in_para_key.open((string(PATH1) + string("parameters_key.csv")).c_str(), ios::in);
	line_count = 0;

	while (getline(myfile_in_para_key, line)) {

		//getline(myfile_in_simdata,line);
		stringstream ss(line);
		//string field;
		field_count = 0;

		while (getline(ss, field, ',')) {
			stringstream fs(field);
			if ((line_count == 1) & (field_count == 0)) fs >> para_key_arg.alpha;
			if ((line_count == 1) & (field_count == 1)) fs >> para_key_arg.beta;
			if ((line_count == 1) & (field_count == 2)) fs >> para_key_arg.mu_1;
			if ((line_count == 1) & (field_count == 3)) fs >> para_key_arg.mu_2;
			if ((line_count == 1) & (field_count == 4)) fs >> para_key_arg.a;
			if ((line_count == 1) & (field_count == 5)) fs >> para_key_arg.b;
			if ((line_count == 1) & (field_count == 6)) fs >> para_key_arg.c;
			if ((line_count == 1) & (field_count == 7)) fs >> para_key_arg.d;
			if ((line_count == 1) & (field_count == 8)) fs >> para_key_arg.k_1;
			if ((line_count == 1) & (field_count == 9)) fs >> para_key_arg.p_ber;
			if ((line_count == 1) & (field_count == 10)) fs >> para_key_arg.phi_inf1;
			if ((line_count == 1) & (field_count == 11)) fs >> para_key_arg.phi_inf2;
			if ((line_count == 1) & (field_count == 12)) fs >> para_key_arg.rho_susc1;
			if ((line_count == 1) & (field_count == 13)) fs >> para_key_arg.rho_susc2;
			if ((line_count == 1) & (field_count == 14)) fs >> para_key_arg.nu_inf;
			if ((line_count == 1) & (field_count == 15)) fs >> para_key_arg.tau_susc;
			////////////////////////////////////////////////////////////////////////////////////////////////////////
			if ((line_count == 1) & (field_count == 16)) fs >> para_key_arg.beta_m;
			////////////////////////////////////////////////////////////////////////////////////////////////////////
			field_count = field_count + 1;
		}

		line_count = line_count + 1;

	}

	myfile_in_para_key.close();


	/*
	myfile_out_para_key.open((string(PATH2) + string("parameters_key.csv")).c_str(), ios::out);
	myfile_out_para_key << "alpha" << "," << "beta" << "," << "mu_1" << "," << "mu_2" << "," << "a" << "," << "b" << "," << "c" << "," << "d" << "," << "k_1" << "," << "p_ber" << "," << "phi_inf1" << "," << "phi_inf2" << "," << "rho_susc1" << "," << "rho_susc2" << ","  << "nu_inf" << ","  <<"tau_susc" << "," << "beta_m" <<endl;
	myfile_out_para_key << para_key_arg.alpha << "," << para_key_arg.beta << "," << para_key_arg.mu_1 << "," << para_key_arg.mu_2 << "," << para_key_arg.a<< "," << para_key_arg.b<< "," << para_key_arg.c<< "," << para_key_arg.d << "," << para_key_arg.k_1 << "," << para_key_arg.p_ber << "," << para_key_arg.phi_inf1 << "," << para_key_arg.phi_inf2 << "," << para_key_arg.rho_susc1 << "," << para_key_arg.rho_susc2 << "," << para_key_arg.nu_inf << "," << para_key_arg.tau_susc <<  "," << para_key_arg.beta_m<< endl;
	myfile_out_para_key.close();
	*/
}

//---------


void IO_para_epi(epi_struct_sim& epi_data_arg, para_aux_struct& para_other_arg) {

	epi_data_arg.k.resize(para_other_arg.n);
	epi_data_arg.coor_x.resize(para_other_arg.n);
	epi_data_arg.coor_y.resize(para_other_arg.n);
	epi_data_arg.t_e.resize(para_other_arg.n);
	epi_data_arg.t_i.resize(para_other_arg.n);
	epi_data_arg.t_r.resize(para_other_arg.n);
	epi_data_arg.ftype0.resize(para_other_arg.n);
	epi_data_arg.ftype1.resize(para_other_arg.n);
	epi_data_arg.ftype2.resize(para_other_arg.n);
	epi_data_arg.herdn.resize(para_other_arg.n);
	epi_data_arg.status.resize(para_other_arg.n);

	ifstream myfile_in_para_epi;

	string line, field;
	int line_count = 0, field_count = 0;

	myfile_in_para_epi.open((string(PATH1) + string("epi_in.csv")).c_str(), ios::in);
	line_count = 0;

	while (getline(myfile_in_para_epi, line)) {

		stringstream ss(line);
		field_count = 0;

		while (getline(ss, field, ',')) {
			stringstream fs(field);
			if ((line_count >= 1) & (field_count == 0)) fs >> epi_data_arg.k.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 1)) fs >> epi_data_arg.coor_x.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 2)) fs >> epi_data_arg.coor_y.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 3)) fs >> epi_data_arg.t_e.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 4)) fs >> epi_data_arg.t_i.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 5)) fs >> epi_data_arg.t_r.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 6)) fs >> epi_data_arg.ftype0.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 7)) fs >> epi_data_arg.ftype1.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 8)) fs >> epi_data_arg.ftype2.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 9)) fs >> epi_data_arg.herdn.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 10)) fs >> epi_data_arg.status.at(line_count - 1);

			field_count = field_count + 1;
		}

		line_count = line_count + 1;
	}

	myfile_in_para_epi.close();

	/*
	ofstream myfile_out_para_epi;
	myfile_out_para_epi.open((string(PATH2)+string("epi_inputs.csv")).c_str(),ios::app);
	myfile_out_para_epi << "k" << "," << "coor_x" << "," << "coor_y" << "," << "t_e" << "," << "t_i"<< "," << "t_r" <<"," << "ftype0" << "," << "ftype1" << "," << "ftype2" << "," << "herdn" << "," << "status"<< endl;
	for (int i=0; i<=(para_other_arg.n-1);i++){
		myfile_out_para_epi << epi_data_arg.k.at(i) << "," << epi_data_arg.coor_x.at(i) << "," << epi_data_arg.coor_y.at(i)<< "," << epi_data_arg.t_e.at(i) << "," << epi_data_arg.t_i.at(i)<< "," << epi_data_arg.t_r.at(i)  <<  "," << epi_data_arg.ftype0.at(i) <<  "," << epi_data_arg.ftype1.at(i) << "," << epi_data_arg.ftype2.at(i) << "," << epi_data_arg.herdn.at(i) << "," << epi_data_arg.status.at(i) <<endl;
	}
	myfile_out_para_epi.close();
	*/
}

//---------


////////////////////////////////////////////////////////////////////////////////////////////////////////

void IO_para_mov(mov_struct& mov_data_arg, para_aux_struct& para_other_arg) {

	mov_data_arg.from_k.resize(para_other_arg.n_mov);
	mov_data_arg.to_k.resize(para_other_arg.n_mov);
	mov_data_arg.t_m.resize(para_other_arg.n_mov);

	ifstream myfile_in_para_mov;


	string line, field;
	int line_count = 0, field_count = 0;

	myfile_in_para_mov.open((string(PATH1) + string("mov_in.csv")).c_str(), ios::in);
	line_count = 0;

	while (getline(myfile_in_para_mov, line)) {

		stringstream ss(line);
		field_count = 0;

		while (getline(ss, field, ',')) {
			stringstream fs(field);
			if ((line_count >= 1) & (field_count == 0)) fs >> mov_data_arg.from_k.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 1)) fs >> mov_data_arg.to_k.at(line_count - 1);
			if ((line_count >= 1) & (field_count == 2)) fs >> mov_data_arg.t_m.at(line_count - 1);


			field_count = field_count + 1;
		}

		line_count = line_count + 1;
	}

	myfile_in_para_mov.close();

	/*
	ofstream myfile_out_para_mov;
	myfile_out_para_mov.open((string(PATH2) + string("mov_inputs.csv")).c_str(), ios::app);
	myfile_out_para_mov << "from_k" << "," << "to_k" << "," << "t_m" << endl;


	for (int i = 0; i < ( para_other_arg.n_mov); i++) {
		myfile_out_para_mov  << mov_data_arg.from_k.at(i) << "," << mov_data_arg.to_k.at(i) << "," << mov_data_arg.t_m.at(i) << endl;
	}
	myfile_out_para_mov.close();
	*/


}

/*-------------------------------------------*/

inline int func_mov_cnt(int i_arg, int j_arg, mov_struct& mov_arg, vector<double> t_e_arg, vector<double> t_i_arg, vector<double> t_r_arg, double t_now_arg) {

	int mov_ij_count = 0;

	for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
		if ((mov_arg.from_k[m] == i_arg) && (mov_arg.to_k[m] == j_arg)) {

			if ((mov_arg.t_m[m] >= t_i_arg.at(i_arg)) && //could incorporate trace_window here as additional if(t_now_arg < mov_arg.t_m[m] + trace_window)
				(mov_arg.t_m[m] <= t_r_arg.at(i_arg)) &&
				(mov_arg.t_m[m] <= t_now_arg) &&
				(mov_arg.t_m[m] <= t_e_arg.at(j_arg))) {

				mov_ij_count++;
			}

		}

	}

	return(mov_ij_count);
}

/*-------------------------------------------*/

inline double func_mov_exp(int i_arg, int j_arg, mov_struct& mov_arg, vector<double> t_e_arg, vector<double> t_i_arg, vector<double> t_r_arg, double time, double t_now_arg) {

	double mov_ij_exposure = 0.0;

	for (int m = 0; m <= (int)(mov_arg.from_k.size() - 1); m++) {
		if ((mov_arg.from_k[m] == i_arg) && (mov_arg.to_k[m] == j_arg)) {

			if ((mov_arg.t_m[m] >= t_i_arg.at(i_arg)) && //could incorporate trace_window here as additional if(t_now_arg < mov_arg.t_m[m] + trace_window)
				(mov_arg.t_m[m] <= t_r_arg.at(i_arg)) &&
				(mov_arg.t_m[m] <= t_now_arg) &&
				(mov_arg.t_m[m] <= t_e_arg.at(j_arg))) {

				mov_ij_exposure = mov_ij_exposure + (min(t_r_arg.at(i_arg), time) - mov_arg.t_m[m]);
			}

		}

	}

	return(mov_ij_exposure);
}
/*-------------------------------------------*/

//inline double func_delta_ij(double delta_arg, int mov_ij_arg) {
//
//	double delta_ij;
//	delta_ij = delta_arg * mov_ij_arg;
//
//
//	return(delta_ij);
//}


////////////////////////////////////////////////////////////////////////////////////////////////////////

double func_kernel_sim(double x_1, double y_1, double x_2, double y_2, double par_kernel_1_arg, string coord_type_arg, string kernel_type_arg) {

	double eucli_dist = 0.0;

	//cartesian coordinate system
	if (coord_type_arg == "cartesian") {
		eucli_dist = sqrt(pow((x_1 - x_2), 2) + pow((y_1 - y_2), 2)); //coordinates inputted in kilometres
	}

	//lat/long coordinate system (x is lat, y is long because of how Max set it up)
	if (coord_type_arg == "longlat") {
		double pi = 3.1415926535897;
		double rad = pi / 180.0;
		double x_1_rad = x_1 * rad;
		double y_1_rad = y_1 * rad;
		double x_2_rad = x_2 * rad;
		double y_2_rad = y_2 * rad;
		double dlon = x_2_rad - x_1_rad;
		double dlat = y_2_rad - y_1_rad;
		double a = pow((sin(dlat / 2.0)), 2.0) + cos(y_1_rad) * cos(y_2_rad) * pow((sin(dlon / 2.0)), 2.0);
		double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
		double R = 6378.145;  //radius of Earth in km, output is scaled in km
		eucli_dist = R * c;
	}

	double func_ker = 0.0;

	if (kernel_type_arg == "exponential") { func_ker = exp((-par_kernel_1_arg)*eucli_dist); }
	if (kernel_type_arg == "power_law") { func_ker = 1.0 / (1.0 + pow(eucli_dist, par_kernel_1_arg)); }
	if (kernel_type_arg == "cauchy") { func_ker = (1 / (par_kernel_1_arg*(1 + pow(eucli_dist / par_kernel_1_arg, 2.0)))); }
	if (kernel_type_arg == "gaussian") {
		func_ker = exp(-pow(eucli_dist, par_kernel_1_arg));
	}

	return(func_ker);
}

//----------------------

double func_latent_ran(rng_type & rng_arg, double par_1, double par_2) {

	double func_lat_ran;

	//func_lat_ran = gsl_ran_gamma(r_c, par_1, par_2);
	func_lat_ran = rgamma_sim(par_1, par_2, rng_arg);
	//func_lat_ran = gsl_ran_exponential(r_c, par_1*par_2); // par1*par2 (from gamma) = mu of exp
	//func_lat_ran = rweibull(par_1*par_2, 1.0, rng_arg);

	return(func_lat_ran);

}

/*-------------------------------------------*/


inline double func_beta_ij_sim(double n_inf, double n_susc, double nu_inf_arg, double tau_susc_arg, double ftype0_inf, double ftype0_susc, double ftype1_inf, double ftype1_susc, double ftype2_inf, double ftype2_susc, double phi_inf1_arg, double phi_inf2_arg, double rho_susc1_arg, double rho_susc2_arg) {


	double func_beta_ij;

	func_beta_ij = ((pow(n_inf, nu_inf_arg) * (1.0 * ftype0_inf)) +  //cattle (reference)
		(pow(n_inf, nu_inf_arg) * (phi_inf1_arg * ftype1_inf)) +  //pigs
		(pow(n_inf, nu_inf_arg) * (phi_inf2_arg * ftype2_inf))) * //other
		((pow(n_susc, tau_susc_arg) * (1.0 * ftype0_susc)) +	//cattle (reference)
		(pow(n_susc, tau_susc_arg) * (rho_susc1_arg * ftype1_susc)) +  //pigs
			(pow(n_susc, tau_susc_arg) * (rho_susc2_arg * ftype2_susc)));	//other
	return(func_beta_ij);
}

/*------------------------------------------------*/

void initialize_beta_ij_mat(vector< vector<double> >& beta_ij_mat_arg, para_aux_struct& para_other_arg, epi_struct_sim& epi_data_arg, para_key_struct& para_key_arg) {

	for (int i = 0; i <= (para_other_arg.n - 1); i++) { //infectives
		for (int j = 0; j <= (para_other_arg.n - 1); j++) { //susceptibles
			if (i == j) beta_ij_mat_arg[i][j] = 0.0;
			if (i != j) beta_ij_mat_arg[i][j] = func_beta_ij_sim(epi_data_arg.herdn.at(i), epi_data_arg.herdn.at(j), para_key_arg.nu_inf, para_key_arg.tau_susc, epi_data_arg.ftype0.at(i), epi_data_arg.ftype0.at(j), epi_data_arg.ftype1.at(i), epi_data_arg.ftype1.at(j), epi_data_arg.ftype2.at(i), epi_data_arg.ftype2.at(j), para_key_arg.phi_inf1, para_key_arg.phi_inf2, para_key_arg.rho_susc1, para_key_arg.rho_susc2);
		}
	}

}

//----------------------

void epi_functions::func_ric(mov_struct& mov_data_arg, epi_struct_sim& epi_data_arg, vector<int> uninfected_arg, vector<int> once_infectious_arg, const vector<double>& norm_const_arg, double t_now_arg, vector< vector<double> >& beta_ij_mat_arg, rng_type & rng_arg) {

	int num_uninfected = int(uninfected_arg.size());

	switch (int(num_uninfected >= 1)) {

	case 1: {
		for (int j = 0; j <= (num_uninfected - 1); j++) {

			int k_arg = uninfected_arg.at(j);

			double ric = 0.0;


			switch (int(once_infectious_arg.size() == 0)) {
			case 1: {  //there are as yet none infected, just background pressure (alpha)
				ric = epi_data_arg.q.at(k_arg) - alpha_Cepi * t_now_arg;

				break;
			}

			case 0: {  //background pressure (alpha) + pressure from those infectious by t_now for the duration of exposure

				double beta_ij_cumulative = 0.0;
				double duration_exposure = 0.0;
				double duration_exposure_mov = 0.0;
				double mov_ij_cumulative = 0;


				for (int i = 0; i <= ((int)once_infectious_arg.size() - 1); i++) {
					if (epi_data_arg.t_r.at(once_infectious_arg.at(i)) != unassigned_time_Cepi) {
						duration_exposure = epi_data_arg.t_r.at(once_infectious_arg.at(i)) - epi_data_arg.t_i.at(once_infectious_arg.at(i));
						duration_exposure_mov = func_mov_exp(once_infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, epi_data_arg.t_r.at(once_infectious_arg.at(i)), t_now_arg);

					}

					if (epi_data_arg.t_r.at(once_infectious_arg.at(i)) == unassigned_time_Cepi) {
						duration_exposure = t_now_arg - epi_data_arg.t_i.at(once_infectious_arg.at(i));
						duration_exposure_mov = func_mov_exp(once_infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, t_now_arg, t_now_arg);
					}

					if (opt_betaij_sim == 0) {
						beta_ij_cumulative = beta_ij_cumulative + (duration_exposure)*distance_mat_Cepi[k_arg][once_infectious_arg.at(i)] / norm_const_arg.at(once_infectious_arg.at(i)); //the beta contribution inside the sum over all infectious at time_now
					}

					if (opt_betaij_sim == 1) {
						beta_ij_cumulative = beta_ij_cumulative + (duration_exposure)*beta_ij_mat_arg[once_infectious_arg.at(i)][k_arg] * distance_mat_Cepi[k_arg][once_infectious_arg.at(i)] / norm_const_arg.at(once_infectious_arg.at(i)); //the beta contribution inside the sum over all infectious at time_now
					}

					if (opt_mov_sim == 0) {
						mov_ij_cumulative = 0;
					}

					if (opt_mov_sim == 1) {
						mov_ij_cumulative = mov_ij_cumulative + duration_exposure_mov;
					}

				}


				ric = epi_data_arg.q.at(k_arg) - alpha_Cepi * t_now_arg - beta_Cepi * beta_ij_cumulative - beta_m_Cepi * mov_ij_cumulative;

				break;
			}
			}

			epi_data_arg.ric.at(k_arg) = ric;

		}

		break;
	}

	case 0: {
		//do nothing
		break;
	}
	}

	//return(ric);

}

//

void epi_functions::func_t_next(mov_struct& mov_data_arg, epi_struct_sim& epi_data_arg, vector<int> infectious_arg, double t_now_arg, const vector<double>& norm_const_arg, int loop_count_arg, vector< vector<double> >& beta_ij_mat_arg, rng_type & rng_arg) {


	int num_infectious = int(infectious_arg.size());

	for (int i = 0; i <= (n_Cepi - 1); i++) {

		double t_next = unassigned_time_Cepi;
		int k_arg = i;

		switch (epi_data_arg.status.at(k_arg)) {
		case 1: {  //1=S

			switch (int((num_infectious) >= 1)) {
			case 1: {
				double beta_ij_now = 0.0; // the additional (secondary) spatial-related contribution to the infectious pressure, from each of those infectious at time_now
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				int mov_ij_now = 0;
				double beta_m_ij_now = 0.0; // the additional (secondary) movement-related contribution to the infectious pressure, from each of those infectious at time_now
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				for (int i = 0; i <= (num_infectious - 1); i++) {
					if (opt_betaij_sim == 0) {
						beta_ij_now = beta_ij_now + distance_mat_Cepi[k_arg][infectious_arg.at(i)] / norm_const_arg.at(infectious_arg.at(i));  //component within the sum over all infectious
					}
					if (opt_betaij_sim == 1) {
						beta_ij_now = beta_ij_now + beta_ij_mat_arg[infectious_arg.at(i)][k_arg] * distance_mat_Cepi[k_arg][infectious_arg.at(i)] / norm_const_arg.at(infectious_arg.at(i));  //component within the sum over all infectious
					}

					////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (opt_mov_sim == 0) {
						mov_ij_now = 0;
					}

					if (opt_mov_sim == 1) {
						mov_ij_now = mov_ij_now + func_mov_cnt(infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, t_now_arg);
					}
					////////////////////////////////////////////////////////////////////////////////////////////////////////


				}
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				if (opt_mov_sim == 0) {
					beta_ij_now = beta_Cepi * beta_ij_now; //any components outside the sum (i.e. base beta)
					t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi + beta_ij_now);
				}
				if (opt_mov_sim == 1) {
					beta_ij_now = beta_Cepi * beta_ij_now; //any components outside the sum (i.e. base beta)
					beta_m_ij_now = beta_m_Cepi * mov_ij_now;
					t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi + beta_ij_now + beta_m_ij_now);
				}
				////////////////////////////////////////////////////////////////////////////////////////////////////////
				break;
			}

			case 0: {
				t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi); //time until infected at background rate
				break;
			}
			}
			break;
		}

		case 2: { //2 = E

		//const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
		//gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
		//gsl_rng_set (r_c, seed_Cepi*k_arg); // set a seed
		//t_next = epi_data_arg.t_e.at(k_arg)+ func_latent_ran(r_c,a_Cepi,b_Cepi);
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		//t_next = epi_data_arg.t_e.at(k_arg) + func_latent_ran(rng_arg, a_Cepi, b_Cepi);
			t_next = epi_data_arg.t_e.at(k_arg) + epi_data_arg.lat_period.at(k_arg);
			////////////////////////////////////////////////////////////////////////////////////////////////////////


			//gsl_rng_free(r_c);

			//t_next = epi_data_arg.t_e.at(k_arg)+ gsl_ran_gamma(r_Cepi,a_Cepi,b_Cepi);

			break;
		}

		case 3: {  //3 = I
		//const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
		//gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
		//gsl_rng_set (r_c, seed_Cepi*k_arg); // set a seed
		//t_next = epi_data_arg.t_i.at(k_arg) + gsl_ran_weibull(r_c,c_Cepi,d_Cepi);
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		//t_next = epi_data_arg.t_i.at(k_arg) + rweibull_sim(c_Cepi, d_Cepi, rng_arg);
			t_next = epi_data_arg.t_i.at(k_arg) + epi_data_arg.inf_period.at(k_arg);
			////////////////////////////////////////////////////////////////////////////////////////////////////////


			//gsl_rng_free(r_c);


			break;
		}

		case 4: {  //4 = R
			t_next = unassigned_time_Cepi;
			break;
		}

		}


		epi_data_arg.t_next.at(k_arg) = t_next;

	}

}

//----------------------

void epi_functions::func_ric_j(int k_arg, mov_struct& mov_data_arg, epi_struct_sim& epi_data_arg, vector<int> uninfected_arg, vector<int> once_infectious_arg, const vector<double>& norm_const_arg, double t_now_arg, vector< vector<double> >& beta_ij_mat_arg, rng_type & rng_arg) {



	double ric = 0.0;


	switch (int(once_infectious_arg.size() == 0)) {
	case 1: {  //there are as yet none infected, just background pressure (alpha)
		ric = epi_data_arg.q.at(k_arg) - alpha_Cepi * t_now_arg;

		break;
	}

	case 0: {  //background pressure (alpha) + pressure from those infectious by t_now for the duration of exposure

		double beta_ij_cumulative = 0.0;
		double duration_exposure = 0.0;
		double duration_exposure_mov = 0.0;
		double mov_ij_cumulative = 0;


		for (int i = 0; i <= ((int)once_infectious_arg.size() - 1); i++) {
			if (epi_data_arg.t_r.at(once_infectious_arg.at(i)) != unassigned_time_Cepi) {
				duration_exposure = epi_data_arg.t_r.at(once_infectious_arg.at(i)) - epi_data_arg.t_i.at(once_infectious_arg.at(i));
				duration_exposure_mov = func_mov_exp(once_infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, epi_data_arg.t_r.at(once_infectious_arg.at(i)), t_now_arg);

			}

			if (epi_data_arg.t_r.at(once_infectious_arg.at(i)) == unassigned_time_Cepi) {
				duration_exposure = t_now_arg - epi_data_arg.t_i.at(once_infectious_arg.at(i));
				duration_exposure_mov = func_mov_exp(once_infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, t_now_arg, t_now_arg);
			}

			if (opt_betaij_sim == 0) {
				beta_ij_cumulative = beta_ij_cumulative + (duration_exposure)*distance_mat_Cepi[k_arg][once_infectious_arg.at(i)] / norm_const_arg.at(once_infectious_arg.at(i)); //the beta contribution inside the sum over all infectious at time_now
			}

			if (opt_betaij_sim == 1) {
				beta_ij_cumulative = beta_ij_cumulative + (duration_exposure)*beta_ij_mat_arg[once_infectious_arg.at(i)][k_arg] * distance_mat_Cepi[k_arg][once_infectious_arg.at(i)] / norm_const_arg.at(once_infectious_arg.at(i)); //the beta contribution inside the sum over all infectious at time_now
			}

			if (opt_mov_sim == 0) {
				mov_ij_cumulative = 0;
			}

			if (opt_mov_sim == 1) {
				mov_ij_cumulative = mov_ij_cumulative + duration_exposure_mov;
			}

		}


		ric = epi_data_arg.q.at(k_arg) - alpha_Cepi * t_now_arg - beta_Cepi * beta_ij_cumulative - beta_m_Cepi * mov_ij_cumulative;

		break;
	}
	}

	epi_data_arg.ric.at(k_arg) = ric;





	//return(ric);

}

//----------------------

void epi_functions::func_t_next_j(int k_arg, mov_struct& mov_data_arg, epi_struct_sim& epi_data_arg, vector<int> infectious_arg, double t_now_arg, const vector<double>& norm_const_arg, int loop_count_arg, vector< vector<double> >& beta_ij_mat_arg, rng_type & rng_arg) {

	int num_infectious = int(infectious_arg.size());

	double t_next = unassigned_time_Cepi;


	switch (int((num_infectious) >= 1)) {
	case 1: {
		double beta_ij_now = 0.0; // the additional (secondary) spatial-related contribution to the infectious pressure, from each of those infectious at time_now
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		int mov_ij_now = 0;
		double beta_m_ij_now = 0.0; // the additional (secondary) movement-related contribution to the infectious pressure, from each of those infectious at time_now
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		for (int i = 0; i <= (num_infectious - 1); i++) {
			if (opt_betaij_sim == 0) {
				beta_ij_now = beta_ij_now + distance_mat_Cepi[k_arg][infectious_arg.at(i)] / norm_const_arg.at(infectious_arg.at(i));  //component within the sum over all infectious
			}
			if (opt_betaij_sim == 1) {
				beta_ij_now = beta_ij_now + beta_ij_mat_arg[infectious_arg.at(i)][k_arg] * distance_mat_Cepi[k_arg][infectious_arg.at(i)] / norm_const_arg.at(infectious_arg.at(i));  //component within the sum over all infectious
			}

			////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (opt_mov_sim == 0) {
				mov_ij_now = 0;
			}

			if (opt_mov_sim == 1) {
				mov_ij_now = mov_ij_now + func_mov_cnt(infectious_arg.at(i), k_arg, mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, t_now_arg);
			}
			////////////////////////////////////////////////////////////////////////////////////////////////////////


		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		if (opt_mov_sim == 0) {
			beta_ij_now = beta_Cepi * beta_ij_now; //any components outside the sum (i.e. base beta)
			t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi + beta_ij_now);
		}
		if (opt_mov_sim == 1) {
			beta_ij_now = beta_Cepi * beta_ij_now; //any components outside the sum (i.e. base beta)
			beta_m_ij_now = beta_m_Cepi * mov_ij_now;
			t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi + beta_ij_now + beta_m_ij_now);
		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		break;
	}

	case 0: {
		t_next = t_now_arg + epi_data_arg.ric.at(k_arg) / (alpha_Cepi); //time until infected at background rate
		break;
	}
	}


	epi_data_arg.t_next.at(k_arg) = t_next;



}


//----------------------

vector<int> min_max_functions::min_position_double(vector<double> v, double min_t, int max_leng) {

	vector<int> min_v;
	min_v.reserve(max_leng);

	int m = int(v.size());

	for (int i = 0; i <= (m - 1); i++) {
		if (v.at(i) == min_t) min_v.push_back(i);
		//(DOUBLE_EQ(v.at(i),min_t)) min_v.push_back(i) ;
	}

	return(min_v);
}


//



void epi_functions::func_status_update(epi_struct_sim& epi_data_arg, vector<int> ind_now_arg, rng_type & rng_arg) {

	int num_now = int(ind_now_arg.size());

	for (int i = 0; i <= (num_now - 1); i++) {
		epi_data_arg.status.at(ind_now_arg.at(i)) = min(4, epi_data_arg.status.at(ind_now_arg.at(i)) + 1); //increment by +1 up until 4 (i.e. R)
	}

}

//
void epi_functions::func_time_update(mov_struct& mov_data_arg, epi_struct_sim& epi_data_arg, vector<int>& once_infectious_arg, vector<int>& uninfected_arg, vector<int>& infectious_arg, vector <int>& infected_source_arg, vector<int> ind_now_arg, double t_now_arg, const vector<double>& norm_const_arg, nt_struct_sim& nt_data_arg, vector<int>& con_seq, rng_type & rng_arg) {


	int num_now = int(ind_now_arg.size());
	int num_infectious = int(infectious_arg.size());

	// const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
	// gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
	// gsl_rng_set (r_c,1000*num_infectious*ind_now_arg.at(0)); // set a seed


	ofstream myfile; // this object can be recycled after myfile.close()
	//myfile.open((string(PATH2)+string("num_now.txt")).c_str(),ios::app);
	//myfile << num_now << endl;
	//myfile.close();


	for (int i = 0; i <= (num_now - 1); i++) {

		switch (epi_data_arg.status.at(ind_now_arg.at(i))) { //status already updated, so this is what ind_now has changed into

		case 2: {
			epi_data_arg.t_e.at(ind_now_arg.at(i)) = t_now_arg;
			epi_data_arg.lat_period.at(ind_now_arg.at(i)) = func_latent_ran(rng_arg, a_Cepi, b_Cepi);
			epi_data_arg.inf_period.at(ind_now_arg.at(i)) = rweibull_sim(c_Cepi, d_Cepi, rng_arg);
			uninfected_arg.erase(find(uninfected_arg.begin(), uninfected_arg.end(), ind_now_arg.at(i)));

			//-------

			nt_data_arg.t_last.at(ind_now_arg.at(i)) = t_now_arg;
			//nt_data_arg.t_sample.at(ind_now_arg.at(i)) = min(t_max_Cepi, t_now_arg +   sample_range_Cepi*gsl_rng_uniform(r_c)); // for infected randomly assign a sampling time forward (randomly replace with distribution sample_delay_mu, sample_delay_var)
			nt_data_arg.t_sample.at(ind_now_arg.at(i)) = min(t_max_Cepi, t_now_arg + sample_range_Cepi * runif_sim(0.0, 1.0, rng_arg)); // for infected randomly assign a sampling time forward (randomly replace with distribution sample_delay_mu, sample_delay_var)


		//--------

			switch (int(num_infectious >= 1)) {

			case 1: {
				vector<double> ic(num_infectious + 1);  //one infectious component to add for: alpha, and betaij per infective
				ic.at(0) = alpha_Cepi;
				for (int j = 0; j <= (num_infectious - 1); j++) {
					if (opt_betaij_sim == 0) {
						if (norm_const_arg.at(infectious_arg.at(j)) > 0) {
							ic.at(j + 1) = beta_Cepi * distance_mat_Cepi[ind_now_arg.at(i)][infectious_arg.at(j)] / norm_const_arg.at(infectious_arg.at(j));

						}
					}
					if (opt_betaij_sim == 1) {
						if (norm_const_arg.at(infectious_arg.at(j)) > 0) {
							ic.at(j + 1) = beta_Cepi * beta_ij_mat_Cepi[infectious_arg.at(j)][ind_now_arg.at(i)] * distance_mat_Cepi[ind_now_arg.at(i)][infectious_arg.at(j)] / norm_const_arg.at(infectious_arg.at(j));

						}
					}
					////////////////////////////////////////////////////////////////////////////////////////////////////////
					if (opt_mov_sim == 0) {
						ic.at(j + 1) = ic.at(j + 1);
					}

					if (opt_mov_sim == 1) {
						ic.at(j + 1) = ic.at(j + 1) + beta_m_Cepi * func_mov_cnt(infectious_arg.at(j), ind_now_arg.at(i), mov_data_arg, epi_data_arg.t_e, epi_data_arg.t_i, epi_data_arg.t_r, t_now_arg);
					}
					////////////////////////////////////////////////////////////////////////////////////////////////////////

				}

				//double *P=&ic.at(0); // convert vector to array
				//for (int i = 0; i<int(sizeof(P) / sizeof(P[0])); i++) { if (P[i]<0) { P[i] = 0; } }
				//gsl_ran_discrete_t * g = gsl_ran_discrete_preproc ((int)ic.size(),P);
				//int link= gsl_ran_discrete (r_c, g); // the actual link
				//gsl_ran_discrete_free (g);
				for (int i = 0; i < (int)(ic.size()); i++) { if (ic.at(i) < 0) { ic.at(i) = 0; } }
				int link = edf_sample_sim(ic, rng_arg);

				switch (int(link >= 1)) {
				case 1: {
					int k_source = infected_source_arg.at(ind_now_arg.at(i)) = infectious_arg.at(link - 1); //minus 1 because first element was alpha

					//myfile.open((string(PATH2)+string("t_e_subject==t_s_source.txt")).c_str(),ios::app);
					//myfile <<nt_data_arg.t_sample.at(k_source) <<"," << t_now_arg << endl;
					//if (t_now_arg==nt_data_arg.t_sample.at(k_source)) myfile <<k_source << "," << ind_now_arg.at(i) << endl;
					//myfile.close();

				//---

					switch ((nt_data_arg.t_sample.at(k_source) > nt_data_arg.t_last.at(k_source)) & (nt_data_arg.t_sample.at(k_source) < t_now_arg)) {
					case 1: {
						epi_functions::seq_update_source(nt_data_arg, k_source, rng_arg); // sample the sequence of infectious at sampling time
						nt_data_arg.t_nt[k_source].push_back(nt_data_arg.t_sample.at(k_source));
						nt_data_arg.ind_sample.at(k_source) = 1; // 1 = sampled
						break;
					}
					case 0: {
						//nt_data_arg.t_last.at(k_source) = t_now_arg; // infecting event is recorded in t_last
						break;
					}
					}
					//---
					epi_functions::seq_update_pair(nt_data_arg, k_source, ind_now_arg.at(i), t_now_arg, rng_arg);

					nt_data_arg.t_nt[ind_now_arg.at(i)].push_back(t_now_arg);
					nt_data_arg.t_nt[k_source].push_back(t_now_arg);

					nt_data_arg.t_last.at(k_source) = t_now_arg; // infecting event is recorded in t_last

	// 				nt_data_arg.current_size.at(ind_now_arg.at(i)) = nt_data_arg.current_size.at(ind_now_arg.at(i))  + 1;
	// 				nt_data_arg.current_size.at(k_source) = nt_data_arg.current_size.at(k_source)  + 1;

					break;
				}
				case 0: {
					infected_source_arg.at(ind_now_arg.at(i)) = 9999; // 9999 indicates a background infection
					nt_data_arg.t_nt[ind_now_arg.at(i)].push_back(t_now_arg);

					// 				for (int j=0;j<=(n_base_Cepi-1);j++){
					// 				nt_data_arg.nt[ind_now_arg.at(i)].push_back(gsl_rng_uniform_int(r_c, 4) +1 );//  first part generates r.v. uniformly on [0,4-1]
					// 				}


					for (int j = 0; j <= (n_base_Cepi - 1); j++) {

						//int ber_trial =  gsl_ran_bernoulli (r_c,p_ber_Cepi); // return 1 if a change is to happen
						int ber_trial = rbern_sim(p_ber_Cepi, rng_arg); // return 1 if a change is to happen
						int base_proposed = 0;

						// 				ofstream myfile; // this object can be recycled after myfile.close()
						//
						// 				myfile.open((string(PATH2)+string("22.txt")).c_str(),ios::app);
						// 				myfile <<ber_trial<< endl;
						// 				myfile.close();

						switch (int(ber_trial == 1)) {
						case 1: { // randomly choose one among other 3
							switch (con_seq.at(j)) {
							case 1: {
								int type = runif_int_sim(0, 2, rng_arg);
								switch (type) {
								case 0: {
									base_proposed = 2;
									break;
								}
								case 1: {
									base_proposed = 3;
									break;
								}
								case 2: {
									base_proposed = 4;
									break;
								}
								}
								break;
							}
							case 2: {
								int type = runif_int_sim(0, 2, rng_arg);

								switch (type) {
								case 0: {
									base_proposed = 1;
									break;
								}
								case 1: {
									base_proposed = 3;
									break;
								}
								case 2: {
									base_proposed = 4;
									break;
								}
								}
								break;
							}
							case 3: {
								int type = runif_int_sim(0, 2, rng_arg);
								switch (type) {
								case 0: {
									base_proposed = 1;
									break;
								}
								case 1: {
									base_proposed = 2;
									break;
								}
								case 2: {
									base_proposed = 4;
									break;
								}
								}
								break;
							}
							case 4: {
								int type = runif_int_sim(0, 2, rng_arg);
								switch (type) {
								case 0: {
									base_proposed = 1;
									break;
								}
								case 1: {
									base_proposed = 2;
									break;
								}
								case 2: {
									base_proposed = 3;
									break;
								}
								}
								break;
							}
							}
							//---

	// 						ofstream myfile; // this object can be recycled after myfile.close()
	//
	// 						myfile.open((string(PATH2)+string("00.txt")).c_str(),ios::app);
	// 						myfile <<p_ber_Cepi<< endl;
	// 						myfile.close();
	//
	// 						myfile.open((string(PATH2)+string("11.txt")).c_str(),ios::app);
	// 						myfile <<con_seq.at(j) <<","<< base_proposed << endl;
	// 						myfile.close();

	//

							nt_data_arg.nt[ind_now_arg.at(i)].push_back(base_proposed); //

							break;
						}
						case 0: {
							nt_data_arg.nt[ind_now_arg.at(i)].push_back(con_seq.at(j)); // same as consensus seq
							break;
						}
						}
					}

					//---

					nt_data_arg.current_size.at(ind_now_arg.at(i)) = nt_data_arg.current_size.at(ind_now_arg.at(i)) + 1;

					break;
				}
				}

				// ofstream myfile; // this object can be recycled after myfile.close()
				// myfile.open((string(PATH2)+string("link.txt")).c_str(),ios::app);
				// myfile << link << endl;
				// myfile.close();

				break;
			}
			case 0: {
				infected_source_arg.at(ind_now_arg.at(i)) = 9999; // 9999 indicates a background infection
				nt_data_arg.t_nt[ind_now_arg.at(i)].push_back(t_now_arg);

				// 			for (int j=0;j<=(n_base_Cepi-1);j++){
				// 			nt_data_arg.nt[ind_now_arg.at(i)].push_back(gsl_rng_uniform_int(r_c, 4) +1 );//  first part generates r.v. uniformly on [0,4-1]
				// 			}

				for (int j = 0; j <= (n_base_Cepi - 1); j++) {

					//int ber_trial =  gsl_ran_bernoulli (r_c,p_ber_Cepi); // return 1 if a change is to happen
					int ber_trial = rbern_sim(p_ber_Cepi, rng_arg); // return 1 if a change is to happen
					int base_proposed = 0;

					// 				ofstream myfile; // this object can be recycled after myfile.close()
					//
					// 				myfile.open((string(PATH2)+string("22.txt")).c_str(),ios::app);
					// 				myfile <<ber_trial<< endl;
					// 				myfile.close();

					switch (int(ber_trial == 1)) {
					case 1: { // randomly choose one among other 3
						switch (con_seq.at(j)) {
						case 1: {
							int type = runif_int_sim(0, 2, rng_arg);
							switch (type) {
							case 0: {
								base_proposed = 2;
								break;
							}
							case 1: {
								base_proposed = 3;
								break;
							}
							case 2: {
								base_proposed = 4;
								break;
							}
							}
							break;
						}
						case 2: {
							int type = runif_int_sim(0, 2, rng_arg);

							switch (type) {
							case 0: {
								base_proposed = 1;
								break;
							}
							case 1: {
								base_proposed = 3;
								break;
							}
							case 2: {
								base_proposed = 4;
								break;
							}
							}
							break;
						}
						case 3: {
							int type = runif_int_sim(0, 2, rng_arg);
							switch (type) {
							case 0: {
								base_proposed = 1;
								break;
							}
							case 1: {
								base_proposed = 2;
								break;
							}
							case 2: {
								base_proposed = 4;
								break;
							}
							}
							break;
						}
						case 4: {
							int type = runif_int_sim(0, 2, rng_arg);
							switch (type) {
							case 0: {
								base_proposed = 1;
								break;
							}
							case 1: {
								base_proposed = 2;
								break;
							}
							case 2: {
								base_proposed = 3;
								break;
							}
							}
							break;
						}
						}
						//---

// 						ofstream myfile; // this object can be recycled after myfile.close()
//
// 						myfile.open((string(PATH2)+string("00.txt")).c_str(),ios::app);
// 						myfile <<p_ber_Cepi<< endl;
// 						myfile.close();
//
// 						myfile.open((string(PATH2)+string("11.txt")).c_str(),ios::app);
// 						myfile <<con_seq.at(j) <<","<< base_proposed << endl;
// 						myfile.close();

//

						nt_data_arg.nt[ind_now_arg.at(i)].push_back(base_proposed); //

						break;
					}
					case 0: {
						nt_data_arg.nt[ind_now_arg.at(i)].push_back(con_seq.at(j)); // same as consensus seq
						break;
					}
					}
				}

				//---

				nt_data_arg.current_size.at(ind_now_arg.at(i)) = nt_data_arg.current_size.at(ind_now_arg.at(i)) + 1;

				break;
			}
			}
			//--------


			break;
		}

		case 3: {
			epi_data_arg.t_i.at(ind_now_arg.at(i)) = t_now_arg;

			once_infectious_arg.push_back(ind_now_arg.at(i));
			infectious_arg.push_back(ind_now_arg.at(i));

			//--
			// 		switch((nt_data_arg.t_sample.at(ind_now_arg.at(i))>nt_data_arg.t_last.at(ind_now_arg.at(i))) & (nt_data_arg.t_sample.at(ind_now_arg.at(i))<t_now_arg)){
			// 		case 1:{
			// 		epi_functions::seq_update_source(nt_data_arg, ind_now_arg.at(i)); // sample the sequence of just-become-infectious at sampling time
			// 		nt_data_arg.t_nt[ind_now_arg.at(i)].push_back(nt_data_arg.t_sample.at(ind_now_arg.at(i)));
			// 		nt_data_arg.ind_sample.at(ind_now_arg.at(i)) = 1; // 1 = sampled
			// 		break;
			// 		}
			// 		case 0:{
			// 		break;
			// 		}
			// 		}
			//
			// 	nt_data_arg.t_last.at(ind_now_arg.at(i)) = t_now_arg;

			//--
			break;
		}

		case 4: {
			epi_data_arg.t_r.at(ind_now_arg.at(i)) = t_now_arg;

			infectious_arg.erase(find(infectious_arg.begin(), infectious_arg.end(), ind_now_arg.at(i)));

			//--
					// for recoveries, simulate final sampled sequence
			switch ((nt_data_arg.t_sample.at(ind_now_arg.at(i)) > nt_data_arg.t_last.at(ind_now_arg.at(i))) & (nt_data_arg.t_sample.at(ind_now_arg.at(i)) < t_now_arg)) {
			case 1: {
				epi_functions::seq_update_source(nt_data_arg, ind_now_arg.at(i), rng_arg); // sample the sequence of just-become-infectious at sampling time
				nt_data_arg.t_nt[ind_now_arg.at(i)].push_back(nt_data_arg.t_sample.at(ind_now_arg.at(i)));
				nt_data_arg.ind_sample.at(ind_now_arg.at(i)) = 1; // 1 = sampled
				break;
			}
			case 0: {
				break;
			}
			}
			//
			// 	nt_data_arg.t_last.at(ind_now_arg.at(i)) = t_now_arg;

			//--

			break;
		}

		case 1: {
			//do nothing
			break;
		}

		}

	}

}

//

void epi_functions::seq_update_source(nt_struct_sim& nt_data_arg, int  k_source_arg, rng_type & rng_arg) { // sample sequences of infectious at sampling time

	int current_size_source = nt_data_arg.current_size.at(k_source_arg);
	double dt = nt_data_arg.t_sample.at(k_source_arg) - nt_data_arg.t_nt[k_source_arg][current_size_source - 1];

	double p_1, p_2, p_3, p_4;
	p_1 = p_2 = p_3 = p_4 = 0;

	if (opt_k80_sim == 0) {
		p_1 = 0.25 + 0.25*exp(-4.0*mu_2_Cepi*dt) + 0.5*exp(-2.0*(mu_1_Cepi + mu_2_Cepi)*dt); // pr of a base not changing
		p_2 = 0.25 + 0.25*exp(-4.0*mu_2_Cepi*dt) - 0.5*exp(-2.0*(mu_1_Cepi + mu_2_Cepi)*dt); // pr of a transition of a base
		p_3 = 1.0*(0.25 - 0.25*exp(-4.0*mu_2_Cepi*dt));  // pr of a transversion (two possible events)
		p_4 = p_3;
		//p_1 = 1.0- p_2 - 2.0*p_3;
	}

	if (opt_k80_sim == 1) {
		//K80: mu1 = alpha; mu2 = beta
		p_2 = 0.25 - 0.5*exp(-4.0*(mu_1_Cepi + mu_2_Cepi)*dt) + 0.25*exp(-8.0*mu_2_Cepi*dt); // P = pts
		p_3 = 0.25 - 0.25*exp(-8.0*mu_2_Cepi*dt);  // Q = ptv (2 options)
		p_4 = p_3;
		p_1 = 1 - p_2 - p_3 - p_4; // R = no change
							 //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
	}



	//double P[4] = { p_1, p_2, p_3, p_4 };
	vector<double> P = { p_1, p_2, p_3, p_4 };
	for (int i = 0; i < (int)(P.size()); i++) { if (P.at(i) < 0) { P.at(i) = 0; } }

	//const gsl_rng_type* T_c = gsl_rng_ranlux;  // T is pointer points to the type of generator
	//gsl_rng *r_c = gsl_rng_alloc(T_c); // r is pointer points to an object with Type T
									   //gsl_rng_set (r_c,-1000*k_source_arg); // set a seed
	//for (int i = 0; i<int(sizeof(P) / sizeof(P[0])); i++) { if (P[i]<0) { P[i] = 0; } }
	//gsl_ran_discrete_t * g = gsl_ran_discrete_preproc(sizeof(P) / sizeof(P[0]), P);

	int count_1, count_2, count_3, count_4;

	count_1 = count_2 = count_3 = count_4 = 0;

	// ofstream myfile; // this object can be recycled after myfile.close()
	//
	// myfile.open((string(PATH2)+string("nt_sourcetxt")).c_str(),ios::app);
	// myfile << nt_data_arg.nt[k_source_arg][0][0]<< endl;
	// myfile.close();
	//
	// myfile.open((string(PATH2)+string("k_source.txt")).c_str(),ios::app);
	// myfile <<k_source_arg<< endl;
	// myfile.close();
	//
	// myfile.open((string(PATH2)+string("current_size_source.txt")).c_str(),ios::app);
	// myfile <<current_size_source<< endl;
	// myfile.close();

	//gsl_rng_set(r_c, -1000 * k_source_arg*(current_size_source + 1)); // set a seed


	for (int j = 0; j <= (n_base_Cepi - 1); j++) {


		//int type = gsl_ran_discrete(r_c, g) + 1;
		int type = edf_sample_sim(P, rng_arg) + 1;

		//ofstream myfile; // this object can be recycled after myfile.close()
		// 	myfile.open((string(PATH2)+string("type.txt")).c_str(),ios::app);
		// 	myfile << type << endl;
		// 	myfile.close();

		switch (nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]) {

		case 1: { // was an A
			switch (type) {
			case 1: {  //remains unchanged (A)
				//nt_data_arg.nt[k_source_arg][current_size_source].push_back(nt_data_arg.nt[k_source_arg][current_size_source-1][j]);
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;

				break;
			}
			case 2: {  // transition change (to a G)
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;


				break;
			}
			case 3: {  //transversion change (to a C)

				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;



				//		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion

				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(3);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(4);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {  //transversion change (to a T)
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;

				break;
			}
			}
			break;
		}

		case 2: { // a G
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;

				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;

				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;



				//		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion

				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(3);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(4);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;


				break;
			}
			}
			break;
		}

		case 3: { // a T
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;


				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;


				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;



				//		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion

				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(1);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(2);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;


				break;
			}
			}
			break;
		}

		case 4: { // a C
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;


				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;

				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;

				/*		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion

				switch(type_trans){
				case 0:{
				nt_data_arg.nt[k_source_arg].push_back(1);
				break;
				}
				case 1:{
				nt_data_arg.nt[k_source_arg].push_back(2);
				break;
				}
				}*/

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;


				break;
			}
			}
			break;
		}

		}

	}

	//gsl_ran_discrete_free(g);

	nt_data_arg.current_size.at(k_source_arg) = current_size_source + 1;

	//nt_data_arg.nt[k_arg].push_back(nt_data_arg.nt[k_source_arg][current_size_source]);

	ofstream myfile; // this object can be recycled after myfile.close()
	//myfile.open((string(PATH2) + string("P_source.txt")).c_str(), ios::app);
	//myfile << P[0] << "," << P[1] << "," << P[2] << "," << P[3] << "," << P[0] + P[1] + P[2] + P[3] << endl;
	//myfile.close();

	//myfile.open((string(PATH2) + string("count_source.txt")).c_str(), ios::app);
	//myfile << count_1 << "," << count_2 << "," << count_3 << "," << count_4 << "," << count_1 + count_2 + count_3 + count_4 << endl;
	//myfile.close();

	//unsigned int M[4] = { 0,0,0,0 };
	//gsl_ran_multinomial(r_c, 4, n_base_Cepi, P, M);
	vector<int> M = { 0,0,0,0 };
	int tmp;
	for (int i = 0; i < n_base_Cepi; i++) {
		tmp = edf_sample_sim(P, rng_arg);
		M.at(tmp) = M.at(tmp) + 1;
	}

	//myfile.open((string(PATH2) + string("count_multinomial_source.txt")).c_str(), ios::app);
	//myfile << M[0] << "," << M[1] << "," << M[2] << "," << M[3] << "," <<
	//		M[0] + M[1] + M[2] + M[3] << endl;
	//myfile.close();

	nt_data_arg.total_count_1 = nt_data_arg.total_count_1 + count_1;
	nt_data_arg.total_count_2 = nt_data_arg.total_count_2 + count_2;
	nt_data_arg.total_count_3 = nt_data_arg.total_count_3 + count_3 + count_4;


}
//

void epi_functions::seq_update_pair(nt_struct_sim& nt_data_arg, int  k_source_arg, int k_arg, double t_now_arg, rng_type & rng_arg) { // update sequences of infectious-infected pair

	int current_size_source = nt_data_arg.current_size.at(k_source_arg);
	double t_null = nt_data_arg.t_nt[k_source_arg][current_size_source - 1];

	double dt = t_now_arg - t_null;

	double p_1, p_2, p_3, p_4;
	p_1 = p_2 = p_3 = p_4 = 0;

	if (opt_k80_sim == 0) {
		p_1 = 0.25 + 0.25*exp(-4.0*mu_2_Cepi*dt) + 0.5*exp(-2.0*(mu_1_Cepi + mu_2_Cepi)*dt); // pr of a base not changing
		p_2 = 0.25 + 0.25*exp(-4.0*mu_2_Cepi*dt) - 0.5*exp(-2.0*(mu_1_Cepi + mu_2_Cepi)*dt); // pr of a transition of a base
		p_3 = 1.0*(0.25 - 0.25*exp(-4.0*mu_2_Cepi*dt));  // pr of a transversion (two possible events)
		p_4 = p_3;
		//p_1 = 1.0- p_2 - 2.0*p_3;
	}

	if (opt_k80_sim == 1) {
		//K80: mu1 = alpha; mu2 = beta
		p_2 = 0.25 - 0.5*exp(-4.0*(mu_1_Cepi + mu_2_Cepi)*dt) + 0.25*exp(-8.0*mu_2_Cepi*dt); // P = pts
		p_3 = 0.25 - 0.25*exp(-8.0*mu_2_Cepi*dt);  // Q = ptv (2 options)
		p_4 = p_3;
		p_1 = 1 - p_2 - p_3 - p_4; // R = no change
								   //p_1 = 0.25 + 0.5*exp(-4.0*(mu_1 + mu_2)*abs_dt) + 0.25*exp(-8.0*mu_2*abs_dt); // R = no change
	}


	//double P[4] = {p_1, p_2, p_3,p_4};
	vector<double> P = { p_1, p_2, p_3, p_4 };
	for (int i = 0; i < (int)(P.size()); i++) { if (P.at(i) < 0) { P.at(i) = 0; } }

	//const gsl_rng_type* T_c= gsl_rng_ranlux;  // T is pointer points to the type of generator
	//gsl_rng *r_c = gsl_rng_alloc (T_c); // r is pointer points to an object with Type T
	//for (int i = 0; i<int(sizeof(P) / sizeof(P[0])); i++) { if (P[i]<0) { P[i] = 0; } }
	//gsl_ran_discrete_t * g = gsl_ran_discrete_preproc (sizeof(P)/sizeof(P[0]),P);

	int count_1, count_2, count_3, count_4;

	count_1 = count_2 = count_3 = count_4 = 0;

	// gsl_rng_set (r_c,-1000*k_arg); // set a seed

	// ofstream myfile; // this object can be recycled after myfile.close()
	//
	// myfile.open((string(PATH2)+string("nt_sourcetxt")).c_str(),ios::app);
	// myfile << nt_data_arg.nt[k_source_arg][0][0]<< endl;
	// myfile.close();
	//
	// myfile.open((string(PATH2)+string("k_source.txt")).c_str(),ios::app);
	// myfile <<k_source_arg<< endl;
	// myfile.close();
	//
	// myfile.open((string(PATH2)+string("current_size_source.txt")).c_str(),ios::app);
	// myfile <<current_size_source<< endl;
	// myfile.close();

	// ofstream myfile; // this object can be recycled after myfile.close()
	// myfile.open((string(PATH2)+string("P.txt")).c_str(),ios::app);
	// myfile << P[0] <<"," << P[1]  << "," << P[2] <<"," << P[3] << endl;
	// myfile.close();

	//gsl_rng_set (r_c,-1000*k_arg*k_source_arg*(current_size_source+1)); // set a seed

	//simulate sequence for infectious (k_source_arg) at time of infection of infected (k_arg)
	for (int j = 0; j <= (n_base_Cepi - 1); j++) {


		//int type= gsl_ran_discrete (r_c, g) + 1;
		int type = edf_sample_sim(P, rng_arg) + 1;

		// 	ofstream myfile; // this object can be recycled after myfile.close()
		// 	myfile.open((string(PATH2)+string("type.txt")).c_str(),ios::app);
		// 	myfile << type << endl;
		// 	myfile.close();

		switch (nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]) {

		case 1: { // an A
			switch (type) {
			case 1: {
				//nt_data_arg.nt[k_source_arg][current_size_source].push_back(nt_data_arg.nt[k_source_arg][current_size_source-1][j]);
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;
				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;


				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;


				// 		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion
				//
				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(3);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(4);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;

				break;
			}
			}
			break;
		}

		case 2: { // a G
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;

				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;

				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;


				// 		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion
				//
				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(3);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(4);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;

				break;
			}
			}
			break;
		}

		case 3: { // a T
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;

				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(4);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;


				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;



				// 		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion
				//
				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(1);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(2);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;


				break;
			}
			}
			break;
		}

		case 4: { // a C
			switch (type) {
			case 1: {
				nt_data_arg.nt[k_source_arg].push_back(nt_data_arg.nt[k_source_arg][(current_size_source - 1)*n_base_Cepi + j]);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_1);
				count_1 = count_1 + 1;


				break;
			}
			case 2: {
				nt_data_arg.nt[k_source_arg].push_back(3);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_2);
				count_2 = count_2 + 1;


				break;
			}
			case 3: {

				nt_data_arg.nt[k_source_arg].push_back(1);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_3);
				count_3 = count_3 + 1;



				// 		int type_trans = gsl_rng_uniform_int (r_c, 2);//  uniformly drawn from [0,2) to determine the exact type of transversion
				//
				// 			switch(type_trans){
				// 				case 0:{
				// 					nt_data_arg.nt[k_source_arg].push_back(1);
				// 				break;
				// 				}
				// 				case 1:{
				// 					nt_data_arg.nt[k_source_arg].push_back(2);
				// 				break;
				// 				}
				// 			}

				break;
			}
			case 4: {
				nt_data_arg.nt[k_source_arg].push_back(2);

				nt_data_arg.log_f_S = nt_data_arg.log_f_S + log(p_4);
				count_4 = count_4 + 1;


				break;
			}
			}
			break;
		}

		}

	}

	//gsl_ran_discrete_free (g);

	//simulate sequence for infected (k_arg)
	for (int j = 0; j <= (n_base_Cepi - 1); j++) {
		nt_data_arg.nt[k_arg].push_back(nt_data_arg.nt[k_source_arg][current_size_source*n_base_Cepi + j]);
	}

	nt_data_arg.current_size.at(k_arg) = nt_data_arg.current_size.at(k_arg) + 1;
	nt_data_arg.current_size.at(k_source_arg) = current_size_source + 1;

	ofstream myfile; // this object can be recycled after myfile.close()
	//myfile.open((string(PATH2)+string("P_pair.txt")).c_str(),ios::app);
	//myfile << P[0] <<"," << P[1]  << "," << P[2] <<"," << P[3] << "," << P[0]+P[1]+P[2]+P[3]<< endl;
	//myfile.close();

	//myfile.open((string(PATH2)+string("count_pair.txt")).c_str(),ios::app);
	//myfile << count_1 <<"," << count_2  << "," << count_3 <<"," << count_4 << "," << count_1 + count_2 + count_3 + count_4 << endl;
	//myfile.close();

	//unsigned int M[4] = {0,0,0,0};
	//gsl_ran_multinomial(r_c, 4, n_base_Cepi, P, M);
	vector<int> M = { 0,0,0,0 };
	int tmp;
	for (int i = 0; i < n_base_Cepi; i++) {
		tmp = edf_sample_sim(P, rng_arg);
		M.at(tmp) = M.at(tmp) + 1;
	}

	//myfile.open((string(PATH2)+string("count_multinomial_pair.txt")).c_str(),ios::app);
	//myfile << M[0]<<"," << M[1]  << "," << M[2] <<"," << M[3] << "," <<
	//M[0] + M[1] + M[2] + M[3] << endl;
	//myfile.close();

	nt_data_arg.total_count_1 = nt_data_arg.total_count_1 + count_1;
	nt_data_arg.total_count_2 = nt_data_arg.total_count_2 + count_2;
	nt_data_arg.total_count_3 = nt_data_arg.total_count_3 + count_3 + count_4;

}

/*-------------------------------------------------------*/
///// round a double to y decimal places
double roundx_sim(double x, int y) {
	x = round(x * pow(10, y)) / pow(10, y);
	return x;
}

/*-------------------------------------------------------*/


/*----------------------------------------------------------*/
/*- Main call of function to export & inputs from R --------*/
/*----------------------------------------------------------*/

// [[Rcpp::export]]
Rcpp::List infer_cpp() {
  /*----------------------------*/
  /*- int main -----------------*/
  /*----------------------------*/
	ifstream myfile_in; //recycled instream (not in loops)
	ofstream myfile_out; //recycled outstream (not in loops)

	para_priors_etc para_priorsetc;
	IO_parapriorsetc(para_priorsetc);
	ind_n_base_part = para_priorsetc.ind_n_base_part;// =1 if the seq data is partial
	n_base_part = para_priorsetc.n_base_part; // the partial length used if ind_n_base_part =1

	para_key_init para_init;
	IO_parakeyinit(para_init);

	para_scaling_factors para_scalingfactors;
	IO_parascalingfactors(para_scalingfactors);

	para_aux para_other;
	IO_para_aux(para_other);  //Importing parameters
	opt_latgamma = para_other.opt_latgamma;
	opt_k80 = para_other.opt_k80;
	opt_betaij = para_other.opt_betaij;
	opt_ti_update = para_other.opt_ti_update;
	opt_mov = para_other.opt_mov;
	debug = para_other.debug;

	epi_struct epi_final;
	nt_struct nt_data;
	moves_struct moves;
	vector<int> index;
	vector<int> seeds;
	vector < vector<double> > coordinate(NLIMIT, vector<double>(2));
	vector<int> con_seq, con_seq_estm;
	IO_data(para_other, coordinate, epi_final, nt_data, index, con_seq_estm, seeds, moves); //Importing  data


	// set a universal seed
	//const gsl_rng_type* T_c_unvs = gsl_rng_default;  // T is pointer points to the type of generator
	//gsl_rng *r_c_unvs = gsl_rng_alloc(T_c_unvs); // r is pointer points to an object with Type T


	//sf edit
	//gsl_rng_set(r_c_unvs, 1); // set a universal seed
	int seed = 1234;

	if (para_other.np == 1) {	// on single computer
		//gsl_rng_set(r_c_unvs, seeds.at(0)); // set a universal seed
		seed = seeds.at(0); //set a universal seed
	}

	if (para_other.np > 1) {	// on cluster
		char *env_tid = getenv("SLURM_ARRAY_TASK_ID");
		char tid_ch = *env_tid;

		//int tid_l = strlen(&tid_ch);
		//cout << tid_l << endl;
		//for (int i = 0; i < tid_l; i++) {
		int tid = atoi(&tid_ch); //just takes first character, &tid_ch[0], need whole string
		//gsl_rng_set(r_c_unvs, seeds.at(tid-1)); // set a universal seed
		seed = seeds.at(tid - 1); //set a universal seed
		cout << "tid: " << tid << "  seed: " << seeds.at(tid - 1) << endl;
	}

	rng_type rng(seed); //set a universal seed


	vec2int sample_data; // 2-d vector contains the sampled sequences; non-sampled premises would have unexpected values
	sample_data.resize(NLIMIT);
	for (int i = 0; i <= (NLIMIT - 1); i++) {
		sample_data[i].reserve(SEQLLIMIT);
	}



	string line, field;
	int line_count = 0;

	myfile_in.open((string(PATH1) + string("seq.csv")).c_str(), ios::in);

	while (getline(myfile_in, line)) {
		stringstream ss(line);
		//field_count=0;

		while (getline(ss, field, ',')) {
			stringstream fs(field);
			int t;
			fs >> t;
			sample_data[line_count].push_back(t);

			//field_count = field_count + 1;
		}
		line_count = line_count + 1;

	}// end while for getline
	myfile_in.close();


	// upload the true infected sources vector
	vector<int> atab_from; // vector to hold true sources (from accuracy table comparison file)
	atab_from.resize(NLIMIT);
	myfile_in.open((string(PATH1) + string("atab_from.csv")).c_str(), ios::in);
	line_count = 0;
	while (getline(myfile_in, line)) {

		stringstream ss(line);

		while (getline(ss, field)) {
			stringstream fs(field);
			int t;
			fs >> t;
			atab_from.at(line_count) = t;
		}

		line_count = line_count + 1;
	}
	myfile_in.close();

	if (debug == 1) {
		myfile_out.open((string(PATH2) + string("atab_from.csv")).c_str(), ios::app);
		for (int i = 0; i <= (para_other.n - 1); i++) {
			myfile_out << atab_from.at(i) << endl;
		}
		myfile_out.close();
	}


	if (debug == 1) {
		for (int i = 0; i <= (para_other.n - 1); i++) {

			myfile_out.open((string(PATH2) + string("sample_data.csv")).c_str(), ios::app);
			//	if (nt_data.t_sample.at(i)!=para_other.unassigned_time){
			for (int j = 0; j <= ((int)sample_data.at(i).size() - 1); j++) {
				int rem = (j + 1) % para_other.n_base;
				if ((rem != 0) | (j == 0)) myfile_out << sample_data[i][j] << ",";
				if ((rem == 0) & (j != 0)) myfile_out << sample_data[i][j] << " " << endl;
			}
			//	}
			myfile_out.close();
		}
	}

	/*----------------------------*/

	vector<int> xi_U, xi_E, xi_E_minus, xi_I, xi_R, xi_EnI, xi_EnIS, xi_InR; // indices sets indicating the individuals stay in S OR have gone through the other classes (E OR I OR R), and individuals hve gone through E but not I (EnI) and I but not R (InR)

	vector<int> xi_beta_E; // vector contains the individuals with secondary infection

	xi_U.reserve(NLIMIT);
	xi_E.reserve(NLIMIT);
	xi_E_minus.reserve(NLIMIT);
	xi_I.reserve(NLIMIT);
	xi_R.reserve(NLIMIT);
	xi_EnI.reserve(NLIMIT);
	xi_EnIS.reserve(NLIMIT);
	xi_InR.reserve(NLIMIT);
	xi_beta_E.reserve(NLIMIT);


	for (int i = 0; i <= (para_other.n - 1); i++) {
		if (epi_final.t_e.at(i) == para_other.unassigned_time) xi_U.push_back(i);
		if (epi_final.t_e.at(i) != para_other.unassigned_time) xi_E.push_back(i);
		// if (epi_final.t_i.at(i)!=para_other.unassigned_time) xi_I.push_back(i);
		// if (epi_final.t_r.at(i)!=para_other.unassigned_time) xi_R.push_back(i);
		if ((epi_final.t_i.at(i) != para_other.unassigned_time) & (epi_final.t_i.at(i) < epi_final.t_r.at(i))) xi_I.push_back(i); // the seocnd condition handles the situation that i=for a few cases t_r<t_i (ignore their contibution of t_i and t_r)
		if ((epi_final.t_r.at(i) != para_other.unassigned_time) & (epi_final.t_i.at(i) < epi_final.t_r.at(i))) xi_R.push_back(i); // the seocnd condition handles the situation that i=for a few cases t_r<t_i (ignore their contibution of t_i and t_r)

	}

	xi_E_minus = xi_E;
	for (int i = 0; i <= (int)(index.size() - 1); i++) {
		xi_E_minus.erase(find(xi_E_minus.begin(), xi_E_minus.end(), index.at(i)));
	} // E set excluding index

	xi_EnIS = xi_EnI;
	for (int i = 0; i <= (int)(xi_EnI.size() - 1); i++) {
		if (nt_data.t_sample.at(xi_EnI.at(i)) != para_other.unassigned_time) {
			xi_EnIS.erase(find(xi_EnIS.begin(), xi_EnIS.end(), xi_EnI.at(i)));
		}
	} // E set excluding I and sampled


	xi_InR = xi_I;
	for (int i = 0; i <= (int)(xi_R.size() - 1); i++) {
		xi_InR.erase(find(xi_InR.begin(), xi_InR.end(), xi_R.at(i)));
	} // I set excluding R

	for (int i = 0; i <= (int)(para_other.n - 1); i++) {
		if ((epi_final.infected_source.at(i) != 9999) & (epi_final.infected_source.at(i) != -99)) xi_beta_E.push_back(i);
	}



	/*----------------------------*/

	lh_SQUARE lh_square; // the object contains the information of the likelihood contribution of each individual, and it is dynamic and changed during MCMC sampling

	lh_square.f_U.assign(para_other.n, 1.0);  //contribution to likelihood of those uninfected at time t (xi_U), Paper equation (3) line 1, 3rd component
	lh_square.q_T.assign(para_other.n, 0.0);  //contribution to likelihood of survival til tmax for xi_U, Paper equation (3) line 1, 3rd component, using equation (5)
	lh_square.kt_sum_U.assign(para_other.n, 0.0); //for each xi_U = (sum of kernel contributions across all infectious) x delta_t (time exposed to each infectious), part of Paper equation (5) with dt

	lh_square.f_E.assign(para_other.n, 1.0);  //contribution to likelihood of those exposed at time t (xi_E), Paper equation (3) line 1, 1st two components
	lh_square.g_E.assign(para_other.n, 1.0);  //contribution to likelihood of those exposed at time t (xi_E), Paper equation (3) line 4, 1st component
	lh_square.h_E.assign(para_other.n, 1.0);  //contribution to likelihood of those exposed at time t (xi_E), Paper equation (3) line 4, 2nd component
	lh_square.k_sum_E.assign(para_other.n, 0.0); //kernel likelihood from proposed source to each xi_E_minus, if primary case = 0, otherwise beta x kernel, etc, equation (4), no dt
	lh_square.q_E.assign(para_other.n, 0.0);  //contribution to likelihood of survival til time t unexposed for xi_E (minus any primary infections), Paper equation (3) line 1, 2nd component, equation (5)
	lh_square.kt_sum_E.assign(para_other.n, 0.0);  //for each xi_E = (sum of kernel contributions across all infectious prior to exposure time) x delta_t (time exposed to each infectious), part of equation (5) with dt

	lh_square.f_I.assign(para_other.n, 1.0);  //contribution to likelihood of those infectious at time t (xi_I), likelihood of latent period sojourn time

	lh_square.f_R.assign(para_other.n, 1.0);  //contribution to likelihood of those recovered at time t (xi_R), likelihood of infectious period sojourn time

	lh_square.f_EnI.assign(para_other.n, 1.0); //contribution to likelihood of those exposed but not yet infectious at time t (xi_UnI)
	lh_square.f_InR.assign(para_other.n, 1.0); //contribution to likelihood of those infectious but not yet recovered/removed at time t (xi_InR)

	lh_square.log_f_S.assign(para_other.n, 0.0); //contribution to likelihood of the set of sequences of an exposed individual
	lh_square.log_f_Snull.assign(para_other.n, 0.0); //contribution to likelihood of the set of sequences of an exposed individual if as yet none recorded

	lh_square.movest_sum_U.assign(para_other.n, 0); //count of moves to those uninfected at tmax, time dependent
	lh_square.movest_sum_E.assign(para_other.n, 0); //count of moves to those exposed before tmax, time dependent
	lh_square.moves_sum_E.assign(para_other.n, 0);  //count of moves to those exposed before tmax, ignoring time


	/*--------------------------------Start of MCMC sampling------------------------------------------*/

	para_key para_current;
	vector<int> xi_I_current = xi_I;
	vector<int> xi_U_current = xi_U;
	vector<int> xi_E_current = xi_E;
	vector<int> xi_E_minus_current = xi_E_minus;
	vector<int> xi_R_current = xi_R;
	vector<int> xi_EnI_current = xi_EnI;
	vector<int> xi_EnIS_current = xi_EnIS;
	vector <int> xi_InR_current = xi_InR;
	vector<double> t_e_current = epi_final.t_e;
	vector<double>t_i_current = epi_final.t_i;
	vector<double>t_r_current = epi_final.t_r;
	vector<int>index_current = index;
	vector<int> xi_beta_E_current = xi_beta_E;

	//if (debug == 1) {
	//	for (int i = 0; i <= ((int)para_other.n - 1); i++) {
	//		cout << "t_i-e: " << t_i_current.at(i) - t_e_current.at(i) << "t_e: " << t_e_current.at(i) << " t_i: " << t_i_current.at(i) << endl;
	//	}
	//}

	if (debug == 1) {
		ofstream myfile_out_test;
		myfile_out_test.open(string(PATH2) + string("test.txt").c_str(), ios::out);
		myfile_out_test << "xi_U_Clh.size() = " << xi_U_current.size() << endl;
		myfile_out_test << "xi_E_Clh.size() = " << xi_E_current.size() << endl;
		myfile_out_test << "xi_I_Clh.size() = " << xi_I_current.size() << endl;
		myfile_out_test << "xi_R_Clh.size() = " << xi_R_current.size() << endl;
		myfile_out_test << "xi_EnI_Clh.size() = " << xi_EnI_current.size() << endl;
		myfile_out_test << "xi_InR_Clh.size() = " << xi_InR_current.size() << endl;

		myfile_out_test << endl << endl;
		for (int i = 0; i <= ((int)xi_I_current.size() - 1); i++) {
			myfile_out_test << xi_I_current.at(i) << endl;
		}
		myfile_out_test.close();


	}

	vector<int> con_seq_current = con_seq_estm;

	vector<int> infected_source_current = epi_final.infected_source;

	nt_struct nt_data_current;
	nt_data_current.t_sample = nt_data.t_sample;


	vector<double> t_onset = epi_final.t_i;// used as a ref point to sample t_i; assumed to be given,e.g, the onset time  (in simulation, this may be taken to be the true t_i)

	//Note: struct copy is fragile: nt_data_current=nt_data wont work!!//


	lh_SQUARE lh_square_current;
	lh_square_current.f_U.assign(para_other.n, 1.0);
	lh_square_current.q_T.assign(para_other.n, 0.0);
	lh_square_current.kt_sum_U.assign(para_other.n, 0.0);
	lh_square_current.f_E.assign(para_other.n, 1.0);
	lh_square_current.g_E.assign(para_other.n, 1.0);
	lh_square_current.h_E.assign(para_other.n, 1.0);
	lh_square_current.k_sum_E.assign(para_other.n, 0.0);
	lh_square_current.q_E.assign(para_other.n, 0.0);
	lh_square_current.kt_sum_E.assign(para_other.n, 0.0);
	lh_square_current.f_I.assign(para_other.n, 1.0);
	lh_square_current.f_R.assign(para_other.n, 1.0);
	lh_square_current.f_EnI.assign(para_other.n, 1.0);
	lh_square_current.f_InR.assign(para_other.n, 1.0);
	lh_square_current.log_f_S.assign(para_other.n, 0.0);
	lh_square_current.log_f_Snull.assign(para_other.n, 0.0);
	lh_square_current.movest_sum_U.assign(para_other.n, 0.0);
	lh_square_current.movest_sum_E.assign(para_other.n, 0.0);
	lh_square_current.moves_sum_E.assign(para_other.n, 0.0);

	vector < vector<double> > kernel_mat_current(NLIMIT, vector<double>(NLIMIT)); // a dynamic matrix contain the "kernel distance"
	vector <double> norm_const_current(NLIMIT);
	vector < vector<double> > delta_mat_current(NLIMIT, vector<double>(NLIMIT)); // a dynamic matrix contain the "exposure time delta, dt between i and j"
	vector < vector<double> > beta_ij_mat_current(NLIMIT, vector<double>(NLIMIT)); // a dynamic matrix containing the "covariate pattern" effect
	vector < vector<double> > delta_mat_mov_current(NLIMIT, vector<double>(NLIMIT)); // a dynamic matrix contain the movt "exposure time delta, dt between i and j"
	vector<double> beta_ij_inf_current(NLIMIT, 1.0); // the "covariate pattern" effect on infectivity, normalised
	vector<double> beta_ij_susc_current(NLIMIT, 1.0); // the "covariate pattern" effect on susceptibility, normalised

	FUNC func_mcmc;
	initialize_mcmc(para_init, para_current, para_other, para_priorsetc, xi_I_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_R_current, xi_EnI_current, xi_EnIS_current, xi_InR_current, t_e_current, t_i_current, t_r_current, index_current, infected_source_current, kernel_mat_current, norm_const_current, sample_data, t_onset, nt_data_current, con_seq_current, beta_ij_mat_current); // initialze the parameters/unobserved data for mcmc
	//if (debug == 1) {
	//	for (int i = 0; i <= ((int)para_other.n - 1); i++) {
	//		cout << "t_i-e: " << t_i_current.at(i) - t_e_current.at(i) << "t_e: " << t_e_current.at(i) << " t_i: " << t_i_current.at(i) << endl;
	//	}
	//}

	func_mcmc.set_para(para_current, para_other, coordinate, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_R_current, xi_EnI_current, xi_InR_current, t_e_current, t_i_current, t_r_current, index_current, infected_source_current);
	func_mcmc.initialize_kernel_mat(kernel_mat_current, norm_const_current); // initialize the kernel matrix
	func_mcmc.initialize_delta_mat(delta_mat_current); // initialize the exposure time matrix
	func_mcmc.initialize_delta_mat_mov(delta_mat_mov_current, moves); //initialise the movement exposure time matrix
	if (opt_betaij <= 1) {
		func_mcmc.initialize_beta_ij_mat(beta_ij_mat_current, epi_final.herdn, epi_final.ftype0, epi_final.ftype1, epi_final.ftype2); // initialize the covariate matrix
	}
	/*
	if (opt_betaij >= 2) {
		func_mcmc.initialize_beta_ij_mat_inf(beta_ij_inf_current, epi_final.herdn, epi_final.ftype0, epi_final.ftype1, epi_final.ftype2);
		func_mcmc.initialize_beta_ij_mat_susc(beta_ij_susc_current, epi_final.herdn, epi_final.ftype0, epi_final.ftype1, epi_final.ftype2);
		func_mcmc.initialize_beta_ij_mat_norm(beta_ij_mat_current, beta_ij_inf_current, beta_ij_susc_current);
	}
	*/
	func_mcmc.initialize_lh_square(lh_square_current, kernel_mat_current, delta_mat_current, norm_const_current, nt_data_current, con_seq_current, beta_ij_mat_current, moves, para_priorsetc, delta_mat_mov_current); //initialize lh_square


	double log_lh_current = log_lh_func(lh_square_current, para_other.n); // initialization of log-likelihood value

	if (debug == 1) {
		myfile_out.open((string(PATH2) + string("initial_norm_const.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << norm_const_current.at(i) << endl;
		}
		myfile_out.close();

		myfile_out.open((string(PATH2) + string("initial_f_U.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_U.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_kt_sum_U.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.kt_sum_U.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_f_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_g_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.g_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_h_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.h_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_k_sum_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.k_sum_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_q_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.q_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_kt_sum_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.kt_sum_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_f_I.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_I.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_f_R.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_R.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_f_EnI.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_EnI.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_f_InR.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.f_InR.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_log_f_S.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.log_f_S.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_log_f_Snull.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.log_f_Snull.at(i) << endl;
		}
		myfile_out.close();

		myfile_out.open((string(PATH2) + string("initial_lh.csv")).c_str(), ios::app);
		myfile_out << log_lh_current << endl; //NOTE: this must be defined, otherwise has to re-initialize some components
		myfile_out.close();

		myfile_out.open((string(PATH2) + string("initial_moves_sum_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.moves_sum_E.at(i) << endl;
		}
		myfile_out.close();
		myfile_out.open((string(PATH2) + string("initial_movest_sum_U.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.movest_sum_U.at(i) << endl;
		}
		myfile_out.close();

		myfile_out.open((string(PATH2) + string("initial_movest_sum_E.csv")).c_str(), ios::app);
		for (int i = 0; i <= ((int)para_other.n - 1); i++) {
			myfile_out << lh_square_current.movest_sum_E.at(i) << endl;
		}
		myfile_out.close();

	}

	//-------------------
	int total_count_1_initial, total_count_2_initial, total_count_3_initial;
	total_count_1_initial = total_count_2_initial = total_count_3_initial = 0;

	count_type_all(nt_data_current, xi_E_current, para_other.n_base, total_count_1_initial, total_count_2_initial, total_count_3_initial);

	/*
	myfile_out.open((string(PATH2)+string("count_type_initial.csv")).c_str(),ios::app);
	myfile_out<<total_count_1_initial <<"," << total_count_2_initial <<"," << total_count_3_initial <<endl;
	myfile_out.close();
	*/

	//-------------
	/*
	myfile_out.open((string(PATH2)+string("initial_f_I.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_I_current.size()-1);i++){
	myfile_out<<lh_square_current.f_I.at(xi_I_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_f_EnI.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_EnI_current.size()-1);i++){
	myfile_out<<lh_square_current.f_EnI.at(xi_EnI_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_f_R.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_R_current.size()-1);i++){
	myfile_out<<lh_square_current.f_R.at(xi_R_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_f_InR.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_InR_current.size()-1);i++){
	myfile_out<<lh_square_current.f_InR.at(xi_InR_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_kt_sum_E.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_E_minus_current.size()-1);i++){
	myfile_out<<lh_square_current.kt_sum_E.at(xi_E_minus_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_g_E.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_E_minus_current.size()-1);i++){
	myfile_out<<lh_square_current.g_E.at(xi_E_minus_current.at(i))<<endl;
	}
	myfile_out.close();myfile_out.open((string(PATH2)+string("initial_h_E.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_E_minus_current.size()-1);i++){
	myfile_out<<lh_square_current.h_E.at(xi_E_minus_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_q_E.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_E_minus_current.size()-1);i++){
	myfile_out<<lh_square_current.q_E.at(xi_E_minus_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_f_E.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_E_minus_current.size()-1);i++){
	myfile_out<<lh_square_current.f_E.at(xi_E_minus_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_f_U.csv")).c_str(),ios::app);
	for (int i=0;i<=((int)xi_U_current.size()-1);i++){
	myfile_out<<lh_square_current.f_U.at(xi_U_current.at(i))<<endl;
	}
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("initial_log_f_S.csv")).c_str(),ios::app);
	for (int i=0;i<=(para_other.n-1);i++){
	myfile_out<<lh_square_current.log_f_S.at(i)<<endl;
	}
	myfile_out.close();
	*/

	/*--------------------*/
	ofstream myfile1_out, myfile2_out, myfile3_out, myfile4_out, myfile5_out, myfile6_out, myfile7_out, myfile8_out;

	mcmc_UPDATE mcmc_update;
	mcmc_update.set_para(para_other, coordinate, epi_final);

	myfile1_out.open((string(PATH2) + string("parameters_current.log")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile1_out.close();
		myfile1_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("parameters_current.log")).c_str(), ios::app);
	}

	myfile1_out << "sample" << "\t" << "log_likelihood" << "\t" << "corr" << "\t" << "coverage" << "\t" << "alpha" << "\t" << "beta" << "\t" << "lat_mu" << "\t" << "lat_sd" << "\t" << "c" << "\t" << "d" << "\t" << "k_1" << "\t" << "mu_1" << "\t" << "mu_2" << "\t" << "p_ber" << "\t" << "phi_inf1" << "\t" << "phi_inf2" << "\t" << "rho_susc1" << "\t" << "rho_susc2" << "\t" << "nu_inf" << "\t" << "tau_susc" << "\t" << "beta_m" << endl;
	//myfile1_out.close();

	if (debug == 1) {
		myfile2_out.open((string(PATH2) + string("lh_current.csv")).c_str(), ios::app);
		if (para_other.np > 1) {	// on cluster
			myfile2_out.close();
			myfile2_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("lh_current.csv")).c_str(), ios::app);
		}
	}

	myfile3_out.open((string(PATH2) + string("infected_source_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile3_out.close();
		myfile3_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("infected_source_current.csv")).c_str(), ios::app);
	}

	//record timing of exposure
	myfile4_out.open((string(PATH2) + string("t_e_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile4_out.close();
		myfile4_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("t_e_current.csv")).c_str(), ios::app);
	}

	//record timing of infectious onset
	ofstream myfile_t_i_current_out;
	myfile_t_i_current_out.open((string(PATH2) + string("t_i_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile_t_i_current_out.close();
		myfile_t_i_current_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("t_i_current.csv")).c_str(), ios::app);
	}



	// record the consensus sequence every n_output_gm cycles
	myfile5_out.open((string(PATH2) + string("con_seq_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile5_out.close();
		myfile5_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("con_seq_current.csv")).c_str(), ios::app);
	}


	// record the sequence data every n_output_gm cycles
	myfile6_out.open((string(PATH2) + string("seqs_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile6_out.close();
		myfile6_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("seqs_current.csv")).c_str(), ios::app);
	}

	myfile7_out.open((string(PATH2) + string("seqs_t_current.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile7_out.close();
		myfile7_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("seqs_t_current.csv")).c_str(), ios::app);
	}

	myfile8_out.open((string(PATH2) + string("sample_percentage.csv")).c_str(), ios::app);
	if (para_other.np > 1) {	// on cluster
		myfile8_out.close();
		myfile8_out.open((string(PATH2) + string(getenv("SLURM_ARRAY_TASK_ID")) + string("_") + string("sample_percentage.csv")).c_str(), ios::app);
	}


	vector<int> list_update; // would contain the subjects (descended from subject_proposed below) whose FIRST sequence would be updated, with a sequential order (i.e., level-wise and time-wise) of updating (note: as each event needed to be updated corresponds to an infection event, it would be sufficient to update the first sequence of necessary subjects so as to update all downstream seq)
	list_update.reserve(1);


	int n_iter = static_cast<int>(para_other.n_iterations); //number of iterations for MCMC
	int n_freq = para_other.n_frequ; // frequency to translate an infection time


	double corr, coverage, sample_percentage;

	// the MCMC iterations (+10 for Tracer log output):
	for (int i = 0; i < (n_iter + 10); i++) {



		if (opt_betaij >= 1) {

			mcmc_update.rho_susc1_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
			mcmc_update.rho_susc2_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
			mcmc_update.phi_inf1_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
			mcmc_update.phi_inf2_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
			mcmc_update.tau_susc_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
			mcmc_update.nu_inf_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, beta_ij_mat_current, para_priorsetc, para_scalingfactors, beta_ij_inf_current, beta_ij_susc_current, moves, i, rng, delta_mat_mov_current);
		}

		if ((opt_mov == 1) | (opt_mov == 2)) {
			mcmc_update.beta_m_update(lh_square_current, log_lh_current, xi_U_current, xi_E_minus_current, t_e_current, index_current, infected_source_current, para_current, para_priorsetc, para_scalingfactors, i, rng);
		}


		if (para_other.kernel_type == "exponential") {
			mcmc_update.k_1_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, para_priorsetc, para_scalingfactors, beta_ij_mat_current, moves, i, rng, delta_mat_mov_current);
		}

		if ((para_other.kernel_type == "power_law") | (para_other.kernel_type == "cauchy") | (para_other.kernel_type == "gaussian")) {
			mcmc_update.k_1_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_minus_current, xi_I_current, t_r_current, t_i_current, t_e_current, index_current, para_current, infected_source_current, norm_const_current, para_priorsetc, para_scalingfactors, beta_ij_mat_current, moves, i, rng, delta_mat_mov_current);
		}


		mcmc_update.beta_update(lh_square_current, log_lh_current, xi_U_current, xi_E_minus_current, t_e_current, t_i_current, t_r_current, index_current, infected_source_current, para_current, para_priorsetc, para_scalingfactors, moves, i, rng);

		mcmc_update.alpha_update(lh_square_current, log_lh_current, xi_U_current, xi_E_minus_current, xi_I_current, t_e_current, t_i_current, t_r_current, index_current, infected_source_current, para_current, para_priorsetc, para_scalingfactors, moves, i, rng);

		mcmc_update.lat_mu_update(lh_square_current, log_lh_current, xi_I_current, xi_EnI_current, t_i_current, t_e_current, index_current, para_current, para_priorsetc, para_scalingfactors, i, rng);
		mcmc_update.lat_var_update(lh_square_current, log_lh_current, xi_I_current, xi_EnI_current, t_i_current, t_e_current, index_current, para_current, para_priorsetc, para_scalingfactors, i, rng);

		mcmc_update.c_update(lh_square_current, log_lh_current, xi_R_current, xi_InR_current, t_r_current, t_i_current, index_current, para_current, para_priorsetc, para_scalingfactors, i, rng);
		mcmc_update.d_update(lh_square_current, log_lh_current, xi_R_current, xi_InR_current, t_r_current, t_i_current, index_current, para_current, para_priorsetc, para_scalingfactors, i, rng);


		mcmc_update.mu_1_update(lh_square_current, log_lh_current, xi_E_current, para_current, nt_data_current, para_priorsetc, para_scalingfactors, i, rng);
		mcmc_update.mu_2_update(lh_square_current, log_lh_current, xi_E_current, para_current, nt_data_current, para_priorsetc, para_scalingfactors, i, rng);

		mcmc_update.p_ber_update(lh_square_current, log_lh_current, xi_E_current, para_current, nt_data_current, infected_source_current, con_seq_current, para_priorsetc, para_scalingfactors, i, rng);

		for (int k = 0; k <= (para_other.n_base - 1); k++) {
			mcmc_update.con_seq_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, t_r_current, t_i_current, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, xi_beta_E_current, con_seq_current, para_priorsetc, para_scalingfactors, k, rng);
		}

		//--------------------//

		for (int j = 0; j <= ((int)(n_freq / 10.0) - 1); j++) {

			//	int subject_proposed_2 = xi_E_minus_current.at(gsl_rng_uniform_int (r_c_unvs, xi_E_minus_current.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
				//int subject_proposed_2 = xi_E_current.at(gsl_rng_uniform_int (r_c_unvs, xi_E_current.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
			int subject_proposed_2 = xi_E_current.at(runif_int(0, (int)(xi_E_current.size()) - 1, rng)); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn

			for (int k = 0; k <= (para_other.n_base - 1); k++) {
				mcmc_update.seq_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, t_r_current, t_i_current, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, xi_beta_E_current, con_seq_current, subject_proposed_2, para_priorsetc, para_scalingfactors, k, rng);
			}

		}

		//--------------------//

		for (int j = 0; j <= (n_freq - 1); j++) {

			// int subject_proposed = xi_E_minus_current.at(gsl_rng_uniform_int (r_c_unvs, xi_E_minus_current.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
			//int subject_proposed = xi_E_current.at(gsl_rng_uniform_int (r_c_unvs, xi_E_current.size())); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn
			int subject_proposed = xi_E_current.at(runif_int(0, (int)(xi_E_current.size()) - 1, rng)); // gsl_rng_uniform_int : a int random number in [0,xi_E_arg.size()-1] will be drawn


			mcmc_update.t_e_seq(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, t_r_current, t_i_current, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, nt_data_current.infecting_list, nt_data_current.infecting_size, xi_beta_E_current, con_seq_current, subject_proposed, para_priorsetc, para_scalingfactors, beta_ij_mat_current, moves, i + 1, rng, delta_mat_mov_current, moves);

			mcmc_update.source_t_e_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, t_r_current, t_i_current, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, nt_data_current.infecting_list, nt_data_current.infecting_size, xi_beta_E_current, subject_proposed, list_update, con_seq_current, para_priorsetc, para_scalingfactors, beta_ij_mat_current, moves, i + 1, rng, delta_mat_mov_current);

		}




		for (int jk = 0; jk <= ((int)index_current.size() - 1); jk++) {// update the indexes first seq

			for (int k = 0; k <= (para_other.n_base - 1); k++) {
				mcmc_update.seq_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, t_r_current, t_i_current, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, xi_beta_E_current, con_seq_current, index_current.at(jk), para_priorsetc, para_scalingfactors, k, rng);
			}

		}


		if (opt_ti_update == 1) {
			mcmc_update.t_i_update(lh_square_current, log_lh_current, kernel_mat_current, delta_mat_current, xi_U_current, xi_E_current, xi_E_minus_current, xi_I_current, xi_EnI_current, xi_R_current, xi_InR_current, t_r_current, t_i_current, t_onset, t_e_current, index_current, para_current, norm_const_current, infected_source_current, nt_data_current.t_sample, nt_data_current.current_size, nt_data_current.nt, nt_data_current.t_nt, nt_data_current.infecting_list, nt_data_current.infecting_size, para_priorsetc, para_scalingfactors, beta_ij_mat_current, moves, i + 1, rng, delta_mat_mov_current, moves);
		}


		//-------------------------------//

		div_t div_iter_lh;
		//div_iter_lh = div (i,1);
		div_iter_lh = div(i, para_other.n_output_source);//sf: output every
		if (div_iter_lh.rem == 0) {


			//-----


			for (int js = 0; js <= (para_other.n - 1); js++) {
				int rem = (js + 1) % para_other.n;
				if ((rem != 0) | (js == 0)) myfile3_out << infected_source_current.at(js) << ",";
				if ((rem == 0) & (js != 0)) myfile3_out << infected_source_current.at(js) << " " << endl;
			}


			//--

			for (int ii = 0; ii <= (int)(t_e_current.size() - 1); ii++) {
				if (ii<int(t_e_current.size() - 1)) myfile4_out << t_e_current.at(ii) << ",";
				if (ii == int(t_e_current.size() - 1))   myfile4_out << t_e_current.at(ii) << endl;
			}

			for (int ii = 0; ii <= (int)(t_i_current.size() - 1); ii++) {
				if (ii<int(t_i_current.size() - 1)) myfile_t_i_current_out << t_i_current.at(ii) << ",";
				if (ii == int(t_i_current.size() - 1)) myfile_t_i_current_out << t_i_current.at(ii) << endl;
			}


			//---------

			if (debug == 1) {
				myfile_out.open((string(PATH2) + string("real_time_index.csv")).c_str(), ios::app);
				for (int ii = 0; ii <= (int)(index_current.size() - 1); ii++) {
					myfile_out << index_current.size() << "," << index_current.at(ii) << "," << t_e_current.at(index_current.at(ii)) << endl;
				}
				myfile_out.close();
			}


		}

		//------------------
		//sf console output
		if ((debug == 1) & (i + 1 == 20)) { //checksum based on LH of 20th mcmc cycle Max Lau's original run,
						  //under para_aux input all opts == 0, lh_check == 0 if same config
			if ((opt_latgamma == 0) & (opt_k80 == 0) & (opt_betaij == 0) & (opt_ti_update == 0)) {
				cout << "lh_check original(a52): " << roundx(abs(739100 + log_lh_current), 0) << ", " << log_lh_current << endl; //original
			}

			if ((opt_latgamma == 1) & (opt_k80 == 1) & (opt_betaij == 0) & (opt_ti_update == 0)) {
				cout << "lh_check 1(a52): " << roundx(abs(715451 + log_lh_current), 0) << ", " << log_lh_current << endl; //sf_edit1
				cout << "lh_check 1(jpn): " << roundx(abs(377755 + log_lh_current), 0) << ", " << log_lh_current << endl; //original

				if ((opt_mov == 0)) {
					cout << "lh_check 1(sk100_mov0): " << roundx(abs(508682 + log_lh_current), 0) << ", " << log_lh_current << endl;
				}
				if ((opt_mov == 1)) {
					cout << "lh_check 1(sk100_mov1): " << roundx(abs(489075 + log_lh_current), 0) << ", " << log_lh_current << endl;
				}
				if ((opt_mov == 2)) {
					cout << "lh_check 1(sk100_mov2): " << roundx(abs(575380 + log_lh_current), 0) << ", " << log_lh_current << endl;
				}

			}

			if ((opt_latgamma == 1) & (opt_k80 == 1) & (opt_betaij == 1) & (opt_ti_update == 0)) {
				cout << "lh_check 1.betaij(a52): " << roundx(abs(635001 + log_lh_current), 0) << ", " << log_lh_current << endl; //sf_edit2 (beta_ij)
			}

			if ((opt_latgamma == 1) & (opt_k80 == 1) & (opt_betaij == 0) & (opt_ti_update == 1)) {
				cout << "lh_check 1.ti(a52): " << roundx(abs(650105 + log_lh_current), 0) << ", " << log_lh_current << endl; //t_i_update
			}



		}


		div_t div_iter_cout;
		//div_iter_cout = div(i + 1, 10);//sf: output every n_cout, starting at n_cout
		div_iter_cout = div(i + 1, para_other.n_cout);//sf: output every n_cout, starting at n_cout

		// calculate how many are correctly identified at this stage
		int n_infected = xi_E.size(); // Total number of infected farms
		corr = 0;// calculate how many are correctly identified at this stage
		sample_percentage = 0;
		for (int cs = 0; cs <= (para_other.n - 1); cs++) {
			if (infected_source_current.at(cs) == atab_from.at(cs)) corr++;
			if (nt_data.t_sample.at(cs) != para_other.unassigned_time) sample_percentage++;
		}
		coverage = corr / n_infected;
		sample_percentage = sample_percentage / n_infected;

		if ((i + 1) < 2) { myfile8_out << sample_percentage << endl; }


		if (div_iter_cout.rem == 0) { // for console output (only at every n_cout iterations)
			if ((i + 1) >= 10) {
				cout << i + 1 << ": corr = " << corr << " coverage = " << roundx(coverage, 2) << ", c = " << roundx(para_current.c, 3) << ", d = " << roundx(para_current.d, 3) << ", mu_1 = " << para_current.mu_1 << ", mu_2 = " << para_current.mu_2 << ", beta_m = " << roundx(para_current.beta_m, 3) << endl;
			}
		}


		//--parameters_current output -----------------------------//
		//remove first 10 rows so can load directly in Tracer
		if ((i + 1) > 10) { myfile1_out << i + 1 << "\t" << log_lh_current << "\t" << corr << "\t" << coverage << "\t" << para_current.alpha << "\t" << para_current.beta << "\t" << para_current.lat_mu << "\t" << sqrt(para_current.lat_var) << "\t" << para_current.c << "\t" << para_current.d << "\t" << para_current.k_1 << "\t" << para_current.mu_1 << "\t" << para_current.mu_2 << "\t" << para_current.p_ber << "\t" << para_current.phi_inf1 << "\t" << para_current.phi_inf2 << "\t" << para_current.rho_susc1 << "\t" << para_current.rho_susc2 << "\t" << para_current.nu_inf << "\t" << para_current.tau_susc << "\t" << para_current.beta_m << endl; }

		if (debug == 1) {
			myfile2_out << log_lh_current << endl;
		}
		//------------------


		div_t div_iter_seq;
		//div_iter_seq = div (i,2000);
		div_iter_seq = div(i, para_other.n_output_gm);

		switch (int(div_iter_seq.rem == 0)) {

		case 1: {

			//--- output consensus sequence
			for (int js = 0; js <= (para_other.n_base - 1); js++) {
				int rem = (js + 1) % para_other.n_base;
				if ((rem != 0) | (js == 0)) myfile5_out << con_seq_current.at(js) << ",";
				if ((rem == 0) & (js != 0)) myfile5_out << con_seq_current.at(js) << " " << endl;
			}
			//--- output inferred sequences
			//[structured in rows: iteration n_output_gm*1 seqeunces 1:n;
			//                     iteration n_output_gm*2 seqeunces 1:n; etc]
			//so for 1st sequence take every seq(1,niter,by=n_output_gm) row
			for (int is = 0; is <= (int)(para_other.n - 1); is++) {

				for (int js = 0; js <= (int)(nt_data_current.nt[is].size() - 1); js++) {
					int rem = (js + 1) % para_other.n_base;
					if ((rem != 0) | (js == 0)) myfile6_out << nt_data_current.nt[is][js] << ",";
					if ((rem == 0) & (js != 0)) myfile6_out << nt_data_current.nt[is][js] << " " << endl;
				}
			}

			for (int is = 0; is <= (int)(para_other.n - 1); is++) {
				for (int js = 0; js <= (int)(nt_data_current.t_nt[is].size() - 1); js++) {
					if ((js < (int)(nt_data_current.t_nt[is].size() - 1))) myfile7_out << nt_data_current.t_nt[is][js] << ",";
					if ((js == (int)(nt_data_current.t_nt[is].size() - 1))) myfile7_out << nt_data_current.t_nt[is][js] << " " << endl;
				}
			}

			break;
		}

		default: {
			break;
		}

		}

		//------------------

	} // end of MCMC loop




	myfile1_out.close();  //parameters_current.csv
	if (debug == 1) { myfile2_out.close(); } //lh_current.csv
	myfile3_out.close();  //infected_source_current.csv
	myfile4_out.close();  // timing of exposure (t_e_current.csv)
	myfile_t_i_current_out.close();  // timing of onset of infectiousness (t_i_current.csv)
	myfile5_out.close();  //consensus sequence every n_output_gm cycles
	myfile6_out.close();  //inferred sequences every n_output_gm cycles
	myfile7_out.close();  //inferred sequence_timings every n_output_gm cycles
	myfile8_out.close(); // percentage of known infected hosts that are sampled


  
  /*----------------------------*/
  /*----------------------------*/
  /*----------------------------*/
	//  Returns to R ----------------------------------------------
	/*
	vector<double> r_epi(epi_final.k.begin(), epi_final.k.end());
	r_epi.insert(r_epi.end(), epi_final.coor_x.begin(), epi_final.coor_x.end());
	r_epi.insert(r_epi.end(), epi_final.coor_y.begin(), epi_final.coor_y.end());
	r_epi.insert(r_epi.end(), epi_final.ftype0.begin(), epi_final.ftype0.end());
	r_epi.insert(r_epi.end(), epi_final.ftype1.begin(), epi_final.ftype1.end());
	r_epi.insert(r_epi.end(), epi_final.ftype2.begin(), epi_final.ftype2.end());
	r_epi.insert(r_epi.end(), epi_final.herdn.begin(), epi_final.herdn.end());
	Rcpp::NumericVector r_epi_v(r_epi.begin(), r_epi.end());
	r_epi_v.attr("dim") = Rcpp::Dimension(epi_final.k.size(), 7);
	Rcpp::NumericMatrix r_epi_m = as<Rcpp::NumericMatrix>(r_epi_v);
	colnames(r_epi_m) = CharacterVector::create("k", "coor_x", "coor_y", "ftype0", "ftype1", "ftype2", "herdn");
	*/
	/*----------------------------*/
	/*
	vector<double> r_moves(moves.from_k.begin(), moves.from_k.end());
	r_moves.insert(r_moves.end(), moves.to_k.begin(), moves.to_k.end());
	r_moves.insert(r_moves.end(), moves.t_m.begin(), moves.t_m.end());
	Rcpp::NumericVector r_moves_v(r_moves.begin(), r_moves.end());
	r_moves_v.attr("dim") = Rcpp::Dimension(moves.to_k.size(), 3);
	Rcpp::NumericMatrix r_moves_m = as<Rcpp::NumericMatrix>(r_moves_v);
	colnames(r_moves_m) = CharacterVector::create("from_k", "to_k", "t_m");
	*/
	/*----------------------------*/
	/*
	Rcpp::NumericVector r_par_aux;
	r_par_aux.push_back(para_other.n);
	r_par_aux.push_back(para_other.t_max);
	r_par_aux.push_back(para_other.np);
	r_par_aux.push_back(para_other.n_seq);
	r_par_aux.push_back(para_other.n_base);
	r_par_aux.push_back(para_other.n_iterations);
	r_par_aux.push_back(para_other.n_frequ);
	r_par_aux.push_back(para_other.n_output_source);
	r_par_aux.push_back(para_other.n_output_gm);
	r_par_aux.push_back(para_other.n_cout);
	r_par_aux.push_back(para_other.opt_latgamma);
	r_par_aux.push_back(para_other.opt_k80);
	r_par_aux.push_back(para_other.opt_betaij);
	r_par_aux.push_back(para_other.opt_ti_update);
	r_par_aux.push_back(para_other.opt_mov);
	vector<string> r_par_aux_names = { "n", "t_max", "nprocessors", "n_seq", "n_base", "n_iterations",
		"n_frequ", "n_output_source", "n_output_gm", "n_cout", "opt_latgamma", "opt_k80", "opt_betaij",
		"opt_ti_update", "opt_mov" };
	r_par_aux.attr("names") = r_par_aux_names;
	*/
	/*----------------------------*/
	/*
	Rcpp::NumericVector r_par_init;
	r_par_init.push_back(para_init.alpha);
	r_par_init.push_back(para_init.beta);
	r_par_init.push_back(para_init.lat_mu);
	r_par_init.push_back(para_init.lat_var);
	r_par_init.push_back(para_init.c);
	r_par_init.push_back(para_init.d);
	r_par_init.push_back(para_init.k_1);
	r_par_init.push_back(para_init.mu_1);
	r_par_init.push_back(para_init.mu_2);
	r_par_init.push_back(para_init.p_ber);
	r_par_init.push_back(para_init.phi_inf1);
	r_par_init.push_back(para_init.phi_inf2);
	r_par_init.push_back(para_init.rho_susc1);
	r_par_init.push_back(para_init.rho_susc2);
	r_par_init.push_back(para_init.nu_inf);
	r_par_init.push_back(para_init.tau_susc);
	r_par_init.push_back(para_init.beta_m);
	vector<string> r_par_init_names = { "alpha", "beta", "mu_lat", "var_lat",
		"c", "d", "k_1", "mu_1", "mu_2", "p_ber", "phi_inf1", "phi_inf2", "rho_susc1",
		"rho_susc2", "nu_inf", "tau_susc", "beta_m" };
	r_par_init.attr("names") = r_par_init_names;
	*/
	/*----------------------------*/
	/*
	Rcpp::NumericVector r_par_priors;
	r_par_priors.push_back(para_priorsetc.t_range);
	r_par_priors.push_back(para_priorsetc.t_back);
	r_par_priors.push_back(para_priorsetc.t_bound_hi);
	r_par_priors.push_back(para_priorsetc.rate_exp_prior);
	r_par_priors.push_back(para_priorsetc.ind_n_base_part);
	r_par_priors.push_back(para_priorsetc.n_base_part);
	r_par_priors.push_back(para_priorsetc.alpha_hi);
	r_par_priors.push_back(para_priorsetc.beta_hi);
	r_par_priors.push_back(para_priorsetc.lat_mu_hi);
	r_par_priors.push_back(para_priorsetc.lat_var_lo);
	r_par_priors.push_back(para_priorsetc.lat_var_hi);
	r_par_priors.push_back(para_priorsetc.c_hi);
	r_par_priors.push_back(para_priorsetc.d_hi);
	r_par_priors.push_back(para_priorsetc.k_1_hi);
	r_par_priors.push_back(para_priorsetc.mu_1_hi);
	r_par_priors.push_back(para_priorsetc.mu_2_hi);
	r_par_priors.push_back(para_priorsetc.p_ber_hi);
	r_par_priors.push_back(para_priorsetc.phi_inf1_hi);
	r_par_priors.push_back(para_priorsetc.phi_inf2_hi);
	r_par_priors.push_back(para_priorsetc.rho_susc1_hi);
	r_par_priors.push_back(para_priorsetc.rho_susc2_hi);
	r_par_priors.push_back(para_priorsetc.nu_inf_lo);
	r_par_priors.push_back(para_priorsetc.nu_inf_hi);
	r_par_priors.push_back(para_priorsetc.tau_susc_lo);
	r_par_priors.push_back(para_priorsetc.tau_susc_hi);
	r_par_priors.push_back(para_priorsetc.beta_m_hi);
	r_par_priors.push_back(para_priorsetc.trace_window);
	vector<string> priors_names = { "t_range", "t_back", "t_bound_hi", "rate_exp_prior",
		"ind_n_base_part", "n_base_part", "alpha_hi", "beta_hi", "lat_mu_hi", "lat_var_lo", "lat_var_hi",
		"c_hi", "d_hi", "k_1_hi", "mu_1_hi", "mu_2_hi", "p_ber_hi", "phi_inf1_hi", "phi_inf2_hi", "rho_susc1_hi",
		"rho_susc2_hi", "nu_inf_lo", "nu_inf_hi", "tau_susc_lo", "tau_susc_hi", "beta_m_hi", "trace_window" };
	r_par_priors.attr("names") = priors_names;
	*/
	/*----------------------------*/
	/*
	Rcpp::NumericVector r_par_sf;
	r_par_sf.push_back(para_scalingfactors.alpha_sf);
	r_par_sf.push_back(para_scalingfactors.beta_sf);
	r_par_sf.push_back(para_scalingfactors.lat_mu_sf);
	r_par_sf.push_back(para_scalingfactors.lat_var_sf);
	r_par_sf.push_back(para_scalingfactors.c_sf);
	r_par_sf.push_back(para_scalingfactors.d_sf);
	r_par_sf.push_back(para_scalingfactors.k_1_sf);
	r_par_sf.push_back(para_scalingfactors.mu_1_sf);
	r_par_sf.push_back(para_scalingfactors.mu_2_sf);
	r_par_sf.push_back(para_scalingfactors.p_ber_sf);
	r_par_sf.push_back(para_scalingfactors.phi_inf1_sf);
	r_par_sf.push_back(para_scalingfactors.phi_inf2_sf);
	r_par_sf.push_back(para_scalingfactors.rho_susc1_sf);
	r_par_sf.push_back(para_scalingfactors.rho_susc2_sf);
	r_par_sf.push_back(para_scalingfactors.nu_inf_sf);
	r_par_sf.push_back(para_scalingfactors.tau_susc_sf);
	r_par_sf.push_back(para_scalingfactors.beta_m_sf);
	vector<string> r_par_sf_names = { "alpha_sf", "beta_sf", "mu_lat_sf", "var_lat_sf", "c_sf", "d_sf", "k_1_sf", "mu_1_sf", "mu_2_sf", "p_ber_sf", "phi_inf1_sf", "phi_inf2_sf", "rho_susc1_sf", "rho_susc2_sf", "nu_inf_sf", "tau_susc_sf", "beta_m_sf" };
	r_par_sf.attr("names") = CharacterVector::create();
	*/
	/*----------------------------*/
	/*----------------------------*/

	return Rcpp::List::create(
		/*
				Rcpp::Named("epi.mat") = r_epi_m,
				Rcpp::Named("moves.mat") = r_moves_m,
				Rcpp::Named("index") = index,
				Rcpp::Named("par.aux") = r_par_aux,
				Rcpp::Named("par.aux.unassigned") = para_other.unassigned_time,
				Rcpp::Named("par.aux.kernel") = para_other.kernel_type,
				Rcpp::Named("par.aux.coord") = para_other.coord_type,
				Rcpp::Named("par.init") = r_par_init,
				Rcpp::Named("par.priors") = r_par_priors,
				Rcpp::Named("par.sf") = r_par_sf,
		*/
		Rcpp::Named("seed") = seeds,
		Rcpp::Named("t_sample") = nt_data.t_sample);
}

/*----------------------------------------------------------*/
/*- Main call of function to export & inputs from R --------*/
/*----------------------------------------------------------*/

// [[Rcpp::export]]
Rcpp::List sim_cpp() {
  /*----------------------------*/
  /*- int main -----------------*/
  /*----------------------------*/  
	para_aux_struct para_other;
	IO_para_aux(para_other);  //Importing parameters
	opt_k80_sim = para_other.opt_k80;
	opt_betaij_sim = para_other.opt_betaij;
	debug_sim = para_other.debug;
	opt_mov_sim = para_other.opt_mov;


	para_key_struct para_key;
	IO_para_key(para_key);  //Importing parameters


	// simulate con_seq (GM sequence)
	// srand(time(NULL)); //initializes the random seed
	// int seed= rand()%100;
	//const gsl_rng_type* T= gsl_rng_ranlux;  // T is pointer points to the type of generator
	//gsl_rng *r = gsl_rng_alloc (T); // r is pointer points to an object with Type T
	//gsl_rng_set (r, para_other.seed); // set a seed
	seed_sim = para_other.seed; //set a universal seed
	rng_type rng_sim(seed_sim); //set a universal seed

	vector<int> con_seq(para_other.n_base); // the consensus seq to be perturbed for background infection

	for (int i = 0; i <= (para_other.n_base - 1); i++) {
		//con_seq.at(i)=(gsl_rng_uniform_int(r, 4) +1 );//  first part generates r.v. uniformly on [0,3]
		con_seq.at(i) = runif_int_sim(1, 4, rng_sim);//  first part generates r.v. uniformly on [1,4]
	}

	ofstream myfile; // this object can be recycled after myfile.close()

	//---------------------------

	myfile.open((string(PATH2) + string("con_seq_estm.csv")).c_str(), ios::app);  // Note: filename is also an OBJECT of CLASS String

	for (int i = 0; i <= (para_other.n_base - 1); i++) {
		int rem = (i + 1) % para_other.n_base;
		if ((rem != 0) | (i == 0)) myfile << con_seq.at(i) << ",";
		if ((rem == 0) & (i != 0)) myfile << con_seq.at(i) << " " << endl;
	}
	myfile.close();

	min_max_functions min_max;

	epi_struct_sim epi_data; // epi_struct is a user-defined data structure in the header file
	IO_para_epi(epi_data, para_other);


	mov_struct mov_data; // mov_struct is a user-defined data structure in the header file
	if (opt_mov_sim == 1) {
		IO_para_mov(mov_data, para_other);
	}


	//double coordinate[n][2]; // coordinates of the subjects
	vector< vector<double> > coordinate(para_other.n, vector<double>(2));
	for (int i = 0; i <= (para_other.n - 1); i++) {
		for (int j = 0; j <= 1; j++) {
			if (j == 0) { coordinate[i][j] = epi_data.coor_x.at(i); }
			if (j == 1) { coordinate[i][j] = epi_data.coor_y.at(i); }
		}
	}


	//vector< vector<int> > infected_source(n,1); //infection souces of the infected cases (the size 1 would be resized during the execution according to the actual number of infection sources)
	//int infect_source[n][n]; //infection souces of the infected cases

	epi_data.q.resize(para_other.n);
	epi_data.ric.resize(para_other.n);
	epi_data.t_next.resize(para_other.n);
	epi_data.lat_period.resize(para_other.n);
	epi_data.inf_period.resize(para_other.n);

	//myfile.open((string(PATH2)+string("sellkes.txt")).c_str(),ios::app);
	//myfile << "k" << "," << "q" << endl;

	for (int i = 0; i <= (para_other.n - 1); i++) {

		//epi_data.k.at(i) = i;
		//epi_data.coor_x.at(i) = coordinate[i][0];
		//epi_data.coor_y.at(i) = coordinate[i][1];
		//epi_data.t_e.at(i) = epi_data.t_i.at(i) = epi_data.t_r.at(i) = para_other.unassigned_time; // initialize the event times
		//ftype1, ftype2, herdn
		//epi_data.status.at(i) = 1; //1=S, 2=E, 3=I,4=R

		//epi_data.q.at(i) = gsl_ran_exponential(r,1.0); //set sellke thresholds
		epi_data.q.at(i) = rweibull_sim(1.0, 1.0, rng_sim); //set sellke thresholds
		epi_data.ric.at(i) = epi_data.q.at(i); // the remaining infective challenge needed to get infected (i.e.,from S to E)
		epi_data.t_next.at(i) = para_other.unassigned_time; // time of next event
		epi_data.lat_period.at(i) = func_latent_ran(rng_sim, para_key.a, para_key.b);
		epi_data.inf_period.at(i) = rweibull_sim(para_key.c, para_key.d, rng_sim);

		//myfile << epi_data.k.at(i) << "," << epi_data.q.at(i) << endl;
	}

	//myfile.close();


	//---------------------------



	nt_struct_sim nt_data;

	nt_data.log_f_S = 0.0;
	nt_data.total_count_1 = nt_data.total_count_2 = nt_data.total_count_3 = 0;

	nt_data.nt.resize(para_other.n);
	for (int i = 0; i <= (para_other.n - 1); i++) {
		nt_data.nt[i].reserve(para_other.n_seq*para_other.n_base);
	}

	nt_data.t_nt.resize(para_other.n);
	for (int i = 0; i <= (para_other.n - 1); i++) {
		nt_data.t_nt[i].reserve(para_other.n_seq);
	}


	nt_data.current_size.resize(para_other.n);
	nt_data.t_sample.resize(para_other.n); // sampling times upon infection; if sample time > t_r or t_max, it would not have a corresponding sequence
	nt_data.t_last.resize(para_other.n);
	nt_data.ind_sample.resize(para_other.n);


	//------------------------------------------------------------------//

	vector<int> index;

	//switch (alpha>0.0) { // this block determines the index cases depending on if alpah==0

	//  case 1:
	//   {
	// 	double min_q = *min_element(epi_data.q.begin(), epi_data.q.end()); // the smallest threshold
	// 	index = min_max.min_position_double(epi_data.q,min_q,n); //vector contains the index for subjects with smallest threshold
	//
	// 	for (int i=0; i<=((int)index.size()-1);i++) {
	// 	//epi_data.t_e.at(index.at(i)) = 0.0; // scale the time of first event to be zero
	// 	epi_data.t_e.at(index.at(i)) = epi_data.q.at(index.at(i))/alpha;
	// 	epi_data.status.at(index.at(i)) = 2;
	// 	epi_data.ric.at(index.at(i)) = 0.0;
	//         epi_data.coor_x.at(index.at(i)) = dimen_x/2.0; //manually set the index to be placed in the middle of the field
	//         epi_data.coor_y.at(index.at(i)) = dimen_y/2.0;
	//         coordinate[index.at(i)][0] = dimen_x/2.0;
	//         coordinate[index.at(i)][1] = dimen_y/2.0;
	//
	// 	nt_data.t_nt[index.at(i)].push_back(epi_data.t_e.at(index.at(i)));
	// 	nt_data.current_size.at(index.at(i)) = 0;
	// 	for (int j=0;j<=(n_base-1);j++){
	// 	nt_data.nt[index.at(i)].push_back(gsl_rng_uniform_int(r, 4) +1 );//  first part generates r.v. uniformly on [0,4-1]
	// 	}
	// 	nt_data.current_size.at(index.at(i)) = nt_data.current_size.at(index.at(i)) + 1;
	//
	// 	nt_data.t_last.at(index.at(i)) = epi_data.t_e.at(index.at(i));
	// 	nt_data.t_sample.at(index.at(i)) = min( t_max, epi_data.t_e.at(index.at(i)) + 5.0*gsl_rng_uniform(r)); // randomly assign a sampling time forward
	//
	// 	}
	//
	//   break;
	//   }


	//  case 0: {


		// next block randomly selects index cases and put them into index vector using a pointer
		/*
		//index.resize(para_other.n_index);

		int *index_array = &index.at(0); // the later function gsl_ran_choose needs a pointer to an array in the arguments

		vector<int> index_pool(para_other.n); //infection souces of the infected cases
		for (int i=0;i<=(para_other.n-1); i++ ){
		index_pool.at(i) = i;
		}

		int* index_pool_arr = &index_pool[0]; //convert to array for gsl_function
		//const gsl_rng_type* T= gsl_rng_default;
		//gsl_rng *r = gsl_rng_alloc (T);
		//gsl_rng_set (r, seed); // set a seed
		gsl_ran_choose(r,index_array,para_other.n_index,index_pool_arr,para_other.n,sizeof(int)); // this chooses index cases without replacements

		//gsl_rng_free (r);
		*/

	for (int i = 0; i < para_other.n; i++) {
		if ((epi_data.status.at(i) == 2) | (epi_data.status.at(i) == 3)) {
			index.push_back(i);
		}
	}

	for (int i = 0; i <= ((int)index.size() - 1); i++) {

		//epi_data.t_e.at(index.at(i))= 0.0; // scale the time of first event to be zero
		//epi_data.status.at(index.at(i))= 2;
		epi_data.ric.at(index.at(i)) = 0.0;
		//epi_data.coor_x.at(index.at(i)) = para_other.dimen_x/2.0 + i*100; //manually set the index to be placed in the middle of the field
		//epi_data.coor_y.at(index.at(i)) = para_other.dimen_y/2.0 + i*100;
		//coordinate[index.at(i)][0] = para_other.dimen_x/2.0 + i*100;
		//coordinate[index.at(i)][1] = para_other.dimen_y/2.0 + i*100;

		nt_data.t_nt[index.at(i)].push_back(epi_data.t_e.at(index.at(i)));
		nt_data.current_size.at(index.at(i)) = 0;

		// 	for (int j=0;j<=(n_base-1);j++){
		// 	nt_data.nt[index.at(i)].push_back(gsl_rng_uniform_int(r, 4) +1 );//  first part generates r.v. uniformly on [0,4-1]
		// 	}
		for (int j = 0; j <= (para_other.n_base - 1); j++) {


			//int ber_trial =  gsl_ran_bernoulli (r, para_key.p_ber); // return 1 if a change is to happen
			int ber_trial = rbern_sim(para_key.p_ber, rng_sim); // return 1 if a change is to happen
			int base_proposed = 0;

			// 		boost::bernoulli_distribution<> ber_dist(p_ber);
			// 		boost::variate_generator<base_generator_type&, boost::bernoulli_distribution<> > ber(base_generator_type, ber_dist);
			// 		ber_trial = ber();

			switch (int(ber_trial == 1)) {

			case 1: { // randomly choose one among other 3
				switch (con_seq.at(j)) {
				case 1: {
					//int type = gsl_rng_uniform_int (r, 3);
					int type = runif_int_sim(0, 2, rng_sim);
					switch (type) {
					case 0: {
						base_proposed = 2;
						break;
					}
					case 1: {
						base_proposed = 3;
						break;
					}
					case 2: {
						base_proposed = 4;
						break;
					}
					}
					break;
				}
				case 2: {
					int type = runif_int_sim(0, 2, rng_sim);

					switch (type) {
					case 0: {
						base_proposed = 1;
						break;
					}
					case 1: {
						base_proposed = 3;
						break;
					}
					case 2: {
						base_proposed = 4;
						break;
					}
					}
					break;
				}
				case 3: {
					int type = runif_int_sim(0, 2, rng_sim);
					switch (type) {
					case 0: {
						base_proposed = 1;
						break;
					}
					case 1: {
						base_proposed = 2;
						break;
					}
					case 2: {
						base_proposed = 4;
						break;
					}
					}
					break;
				}
				case 4: {
					int type = runif_int_sim(0, 2, rng_sim);
					switch (type) {
					case 0: {
						base_proposed = 1;
						break;
					}
					case 1: {
						base_proposed = 2;
						break;
					}
					case 2: {
						base_proposed = 3;
						break;
					}
					}
					break;
				}
				}

				nt_data.nt[index.at(i)].push_back(base_proposed);

				break;
			}
			case 0: {
				nt_data.nt[index.at(i)].push_back(con_seq.at(j)); // same as consensus seq
				break;
			}
			}
		}


		//---

		nt_data.current_size.at(index.at(i)) = nt_data.current_size.at(index.at(i)) + 1;

		nt_data.t_last.at(index.at(i)) = epi_data.t_e.at(index.at(i));
		//nt_data.t_sample.at(index.at(i)) = min(para_other.t_max, epi_data.t_e.at(index.at(i)) + para_other.sample_range*gsl_rng_uniform(r)); // randomly assign a sampling time forward dependant on sample_range (replace with distribution)
		nt_data.t_sample.at(index.at(i)) = min(para_other.t_max, epi_data.t_e.at(index.at(i)) + para_other.sample_range*runif_sim(0.0, 1.0, rng_sim)); // randomly assign a sampling time forward dependant on sample_range (replace with distribution)

	}

	//   break;
	//   }
	//
	// }

	//------------------------------------------------------------------//


	//myfile.open((string(PATH2)+string("index.txt")).c_str(),ios::app);
	//myfile << "k" << endl;
	//for (int i=0; i<=((int)index.size()-1);i++){
	//myfile << index.at(i) << endl;
	//}
	//myfile.close();


	int total_mismatch = 0;
	for (int j = 0; j <= (para_other.n_base - 1); j++) {
		if (nt_data.nt[index.at(0)][j] != con_seq.at(j)) total_mismatch = total_mismatch + 1;
	}
	//myfile.open((string(PATH2)+string("index_mismatch.txt")).c_str(),ios::app);
	//myfile << total_mismatch;
	//myfile.close();


	/*
	myfile.open((string(PATH2)+string("coordinate.txt")).c_str(),ios::app);
	for (int i=0;i<=(para_other.n-1);i++){
	myfile << coordinate[i][0] << "," << coordinate[i][1]<< endl;
	}
	myfile.close();
	*/

	/*
	myfile.open((string(PATH2)+string("index_t_nt_initial.txt")).c_str(),ios::app);
	for (int i=0; i<=((int)index.size()-1);i++){
	myfile <<nt_data.t_nt[index.at(i)][0]<< endl;
	}
	myfile.close();

	myfile.open((string(PATH2)+string("index_nt_initial.txt")).c_str(),ios::app);
	for (int i=0; i<=((int)index.size()-1);i++){
	for (int j=0;j<=(para_other.n_base-1);j++) {
	if (j<(para_other.n_base-1)) myfile << nt_data.nt[index.at(i)][j] << ",";
	if (j==(para_other.n_base-1)) myfile << nt_data.nt[index.at(i)][j] << " " << endl;
	}
	}
	myfile.close();


	myfile.open((string(PATH2)+string("index_inititial_size.txt")).c_str(),ios::app);
	for (int i=0; i<=((int)index.size()-1);i++){
	myfile <<nt_data.current_size.at(index.at(i)) << endl;
	}
	myfile.close();
	*/

	vector<int> ind_now = index; // indicates which subjects correspond to nearest events

	double t_now = epi_data.t_e.at(index.at(0)); // the time of the nearest event
	//vector<int> e_now_vector; //the vector for indicatings the nearest events
	//e_now_vector.assign(index.size(),2); // the initial first events are S -> E


	epi_functions epi_func;

	vector< vector<double> > distance_mat(para_other.n, vector<double>(para_other.n)); // nxn matrix to store the distance between subjects, as a look-up table

	for (int i = 0; i <= (para_other.n - 1); i++) {
		for (int j = 0; j <= (para_other.n - 1); j++) {
			if (i == j) distance_mat[i][j] = 0;
			if (i < j) distance_mat[i][j] = func_kernel_sim(coordinate[i][0], coordinate[i][1], coordinate[j][0], coordinate[j][1], para_key.k_1, para_other.coord_type, para_other.kernel_type);
			if (i > j) distance_mat[i][j] = distance_mat[j][i];
		}
	}

	/*
	myfile.open((string(PATH2) + string("distance_matrix.txt")).c_str(), ios::app);
	for (int i = 0; i <= (para_other.n - 1); i++) {
		for (int j = 0; j <= (para_other.n - 1); j++) {
			if (j<(para_other.n - 1)) myfile << distance_mat[i][j] << ",";
			if (j == (para_other.n - 1)) myfile << distance_mat[i][j] << " " << endl;
		}
	}
	myfile.close();
	*/

	//---------------------------------------------------------------------------------------//
	vector <double> norm_const(para_other.n);

	for (int j = 0; j <= (para_other.n - 1); j++) {
		norm_const.at(j) = 0.0;
		for (int i = 0; (i <= (para_other.n - 1)); i++) {
			norm_const.at(j) = norm_const.at(j) + distance_mat[i][j];
		}
		if (norm_const.at(j) < 0.0) { norm_const.at(j) = 0.0; } //if a few spatially distant individuals can get norm_const<0
	   //norm_const.at(j) = 1.0; // when no normalization
	}

	/*
	myfile.open((string(PATH2)+string("norm_const.txt")).c_str(),ios::app);
	for (int i=0;i<=(para_other.n-1);i++){
	myfile << norm_const.at(i) << endl;
	}
	myfile.close();
	*/

	//----------------------------------------------------------------------------------------//

	//initialise beta_ij_mat
	vector < vector<double> > beta_ij_mat(para_other.n, vector<double>(para_other.n)); // a dynamic matrix containing the "covariate pattern" effect
	initialize_beta_ij_mat(beta_ij_mat, para_other, epi_data, para_key);

	epi_func.set_para(para_key, para_other, rng_sim, distance_mat, beta_ij_mat);


	//int total_infected = -99;
	//int total_recovered = -99;

	vector<int> infected_source(para_other.n); //infection souces of the infected cases
	infected_source.assign(para_other.n, -99); // -99 indicates not being infected

	for (int i = 0; i <= ((int)index.size() - 1); i++) {
		infected_source.at(index.at(i)) = 9999; // 9999  indicates  being infected by background
	}

	vector<int> uninfected; // indicate who are uninfected
	vector<int> infectious; // indicate who are infectious (not yet recovered)
	vector<int> once_infectious; // indicate who have already been in class I (may or may not have recovered)

	/*int uninfected_array[n];
	for (int i=0;i<=(n-1);i++){
	uninfected_array[i] = i;
	}
	uninfected.assign(uninfected_array,uninfected_array+n);*/

	uninfected.reserve(para_other.n);
	once_infectious.reserve(para_other.n);
	infectious.reserve(para_other.n);

	for (int i = 0; i <= (para_other.n - 1); i++) {
		if (epi_data.status.at(i) == 1) uninfected.push_back(i);
		if (epi_data.status.at(i) == 3) infectious.push_back(i);
		if (epi_data.t_i.at(i) != para_other.unassigned_time) once_infectious.push_back(i);
	}





	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	// the simulation loop
	// --------------------------------------------------------------------------------
	// --------------------------------------------------------------------------------
	int loop_count = 0;


	while (t_now < para_other.t_max) {
		//while ((total_infected<n) & (t_now<t_max)){
	//for (int k=0;k<=7;k++){

		loop_count = loop_count + 1;


		////////////////////////////////////////////////////////////////////////////////////////////////////////
		epi_func.func_ric(mov_data, epi_data, uninfected, once_infectious, norm_const, t_now, beta_ij_mat, rng_sim); // calculate the remaining challenges needed to get infected
		epi_func.func_t_next(mov_data, epi_data, infectious, t_now, norm_const, loop_count, beta_ij_mat, rng_sim); // calculate the times of next event

		//t_now = *min_element(epi_data.t_next.begin(), epi_data.t_next.end());

		//find minimum t_next																								//find minimum of vector
		double t_next_min;
		t_next_min = *min_element(epi_data.t_next.begin(), epi_data.t_next.end());


		if (opt_mov_sim == 1) {
			//if a movement before next t_next then an exposure could occur at that point
			//check if any movements onto an S before t_next.(S)

			vector<int> xi_S_movt; //each S that received any movts from an I between t_now and t_next_min, used later to determine type of change at t_next
			for (int j = 0; j <= int(uninfected.size() - 1); j++) {

				int k_arg = uninfected.at(j);
				vector<double> t_ms_k; //timing of any movements onto k from an I in appropriate timeframes

				for (int m = 0; m <= (para_other.n_mov - 1); m++) {
					if (mov_data.to_k.at(m) == k_arg) {
						if ((epi_data.status.at(mov_data.from_k.at(m)) == 3) &&  //from an infective
							(mov_data.t_m.at(m) > t_now) &&						 //between t_now and next proposed event time
							(mov_data.t_m.at(m) <= t_next_min)) {
							t_ms_k.push_back(mov_data.t_m.at(m));
						}
					}
				}

				if (t_ms_k.size() > 0) {
					xi_S_movt.push_back(k_arg);

					double t_prop;
					t_prop = *min_element(t_ms_k.begin(), t_ms_k.end());
					//if so recalc its ric and t_next
					if (t_prop < t_next_min) {
						epi_func.func_ric_j(k_arg, mov_data, epi_data, uninfected, once_infectious, norm_const, t_prop, beta_ij_mat, rng_sim); // calculate the remaining challenges needed to get infected
						epi_func.func_t_next_j(k_arg, mov_data, epi_data, infectious, t_prop, norm_const, loop_count, beta_ij_mat, rng_sim); // calculate the times of next event
					}
				}

			}

			//recalc and set t_now as t_next_min
			t_next_min = *min_element(epi_data.t_next.begin(), epi_data.t_next.end());

		}

		t_now = t_next_min;


		////////////////////////////////////////////////////////////////////////////////////////////////////////

		if ((debug_sim == 1) & (loop_count == 100)) { //checksum based on t_now at 100th cycle,
			if (opt_betaij_sim == 0 && opt_mov_sim == 0) {
				cout << "t_now check (vanilla): " << 18.31168342630719069 - t_now << endl; //original
			}
			if (opt_betaij_sim == 1 && opt_mov_sim == 0) {
				cout << "t_now check (betaij): " << 28.0195892480458193 - t_now << endl; //original
			}
			if (opt_betaij_sim == 0 && opt_mov_sim == 1) {
				cout << "t_now check (movt): " << 26.17170870820149214 - t_now << endl;
			}
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		//assign t_now to appropriate place & updating the vectors regarding the infectious status & construct the transmission path
		switch ((t_now == para_other.unassigned_time) | (t_now >= para_other.t_max)) {

		case 0: {
			ind_now = min_max.min_position_double(epi_data.t_next, t_now, para_other.n);
			epi_func.func_status_update(epi_data, ind_now, rng_sim); //update the status
			epi_func.func_time_update(mov_data, epi_data, once_infectious, uninfected, infectious, infected_source, ind_now, t_now, norm_const, nt_data, con_seq, rng_sim);
			// myfile.open((string(path2)+string("t_i_after.txt")).c_str(),ios::app);
			// myfile << "k" << "," << "t_i" << "," << "status" << endl;
			// myfile << index.at(0)<< "," << epi_data.t_i.at(index.at(0)) << "," << epi_data.status.at(index.at(0)) << endl;
			// myfile.close();

			break;
		}

		case 1: {
			// do nothing
			break;
		}

		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		/*
		myfile.open((string(PATH2)+string("uninfected.txt")).c_str(),ios::app);
		myfile <<endl;
		myfile << "k" << "," <<"loop_count"<< endl;
		if (uninfected.size()>=1){
		for (int i=0; i<=((int)uninfected.size()-1);i++){
		myfile << uninfected.at(i) << "," <<loop_count <<" " ;
		}
		}
		myfile <<endl;
		myfile.close();


		myfile.open((string(PATH2)+string("infectious.txt")).c_str(),ios::app);
		myfile <<endl;
		myfile << "k" << "," << "loop_count" << endl;
		if (infectious.size()>=1){
		for (int i=0; i<=((int)infectious.size()-1);i++){
		myfile << infectious.at(i) << "," <<loop_count << " ";
		}
		}
		myfile <<endl;
		myfile.close();

		myfile.open((string(PATH2)+string("once_infectious.txt")).c_str(),ios::app);
		myfile <<endl;
		myfile << "k" << "," << "loop_count"  << endl;
		if (once_infectious.size()>=1){
		for (int i=0; i<=((int)once_infectious.size()-1);i++){
		myfile << once_infectious.at(i) << "," <<loop_count << " ";
		}
		}
		myfile <<endl;
		myfile.close();

		myfile.open((string(PATH2)+string("ric.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "ric" << "," <<"loop_count"<< endl;
		for(int i=0;i<=((int)epi_data.ric.size()-1);i++){
		myfile << epi_data.ric.at(i) << "," <<loop_count << endl;
		}
		myfile.close();


		myfile.open((string(PATH2)+string("t_next.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "t_next" << "," <<"loop_count" << endl;
		for(int i=0;i<=((int)epi_data.t_next.size()-1);i++){
		myfile << epi_data.t_next.at(i) << "," <<loop_count << endl;
		}
		myfile.close();

		myfile.open((string(PATH2)+string("t_now.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "t_now" << "," <<"loop_count"<< endl;
		myfile << t_now << "," <<loop_count << endl;
		myfile.close();

		myfile.open((string(PATH2)+string("ind_now.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "ind_now" << "," <<"loop_count"<< endl;
		for ( int i=0; i<=((int)ind_now.size()-1);i++){
		myfile <<  ind_now.at(i) << "," <<loop_count << endl;
		}
		myfile.close();

		myfile.open((string(PATH2)+string("epi_track.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "k" << "," << "q" << "," << "t_e" << "," << "t_i"<< "," << "t_r" << "," << "status"<< "," << "ric" << "," <<  "t_next" << "," <<"loop_count" << endl;
		for (int i=0; i<=(para_other.n-1);i++){
		myfile << epi_data.k.at(i) << "," << epi_data.q.at(i) << "," << epi_data.t_e.at(i) << "," << epi_data.t_i.at(i)<< "," << epi_data.t_r.at(i) << "," << epi_data.status.at(i)<< "," << epi_data.ric.at(i) << "," <<  epi_data.t_next.at(i) << "," <<loop_count << endl;
		}
		myfile.close();
		*/

		//total_infected = para_other.n - count(epi_data.t_e.begin(),epi_data.t_e.end(),para_other.unassigned_time);
		//total_recovered = para_other.n - count(epi_data.t_r.begin(),epi_data.t_r.end(),para_other.unassigned_time);

		/*
		myfile.open((string(PATH2)+string("total_infected.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "total_infected" << "," <<"loop_count" << endl;
		myfile << total_infected << "," <<loop_count << endl;
		myfile.close();

		myfile.open((string(PATH2)+string("total_recovered.txt")).c_str(),ios::app);
		myfile << endl;
		myfile << "total_recovered" << "," <<"loop_count" << endl;
		myfile << total_recovered << "," <<loop_count << endl;
		myfile.close();
		*/

		//uninfected.clear();
		//infectious.clear();
		//infected.clear();

	}// end while (total_infected/total_recovered<para_other.n & t_now<t_max)


	//-- sample the sequences which infected but not recovered and the algorithm does not sample them--//
	for (int i = 0; i <= (para_other.n - 1); i++) {
		if ((nt_data.ind_sample.at(i) == 0) & (epi_data.t_e.at(i) != para_other.unassigned_time) & (epi_data.t_r.at(i) == para_other.unassigned_time)) {
			epi_func.seq_update_source(nt_data, i, rng_sim);
			nt_data.t_nt[i].push_back(nt_data.t_sample.at(i));
			nt_data.ind_sample.at(i) = 1; // 1 = sampled
		}
	}

	//---- if not sampled, the sampling time drawn at infection is ineffective//
	int total_sampled = 0;

	for (int i = 0; i <= (para_other.n - 1); i++) {
		if (nt_data.ind_sample.at(i) == 0) nt_data.t_sample.at(i) = para_other.unassigned_time;
		if (nt_data.ind_sample.at(i) == 1) total_sampled = total_sampled + 1;
	}
	//----

	myfile.open((string(PATH2) + string("infected_source.txt")).c_str(), ios::app);
	for (int i = 0; i <= (para_other.n - 1); i++) {
		myfile << infected_source[i] << endl;
	}
	myfile.close();


	myfile.open((string(PATH2) + string("epi_sim.csv")).c_str(), ios::app);
	myfile << "k" << "," << "coor_x" << "," << "coor_y" << "," << "t_e" << "," << "t_i" << "," << "t_r" << "," << "ftype0" << "," "ftype1" << "," << "ftype2" << "," << "herdn" << endl;
	for (int i = 0; i <= (para_other.n - 1); i++) {
		myfile << epi_data.k.at(i) << "," << epi_data.coor_x.at(i) << "," << epi_data.coor_y.at(i) << "," << epi_data.t_e.at(i) << "," << epi_data.t_i.at(i) << "," << epi_data.t_r.at(i) << "," << epi_data.ftype0.at(i) << "," << epi_data.ftype1.at(i) << "," << epi_data.ftype2.at(i) << "," << epi_data.herdn.at(i) << endl;
	}
	myfile.close();

	if (opt_mov_sim == 1) {
		myfile.open((string(PATH2) + string("mov_inputs.csv")).c_str(), ios::app);
		myfile << "from_k" << "," << "to_k" << "," << "t_m" << endl;
		for (int i = 0; i <= (para_other.n_mov - 1); i++) {
			myfile << mov_data.from_k.at(i) << "," << mov_data.to_k.at(i) << "," << mov_data.t_m.at(i) << endl;
		}
		myfile.close();
	}
	/*
	myfile.open((string(PATH2) + string("epi_other.csv")).c_str(), ios::app);
	myfile << "q" << "," << "status" << ","  << endl;
	for (int i = 0; i <= (para_other.n - 1); i++) {
		myfile << epi_data.q.at(i) << "," << epi_data.status.at(i) << endl;
	}
	myfile.close();
	*/


	//--
	for (int i = 0; i <= (para_other.n - 1); i++) {

		if (infected_source[i] == 9999) {
			int total_mismatch = 0;
			for (int j = 0; j <= (para_other.n_base - 1); j++) {
				if (nt_data.nt[i][j] != con_seq.at(j)) total_mismatch = total_mismatch + 1;
			}
			//myfile.open((string(PATH2)+string("num_mismatch.txt")).c_str(),ios::app);
			//myfile << total_mismatch << endl;
			//myfile.close();
		}

	}

	/*----------------------------*/
	for (int i = 0; i <= (para_other.n - 1); i++) {

		ostringstream convert;
		convert << i;

		myfile.open((string(PATH2) + string("subject_").c_str() + string(convert.str()) + string("_t_nt.txt")).c_str(), ios::app);
		for (int j = 0; j <= (nt_data.current_size.at(i) - 1); j++) {
			myfile << nt_data.t_nt[i].at(j) << endl;
		}
		myfile.close();

		myfile.open((string(PATH2) + string("subject_").c_str() + string(convert.str()) + string("_nt.txt")).c_str(), ios::app);

		switch (int(para_other.partial_seq_out == 1)) {
		case 0: {// output full seq
			for (int j = 0; j <= (nt_data.current_size.at(i)*para_other.n_base - 1); j++) {
				int rem = (j + 1) % para_other.n_base;
				if ((rem != 0) | (j == 0)) myfile << nt_data.nt[i][j] << ",";
				if ((rem == 0) & (j != 0)) myfile << nt_data.nt[i][j] << " " << endl;
			}
			myfile.close();
			break;
		}

		case 1: {// output partial seq
			for (int j = 0; j <= (nt_data.current_size.at(i) - 1); j++) {
				int position_start = j * para_other.n_base;
				vector<int>seq_partial(para_other.n_base_part);
				seq_partial.assign(nt_data.nt.at(i).begin() + position_start, nt_data.nt.at(i).begin() + position_start + para_other.n_base_part);
				for (int k = 0; k <= (para_other.n_base_part - 1); k++) {
					int rem = (k + 1) % para_other.n_base_part;
					if ((rem != 0) | (k == 0)) myfile << seq_partial.at(k) << ",";
					if ((rem == 0) & (k != 0)) myfile << seq_partial.at(k) << " " << endl;
				}
			}
			myfile.close();

			//myfile.open((string(PATH2)+string("n_base_part.txt")).c_str(),ios::out);
			//myfile <<  "n_base_part"  << endl;
			//myfile << para_other.n_base_part  << endl;
			//myfile.close();

			break;
		}
		}

	}


	//myfile.open((string(PATH2)+string("final_size.txt")).c_str(),ios::app);
	//for (int i=0;i<=(para_other.n-1);i++){
	//myfile <<nt_data.current_size.at(i)<< endl;
	//}
	//myfile.close();






	myfile.open((string(PATH2) + string("t_sample.csv")).c_str(), ios::app);
	for (int i = 0; i <= (para_other.n - 1); i++) {
		myfile << nt_data.t_sample.at(i) << endl;
	}
	myfile.close();

	//myfile.open((string(PATH2)+string("log_f_S_sum.txt")).c_str(),ios::app);
	//myfile << nt_data.log_f_S<<endl;
	//myfile.close();

	//myfile.open((string(PATH2)+string("count_type.txt")).c_str(),ios::app);
	//myfile << nt_data.total_count_1 <<"," <<  nt_data.total_count_2 <<"," << nt_data.total_count_3 <<endl ;
	//myfile.close();


	/*----------------------------*/

	ifstream myfile_in;
	ofstream myfile_out;

	vector<int> xi_U, xi_E, xi_E_minus, xi_I, xi_R, xi_EnI, xi_InR; // indices sets indicating the individuals stay in S OR have gone through the other classes (E OR I OR R), and individuals hve gone through E but not I (EnI) and I but not R (InR)

	xi_U.reserve(para_other.n); //dynamically updated if necessary
	xi_E.reserve(para_other.n);
	xi_E_minus.reserve(para_other.n);
	xi_I.reserve(para_other.n);
	xi_R.reserve(para_other.n);
	xi_EnI.reserve(para_other.n);
	xi_InR.reserve(para_other.n);

	for (int i = 0; i <= (para_other.n - 1); i++) {
		if (epi_data.t_e.at(i) == para_other.unassigned_time) xi_U.push_back(i);
		if (epi_data.t_e.at(i) != para_other.unassigned_time) xi_E.push_back(i);
		if (epi_data.t_i.at(i) != para_other.unassigned_time) xi_I.push_back(i);
		if (epi_data.t_r.at(i) != para_other.unassigned_time) xi_R.push_back(i);
	}

	xi_E_minus = xi_E;
	for (int i = 0; i <= (int)(index.size() - 1); i++) {
		xi_E_minus.erase(find(xi_E_minus.begin(), xi_E_minus.end(), index.at(i)));
	} // E set excluding index

	xi_EnI = xi_E;
	for (int i = 0; i <= (int)(xi_I.size() - 1); i++) {
		xi_EnI.erase(find(xi_EnI.begin(), xi_EnI.end(), xi_I.at(i)));
	} // E set excluding I

	xi_InR = xi_I;
	for (int i = 0; i <= (int)(xi_R.size() - 1); i++) {
		xi_InR.erase(find(xi_InR.begin(), xi_InR.end(), xi_R.at(i)));
	} // I set excluding R


	int num_import = 0;
	for (int i = 0; i <= ((int)xi_E.size() - 1); i++) {
		if (infected_source.at(i) == 9999) num_import = num_import + 1;
	}
	//myfile_out.open((string(PATH2)+string("num_import.txt")).c_str(),ios::out);
	//myfile_out << num_import;
	//myfile_out.close();

	/*
	myfile_out.open((string(PATH2)+string("xi_E_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_E.empty()!=1){
	for (int i=0; i<=((int)xi_E.size()-1);i++){
	myfile_out << xi_E.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_E.size();
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("xi_E_minus_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_E_minus.empty()!=1){
	for (int i=0; i<=((int)xi_E_minus.size()-1);i++){
	myfile_out << xi_E_minus.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_E_minus.size();
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("xi_U_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_U.empty()!=1){
	for (int i=0; i<=((int)xi_U.size()-1);i++){
	myfile_out << xi_U.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_U.size();
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("xi_I_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_I.empty()!=1){
	for (int i=0; i<=((int)xi_I.size()-1);i++){
	myfile_out << xi_I.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_I.size();
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("xi_EnI_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_EnI.empty()!=1){
	for (int i=0; i<=((int)xi_EnI.size()-1);i++){
	myfile_out << xi_EnI.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_EnI.size();
	myfile_out.close();

	myfile_out.open((string(PATH2)+string("xi_InR_true.txt")).c_str(),ios::out);
	myfile_out << "k" << endl;
	if (xi_InR.empty()!=1){
	for (int i=0; i<=((int)xi_InR.size()-1);i++){
	myfile_out << xi_InR.at(i) << endl;
	}
	}
	myfile_out << "size" << endl;
	myfile_out << xi_InR.size();
	myfile_out.close();


	myfile_out.open((string(PATH2)+string("EnI_perct.txt")).c_str(),ios::out);
	myfile_out << (double)xi_EnI.size()/(double)xi_E.size() ;
	myfile_out.close();
	*/
	myfile_out.open((string(PATH2) + string("sampled_perct.txt")).c_str(), ios::out);
	myfile_out << ((double)total_sampled) / (double)xi_E.size();
	myfile_out.close();

	/*----------------------------*/


 
  /*----------------------------*/
  /*----------------------------*/
  /*----------------------------*/
  //  Returns to R ----------------------------------------------
	vector<double> r_epi(epi_data.k.begin(), epi_data.k.end());
	r_epi.insert(r_epi.end(), epi_data.coor_x.begin(), epi_data.coor_x.end());
	r_epi.insert(r_epi.end(), epi_data.coor_y.begin(), epi_data.coor_y.end());
	r_epi.insert(r_epi.end(), epi_data.t_e.begin(), epi_data.t_e.end());
	r_epi.insert(r_epi.end(), epi_data.t_i.begin(), epi_data.t_i.end());
	r_epi.insert(r_epi.end(), epi_data.t_r.begin(), epi_data.t_r.end());
	r_epi.insert(r_epi.end(), epi_data.ftype0.begin(), epi_data.ftype0.end());
	r_epi.insert(r_epi.end(), epi_data.ftype1.begin(), epi_data.ftype1.end());
	r_epi.insert(r_epi.end(), epi_data.ftype2.begin(), epi_data.ftype2.end());
	r_epi.insert(r_epi.end(), epi_data.herdn.begin(), epi_data.herdn.end());
	Rcpp::NumericVector r_epi_v(r_epi.begin(), r_epi.end());
	r_epi_v.attr("dim") = Rcpp::Dimension(epi_data.k.size(), 10);
	Rcpp::NumericMatrix r_epi_m = as<Rcpp::NumericMatrix>(r_epi_v);
	colnames(r_epi_m) = CharacterVector::create("k", "coor_x", "coor_y", "t_e", "t_i", "t_r",
		"ftype0", "ftype1", "ftype2", "herdn");

	/*----------------------------*/
	/*----------------------------*/
	return Rcpp::List::create(
		Rcpp::Named("epi.sim") = r_epi_m,
		Rcpp::Named("infected_source") = infected_source,
		Rcpp::Named("t_sample") = nt_data.t_sample,
		Rcpp::Named("sampled_perct") = ((double)total_sampled) / (double)xi_E.size(),
		Rcpp::Named("cons_seq") = con_seq);
}

