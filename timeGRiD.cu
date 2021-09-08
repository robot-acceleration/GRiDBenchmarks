/***
nvcc -std=c++11 -o timeGRiD.exe timeGRiD.cu -gencode arch=compute_86,code=sm_86 -O3 -ftz=true -prec-div=false -prec-sqrt=false
***/

#include "util/experiment_helpers.h" // include constants and other experiment consistency helpers
#include "grid.cuh"

dim3 dimms(grid::SUGGESTED_THREADS,1,1); // all loops are single loops (all mat mult flattened into column opps)
#define GRAVITY 9.81

template <typename T, int TEST_ITERS>
__host__
void test(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
   	#if TEST_FOR_EQUIVALENCE
		printf("q,qd,u\n");
		printMat<T,1,grid::NUM_JOINTS>(hd_data->h_q_qd_u,1);
		printMat<T,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[grid::NUM_JOINTS],1);
		printMat<T,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[2*grid::NUM_JOINTS],1);

		grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
		grid::direct_minv<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
		grid::forward_dynamics<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
		grid::inverse_dynamics_gradient<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
		grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);

		printf("c\n");
		printMat<T,1,grid::NUM_JOINTS>(hd_data->h_c,1);

		printf("Minv\n");
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_Minv,grid::NUM_JOINTS);

		printf("qdd\n");
		printMat<T,1,grid::NUM_JOINTS>(hd_data->h_qdd,1);

		printf("dc_dq\n");
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_dc_du,grid::NUM_JOINTS);

		printf("dc_dqd\n");
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_dc_du[grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);

		printf("df_dq\n");
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_df_du,grid::NUM_JOINTS);

		printf("df_dqd\n");
		printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_df_du[grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);
		
   	#else
		// Setup timer
	   	struct timespec start, end;
	   	std::vector<double> times = {};

		if(NUM_TIMESTEPS == 1){
			// first one is done twice to wake up the GPU and get it up to full speed
			grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
    		grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::direct_minv_single_timing<T,true>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::forward_dynamics_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::inverse_dynamics_gradient_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::forward_dynamics_gradient_single_timing<T,false>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
		}
		else{
			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::direct_minv<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: Minv WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::direct_minv_compute_only<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: Minv COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_gradient<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID_DU WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_gradient_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID_DU COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD_DU WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_gradient_compute_only<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD_DU COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();
		}
	#endif
}

template<typename T, int TEST_ITERS>
void run_all_tests(){
	// allocate memory for max of what we need
	const int MAX_TIMESTEPS = 256;
	cudaStream_t *streams = grid::init_grid<T>();
	grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
	grid::gridData<T> *hd_data = grid::init_gridData<T,MAX_TIMESTEPS>();

	// load q,qd,u
	for(int k = 0; k < MAX_TIMESTEPS; k++){
		for(int ind = 0; ind < grid::NUM_JOINTS; ind++){
			// get values
			T val1 = getRand<double>(); T val2 = getRand<double>(); T val3 = getRand<double>();
			hd_data->h_q_qd_u[k*3*grid::NUM_JOINTS + ind] = val1;
			hd_data->h_q_qd_u[k*3*grid::NUM_JOINTS + grid::NUM_JOINTS + ind] = val2;
			hd_data->h_q_qd_u[k*3*grid::NUM_JOINTS + 2*grid::NUM_JOINTS + ind] = val3;
			// load into alternate memory sizes
			hd_data->h_q_qd[k*2*grid::NUM_JOINTS + ind] = val1;
			hd_data->h_q_qd[k*2*grid::NUM_JOINTS + grid::NUM_JOINTS + ind] = val2;
			hd_data->h_q[k*grid::NUM_JOINTS + ind] = val1;
		}
	}
	// copy values onto the GPU as default values (we will do more transfers later but this ensures things are initialized)
	gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(hd_data->d_q_qd,hd_data->h_q_qd,2*grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(hd_data->d_q,hd_data->h_q,grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// then run the tests
	test<T,TEST_ITERS*10>(1,streams,d_robotModel,hd_data); // more iters for single test
	#if !TEST_FOR_EQUIVALENCE
		test<T,TEST_ITERS>(16,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(32,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(64,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(128,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(256,streams,d_robotModel,hd_data);
	#endif
	
	// free all memory and exit
	grid::close_grid<T>(streams,d_robotModel,hd_data);
}

int main(void){
	run_all_tests<float,TEST_ITERS_GLOBAL>(); return 0;
}