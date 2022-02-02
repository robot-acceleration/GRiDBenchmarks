/*
 * This timing code is based on the benchmarking code as written in the Pinocchio repository
 * clang++-12 -std=c++11 -o timePinocchio.exe timePinocchio.cpp -O3 -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_WITH_URDFDOM -lboost_system -lpinocchio -lurdfdom_model -lpthread -ldl
 * eample usage: timePinocchio.exe urdfs/atlas.urdf
 */
#include "util/experiment_helpers.h" // include constants and other experiment consistency helpers
#include "ReusableThreads/ReusableThreads.h" // multi-threading wrapper

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"

#include "pinocchio/codegen/cppadcg.hpp"
#include "pinocchio/codegen/code-generator-algo.hpp"

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"

#include "pinocchio/container/aligned-vector.hpp"

#include <Eigen/StdVector>
// EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::VectorXd)

using namespace Eigen;
using namespace pinocchio;

#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))

template<typename T>
void inverseDynamicsThreaded_codegen_inner(CodeGenRNEA<T> *rnea_code_gen, int nq, int nv, \
                                           Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, int tid, int kStart, int kMax){
    Matrix<T, Dynamic, 1> zeros = Matrix<T, Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void inverseDynamicsThreaded_codegen(CodeGenRNEA<T> **rnea_code_gen_arr, int nq, int nv, \
                                     Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, ReusableThreads<NUM_THREADS> *threads){
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
            threads->addTask(tid, &inverseDynamicsThreaded_codegen_inner<T>, std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void minvThreaded_codegen_inner(CodeGenMinv<T> *minv_code_gen, int nq, int nv, Matrix<T, Dynamic, 1> *qs, int tid, int kStart, int kMax){
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void minvThreaded_codegen(CodeGenMinv<T> **minv_code_gen_arr, int nq, int nv, Matrix<T, Dynamic, 1> *qs, ReusableThreads<NUM_THREADS> *threads){
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
            threads->addTask(tid, &minvThreaded_codegen_inner<T>, std::ref(minv_code_gen_arr[tid]), nq, nv, std::ref(qs), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void forwardDynamicsThreaded_codegen_inner(CodeGenMinv<T> *minv_code_gen, CodeGenRNEA<T> *rnea_code_gen, int nq, int nv, \
                                           Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, Matrix<T, Dynamic, 1> *qdds, \
                                           Matrix<T, Dynamic, 1> *us, int tid, int kStart, int kMax){
    Matrix<T, Dynamic, 1> zeros = Matrix<T, Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
        minv_code_gen->Minv.template triangularView<Eigen::StrictlyLower>() = 
            minv_code_gen->Minv.transpose().template triangularView<Eigen::StrictlyLower>();
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
        qdds[k].noalias() = minv_code_gen->Minv*(us[k] - rnea_code_gen->res);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void forwardDynamicsThreaded_codegen(CodeGenMinv<T> **minv_code_gen_arr, CodeGenRNEA<T> **rnea_code_gen_arr, int nq, int nv, \
                                     Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, Matrix<T, Dynamic, 1> *qdds, \
                                     Matrix<T, Dynamic, 1> *us, ReusableThreads<NUM_THREADS> *threads){
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
            threads->addTask(tid, &forwardDynamicsThreaded_codegen_inner<T>, std::ref(minv_code_gen_arr[tid]), 
                                                                             std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                             std::ref(qs), std::ref(qds), std::ref(qdds), std::ref(us), 
                                                                             tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void inverseDynamicsGradientThreaded_codegen_inner(CodeGenRNEADerivatives<T> *rnea_derivatives_code_gen, \
                                                   int nq, int nv, Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, \
                                                   int tid, int kStart, int kMax){
    Matrix<T, Dynamic, 1> zeros = Matrix<T, Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        rnea_derivatives_code_gen->evalFunction(qs[k],qds[k],zeros);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void inverseDynamicsGradientThreaded_codegen(CodeGenRNEADerivatives<T> **rnea_derivatives_code_gen_arr, \
                                             int nq, int nv, Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, 
                                             ReusableThreads<NUM_THREADS> *threads){
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
            threads->addTask(tid, &inverseDynamicsGradientThreaded_codegen_inner<T>, std::ref(rnea_derivatives_code_gen_arr[tid]), 
                                                                                     nq, nv, std::ref(qs), std::ref(qds), tid, kStart, kMax);
        }
        threads->sync();
}

template<typename T>
void forwardDynamicsGradientThreaded_codegen_inner(CodeGenRNEADerivatives<T> *rnea_derivatives_code_gen, \
                                                   CodeGenMinv<T> *minv_code_gen, CodeGenRNEA<T> *rnea_code_gen, \
                                                   int nq, int nv, Matrix<T, Dynamic, Dynamic> *dqdd_dqs, Matrix<T, Dynamic, Dynamic> *dqdd_dvs, \
                                                   Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, Matrix<T, Dynamic, 1> *us, \
                                                   int tid, int kStart, int kMax){
    Matrix<T, Dynamic, 1> zeros = Matrix<T, Dynamic, 1>::Zero(nv);
    for(int k = kStart; k < kMax; k++){
        minv_code_gen->evalFunction(qs[k]);
        minv_code_gen->Minv.template triangularView<Eigen::StrictlyLower>() = 
            minv_code_gen->Minv.transpose().template triangularView<Eigen::StrictlyLower>();
        rnea_code_gen->evalFunction(qs[k],qds[k],zeros);
        Matrix<T, Dynamic, 1> qdd = minv_code_gen->Minv*(us[k] - rnea_code_gen->res);
        rnea_derivatives_code_gen->evalFunction(qs[k],qds[k],qdd);
        dqdd_dqs[k].noalias() = -(minv_code_gen->Minv)*(rnea_derivatives_code_gen->dtau_dq);
        dqdd_dvs[k].noalias() = -(minv_code_gen->Minv)*(rnea_derivatives_code_gen->dtau_dv);
    }
}

template<typename T, int NUM_THREADS, int NUM_TIME_STEPS>
void forwardDynamicsGradientThreaded_codegen(CodeGenRNEADerivatives<T> **rnea_derivatives_code_gen_arr, \
                                             CodeGenMinv<T> **minv_code_gen_arr, CodeGenRNEA<T> **rnea_code_gen_arr, \
                                             int nq, int nv, Matrix<T, Dynamic, Dynamic> *dqdd_dqs, Matrix<T, Dynamic, Dynamic> *dqdd_dvs, \
                                             Matrix<T, Dynamic, 1> *qs, Matrix<T, Dynamic, 1> *qds, Matrix<T, Dynamic, 1> *us, \
                                             ReusableThreads<NUM_THREADS> *threads){
        for (int tid = 0; tid < NUM_THREADS; tid++){
            int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
            if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
            threads->addTask(tid, &forwardDynamicsGradientThreaded_codegen_inner<T>, std::ref(rnea_derivatives_code_gen_arr[tid]), 
                                                                                     std::ref(minv_code_gen_arr[tid]), 
                                                                                     std::ref(rnea_code_gen_arr[tid]), nq, nv,
                                                                                     std::ref(dqdd_dqs), std::ref(dqdd_dvs), 
                                                                                     std::ref(qs), std::ref(qds), std::ref(us),
                                                                                     tid, kStart, kMax);
    }
        threads->sync();
}

template<typename T, int TEST_ITERS, int NUM_THREADS, int NUM_TIME_STEPS>
void test(std::string urdf_filepath){
    // Setup timer
    struct timespec start, end;

    // Matrix typedefs
    typedef Matrix<T, Dynamic, Dynamic> MatrixXT;
    typedef Matrix<T, Dynamic, 1> VectorXT;

    // Import URDF model and prepare pinnochio
    Model model;
    pinocchio::urdf::buildModel(urdf_filepath,model);
    // model.gravity.setZero();
    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
    Data datas[NUM_THREADS]; 
    for(int i = 0; i < NUM_THREADS; i++){datas[i] = Data(model);}

    // generate the code_gen
    CodeGenRNEA<T> rnea_code_gen(model.cast<T>());
    rnea_code_gen.initLib();
    rnea_code_gen.loadLib();

    CodeGenRNEA<T> *rnea_code_gen_arr[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++){
        rnea_code_gen_arr[i] = new CodeGenRNEA<T>(model.cast<T>());
        rnea_code_gen_arr[i]->initLib();
        rnea_code_gen_arr[i]->loadLib();
    }

    CodeGenMinv<T> minv_code_gen(model.cast<T>());
    minv_code_gen.initLib();
    minv_code_gen.loadLib();

    CodeGenMinv<T> *minv_code_gen_arr[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++){
        minv_code_gen_arr[i] = new CodeGenMinv<T>(model.cast<T>());
        minv_code_gen_arr[i]->initLib();
        minv_code_gen_arr[i]->loadLib();
    }

    CodeGenRNEADerivatives<T> rnea_derivatives_code_gen(model.cast<T>());
    rnea_derivatives_code_gen.initLib();
    rnea_derivatives_code_gen.loadLib();

    CodeGenRNEADerivatives<T> *rnea_derivatives_code_gen_arr[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++){
        rnea_derivatives_code_gen_arr[i] = new CodeGenRNEADerivatives<T>(model.cast<T>());
        rnea_derivatives_code_gen_arr[i]->initLib();
        rnea_derivatives_code_gen_arr[i]->loadLib();
    }

    // allocate and load on CPU
    VectorXT qs[NUM_TIME_STEPS];
    VectorXT qds[NUM_TIME_STEPS];
    VectorXT qdds[NUM_TIME_STEPS];
    VectorXT us[NUM_TIME_STEPS];
    MatrixXT dqdd_dqs[NUM_TIME_STEPS];
    MatrixXT dqdd_dvs[NUM_TIME_STEPS];
    for(int i = 0; i < NUM_TIME_STEPS; i++){
        qs[i] = VectorXT::Zero(model.nq);
        qds[i] = VectorXT::Zero(model.nv);
        qdds[i] = VectorXT::Zero(model.nv);
        us[i] = VectorXT::Zero(model.nv);
        dqdd_dqs[i] = MatrixXT::Zero(model.nv,model.nq);
        dqdd_dvs[i] = MatrixXT::Zero(model.nv,model.nv);
        for(int j = 0; j < model.nq; j++){qs[i][j] = getRand<T>(); qds[i][j] = getRand<T>(); us[i][j] = getRand<T>();}
    }

    #if TEST_FOR_EQUIVALENCE
        std::cout << "q,qd,u" << std::endl;
        std::cout << qs[0].transpose() << std::endl;
        std::cout << qds[0].transpose() << std::endl;
        std::cout << us[0].transpose() << std::endl;
        // Minv
        computeMinverse(model,datas[0],qs[0]); 
        std::cout << "Minv" << std::endl << datas[0].Minv << std::endl;
        datas[0].Minv.template triangularView<Eigen::StrictlyLower>() = 
            datas[0].Minv.transpose().template triangularView<Eigen::StrictlyLower>();
        MatrixXT Minv = datas[0].Minv;
        // qdd
        aba(model,datas[0],qs[0],qds[0],us[0]);
        qdds[0] = datas[0].ddq;
        std::cout << "qdd" << std::endl << qdds[0].transpose() << std::endl;
        // dc/du with qdd=0
        MatrixXT drnea_dq = MatrixXT::Zero(model.nq,model.nq);
        MatrixXT drnea_dv = MatrixXT::Zero(model.nv,model.nv);
        MatrixXT drnea_da = MatrixXT::Zero(model.nv,model.nv);
        computeRNEADerivatives(model,datas[0],qs[0],qds[0],VectorXT::Zero(model.nv),drnea_dq,drnea_dv,drnea_da);
        std::cout << "dc_dq" << std::endl << drnea_dq << std::endl;
        std::cout << "dc_dqd" << std::endl << drnea_dv << std::endl;
        // df/du (via dc/du with qdd)
        computeRNEADerivatives(model,datas[0],qs[0],qds[0],qdds[0],drnea_dq,drnea_dv,drnea_da);
        dqdd_dqs[0] = -Minv*drnea_dq;
        dqdd_dvs[0] = -Minv*drnea_dv;
        std::cout << "df_dq" << std::endl << dqdd_dqs[0] << std::endl;
        std::cout << "df_dqd" << std::endl << dqdd_dvs[0] << std::endl;
    #else
        // Single call
        if(NUM_TIME_STEPS == 1){
            VectorXT zeros = VectorXT::Zero(model.nv);

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                rnea_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("ID codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                minv_code_gen.evalFunction(qs[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("Minv codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                minv_code_gen.evalFunction(qs[0]);
                minv_code_gen.Minv.template triangularView<Eigen::StrictlyLower>() = 
                    minv_code_gen.Minv.transpose().template triangularView<Eigen::StrictlyLower>();
                rnea_code_gen.evalFunction(qs[0],qds[0],zeros);
                qdds[0].noalias() = minv_code_gen.Minv*(us[0] - rnea_code_gen.res);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("FD codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                rnea_derivatives_code_gen.evalFunction(qs[0],qds[0],qdds[0]);
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("ID_DU codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

            clock_gettime(CLOCK_MONOTONIC,&start);
            for(int i = 0; i < TEST_ITERS; i++){
                minv_code_gen.evalFunction(qs[0]);
                minv_code_gen.Minv.template triangularView<Eigen::StrictlyLower>() = 
                    minv_code_gen.Minv.transpose().template triangularView<Eigen::StrictlyLower>();
                rnea_code_gen.evalFunction(qs[0],qds[0],zeros);
                VectorXT qdd = minv_code_gen.Minv*(us[0] - rnea_code_gen.res);
                rnea_derivatives_code_gen.evalFunction(qs[0],qds[0],qdd);
                dqdd_dqs[0].noalias() = -minv_code_gen.Minv*rnea_derivatives_code_gen.dtau_dq;
                dqdd_dvs[0].noalias() = -minv_code_gen.Minv*rnea_derivatives_code_gen.dtau_dv;
            }
            clock_gettime(CLOCK_MONOTONIC,&end);
            printf("FD_DU codegen %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(TEST_ITERS));

        }
        // multi call with threadPools
        else{
            ReusableThreads<NUM_THREADS> threads;
            std::vector<double> times = {};

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                inverseDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_code_gen_arr,
                                                                              model.nq,model.nv,qs,qds,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: ID codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                minvThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,model.nq,model.nv,qs,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: Minv codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                forwardDynamicsThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(minv_code_gen_arr,rnea_code_gen_arr,
                                                                              model.nq,model.nv,qs,qds,qdds,us,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: FD codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                inverseDynamicsGradientThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_derivatives_code_gen_arr,
                                                                                      model.nq,model.nv,qs,qds,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: ID_DU codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");

            for(int iter = 0; iter < TEST_ITERS; iter++){
                clock_gettime(CLOCK_MONOTONIC,&start);
                forwardDynamicsGradientThreaded_codegen<T,NUM_THREADS,NUM_TIME_STEPS>(rnea_derivatives_code_gen_arr,
                                                                                    minv_code_gen_arr,rnea_code_gen_arr,
                                                                                    model.nq,model.nv,dqdd_dqs,dqdd_dvs,
                                                                                    qs,qds,us,&threads);
                clock_gettime(CLOCK_MONOTONIC,&end);
                times.push_back(time_delta_us_timespec(start,end));
            }
            printf("[N:%d]: FD_DU codegen: ",NUM_TIME_STEPS); printStats(&times); times.clear();
            printf("----------------------------------------\n");
        }
    #endif

    // make sure to delete objs
    for (int i = 0; i < NUM_THREADS; i++){delete rnea_derivatives_code_gen_arr[i];}// delete rnea_code_gen_arr[i]; delete minv_code_gen_arr[i];}
}

template<typename T, int TEST_ITERS, int CPU_THREADS>
void run_all_tests(std::string urdf_filepath){
    test<T,10*TEST_ITERS,CPU_THREADS,1>(urdf_filepath);
    #if !TEST_FOR_EQUIVALENCE
        test<T,TEST_ITERS,CPU_THREADS,16>(urdf_filepath);
        test<T,TEST_ITERS,CPU_THREADS,32>(urdf_filepath);
        test<T,TEST_ITERS,CPU_THREADS,64>(urdf_filepath);
        test<T,TEST_ITERS,CPU_THREADS,128>(urdf_filepath);
        test<T,TEST_ITERS,CPU_THREADS,256>(urdf_filepath);
    #endif
}

int main(int argc, const char ** argv){
    std::string urdf_filepath;
    if(argc>1){urdf_filepath = argv[1];}
    else{printf("Usage is: urdf_filepath\n"); return 1;}

    run_all_tests<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL>(urdf_filepath);
    return 0;
}