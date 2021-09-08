#!/usr/bin/python3
import GRiD.util as util
import subprocess
import sys

def main():
    URDF_PATH, DEBUG_MODE = util.parseInputs()
    util.validateFile(URDF_PATH)

    if util.fileExists("timePinocchio.exe"):
        print("-----------------")
        print("Found timePinocchio binary")
        print("-----------------")        
    else:
        print("-----------------")
        print("Compiling timePinocchio")
        print("   this may take a few minutes")
        print("-----------------")
        result = subprocess.run( \
            ["clang++-12", "-std=c++11", "-o", "timePinocchio.exe", "timePinocchio.cpp", "-O3", \
             "-DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR", "-DPINOCCHIO_WITH_URDFDOM", \
             "-lboost_system", "-lpinocchio", "-lurdfdom_model", "-lpthread", "-ldl"], \
            capture_output=True, text=True \
        )
        if result.stderr:
            print("Compilation errors follow:")
            print(result.stderr)
            exit()

    print("-----------------")
    print("Running timePinocchio")
    print("-----------------")
    print("This may take a few minutes....")
    print("     Outputs will show up at the end")
    print("-----------------")
    result = subprocess.run(["./timePinocchio.exe", URDF_PATH], capture_output=True, text=True)
    if result.stderr:
        print("Runtime errors follow:")
        print(result.stderr)
        exit()

    print(result.stdout)

if __name__ == "__main__":
    main()