#!/usr/bin/python3
import GRiD.util as util
import GRiD.URDFParser.URDFParser as URDFParser
import GRiD.GRiDCodeGenerator.GRiDCodeGenerator as GRiDCodeGenerator
import subprocess
import sys

def main():
    inputs = util.parseInputs(NO_ARG_OPTION = True)
    if not inputs is None:
        URDF_PATH, DEBUG_MODE = inputs
        parser = URDFParser()
        robot = parser.parse(URDF_PATH)

        util.validateRobot(robot, NO_ARG_OPTION = True)

        codegen = GRiDCodeGenerator(robot,DEBUG_MODE,True)
        print("-----------------")
        print("Generating GRiD.cuh")
        print("-----------------")
        codegen.gen_all_code()
        print("New code generated and saved to grid.cuh!")

    print("-----------------")
    print("Compiling timeGRiD")
    print("-----------------")
    result = subprocess.run( \
        ["nvcc", "-std=c++11", "-o", "timeGRiD.exe", "timeGRiD.cu", \
         "-gencode", "arch=compute_86,code=sm_86", \
         "-O3", "-ftz=true", "-prec-div=false", "-prec-sqrt=false"], \
        capture_output=True, text=True \
    )
    if result.stderr:
        print("Compilation errors follow:")
        print(result.stderr)
        exit()

    print("-----------------")
    print("Running timeGRiD")
    print("-----------------")
    print("This may take a few minutes....")
    print("     Outputs will show up at the end")
    print("     ID single will show up twice as")
    print("        this is used to warm up the GPU")
    print("-----------------")
    result = subprocess.run(["./timeGRiD.exe"], capture_output=True, text=True)
    if result.stderr:
        print("Runtime errors follow:")
        print(result.stderr)
        exit()

    print(result.stdout)

if __name__ == "__main__":
    main()