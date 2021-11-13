#!/usr/bin/python3
import GRiD.util as util
import GRiD.URDFParser.URDFParser as URDFParser
import GRiD.RBDReference as RBDReference
import GRiD.GRiDCodeGenerator.GRiDCodeGenerator as GRiDCodeGenerator
import subprocess
import sys

def main():
    inputs = util.parseInputs(NO_ARG_OPTION = True)
    if not inputs is None:
        URDF_PATH, DEBUG_MODE = inputs
        parser = URDFParser()
        robot = parser.parse(URDF_PATH)
        Imats = robot.get_Imats_ordered_by_id()
        print(Imats[2])
        # for i in range(len(Imats)):
        #     print(i)
        #     print(Imats[i])
        #     print()

if __name__ == "__main__":
    main()