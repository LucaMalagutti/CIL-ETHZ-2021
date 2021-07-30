"""
Generates a bash script that converts data to LibFM format, see instructions in readme.md
"""

import os
import sys


def main(args):
    DEBUG = False
    features = args[1]
    script_file = "script"
    script_file += features + ".sh"

    with open(script_file, "w") as outf:
        BayesianSVD_dir = os.path.abspath(os.path.dirname(script_file))

        # convert to binary format
        outf.write("cd /home/libfm/\n")
        for split in ["90aug.train", "100aug.train", "aug.valid", "aug.submission"]:
            for block in ["user", "item"]:
                command_str = "./bin/convert -ifile "
                file_str = BayesianSVD_dir
                file_str += "/data_features/CIL_data"
                file_str += split
                file_str += ".libfmrel_" + block
                command_str += file_str + ".libfm"
                command_str += " -ofilex " + file_str + ".x"
                command_str += " -ofiley " + BayesianSVD_dir + "/data_features/totrash.y"
                outf.write(command_str + "\n")

                # generate transposed matrices
                command_str = "./bin/transpose -ifile "
                file_str = BayesianSVD_dir
                file_str += "/data_features/CIL_data"
                file_str += split
                file_str += ".libfmrel_" + block
                command_str += file_str + ".x"
                command_str += " -ofile " + file_str + ".xt"
                outf.write(command_str + "\n")

        # delete all .libfm files in data_features/ (as they are already converted to .x and .xt files)
        # delete all files *.valid.*.x , *.valid.*.xt , *.valid.*.groups, *.submission.*.x , *.submission.*.xt , *.submission.*.groups (KEEP .test files)
        outf.write("cd " + BayesianSVD_dir + "/data_features\n")
        for file in os.listdir(BayesianSVD_dir + "/data_features"):
            if file[-6:] == ".libfm":
                outf.write("rm " + file + "\n")
            elif file[-5:] != ".test" and (".valid." in file or ".submission." in file):
                outf.write("rm " + file + "\n")

        # delete totrash.y
        outf.write("rm totrash.y\n")

        # move data90aug and dataaug.valid files to /Train90Val10
        # move data100aug and dataaug.submission files to /Train100Submission
        outf.write("mkdir Train90Val10\n")
        outf.write("mkdir Train100Submission\n")
        for file in os.listdir(BayesianSVD_dir + "/data_features"):
            if file[-6:] != ".libfm" and "totrash" not in file and not (file[-5:] != ".test" and (".valid." in file or ".submission." in file)):
                if "data90aug" in file or "valid" in file:
                    outf.write("mv " + file + " Train90Val10\n")
                elif "data100aug" in file or "submission" in file:
                    outf.write("mv " + file + " Train100Submission\n")
                else:
                    print("non valid file found: "+file)
        for split in ["90aug.train", "100aug.train", "aug.valid", "aug.submission"]:
            if split == "90aug.train" or split == "aug.valid":
                target_dir = "Train90Val10"
            else:
                target_dir = "Train100Submission"
            for block in ["user", "item"]:
                for ext in [".x",".xt"]:
                    outf.write("mv CIL_data" + split + ".libfmrel_" + block + ext + " " + target_dir + "\n")

        # in both /Train90Val10 and /Train100Submission move *y.train and *y.test files to ./y
        # in /Train90Val10 rename all files *item.* as CIL_data90aug.rel_item.* (keep the file extension)
        # in /Train90Val10 rename all files *user.* as CIL_data90aug.rel_user.* (keep the file extension)
        # in /Train100Submission rename all files *item.* as CIL_data100aug.rel_item.* (keep the file extension)
        # in /Train100Submission rename all files *user.* as CIL_data100aug.rel_user.* (keep the file extension)
        outf.write("mkdir Train90Val10/y\n")
        outf.write("mkdir Train100Submission/y\n")
        for file in os.listdir(BayesianSVD_dir + "/data_features"):
            print("FILE:",file)
            toprint = False
            if file[-6:] != ".libfm" and "totrash" not in file and not (file[-5:] != ".test" and (".valid." in file or ".submission." in file)):
                print("A")
                if "data90aug" in file or "valid" in file:
                    print("B")
                    if "y." in file:
                        outf.write("mv Train90Val10/" + file + " Train90Val10/y\n")
                    elif "item." in file:
                        outf.write("mv Train90Val10/" + file + " Train90Val10/CIL_data90aug.rel_item" + os.path.splitext(file)[1] + "\n")
                    elif "user." in file:
                        outf.write("mv Train90Val10/" + file + " Train90Val10/CIL_data90aug.rel_user" + os.path.splitext(file)[1] + "\n")
                    else:
                        toprint = True
                elif "data100aug" in file or "submission" in file:
                    print("C")
                    if "y." in file:
                        outf.write("mv Train100Submission/" + file + " Train100Submission/y\n")
                    elif "item." in file:
                        outf.write("mv Train100Submission/" + file + " Train100Submission/CIL_data100aug.rel_item" + os.path.splitext(file)[1] + "\n")
                    elif "user." in file:
                        outf.write("mv Train100Submission/" + file + " Train100Submission/CIL_data100aug.rel_user" + os.path.splitext(file)[1] + "\n")
                    else:
                        toprint = True
                else:
                    print("D")
                    toprint = True
            else:
                print("E")
                toprint = True

            if toprint == True and DEBUG:
                print("File", file, "will not be moved or renamed, could be removed")

        for split in ["90aug.train", "100aug.train", "aug.valid", "aug.submission"]:
            if split == "90aug.train" or split == "aug.valid":
                target_dir = "Train90Val10"
                target_num = "90"
            else:
                target_dir = "Train100Submission"
                target_num = "100"
            for block in ["user", "item"]:
                for ext in [".x",".xt"]:
                    outf.write("mv " + target_dir + "/CIL_data" + split + ".libfmrel_" + block + ext + " ")
                    outf.write(target_dir + "/CIL_data" + target_num + "aug.rel_" + block + ext + "\n")


if __name__ == "__main__":
    main(sys.argv)