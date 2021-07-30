""" Converts the predictions from LibFM format to .csv submission format """

import sys


def main(args):
    samplesub_file = args[1]
    libfm_file = args[2]
    out_file = libfm_file + ".csv"

    with open(libfm_file) as predf:
        with open(out_file, "w") as outf:
            with open(samplesub_file) as samplef:
                outf.write("Id,Prediction\n")
                line_num = 0
                for pred_line, sample_line in zip(predf.readlines(), samplef.readlines()[1:]):
                    user_item = sample_line.split(",")[0]
                    rating = float(pred_line)
                    outf.write(f"{user_item},{rating}\n")
                    line_num += 1
                    if line_num % 100000 == 0:
                        print(line_num)

if __name__ == "__main__":
    main(sys.argv)
