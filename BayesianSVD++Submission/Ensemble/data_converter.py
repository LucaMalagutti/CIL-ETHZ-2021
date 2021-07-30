""" Converts .csv submission format files to ensemble input format"""

in_file = "NNMF_sub.csv"
out_file = "NNMF_sub_conv"
with open(in_file, "r") as inf:
    with open(out_file, "w") as outf:
        line_num = 0
        for line in inf.readlines():
            splitline = line.split(",")
            if line_num > 0:
                rating = float(splitline[1][:-1])
                outf.write(str(rating)+"\n")
            line_num += 1