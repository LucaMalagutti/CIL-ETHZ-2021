pred_file = "BS_bags_inndim17_526it_EmpBayes_preds.csv"     # this will be generated, output: change this
samplesub_file = "/home/ico/PycharmProjects/CIL-2021/data/sample_submission.csv"
libfm_file = "/home/ico/PycharmProjects/CIL-2021/baselines/BayesianSVD/submission/BS_bags_EmpBayes17"  # input: change this

with open(libfm_file) as predf:
    with open(pred_file, "w") as outf:
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