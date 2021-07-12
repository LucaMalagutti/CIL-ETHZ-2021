pred_file = "BayesianSVDpp_predictions.csv"     # this will be generated
samplesub_file = "/home/ico/PycharmProjects/CIL-2021/data/sample_submission.csv"
libfm_file = "/home/ico/PycharmProjects/CIL-2021/baselines/DeepRec/data/submission/libfm_deepf_inndim32_pred"

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