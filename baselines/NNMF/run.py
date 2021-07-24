"""Starts up the training process of the model"""

from app.core import run

if __name__ == "__main__":
    # Defines model hyperparameters
    hyper_params = {
        "lambda_value": 39.8,
        "hidden_layer_number": 4,
        "K": 1,
        "D_prime": 60,
        "hidden_unit_number": 50,
        "learning_rate": 1e-3,
    }
    # if none, Batch size == whole training data
    batch_size = None
    valid_rmse, test_rmse = run(batch_size=batch_size, **hyper_params)

    # Prints final summary message
    print("\t".join(sorted(hyper_params.keys())))
    recap_msg = "{}\t{}\t{}".format(
        "\t".join(str(hyper_params[key]) for key in sorted(hyper_params.keys())),
        valid_rmse,
        test_rmse,
    )
    print(recap_msg)
