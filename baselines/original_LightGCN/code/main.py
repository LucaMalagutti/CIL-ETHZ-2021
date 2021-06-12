import time
from os.path import join

import numpy as np
import procedure
import torch
import utils
import world
from tensorboardX import SummaryWriter
from world import cprint

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

model = register.MODELS[world.model_name](world.config, dataset)
model = model.to(world.device)

loss = utils.BPRLoss(model, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        model.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            procedure.test(dataset, model, epoch, w, world.config["multicore"])
        output_information = procedure.BPR_train(
            dataset, model, loss, epoch, w=w
        )
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}")
        torch.save(model.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
