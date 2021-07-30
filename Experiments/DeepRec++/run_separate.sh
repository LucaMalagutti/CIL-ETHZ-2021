#wandb offline

#training items autoencoders
python run.py --num_epochs=10 --batch_size=64 --dense_refeeding_steps=2 --dropout=0.4 --layer1_dim=2048 --layer2_dim=512 --layer3_dim=512 --learning_rate=0.01 --weight_decay=0.001 --major=items --save_every=1 --logdir=model_save/pretrain_emb/items

#training users autoencoders
python run.py --num_epochs=10 --batch_size=64 --dense_refeeding_steps=2 --dropout=0.4 --layer1_dim=512 --layer2_dim=32 --layer3_dim=64 --learning_rate=0.0035 --weight_decay=5.0e-08 --major=users --save_every=1 --logdir=model_save/pretrain_emb/users

#Extracting internal representations
python extract_embeddings.py --hidden_layers=2048,512,512 --major=items --save_path=model_save/pretrain_emb/items/model.last
python extract_embeddings.py --hidden_layers=512,32,64 --major=users --save_path=model_save/pretrain_emb/users/model.last

#Run the MLP with fixed vector representations for users and items
python run_mlp.py --logdir=model_save/mlp/ --num_epochs=30
