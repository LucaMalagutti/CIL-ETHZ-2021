#Extract the Embeddings form the jointly trained embeddings
python extract_embeddings.py --hidden_layers=2048,512,512 --major=items --save_path=model_save/alternate/items_encoder.last
python extract_embeddings.py --hidden_layers=512,32,64 --major=users --save_path=model_save/alternate/users_encoder.last

#Produce the submission file
python infer_mlp.py --logdir=model_save/alternate/model.last --input_size=576
