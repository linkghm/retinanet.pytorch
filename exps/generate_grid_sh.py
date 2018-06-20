exp_id = 1

experiment_paths = ""

backbones = ['resnet18']#, 'resnet34']
n_ways = [10, 5, 20]
n_supports = [4, 8]
emb_sizes = [128, 64, 256]
memory_sizes = [512, 256, 1024]
batch_sizes = [16, 8]
delete_mem_every_episodes = [True]
emb_depths = [1, 2]
fc_sizes = [0, 1]
dropouts = [0, .25]

# for delete_mem_every_episode in delete_mem_every_episodes:
for backbone in backbones:
    for fc_size in fc_sizes:
        for dropout in dropouts:
            for emb_depth in emb_depths:
                for batch_size in batch_sizes:
                    for memory_size in memory_sizes:
                        for emb_size in emb_sizes:
                            for n_way in n_ways:
                                for n_support in n_supports:
                                    str = "/home/hayden/anaconda3/bin/python /media/hayden/Storage21/CODEBASE/retinanet.pytorch/train_mem.py"
                                    str += " --exp voc --log /media/hayden/Storage21/MODELS/PROTINANET/exps/%03d"%(exp_id)
                                    str += " --nway %d"%n_way
                                    str += " --nsup %d"%n_support
                                    str += " --esize %d"%emb_size
                                    str += " --bsize %d"%batch_size
                                    str += " --msize %d"%memory_size
                                    str += " --backbone %s"%backbone
                                    str += " --emb_depth %d"%emb_depth
                                    str += " --fc_size %d"%fc_size
                                    str += " --dropout %f"%dropout

                                    print(str)
                                    exp_id += 1
