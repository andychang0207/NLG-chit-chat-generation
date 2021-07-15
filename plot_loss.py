import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, MultipleLocator
import pickle
from pathlib import Path
import argparse
import os
import json
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_dir",type=Path,default="./fig/",help="Directory to store figure.")
    parse.add_argument("--figure_name",type=str,default="gpt2_testing2",help="The name of figure.")
    parse.add_argument("--source_dir",type=Path,default="./testing2-gpt2",help="Directory where store rouge value.")
    args = parse.parse_args()
    return args

def plot_loss_curve(train_losses, eval_losses,chart_type,file_path,title=''):
    ''' Plot learning curve of your model (train & dev loss) '''
    assert isinstance(train_losses,list) and isinstance(eval_losses,list)
    total_steps = len(train_losses)
    x_1 = range(total_steps)
    x_2 = x_1[len(train_losses)//len(eval_losses)-1::len(train_losses)//len(eval_losses)]
    figure(figsize=(6, 4))
    plt.plot(x_1, train_losses, c='tab:red', label='train')
    plt.plot(x_2, eval_losses, c='tab:cyan', label='validation')
    plt.xlabel('epoch')
    plt.ylabel(chart_type)
    plt.ylim(0,10)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(len(train_losses)//len(eval_losses)))
    ticks = [i+1 for i in range(len(eval_losses))]
    plt.xticks(x_2,ticks)
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig(file_path)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    file_path = os.path.join(args.output_dir,args.figure_name+".png")
    with open(args.source_dir/"train_losses.pkl","rb") as f:
        train_losses = pickle.load(f)
    with open(args.source_dir/"eval_losses.pkl","rb") as f:
        eval_losses = pickle.load(f)
    plot_loss_curve(train_losses,eval_losses,"Loss",file_path,args.figure_name)
    print("Complete Plotting.")