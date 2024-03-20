import argparse
import subprocess
from itertools import product

# 定义参数范围
lamb_u = [0.1, 0.2, 0.3, 0.4]
lamb_i = [0.1, 0.2, 0.3, 0.4]

# 创建参数组合
param_combinations = list(product(lamb_u, lamb_i))

def run_experiment(paramu, parami):
    # 在这里运行你的代码，替换成你的实际命令
    command = f"python main.py --inference false --dataset Music --gpu 0 --model MELT_SASRec --lamb_u {paramu} -lamb_i {parami} --e_max 180 --pareto_rule 0.8 --batch_size 128"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Experiment Script")
    parser.add_argument("--param1", type=int, help="Parameter 1")
    parser.add_argument("--param2", type=float, help="Parameter 2")

    args = parser.parse_args()

    # 如果提供了参数，则运行单一实验
    if args.param1 is not None and args.param2 is not None:
        run_experiment(args.param1, args.param2)
    else:
        # 否则，运行所有的参数组合
        for param1, param2 in param_combinations:
            print(f"Running experiment with param1={param1}, param2={param2}")
            run_experiment(param1, param2)
