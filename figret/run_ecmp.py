import sys
import torch
import os
import shutil
from torch.utils.data import DataLoader
from figret_helper import parse_args
from src.figret_env import FigretEnv
from src.figret_model import Figret, FigretDataset
from src.config import RESULT_DIR


# 构建一个伪装的“传统规则”模型
class ECMPModel(torch.nn.Module):
    def __init__(self, num_paths):
        super().__init__()
        self.num_paths = num_paths

    def forward(self, x):
        # 无论输入什么流量，永远输出全 1 的权重，实现绝对平均的等价多路径分流
        batch_size = x.shape[0]
        return torch.ones((batch_size, self.num_paths), dtype=torch.float64)


def run_ecmp():
    # 模拟终端参数
    sys.argv = ['run_ecmp.py', '--topo_name', 'Facebook_pod_a', '--mode', 'test']
    props = parse_args(sys.argv[1:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FigretEnv(props)
    figret = Figret(props, env, device)

    # 加载测试卷
    test_dataset = FigretDataset(props, env, 'test')
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 重点：不加载 .pt 大脑，而是加载我们写好的传统 ECMP 规则
    model = ECMPModel(env.num_paths).to(device)

    print("🚀 正在执行传统算法基准测试: ECMP (等价多路径路由)...")
    figret.test(test_dl, model, device)

    # 自动重命名并保护成绩单
    old_res = os.path.join(RESULT_DIR, props.topo_name, 'Figret', 'result.txt')
    new_res = os.path.join(RESULT_DIR, props.topo_name, 'Figret', 'result_ecmp.txt')
    if os.path.exists(old_res):
        shutil.move(old_res, new_res)
        print(f"✅ ECMP 测试完成！成绩单已自动保存为: {new_res}")
    else:
        print("❌ 未找到生成的 result.txt，请检查运行路径。")


if __name__ == '__main__':
    run_ecmp()