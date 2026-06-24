import subprocess

configs = [
    "configs/base.yaml",
    "configs/adv_train_fgsm.yaml",
]

for cfg in configs:
    print(f"\nRunning {cfg}")
    subprocess.run([
        "python",
        "-m",
        "src.runner.run_experiment",
        "--config",
        cfg
    ], check=True)