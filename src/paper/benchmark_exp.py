import os
import subprocess
import yaml


def main():
    with open("src/paper/benchmark.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = config["output_dir"]
    os.makedirs(path, exist_ok=True)
    print(f"Saving outputs to {path}")
    experiments = config["experiments"]

    for key, experiment in experiments.items():
        exp = experiment["exp"]
        name = experiment["name"]
        args = experiment.get("args", [])
        output_dir = f"{path}/{key}"
        args.append(f"output_dir={output_dir}")
        args = " ".join(args)
        cmd = f"python -m src.paper.benchmark +experiment={exp} {args}"
        print("=" * 80)
        print(f"Experiment: {name}")
        print(f"Command: {cmd}")
        print("=" * 80)
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
