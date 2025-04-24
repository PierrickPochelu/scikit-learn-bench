from scikit_learn_bench.core import bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST

all_scores = {}
model_counts = {}

# Run benchmarks
categories = [

    ("reg", 1),
    ("reg", 2)
]

for ml_category, num_output in categories:
    scores = bench(
        num_samples=10,
        num_features=2,
        num_output=num_output,
        fix_comp_time=0.1,
        ml_type=ml_category,
        profiler_type="timememory",
        table_print=False
    )

    for model_name, result in scores.items():
        if model_name in model_counts:
            pass
            #model_counts[model_name] += 1
            #all_scores[f"{model_name}{model_counts[model_name]}"] = result
        else:
            model_counts[model_name] = 1
            all_scores[model_name] = result

print_table(all_scores)
print("Number of ML algo retrieved: ", len(all_scores))
print("Number of ML algo inference retrieved: ", len([s[1] for s in all_scores if s != CONST.NANSTR]))

#######

import matplotlib.pyplot as plt

# Filter valid training data points
training_speeds = []
training_memories = []
inference_speeds = []
inference_memories = []
model_names_train = []
model_names_infer = []

for name, vals in all_scores.items():
    try:
        train_speed = float(vals[0])
        train_mem = float(vals[2])
        training_speeds.append(train_speed)
        training_memories.append(train_mem)
        model_names_train.append(name)
    except (ValueError, ZeroDivisionError, TypeError):
        continue

    try:
        infer_speed = float(vals[1])
        infer_mem = float(vals[3])
        inference_speeds.append(infer_speed)
        inference_memories.append(infer_mem)
        model_names_infer.append(name)
    except (ValueError, ZeroDivisionError, TypeError):
        continue

# Create scatter plots


# Plot 1: Training
plt.figure(figsize=(8, 8))
plt.scatter(training_speeds, training_memories, color='blue', alpha=0.7)
for i, name in enumerate(model_names_train):
    plt.annotate(name, (training_speeds[i], training_memories[i]), fontsize=8, alpha=0.6)
plt.title("Training: Throughput vs Memory")
plt.xlabel("Samples per second (Training)")
plt.ylabel("Memory Usage (KB)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("fit.pdf")

# Plot 2: Inference
plt.figure(figsize=(8, 8))
plt.scatter(inference_speeds, inference_memories, color='green', alpha=0.7)
for i, name in enumerate(model_names_infer):
    plt.annotate(name, (inference_speeds[i], inference_memories[i]), fontsize=8, alpha=0.6)
plt.title("Inference: Throughput vs Memory")
plt.xlabel("Samples per second (Inference)")
plt.ylabel("Memory Usage (KB)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("predict.pdf")