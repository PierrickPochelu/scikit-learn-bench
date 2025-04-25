from scikit_learn_bench.all_bench import all_bench
from scikit_learn_bench.display import print_table
from scikit_learn_bench import CONST
CONST.IS_MAX_PROF_TIME=True

scores = all_bench(
    num_samples=1000,
    num_features=1000,
    num_output=2,
    min_prof_time=1.,
    max_prof_time=100.,
    ml_type="all",
    profiler_type="timememory",
    table_print=False
)


print_table(scores)
print("Number of ML algo retrieved: ", len(scores))
print("Number of ML algo inference retrieved: ", len([s[1] for s in scores if s != CONST.NANSTR]))

#######

import matplotlib.pyplot as plt

# Filter valid training data points
training_speeds = []
training_memories = []
inference_speeds = []
inference_memories = []
model_names_train = []
model_names_infer = []

for name, vals in scores.items():
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