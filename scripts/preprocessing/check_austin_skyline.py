import csv

INPUT = "austin_housing_mtp.csv"

with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    data = []

    for row in reader:
        values = [float(x) for x in row[1:]]
        data.append((row[0], values))

n = len(data)
d = len(data[0][1])

print("Rows:", n)
print("Dimensions:", d)

# Skyline for maximization:
# p dominates q if p >= q in every dimension
# and p > q in at least one dimension.

skyline = []

for i, (pid, p) in enumerate(data):

    dominated = False

    for j, (qid, q) in enumerate(data):

        if i == j:
            continue

        ge_all = True
        gt_one = False

        for a, b in zip(q, p):
            if a < b:
                ge_all = False
                break
            if a > b:
                gt_one = True

        if ge_all and gt_one:
            dominated = True
            break

    if not dominated:
        skyline.append(pid)

    if (i + 1) % 500 == 0:
        print("Processed:", i + 1, "/", n)

print()
print("========== SKYLINE RESULT ==========")
print("Original tuples:", n)
print("Skyline tuples :", len(skyline))
print("Removed        :", n - len(skyline))
print("Skyline ratio  :", len(skyline) / n)

print()
print("First skyline IDs:")
print(" ".join(skyline[:30]))