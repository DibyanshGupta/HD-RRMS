import csv
import math

INPUT = "austinHousingData.csv"
OUTPUT = "austin_housing_mtp_12d.csv"

id_col = "zpid"

features = [
    "latitude",
    "longitude",
    "propertyTaxRate",
    "yearBuilt",
    "latestPrice",
    "lotSizeSqFt",
    "livingAreaSqFt",
    "avgSchoolDistance",
    "avgSchoolRating",
    "numOfBathrooms",
    "numOfBedrooms",
    "numOfStories"
]

with open(INPUT, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    required = [id_col] + features

    missing = [c for c in required if c not in reader.fieldnames]

    if missing:
        print("Missing columns:", missing)
        raise SystemExit(1)

    rows = []
    seen = set()
    skipped = 0

    for row in reader:
        pid = row[id_col].strip()

        if not pid or pid in seen:
            skipped += 1
            continue

        values = []
        valid = True

        for col in features:
            value = row[col].strip()

            if not value:
                valid = False
                break

            try:
                x = float(value)

                if not math.isfinite(x):
                    valid = False
                    break

                values.append(x)

            except ValueError:
                valid = False
                break

        if not valid:
            skipped += 1
            continue

        seen.add(pid)
        rows.append([pid] + values)

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([id_col] + features)
    writer.writerows(rows)

print("========== Austin 12D ==========")
print("Rows:", len(rows))
print("Dimensions:", len(features))
print("Skipped:", skipped)
print("Output:", OUTPUT)