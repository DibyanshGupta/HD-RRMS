import csv
import math

INPUT = "austinHousingData.csv"
OUTPUT = "austin_housing_mtp.csv"

id_col = "zpid"

features = [
    "latitude",
    "longitude",
    "propertyTaxRate",
    "garageSpaces",
    "parkingSpaces",
    "yearBuilt",
    "latestPrice",
    "numPriceChanges",
    "latest_salemonth",
    "latest_saleyear",
    "numOfPhotos",
    "numOfAccessibilityFeatures",
    "numOfAppliances",
    "numOfParkingFeatures",
    "numOfPatioAndPorchFeatures",
    "numOfSecurityFeatures",
    "numOfWaterfrontFeatures",
    "numOfWindowFeatures",
    "numOfCommunityFeatures",
    "lotSizeSqFt",
    "livingAreaSqFt",
    "numOfPrimarySchools",
    "numOfElementarySchools",
    "numOfMiddleSchools",
    "numOfHighSchools",
    "avgSchoolDistance",
    "avgSchoolRating",
    "avgSchoolSize",
    "MedianStudentsPerTeacher",
    "numOfBathrooms",
    "numOfBedrooms",
    "numOfStories"
]

with open(INPUT, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    columns = reader.fieldnames

    print("Input columns:", len(columns))
    print("Input rows: reading...")

    # Check requested columns exist
    required = [id_col] + features

    missing_columns = [c for c in required if c not in columns]

    if missing_columns:
        print("\nERROR: Missing columns:")
        for c in missing_columns:
            print("  ", c)
        raise SystemExit(1)

    rows = []
    skipped_missing = 0
    skipped_invalid = 0
    seen_ids = set()

    for row in reader:

        zpid = row[id_col].strip()

        if not zpid:
            skipped_invalid += 1
            continue

        if zpid in seen_ids:
            skipped_invalid += 1
            continue

        values = []
        valid = True

        for col in features:
            value = row[col].strip()

            if value == "":
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
            skipped_missing += 1
            continue

        seen_ids.add(zpid)
        rows.append([zpid] + values)

print("\n========== PREPROCESSING SUMMARY ==========")
print("Original columns :", len(columns))
print("Utility dimensions:", len(features))
print("Original rows     :", len(rows) + skipped_missing + skipped_invalid)
print("Rows skipped      :", skipped_missing + skipped_invalid)
print("  Missing/invalid :", skipped_missing)
print("  Duplicate/invalid IDs:", skipped_invalid)
print("Final rows        :", len(rows))
print("Final dimensions  :", len(features))

# Write output
with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([id_col] + features)

    writer.writerows(rows)

print("\nCreated:", OUTPUT)
