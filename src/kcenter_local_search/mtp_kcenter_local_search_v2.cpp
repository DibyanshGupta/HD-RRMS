#include <bits/stdc++.h>
using namespace std;

using ld = long double;
static constexpr ld EPS = 1e-12L;

/*
============================================================
 MTP K-CENTER + LOCAL SEARCH
=============================================================

Objective:
    F(S) = alpha * ARR(S) + (1-alpha) * MRR(S)

Direction:
    FORWARD / INSERTION

At each iteration:
    For every candidate p not in S:
        S' = S U {p}
        compute ARR(S')
        compute MRR(S') using the MRR-paper LP
        compute F(S')
    insert the candidate giving MINIMUM F(S').

ARR:
    Estimated by sampled non-negative linear utility functions.

MRR:
    Exact LP-based evaluation from:
    "Regret-Minimizing Representative Databases"
    Nanongkai et al., PVLDB 2010.

The MRR LP for a candidate q and current selected set S is:

    maximize x
    subject to:
        (q-s) . v >= x       for every s in S
        q . v = 1
        v[j] >= 0
        x >= 0

For a fixed selected set S:
    MRR(S) = max over q in D-S of LP(q,S).

For forward candidate p:
    S' = S U {p}
    MRR(S') is computed by running the above LP for
    every q in D-S'.

Dataset assumptions:
    - CSV has one header row.
    - Utility dimensions are numeric.
    - All utility dimensions are maximization dimensions.
    - Delimiter can be comma or semicolon.
    - Unwanted columns (e.g. Wine "quality") can be skipped by
      zero-based column index using --skip-cols.
    - The implementation performs min-max normalization on the
      utility dimensions.

Usage:
    ./mtp_forward_greedy <csv> <k> <alpha> <arr_samples>
                         [delimiter] [skip_cols] [max_rows] [seed] [max_local_passes] [max_swap_evaluations]

Examples:
    Wine:
    ./mtp_forward_greedy winequality-red.csv 10 0.5 1000 ';' 11 -1 42

    Comma-separated dataset, all columns are utility dimensions:
    ./mtp_forward_greedy data.csv 10 0.5 1000 ',' "" -1 42

    Skip columns 3 and 8 (zero-based):
    ./mtp_forward_greedy data.csv 10 0.5 1000 ',' "3,8" -1 42

NOTE:
    Winequality-red.csv has 12 columns:
      0..10 = physicochemical attributes
      11    = quality
    so use --/argument "11" to skip quality.
============================================================
*/

struct Point {
    int id = -1;               // original row ID
    vector<ld> x;              // normalized utility dimensions
};

static vector<string> splitCSV(const string& line, char delimiter) {
    vector<string> out;
    string cur;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];

        if (ch == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch == delimiter && !inQuotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }

    out.push_back(cur);
    return out;
}

static string trim(string s) {
    auto notSpace = [](unsigned char c) { return !isspace(c); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), notSpace));
    s.erase(find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

static bool parseLongDouble(const string& token, ld& value) {
    string s = trim(token);
    if (s.empty()) return false;

    try {
        size_t pos = 0;
        value = stold(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

static char detectDelimiter(const string& header) {
    size_t commas = count(header.begin(), header.end(), ',');
    size_t semis = count(header.begin(), header.end(), ';');
    return (semis > commas ? ';' : ',');
}

static vector<int> parseSkipColumns(const string& text) {
    vector<int> result;
    if (text.empty() || text == "-") return result;

    stringstream ss(text);
    string token;

    while (getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;
        int idx = stoi(token);
        if (idx < 0) throw runtime_error("skip column index cannot be negative");
        result.push_back(idx);
    }

    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
}

static bool isSkipped(int column, const vector<int>& skipCols) {
    return binary_search(skipCols.begin(), skipCols.end(), column);
}

static vector<Point> readDataset(
    const string& filename,
    int& dimensions,
    char delimiter,
    const vector<int>& skipCols,
    int maxRows
) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open dataset: " + filename);
    }

    string line;
    if (!getline(file, line)) {
        throw runtime_error("Dataset is empty: " + filename);
    }

    vector<string> header = splitCSV(line, delimiter);
    const int totalColumns = (int)header.size();

    // If the dataset has an explicit ID column, keep it as the tuple
    // identifier but never treat it as a utility dimension.
    auto normalizeHeader = [](string s) {
        s = trim(s);
        transform(s.begin(), s.end(), s.begin(),
                  [](unsigned char ch) { return (char)tolower(ch); });
        return s;
    };
    const bool hasExplicitID =
        !header.empty() && normalizeHeader(header[0]) == "id";

    vector<int> usedColumns;
    for (int c = 0; c < totalColumns; ++c) {
        if (hasExplicitID && c == 0)
            continue;
        if (!isSkipped(c, skipCols))
            usedColumns.push_back(c);
    }

    if (usedColumns.empty()) {
        throw runtime_error("No utility columns remain after --skip-cols.");
    }

    dimensions = (int)usedColumns.size();

    vector<Point> D;
    string parseError;

    int rowID = 0;
    while (getline(file, line)) {
        if (trim(line).empty()) continue;

        if (maxRows > 0 && (int)D.size() >= maxRows)
            break;

        vector<string> fields = splitCSV(line, delimiter);

        if ((int)fields.size() != totalColumns) {
            // Be conservative: skip malformed rows.
            continue;
        }

        Point p;
        if (hasExplicitID) {
            ld idValue;
            if (!parseLongDouble(fields[0], idValue) ||
                !isfinite((double)idValue)) {
                continue;
            }
            p.id = (int)idValue;
        } else {
            p.id = rowID;
        }
        ++rowID;
        p.x.reserve(dimensions);

        bool valid = true;
        for (int c : usedColumns) {
            ld value;
            if (!parseLongDouble(fields[c], value) || !isfinite((double)value)) {
                valid = false;
                break;
            }
            p.x.push_back(value);
        }

        if (valid)
            D.push_back(std::move(p));
    }

    if (D.empty()) {
        throw runtime_error("No valid numeric rows found in dataset.");
    }

    return D;
}

static void normalizeDataset(vector<Point>& D) {
    const int d = (int)D[0].x.size();

    vector<ld> mn(d, numeric_limits<ld>::infinity());
    vector<ld> mx(d, -numeric_limits<ld>::infinity());

    for (const auto& p : D) {
        for (int j = 0; j < d; ++j) {
            mn[j] = min(mn[j], p.x[j]);
            mx[j] = max(mx[j], p.x[j]);
        }
    }

    for (auto& p : D) {
        for (int j = 0; j < d; ++j) {
            if (fabsl(mx[j] - mn[j]) <= EPS) {
                p.x[j] = 0.0L;
            } else {
                p.x[j] = (p.x[j] - mn[j]) / (mx[j] - mn[j]);
            }
        }
    }
}

static bool dominates(const Point& a, const Point& b) {
    bool strictlyBetter = false;

    for (int j = 0; j < (int)a.x.size(); ++j) {
        if (a.x[j] < b.x[j] - EPS)
            return false;

        if (a.x[j] > b.x[j] + EPS)
            strictlyBetter = true;
    }

    return strictlyBetter;
}

static vector<Point> computeSkyline(const vector<Point>& D) {
    vector<Point> skyline;
    skyline.reserve(D.size());

    const int n = (int)D.size();

    for (int i = 0; i < n; ++i) {
        bool dominated = false;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            if (dominates(D[j], D[i])) {
                dominated = true;
                break;
            }
        }

        if (!dominated)
            skyline.push_back(D[i]);
    }

    return skyline;
}

static ld utility(const Point& p, const vector<ld>& w) {
    ld result = 0.0L;
    for (int j = 0; j < (int)p.x.size(); ++j)
        result += p.x[j] * w[j];
    return result;
}

static vector<ld> generateRandomWeight(int d, mt19937_64& rng) {
    exponential_distribution<ld> dist(1.0L);

    vector<ld> w(d);
    ld sum = 0.0L;

    for (int j = 0; j < d; ++j) {
        w[j] = dist(rng);
        sum += w[j];
    }

    if (sum <= EPS) {
        w.assign(d, 1.0L / (ld)d);
        return w;
    }

    for (ld& x : w)
        x /= sum;

    return w;
}

static vector<vector<ld>> generateUtilityFunctions(
    int d,
    int N,
    mt19937_64& rng
) {
    vector<vector<ld>> weights;
    weights.reserve(N);

    for (int i = 0; i < N; ++i)
        weights.push_back(generateRandomWeight(d, rng));

    return weights;
}

static vector<vector<ld>> precomputeUtilities(
    const vector<Point>& D,
    const vector<vector<ld>>& weights
) {
    const int N = (int)weights.size();
    const int n = (int)D.size();

    vector<vector<ld>> values(
        N,
        vector<ld>(n, 0.0L)
    );

    for (int u = 0; u < N; ++u) {
        for (int p = 0; p < n; ++p) {
            values[u][p] = utility(D[p], weights[u]);
        }
    }

    return values;
}

static ld calculateARR(
    const vector<int>& selected,
    const vector<vector<ld>>& utilityValues,
    const vector<ld>& bestD
) {
    const int N = (int)utilityValues.size();

    if (N == 0)
        return 0.0L;

    ld totalRegret = 0.0L;

    for (int u = 0; u < N; ++u) {
        ld bestSelected = 0.0L;

        for (int idx : selected)
            bestSelected = max(bestSelected, utilityValues[u][idx]);

        ld regret = 0.0L;

        if (bestD[u] > EPS) {
            regret = (bestD[u] - bestSelected) / bestD[u];

            if (regret < 0.0L && regret > -1e-10L)
                regret = 0.0L;
        }

        totalRegret += regret;
    }

    return totalRegret / (ld)N;
}

/*
============================================================
Simplex solver

Solves:

    maximize c^T x
    subject to A x <= b
               x >= 0

This is a compact two-phase simplex implementation adapted
for the small LPs used by the MRR formulation.

The MRR LP has:
    variables = d + 1
    constraints = |S| + 2
============================================================
*/
struct LPSolver {
    static constexpr double EPS = 1e-10;
    static constexpr double INF = 1e100;

    int m, n;
    vector<int> B, N;
    vector<vector<double>> T;

    LPSolver(
        const vector<vector<double>>& A,
        const vector<double>& b,
        const vector<double>& c
    )
        : m((int)b.size()),
          n((int)c.size()),
          B(m),
          N(n + 1),
          T(m + 2, vector<double>(n + 2, 0.0))
    {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                T[i][j] = A[i][j];

        for (int i = 0; i < m; ++i) {
            B[i] = n + i;
            T[i][n] = -1.0;
            T[i][n + 1] = b[i];
        }

        for (int j = 0; j < n; ++j) {
            N[j] = j;
            T[m][j] = -c[j];
        }

        N[n] = -1;
        T[m + 1][n] = 1.0;
    }

    void pivot(int r, int s) {
        double inv = 1.0 / T[r][s];

        for (int i = 0; i < m + 2; ++i) {
            if (i == r) continue;

            for (int j = 0; j < n + 2; ++j) {
                if (j == s) continue;

                T[i][j] -= T[r][j] * T[i][s] * inv;
            }
        }

        for (int j = 0; j < n + 2; ++j) {
            if (j != s)
                T[r][j] *= inv;
        }

        for (int i = 0; i < m + 2; ++i) {
            if (i != r)
                T[i][s] *= -inv;
        }

        T[r][s] = inv;
        swap(B[r], N[s]);
    }

    bool simplex(int phase) {
        int objectiveRow = (phase == 1 ? m + 1 : m);

        while (true) {
            int s = -1;

            for (int j = 0; j <= n; ++j) {
                if (phase == 2 && N[j] == -1)
                    continue;

                if (
                    s == -1 ||
                    T[objectiveRow][j] < T[objectiveRow][s] - EPS ||
                    (
                        fabsl(
                            T[objectiveRow][j] -
                            T[objectiveRow][s]
                        ) <= EPS &&
                        N[j] < N[s]
                    )
                ) {
                    s = j;
                }
            }

            if (T[objectiveRow][s] >= -EPS)
                return true;

            int r = -1;

            for (int i = 0; i < m; ++i) {
                if (T[i][s] <= EPS)
                    continue;

                if (r == -1) {
                    r = i;
                } else {
                    double lhs =
                        T[i][n + 1] / T[i][s];

                    double rhs =
                        T[r][n + 1] / T[r][s];

                    if (
                        lhs < rhs - EPS ||
                        (
                            fabsl(lhs - rhs) <= EPS &&
                            B[i] < B[r]
                        )
                    ) {
                        r = i;
                    }
                }
            }

            if (r == -1)
                return false; // unbounded

            pivot(r, s);
        }
    }

    double solve(vector<double>& x) {
        int r = 0;

        for (int i = 1; i < m; ++i) {
            if (T[i][n + 1] < T[r][n + 1])
                r = i;
        }

        // Phase I
        if (T[r][n + 1] < -EPS) {
            pivot(r, n);

            if (
                !simplex(1) ||
                T[m + 1][n + 1] < -EPS
            ) {
                return -INF; // infeasible
            }

            if (fabsl(T[m + 1][n + 1]) > EPS)
                return -INF; // infeasible

            auto it = find(B.begin(), B.end(), -1);

            if (it != B.end()) {
                r = (int)(it - B.begin());

                int s = 0;
                for (int j = 1; j <= n; ++j) {
                    if (
                        T[r][j] < T[r][s] - EPS ||
                        (
                            fabsl(T[r][j] - T[r][s]) <= EPS &&
                            N[j] < N[s]
                        )
                    ) {
                        s = j;
                    }
                }

                pivot(r, s);
            }
        }

        // Phase II
        if (!simplex(2))
            return INF;

        x.assign(n, 0.0);

        for (int i = 0; i < m; ++i) {
            if (B[i] < n)
                x[B[i]] = T[i][n + 1];
        }

        return T[m][n + 1];
    }
};

/*
============================================================
MRR LP FOR ONE CANDIDATE

Returns:
    rr_{S U {candidate}}(S,L)

which is the worst regret of S when candidate is the
only additional point in the temporary database.

Important:
    This function does NOT calculate the MRR of S U {candidate}
    against the whole D.

It evaluates one candidate q relative to S.

============================================================
*/
static ld solveCandidateMRR(
    int candidate,
    const vector<int>& selected,
    const vector<Point>& D,
    vector<ld>* bestV = nullptr
) {
    const int d = (int)D[0].x.size();
    const int variables = d + 1; // v[0..d-1], x

    /*
        maximize x
    */
    vector<double> c(variables, 0.0);
    c[d] = 1.0;

    vector<vector<double>> A;
    vector<double> b;

    /*
        (p-s).v >= x

        becomes

        x - (p-s).v <= 0
    */
    for (int s : selected) {
        vector<double> row(variables, 0.0);

        for (int j = 0; j < d; ++j) {
            row[j] =
                -(double)(
                    D[candidate].x[j] -
                    D[s].x[j]
                );
        }

        row[d] = 1.0;

        A.push_back(row);
        b.push_back(0.0);
    }

    /*
        p.v = 1

        represented as:
            p.v <= 1
           -p.v <= -1
    */
    vector<double> row1(variables, 0.0);
    vector<double> row2(variables, 0.0);

    for (int j = 0; j < d; ++j) {
        row1[j] = (double)D[candidate].x[j];
        row2[j] = -(double)D[candidate].x[j];
    }

    A.push_back(row1);
    b.push_back(1.0);

    A.push_back(row2);
    b.push_back(-1.0);

    LPSolver solver(A, b, c);

    vector<double> solution;
    double result = solver.solve(solution);

    if (result <= -LPSolver::INF / 2) {
        if (bestV) bestV->clear();

        /*
            The candidate cannot be the optimum of any feasible
            non-negative utility direction against S.
            Its contribution to MRR is therefore 0.
        */
        return 0.0L;
    }

    if (result >= LPSolver::INF / 2) {
        if (bestV) bestV->clear();
        throw runtime_error(
            "MRR LP became unexpectedly unbounded."
        );
    }

    if (bestV) {
        bestV->assign(d, 0.0L);
        for (int j = 0; j < d; ++j)
            (*bestV)[j] = (ld)solution[j];
    }

    if (result < 0.0 && result > -1e-8)
        result = 0.0;

    return (ld)result;
}

/*
============================================================
MRR(S)

For a selected set S:

    MRR(S) = max_{q in D-S} LP(q,S)

where LP(q,S) returns:
    rr_{S U {q}}(S,L)

Thus this is the actual MRR of S with respect to D.
============================================================
*/
static ld calculateMRR(
    const vector<int>& selected,
    const vector<Point>& D,
    int* worstTuple = nullptr
) {
    const int n = (int)D.size();

    if (selected.empty()) {
        /*
            With no representative tuple, a user can lose all
            available utility. The normalized maximum regret is 1.
        */
        if (worstTuple) *worstTuple = (n > 0 ? 0 : -1);
        return (n > 0 ? 1.0L : 0.0L);
    }

    vector<char> inSelected(n, false);
    for (int idx : selected)
        inSelected[idx] = true;

    ld mrr = 0.0L;
    int worst = -1;

    for (int q = 0; q < n; ++q) {
        if (inSelected[q])
            continue;

        ld regret = solveCandidateMRR(
            q,
            selected,
            D,
            nullptr
        );

        if (regret > mrr + EPS) {
            mrr = regret;
            worst = q;
        }
    }

    if (worstTuple)
        *worstTuple = worst;

    return mrr;
}

static ld calculateCombinedObjective(
    const vector<int>& selected,
    const vector<Point>& D,
    const vector<vector<ld>>& utilityValues,
    const vector<ld>& bestD,
    ld alpha,
    ld& arr,
    ld& mrr
) {
    if (selected.empty()) {
        arr = 1.0L;
        mrr = 1.0L;
        return alpha * arr + (1.0L - alpha) * mrr;
    }

    arr = calculateARR(
        selected,
        utilityValues,
        bestD
    );

    mrr = calculateMRR(
        selected,
        D
    );

    return
        alpha * arr +
        (1.0L - alpha) * mrr;
}

/*
============================================================
FORWARD GREEDY FOR COMBINED MTP OBJECTIVE

S = empty

At every iteration:

    for each p not in S:
        S' = S U {p}

        calculate ARR(S')
        calculate MRR(S')
        F(S') = alpha*ARR + (1-alpha)*MRR

    choose p minimizing F(S')
    insert p

This is intentionally a straightforward correctness-first
baseline. It can be expensive because every candidate
evaluation recomputes MRR, and each MRR evaluation solves
an LP for every remaining tuple.
============================================================
*/
/*
============================================================
K-CENTER INITIALIZATION + OBJECTIVE-BASED LOCAL SEARCH

Stage 1: Farthest-first traversal (squared Euclidean distance).
Stage 2: Best-improvement 1-for-1 swaps, accepted only when
         the combined ARR/MRR objective strictly decreases.

The MRR objective is evaluated exactly by the existing LP routine
on the skyline dataset. ARR remains the sampled estimate.
============================================================
*/
static ld squaredDistance(const Point& a, const Point& b) {
    ld sum = 0.0L;
    for (int j = 0; j < (int)a.x.size(); ++j) {
        ld diff = a.x[j] - b.x[j];
        sum += diff * diff;
    }
    return sum;
}

static vector<int> kCenterInitialization(
    const vector<Point>& D,
    int k,
    uint64_t seed
) {
    const int n = (int)D.size();
    if (k <= 0 || k > n)
        throw runtime_error("k must be in [1, skyline size].");

    mt19937_64 rng(seed);
    uniform_int_distribution<int> firstDist(0, n - 1);
    int first = firstDist(rng);

    vector<int> selected;
    selected.reserve(k);
    selected.push_back(first);

    vector<char> chosen(n, false);
    chosen[first] = true;

    // nearestDist[p] = squared distance to the nearest selected center.
    vector<ld> nearestDist(n, numeric_limits<ld>::infinity());
    for (int p = 0; p < n; ++p)
        nearestDist[p] = squaredDistance(D[p], D[first]);

    cout << "\nK-center farthest-first initialization\n";
    cout << "Initial tuple: " << D[first].id << '\n';

    while ((int)selected.size() < k) {
        int farthest = -1;
        ld farthestDistance = -1.0L;

        for (int p = 0; p < n; ++p) {
            if (chosen[p]) continue;
            if (nearestDist[p] > farthestDistance + EPS) {
                farthestDistance = nearestDist[p];
                farthest = p;
            }
        }

        if (farthest < 0)
            throw runtime_error("Could not find next farthest tuple.");

        selected.push_back(farthest);
        chosen[farthest] = true;

        for (int p = 0; p < n; ++p) {
            if (chosen[p]) continue;
            nearestDist[p] = min(
                nearestDist[p],
                squaredDistance(D[p], D[farthest])
            );
        }

        cout << "  Center " << selected.size()
             << ": tuple " << D[farthest].id
             << " (nearest-center distance^2 = "
             << setprecision(8) << (double)farthestDistance << ")\n";
    }

    return selected;
}

static vector<int> kCenterLocalSearchMTP(
    const vector<Point>& D,
    int k,
    ld alpha,
    const vector<vector<ld>>& utilityValues,
    const vector<ld>& bestD,
    uint64_t seed,
    int maxPasses,
    long long maxSwapEvaluations,
    const string& logFilename
) {
    const int n = (int)D.size();
    if (k <= 0 || k > n)
        throw runtime_error("k cannot exceed skyline size.");
    if (maxPasses < 0)
        throw runtime_error("max_local_passes must be nonnegative.");
    if (maxSwapEvaluations < 0)
        throw runtime_error("max_swap_evaluations must be nonnegative.");

    vector<int> selected = kCenterInitialization(D, k, seed);
    vector<char> chosen(n, false);
    for (int idx : selected) chosen[idx] = true;

    ld currentARR = 0.0L, currentMRR = 0.0L;
    ld currentF = calculateCombinedObjective(
        selected, D, utilityValues, bestD, alpha,
        currentARR, currentMRR
    );

    ofstream logFile(logFilename);
    if (!logFile.is_open())
        throw runtime_error("Cannot create log file: " + logFilename);

    logFile << "pass,swap_out_id,swap_in_id,ARR,MRR,combined_F,"
               "evaluated_swaps,pass_seconds,cumulative_seconds\n";

    auto totalStart = chrono::steady_clock::now();
    cout << "\nInitial k-center objective: ARR=" << setprecision(12)
         << (double)currentARR << " MRR=" << (double)currentMRR
         << " F=" << (double)currentF << "\n";

    for (int pass = 1; pass <= maxPasses; ++pass) {
        auto passStart = chrono::steady_clock::now();
        ld bestF = currentF;
        ld bestARR = currentARR, bestMRR = currentMRR;
        int bestOutPos = -1, bestIn = -1;
        long long evaluated = 0;
        long long totalSwaps = 1LL * k * (n - k);
        vector<pair<int,int>> moves;
        moves.reserve((size_t)totalSwaps);
        for (int outPos = 0; outPos < k; ++outPos)
            for (int p = 0; p < n; ++p)
                if (!chosen[p]) moves.push_back({outPos, p});

        // Optional randomized neighborhood cap for large skylines.
        mt19937_64 moveRng(seed + 0x9e3779b97f4a7c15ULL * (uint64_t)pass);
        shuffle(moves.begin(), moves.end(), moveRng);
        if (maxSwapEvaluations > 0 &&
            (long long)moves.size() > maxSwapEvaluations)
            moves.resize((size_t)maxSwapEvaluations);
        const long long movesToEvaluate = (long long)moves.size();

        cout << "\nLocal-search pass " << pass << ": evaluating "
             << movesToEvaluate << " of " << totalSwaps
             << " possible one-for-one swaps.\n";

        for (const auto& move : moves) {
            int outPos = move.first;
            int p = move.second;
            vector<int> candidate = selected;
            candidate[outPos] = p;

            ld arr = 0.0L, mrr = 0.0L;
            ld score = calculateCombinedObjective(
                candidate, D, utilityValues, bestD, alpha, arr, mrr
            );
            ++evaluated;

            if (score < bestF - EPS) {
                bestF = score;
                bestARR = arr;
                bestMRR = mrr;
                bestOutPos = outPos;
                bestIn = p;
            }

            if (evaluated % max(1LL, movesToEvaluate / 20) == 0 ||
                evaluated == movesToEvaluate) {
                cout << "\r  evaluated " << evaluated << "/"
                     << movesToEvaluate << " swaps" << flush;
            }
        }
        cout << "\n";

        if (bestOutPos == -1) {
            cout << "No improving 1-for-1 swap found; local optimum reached.\n";
            break;
        }

        int removed = selected[bestOutPos];
        chosen[removed] = false;
        selected[bestOutPos] = bestIn;
        chosen[bestIn] = true;
        currentF = bestF;
        currentARR = bestARR;
        currentMRR = bestMRR;

        auto passEnd = chrono::steady_clock::now();
        double passSeconds = chrono::duration<double>(passEnd-passStart).count();
        double totalSeconds = chrono::duration<double>(passEnd-totalStart).count();

        logFile << pass << ',' << D[removed].id << ',' << D[bestIn].id << ','
                << setprecision(15) << (double)currentARR << ','
                << (double)currentMRR << ',' << (double)currentF << ','
                << evaluated << ',' << setprecision(9) << passSeconds << ','
                << totalSeconds << '\n';
        logFile.flush();

        cout << "Accepted swap: out " << D[removed].id
             << ", in " << D[bestIn].id << '\n'
             << setprecision(12)
             << "  ARR=" << (double)currentARR
             << " MRR=" << (double)currentMRR
             << " F=" << (double)currentF
             << " | pass time=" << passSeconds << " s\n";
        cout << "  Current IDs: ";
        for (int idx : selected) cout << D[idx].id << ' ';
        cout << '\n';
    }

    return selected;
}

static void writeSelectedTuples(
    const string& filename,
    const vector<int>& selected,
    const vector<Point>& D
) {
    ofstream out(filename);

    if (!out.is_open())
        throw runtime_error("Cannot create output file: " + filename);

    out << "tuple_id";
    for (int j = 0; j < (int)D[0].x.size(); ++j)
        out << ",dimension_" << j;
    out << '\n';

    out << setprecision(17);

    for (int idx : selected) {
        out << D[idx].id;

        for (ld x : D[idx].x)
            out << ',' << (double)x;

        out << '\n';
    }
}

static void printUsage(const char* program) {
    cerr
        << "Usage:\n"
        << "  " << program
        << " <dataset.csv> <k> <alpha> <arr_samples>"
        << " [delimiter] [skip_cols] [max_rows] [seed] [max_local_passes] [max_swap_evaluations]\n\n"

        << "Arguments:\n"
        << "  dataset.csv   CSV dataset with one header row\n"
        << "  k             desired number of selected tuples\n"
        << "  alpha         ARR weight, 0 <= alpha <= 1\n"
        << "  arr_samples   number of sampled utility functions for ARR\n"
        << "  delimiter     optional: ',' or ';' (default: auto-detect)\n"
        << "  skip_cols     optional comma-separated zero-based columns\n"
        << "                to ignore, e.g. 11 or 3,8\n"
        << "  max_rows      optional maximum number of rows; -1 = all\n"
        << "  seed          optional random seed (default: 42)\n"
        << "  max_local_passes optional local-search passes (default: 5)\n"
        << "  max_swap_evaluations optional swaps evaluated per pass; 0=all (default: 0)\n\n"

        << "Examples:\n"
        << "  "
        << program
        << " winequality-red.csv 10 0.5 1000 ';' 11 -1 42\n\n"

        << "  "
        << program
        << " data.csv 10 0.5 1000 ',' - -1 42\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 5 || argc > 11) {
            printUsage(argv[0]);
            return 1;
        }

        string filename = argv[1];

        int k = stoi(argv[2]);
        ld alpha = stold(argv[3]);
        int arrSamples = stoi(argv[4]);

        if (k <= 0) {
            cerr << "ERROR: k must be positive.\n";
            return 1;
        }

        if (alpha < 0.0L || alpha > 1.0L) {
            cerr << "ERROR: alpha must be in [0,1].\n";
            return 1;
        }

        if (arrSamples <= 0) {
            cerr << "ERROR: arr_samples must be positive.\n";
            return 1;
        }

        char delimiter = '\0';

        if (argc >= 6) {
            string d = argv[5];

            if (d == ",")
                delimiter = ',';
            else if (d == ";")
                delimiter = ';';
            else if (d == "auto")
                delimiter = '\0';
            else
                throw runtime_error(
                    "delimiter must be ',' ';' or 'auto'."
                );
        }

        string skipText;
        if (argc >= 7)
            skipText = argv[6];

        int maxRows = -1;
        if (argc >= 8)
            maxRows = stoi(argv[7]);

        uint64_t seed = 42;
        if (argc >= 9)
            seed = stoull(argv[8]);

        int maxLocalPasses = 5;
        if (argc >= 10)
            maxLocalPasses = stoi(argv[9]);

        long long maxSwapEvaluations = 0;
        if (argc >= 11)
            maxSwapEvaluations = stoll(argv[10]);

        vector<int> skipCols =
            parseSkipColumns(skipText);

        /*
            Auto-detect delimiter if requested.
        */
        if (delimiter == '\0') {
            ifstream test(filename);
            if (!test.is_open())
                throw runtime_error("Cannot open dataset: " + filename);

            string header;
            getline(test, header);
            delimiter = detectDelimiter(header);
        }

        int d = 0;

        vector<Point> D =
            readDataset(
                filename,
                d,
                delimiter,
                skipCols,
                maxRows
            );

        if (k > (int)D.size()) {
            throw runtime_error(
                "k = " + to_string(k) +
                " is larger than dataset size = " +
                to_string(D.size())
            );
        }

        cout
            << "========================================\n"
            << "MTP K-CENTER + LOCAL SEARCH\n"
            << "========================================\n"
            << "Dataset          : " << filename << '\n'
            << "Rows read        : " << D.size() << '\n'
            << "Dimensions       : " << d << '\n'
            << "Target k         : " << k << '\n'
            << "Alpha            : " << (double)alpha << '\n'
            << "ARR samples      : " << arrSamples << '\n'
            << "Delimiter        : " << delimiter << '\n'
            << "Seed             : " << seed << '\n';

        if (!skipCols.empty()) {
            cout << "Skipped columns  : ";
            for (int x : skipCols)
                cout << x << ' ';
            cout << '\n';
        }

        /*
            Normalize before skyline and utility calculations.
        */
        normalizeDataset(D);

        /*
            Dominated tuples cannot be optimal for any
            non-negative monotone linear utility, so we can
            safely work on the skyline.
        */
        const int originalN = (int)D.size();

        cout << "\nComputing skyline...\n";

        D = computeSkyline(D);

        cout
            << "Original tuples  : " << originalN << '\n'
            << "Skyline tuples   : " << D.size() << '\n'
            << "Removed          : "
            << originalN - (int)D.size()
            << '\n';

        if (k > (int)D.size()) {
            throw runtime_error(
                "After skyline reduction, k = " +
                to_string(k) +
                " is larger than skyline size = " +
                to_string(D.size()) +
                "."
            );
        }

        /*
            ARR utilities.
        */
        mt19937_64 rng(seed);

        cout << "\nGenerating ARR utility samples...\n";

        auto weights =
            generateUtilityFunctions(
                d,
                arrSamples,
                rng
            );

        cout << "Precomputing ARR utilities...\n";

        auto utilityValues =
            precomputeUtilities(
                D,
                weights
            );

        vector<ld> bestD(
            arrSamples,
            0.0L
        );

        for (int u = 0; u < arrSamples; ++u) {
            for (int p = 0; p < (int)D.size(); ++p) {
                bestD[u] =
                    max(
                        bestD[u],
                        utilityValues[u][p]
                    );
            }
        }

        /*
            Run forward greedy.
        */
        string logFilename =
            "kcenter_local_search.csv";

        vector<int> answer =
            kCenterLocalSearchMTP(
                D, k, alpha, utilityValues, bestD, seed,
                maxLocalPasses, maxSwapEvaluations, logFilename
            );

        /*
            Final evaluation.
        */
        ld finalARR = 0.0L;
        ld finalMRR = 0.0L;

        ld finalF =
            calculateCombinedObjective(
                answer,
                D,
                utilityValues,
                bestD,
                alpha,
                finalARR,
                finalMRR
            );

        string selectedFilename =
            "kcenter_local_search_selected.csv";

        writeSelectedTuples(
            selectedFilename,
            answer,
            D
        );

        cout
            << "\n========================================\n"
            << "FINAL RESULT\n"
            << "========================================\n"
            << "Selected tuples : " << answer.size() << '\n'
            << setprecision(15)
            << "ARR             : " << (double)finalARR << '\n'
            << "MRR             : " << (double)finalMRR << '\n'
            << "Combined F      : " << (double)finalF << '\n'
            << "\nSelected tuple IDs:\n";

        for (int idx : answer)
            cout << D[idx].id << ' ';

        cout
            << "\n\nFiles written:\n"
            << "  " << logFilename << '\n'
            << "  " << selectedFilename << '\n';

        return 0;

    } catch (const exception& e) {
        cerr
            << "\nERROR: "
            << e.what()
            << '\n';
        return 1;
    }
}
