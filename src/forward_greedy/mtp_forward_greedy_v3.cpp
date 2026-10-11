#include <bits/stdc++.h>
using namespace std;

using ld = long double;
static constexpr ld EPS = 1e-12L;

/*
============================================================
 MTP FORWARD GREEDY
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
                         [delimiter] [skip_cols] [max_rows] [seed]

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
CHEAP SAMPLED-MRR LOWER BOUND

The exact MRR is a supremum over all nonnegative linear
utility functions. The sampled utility functions used for ARR
are a subset of those utilities, so the maximum regret over
the sampled utilities is a valid LOWER BOUND on exact MRR.

For a candidate set S', this lets us prune a candidate before
solving any MRR LPs if:

    alpha * ARR(S') + (1-alpha) * sampledMRR(S')
        >= current best F.

This is safe because exact MRR(S') >= sampledMRR(S').
============================================================
*/
static pair<ld, ld> calculateARRAndSampleMRR(
    const vector<int>& selected,
    const vector<vector<ld>>& utilityValues,
    const vector<ld>& bestD
) {
    const int N = (int)utilityValues.size();

    if (N == 0)
        return {0.0L, 0.0L};

    ld totalRegret = 0.0L;
    ld sampledMRR = 0.0L;

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
        sampledMRR = max(sampledMRR, regret);
    }

    return {
        totalRegret / (ld)N,
        sampledMRR
    };
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


static ld calculateMRRWithCutoff(
    const vector<int>& selected,
    const vector<Point>& D,
    ld cutoff,
    bool& pruned,
    long long* lpSolves = nullptr
) {
    const int n = (int)D.size();
    pruned = false;
    if (lpSolves) *lpSolves = 0;

    if (selected.empty())
        return (n > 0 ? 1.0L : 0.0L);

    vector<char> inSelected(n, false);
    for (int idx : selected)
        inSelected[idx] = true;

    ld mrr = 0.0L;

    for (int q = 0; q < n; ++q) {
        if (inSelected[q])
            continue;

        ld regret = solveCandidateMRR(q, selected, D, nullptr);

        if (lpSolves)
            ++(*lpSolves);

        if (regret > mrr)
            mrr = regret;

        // MRR is a maximum over all remaining tuples.
        // Therefore partial MRR can only stay the same or increase.
        // Once it reaches the cutoff, this candidate cannot beat
        // the current best combined objective.
        if (mrr >= cutoff - EPS) {
            pruned = true;
            return mrr;
        }
    }

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
static vector<int> forwardGreedyMTP(
    const vector<Point>& D,
    int k,
    ld alpha,
    const vector<vector<ld>>& utilityValues,
    const vector<ld>& bestD,
    const string& logFilename
) {
    const int n = (int)D.size();

    if (k <= 0)
        return {};

    if (k > n)
        throw runtime_error("k cannot exceed the number of tuples.");

    vector<int> selected;
    selected.reserve(k);

    vector<char> chosen(n, false);

    ofstream logFile(logFilename);

    if (!logFile.is_open())
        throw runtime_error("Cannot create log file: " + logFilename);

    logFile
        << "iteration,set_size,inserted_tuple_id,"
        << "ARR,MRR,combined_F,"
        << "iteration_seconds,cumulative_seconds\n";

    auto totalStart = chrono::high_resolution_clock::now();

    cout << "\n========================================\n";
    cout << "FORWARD GREEDY MTP\n";
    cout << "========================================\n";

    for (int iteration = 1; iteration <= k; ++iteration) {
        auto iterationStart = chrono::high_resolution_clock::now();

        ld bestObjective = numeric_limits<ld>::infinity();
        ld bestARR = 0.0L;
        ld bestMRR = 0.0L;
        int bestCandidate = -1;

        const int candidatesRemaining =
            n - (int)selected.size();

        int candidateNumber = 0;

        long long prunedCandidates = 0;
        long long prunedLPSolves = 0;
        long long cheapBoundPrunedCandidates = 0;
        long long fullMRRCandidates = 0;
        long long fullMRRLPSolves = 0;

        for (int p = 0; p < n; ++p) {
            if (chosen[p])
                continue;

            ++candidateNumber;

            vector<int> candidateSet = selected;
            candidateSet.push_back(p);

            // Compute ARR and a cheap sampled-MRR lower bound
            // together. The sampled MRR is a valid lower bound
            // on the exact MRR, so it can safely prune candidates
            // before any LP solve.
            auto [arr, sampledMRR] =
                calculateARRAndSampleMRR(
                    candidateSet,
                    utilityValues,
                    bestD
                );

            ld mrr = 0.0L;
            ld objective = numeric_limits<ld>::infinity();

            bool pruned = false;
            long long lpUsed = 0;

            if (alpha >= 1.0L - EPS) {
                // F = ARR, so MRR does not need to be evaluated.
                objective = arr;
            }
            else {
                // Cheap lower-bound pruning:
                //
                // exact MRR(S') >= sampledMRR(S')
                //
                // therefore
                // alpha*ARR + (1-alpha)*sampledMRR
                // is a lower bound on the true F(S').
                ld sampledObjective =
                    alpha * arr +
                    (1.0L - alpha) * sampledMRR;

                if (sampledObjective >= bestObjective - EPS) {
                    pruned = true;
                }
                else if (alpha * arr >= bestObjective - EPS) {
                    // Even MRR = 0 cannot make this candidate better.
                    pruned = true;
                }
                else {
                    const ld mrrCutoff =
                        (bestObjective - alpha * arr) /
                        (1.0L - alpha);

                    mrr = calculateMRRWithCutoff(
                        candidateSet,
                        D,
                        mrrCutoff,
                        pruned,
                        &lpUsed
                    );

                    if (!pruned) {
                        objective =
                            alpha * arr +
                            (1.0L - alpha) * mrr;
                    }
                }
            }

            if (pruned) {
                ++prunedCandidates;
                prunedLPSolves += lpUsed;

                // lpUsed == 0 means the candidate was eliminated
                // by the cheap sampled-MRR/ARR lower bound.
                if (lpUsed == 0)
                    ++cheapBoundPrunedCandidates;
            }
            else {
                ++fullMRRCandidates;
                fullMRRLPSolves += lpUsed;

                if (objective < bestObjective - EPS) {
                    bestObjective = objective;
                    bestCandidate = p;
                    bestARR = arr;
                    bestMRR = mrr;
                }
            }

            if (
                candidateNumber % max(1, candidatesRemaining / 10) == 0 ||
                candidateNumber == candidatesRemaining
            ) {
                cout
                    << "\rIteration "
                    << iteration
                    << " | tested "
                    << candidateNumber
                    << "/"
                    << candidatesRemaining
                    << flush;
            }
        }

        cout << "\n";

        cout
            << "  Pruned candidates: " << prunedCandidates
            << " | Cheap-bound pruned: " << cheapBoundPrunedCandidates
            << " | LP solves used by pruned candidates: " << prunedLPSolves
            << " | Full MRR candidates: " << fullMRRCandidates
            << " | LP solves for full MRR: " << fullMRRLPSolves
            << "\n";

        if (bestCandidate == -1)
            throw runtime_error("No candidate found during insertion.");

        selected.push_back(bestCandidate);
        chosen[bestCandidate] = true;

        auto iterationEnd = chrono::high_resolution_clock::now();

        double iterationSeconds =
            chrono::duration<double>(
                iterationEnd - iterationStart
            ).count();

        double cumulativeSeconds =
            chrono::duration<double>(
                iterationEnd - totalStart
            ).count();

        cout << fixed << setprecision(12);

        cout
            << "Iteration "
            << iteration
            << " | Insert tuple "
            << D[bestCandidate].id
            << " | ARR = "
            << (double)bestARR
            << " | MRR = "
            << (double)bestMRR
            << " | F = "
            << (double)bestObjective
            << " | Time = "
            << setprecision(3)
            << iterationSeconds
            << " s\n";

        logFile
            << iteration
            << ','
            << selected.size()
            << ','
            << D[bestCandidate].id
            << ','
            << setprecision(15)
            << (double)bestARR
            << ','
            << (double)bestMRR
            << ','
            << (double)bestObjective
            << ','
            << setprecision(9)
            << iterationSeconds
            << ','
            << cumulativeSeconds
            << '\n';

        logFile.flush();

        cout << "Selected IDs: ";
        for (int idx : selected)
            cout << D[idx].id << ' ';
        cout << "\n\n";
    }

    logFile.close();

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
        << " [delimiter] [skip_cols] [max_rows] [seed]\n\n"

        << "Arguments:\n"
        << "  dataset.csv   CSV dataset with one header row\n"
        << "  k             desired number of selected tuples\n"
        << "  alpha         ARR weight, 0 <= alpha <= 1\n"
        << "  arr_samples   number of sampled utility functions for ARR\n"
        << "  delimiter     optional: ',' or ';' (default: auto-detect)\n"
        << "  skip_cols     optional comma-separated zero-based columns\n"
        << "                to ignore, e.g. 11 or 3,8\n"
        << "  max_rows      optional maximum number of rows; -1 = all\n"
        << "  seed          optional random seed (default: 42)\n\n"

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
        if (argc < 5 || argc > 9) {
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
            << "MTP FORWARD GREEDY\n"
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
            "forward_greedy_iterations.csv";

        vector<int> answer =
            forwardGreedyMTP(
                D,
                k,
                alpha,
                utilityValues,
                bestD,
                logFilename
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
            "forward_greedy_selected.csv";

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
