#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <Eigen/Dense>
#include <csv.h>

#define M_PI std::acos(-1)

// ============================================================================
// CONFIGURATION & ESTIMATED PARAMETERS CONSTANTS
// ============================================================================
namespace Config {
    // 1. Estimated Model Parameters
    constexpr double BETA_0 = 0.01;   // Intercept parameter in return equation
    constexpr double BETA_1 = 0.05;   // Autoregressive return parameter
    constexpr double BETA_2 = -0.02;  // Volatility feedback (in-mean) parameter
    constexpr double MU = -0.5;   // Long-run mean of log-volatility
    constexpr double PHI = 0.95;   // Log-volatility persistence (|PHI| < 1)
    constexpr double SIGMA_ETA = 0.25;   // Volatility of log-volatility
    constexpr double NU = 5.0;    // Student-t degrees of freedom (tail parameter)

    // 2. Discretization Settings
    constexpr int    M = 100;   // Number of discrete volatility grid states
    constexpr double STD_DV_RNG = 4.0;   // Range multiplier for grid boundaries

    // 3. Risk Management Settings
    constexpr double ALPHA = 0.05;  // VaR significance level (e.g., 5% VaR)

    // 4. Numerical Solver Settings
    constexpr double ROOT_TOL = 1e-7;  // Convergence tolerance for Brent's algorithm
    constexpr int    MAX_ITER = 100;   // Maximum iterations for root finding
}

// ============================================================================
// HELPER FUNCTIONS: PROBABILITY DISTRIBUTIONS & SPECIAL FUNCTIONS
// ============================================================================

// Standard Normal PDF (Log-scale to prevent numerical underflow)
double log_normal_pdf(double x, double mean, double std_dev) {
    constexpr double LOG_SQRT_2PI = 0.9189385332046727;
    double z = (x - mean) / std_dev;
    return -LOG_SQRT_2PI - std::log(std_dev) - 0.5 * z * z;
}

// Incomplete Gamma / Beta helpers for standard Student-t CDF implementation
// Continuous fraction expansion for Regularized Incomplete Beta function I_x(a, b)
double incbeta_continued_fraction(double a, double b, double x) {
    constexpr int MAX_IT = 200;
    constexpr double EPS = 3.0e-7;
    constexpr double FPMIN = 1.0e-30;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= MAX_IT; ++m) {
        int m2 = 2 * m;
        // Even step
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;

        if (std::abs(del - 1.0) <= EPS) break;
    }
    return h;
}

// Regularized Incomplete Beta Function I_x(a, b)
double incbeta(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
        a * std::log(x) + b * std::log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * incbeta_continued_fraction(a, b, x) / a;
    }
    else {
        return 1.0 - bt * incbeta_continued_fraction(b, a, 1.0 - x) / b;
    }
}

// Student-t Cumulative Distribution Function (CDF) F(z | nu)
double student_t_cdf(double z, double nu) {
    double x = nu / (nu + z * z);
    double ibeta = incbeta(0.5 * nu, 0.5, x);

    if (z >= 0.0) {
        return 1.0 - 0.5 * ibeta;
    }
    else {
        return 0.5 * ibeta;
    }
}

// Student-t Probability Density Function (PDF)
double student_t_pdf(double y, double mean, double scale, double nu) {
    double z = (y - mean) / scale;
    double log_c = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
        - 0.5 * std::log(M_PI * nu) - std::log(scale);
    double log_pdf = log_c - ((nu + 1.0) / 2.0) * std::log(1.0 + (z * z) / nu);
    return std::exp(log_pdf);
}

// ============================================================================
// ROOT FINDING ALGORITHMS (BRACKETING & BRENT'S METHOD)
// ============================================================================

// Automatically expands search boundaries to bracket a zero: f(a) * f(b) <= 0
std::pair<double, double> bracket_root(const std::function<double(double)>& f, double x_init) {
    double a = x_init - 0.5;
    double b = x_init + 0.5;
    double factor = 1.6;

    for (int i = 0; i < 50; ++i) {
        if (f(a) * f(b) <= 0.0) {
            return { a, b };
        }
        if (std::abs(f(a)) < std::abs(f(b))) {
            a += factor * (a - b);
        }
        else {
            b += factor * (b - a);
        }
    }
    throw std::runtime_error("Failed to bracket root.");
}

// Brent's 1D Root-Finding Algorithm
double brent_root_finder(const std::function<double(double)>& f, double a, double b,
    double tol = Config::ROOT_TOL, int max_iter = Config::MAX_ITER) {
    double fa = f(a);
    double fb = f(b);

    if (fa * fb > 0.0) {
        throw std::invalid_argument("Root is not bracketed in [a, b].");
    }

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fb) < tol || std::abs(b - a) < tol) {
            return b; // Convergence reached
        }

        double s;
        if (fa != fc && fb != fc) {
            // Inverse Quadratic Interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc))
                + (b * fa * fc) / ((fb - fa) * (fb - fc))
                + (c * fa * fb) / ((fc - fa) * (fc - fb));
        }
        else {
            // Secant Method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Bisection fallback condition evaluation
        bool cond1 = (s < (3.0 * a + b) / 4.0 || s > b);
        bool cond2 = (mflag && std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = (!mflag && std::abs(s - b) >= std::abs(c - d) / 2.0);
        bool cond4 = (mflag && std::abs(b - c) < tol);
        bool cond5 = (!mflag && std::abs(c - d) < tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0; // Bisection step
            mflag = true;
        }
        else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        }
        else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    return b;
}

void ReadCSVData(std::vector<double>& y_returns) {
    try {
        io::CSVReader<2> in("../../data-processing/gold18_history_log_returns.csv");
        in.read_header(io::ignore_extra_column, "", "Log_Return");
        double x; double ret;
        while (in.read_row(x, ret)) {
            y_returns.push_back(ret);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to read CSV: " + std::string(e.what()));
    }
}

// ============================================================================
// MAIN WORKFLOW EXECUTION
// ============================================================================
int main() {
    int m = Config::M;
    double mu = Config::MU;
    double phi = Config::PHI;
    double sigma_eta = Config::SIGMA_ETA;
    double std_dv_rng = Config::STD_DV_RNG;
    double stationary_sigma = std::sqrt((sigma_eta * sigma_eta) / (1.0 - phi * phi));

    // ------------------------------------------------------------------------
    // STEP 1: INITIALIZE HMM DISCRETIZATION GRID & STATIONARY DELTA
    // ------------------------------------------------------------------------
    Eigen::VectorXd midpoints(m);
    Eigen::VectorXd log_delta(m);

    // User's specified midpoint formula initialization
    for (int i = 1; i <= m; i++) {
        double midpoint = mu - std_dv_rng * sigma_eta * (1.0 - (2.0 * i - 1.0) / m);
        midpoints(i - 1) = midpoint;
        log_delta(i - 1) = log_normal_pdf(midpoint, mu, stationary_sigma);
    }

    // Convert log stationary probabilities back to probability space (Softmax normalized)
    double max_log_delta = log_delta.maxCoeff();
    Eigen::RowVectorXd delta = (log_delta.array() - max_log_delta).exp().matrix().transpose();
    delta /= delta.sum();

    // ------------------------------------------------------------------------
    // STEP 2: CONSTRUCT TRANSITION PROBABILITY MATRIX (GAMMA)
    // ------------------------------------------------------------------------
    Eigen::MatrixXd Gamma(m, m);
    for (int i = 0; i < m; ++i) {
        double cond_mean = mu + phi * (midpoints(i) - mu);
        for (int j = 0; j < m; ++j) {
            // Unnormalized Gaussian transition probability density from state i to state j
            Gamma(i, j) = std::exp(log_normal_pdf(midpoints(j), cond_mean, sigma_eta));
        }
        // Normalize each row so transition probabilities sum to 1
        Gamma.row(i) /= Gamma.row(i).sum();
    }

    // Simulated series of daily returns y_t for T trading days
    std::vector<double> Y;
    ReadCSVData(Y);
    size_t T = Y.size();

    // Output Vector: Daily 5% VaR estimates
    Eigen::VectorXd daily_VaR(T);

    Eigen::RowVectorXd phi_filtered(m);

    // ------------------------------------------------------------------------
    // STEP 3: DAILY FILTERING, PREDICTION, AND VAR CALCULATION LOOP
    // ------------------------------------------------------------------------
    for (size_t t = 0; t < T; ++t) {
        double y_t = Y[t];
        double y_prev = (t == 0) ? 0.0 : Y[t - 1]; // Assume y_0 = 0 for initial day

        // Construct Observation Likelihood Matrix P(y_t)
        Eigen::VectorXd likelihoods(m);
        for (int i = 0; i < m; ++i) {
            double cond_mean = Config::BETA_0 + Config::BETA_1 * y_prev + Config::BETA_2 * std::exp(midpoints(i));
            double cond_scale = std::exp(midpoints(i) / 2.0);
            likelihoods(i) = student_t_pdf(y_t, cond_mean, cond_scale, Config::NU);
        }

        // Forward Filtering Step: Alpha_t and Phi_t
        Eigen::RowVectorXd alpha_t(m);
        if (t == 0) {
            alpha_t = delta.cwiseProduct(likelihoods.transpose());
        }
        else {
            alpha_t = (phi_filtered * Gamma).cwiseProduct(likelihoods.transpose());
        }

        // Normalize Filtered Probabilities
        phi_filtered = alpha_t / alpha_t.sum();

        // One-step-ahead Volatility Prediction: Phi_{t+1 | t}
        Eigen::RowVectorXd phi_pred = phi_filtered * Gamma;

        // Construct Objective Function g(r*) = [ Predictive_CDF(r*) ] - ALPHA = 0
        auto var_objective = [&](double r_star) -> double {
            double cdf_mix = 0.0;
            for (int j = 0; j < m; ++j) {
                double mu_next = Config::BETA_0 + Config::BETA_1 * y_t + Config::BETA_2 * std::exp(midpoints(j));
                double scale_next = std::exp(midpoints(j) / 2.0);
                double z = (r_star - mu_next) / scale_next;

                cdf_mix += phi_pred(j) * student_t_cdf(z, Config::NU);
            }
            return cdf_mix - Config::ALPHA;
            };

        // Solve for VaR using Root Bracketing and Brent's Algorithm
        double init_guess = Config::BETA_0 + Config::BETA_1 * y_t - 2.0 * std::exp(mu / 2.0);
        auto bounds = bracket_root(var_objective, init_guess);
        double var_val = brent_root_finder(var_objective, bounds.first, bounds.second);

        daily_VaR(t) = var_val;
    }

    // ------------------------------------------------------------------------
    // DISPLAY RESULTS
    // ------------------------------------------------------------------------
    std::cout << "=========================================================\n";
    std::cout << "  Daily 5% Value at Risk (VaR) Calculation Results\n";
    std::cout << "=========================================================\n";
    for (size_t t = 0; t < T; ++t) {
        std::cout << "Day " << (t + 1) << " | Return y_t: "
            << std::fixed << std::setprecision(4) << Y[t]
            << " | 5% VaR_{t+1}: " << daily_VaR(t) << "\n";
    }

    return 0;
}