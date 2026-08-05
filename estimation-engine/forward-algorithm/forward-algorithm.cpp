#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <csv.h> // fast-cpp-csv-parser
#include <omp.h>

// =========================================================================
// 1. DATA STRUCTURES
// =========================================================================

#ifdef NDEBUG

#define DOUT if (true) {} else std::cout

#else

#define DOUT std::cout

#endif

struct Parameters {
    double beta0;
    double beta1;
    double beta2;
    double mu;
    double phi;        // Constrained to (-1, 1)
    double sigma_eta;  // Constrained to > 0
};

struct Data {
    const std::vector<double>& y_returns;
    Parameters prior_mean;
    Parameters prior_variance;
};

// =========================================================================
// 2. MATH UTILITIES & NUMERICALLY STABLE HELPERS
// =========================================================================

inline double log_normal_pdf(double x, double mu, double sigma) {
    if (sigma <= 1e-15) return -std::numeric_limits<double>::infinity();
    constexpr double log_sqrt_2pi = 0.918938533204672741780329736406;
    double z = (x - mu) / sigma;
    return -log_sqrt_2pi - std::log(sigma) - 0.5 * z * z;
}

// Log-Sum-Exp trick to prevent arithmetic underflow
inline double log_sum_exp(const Eigen::VectorXd& log_v) {
    double max_val = log_v.maxCoeff();
    if (!std::isfinite(max_val)) return -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (int i = 0; i < log_v.size(); ++i) {
        sum += std::exp(log_v(i) - max_val);
    }
    return max_val + std::log(sum);
}

// EXPLICIT TRANSFORMATION: Constrained -> Unconstrained space
std::vector<double> transformToUnconstrained(const Parameters& pConstrained) {
    std::vector<double> pUnconstrained(6);
    pUnconstrained[0] = pConstrained.beta0;
    pUnconstrained[1] = pConstrained.beta1;
    pUnconstrained[2] = pConstrained.beta2;
    pUnconstrained[3] = pConstrained.mu;
    // Logit transform for phi in (-1, 1)
    double phi_std = std::clamp((pConstrained.phi + 1.0) / 2.0, 1e-15, 1.0 - 1e-15);
    pUnconstrained[4] = std::log(phi_std / (1.0 - phi_std));
    // Log transform for strictly positive parameter
    pUnconstrained[5] = std::log(std::max(1e-15, pConstrained.sigma_eta));
    return pUnconstrained;
}

// EXPLICIT TRANSFORMATION: Unconstrained -> Constrained space
Parameters transformToConstrained(const std::vector<double>& pUnconstrained) {
    Parameters pConstrained;
    pConstrained.beta0 = pUnconstrained[0];
    pConstrained.beta1 = pUnconstrained[1];
    pConstrained.beta2 = pUnconstrained[2];
    pConstrained.mu = pUnconstrained[3];
    // Inverse logit transform for phi
    double exp_phi = std::exp(pUnconstrained[4]);
    pConstrained.phi = 2.0 * (exp_phi / (1.0 + exp_phi)) - 1.0;
    // Inverse log transform
    pConstrained.sigma_eta = std::exp(pUnconstrained[5]);
    return pConstrained;
}

// Log-Jacobian determinant of the Unconstrained -> Constrained transformation
double evaluateLogJacobian(const std::vector<double>& pUnconstrained) {
    // d(phi)/d(u4) = 2 * exp(u4) / (1 + exp(u4))^2
    double exp_u4 = std::exp(pUnconstrained[4]);
    double log_jac_phi = std::log(2.0) + pUnconstrained[4] - 2.0 * std::log(1.0 + exp_u4);
    // d(sigma_eta)/d(u5) = exp(u5)
    double log_jac_sigma = pUnconstrained[5];
    return log_jac_phi + log_jac_sigma;
}

double evaluateLogPrior(const Parameters& pConstrained, const Parameters& prior_mean, const Parameters& prior_variance) {
    auto log_norm = [](double x, double mean, double var) {
        if (var <= 0.0) return -std::numeric_limits<double>::infinity();
        return -0.5 * std::log(2.0 * 3.141592653589793 * var) - std::pow(x - mean, 2) / (2.0 * var);
        };

    double log_p = 0.0;
    log_p += log_norm(pConstrained.beta0, prior_mean.beta0, prior_variance.beta0);
    log_p += log_norm(pConstrained.beta1, prior_mean.beta1, prior_variance.beta1);
    log_p += log_norm(pConstrained.beta2, prior_mean.beta2, prior_variance.beta2);
    log_p += log_norm(pConstrained.mu, prior_mean.mu, prior_variance.mu);
    log_p += log_norm(pConstrained.phi, prior_mean.phi, prior_variance.phi);
    log_p += log_norm(pConstrained.sigma_eta, prior_mean.sigma_eta, prior_variance.sigma_eta);
    return log_p;
}

// =========================================================================
// 3. LOG-SPACE LIKELIHOOD APPROXIMATION
// =========================================================================


double LikelihoodApprox(
    int m, int std_dv_rng,
    double beta_0, double beta_1, double beta_2,
    double mu, double phi, double sigma_eta,
    const std::vector<double>& y_returns)
{
    Eigen::VectorXd log_delta(m);
    Eigen::VectorXd midpoints(m);

    // Variance of the stationary distribution
    double stationary_sigma = sigma_eta / std::sqrt(std::max(1e-15, 1.0 - std::pow(phi, 2)));
    double b = (2.0 * std_dv_rng * sigma_eta) / m;
    double log_b = std::log(b);

    // 1. Parallelize initial grid evaluation (O(m))
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= m; i++) {
        double midpoint = mu - std_dv_rng * sigma_eta * (1.0 - (2.0 * i - 1.0) / m);
        midpoints(i - 1) = midpoint;
        log_delta(i - 1) = log_normal_pdf(midpoint, mu, stationary_sigma);
    }

    log_delta.array() += log_b;
    double log_sum_delta = log_sum_exp(log_delta);
    log_delta.array() -= log_sum_delta;

    // 2. Parallelize transition matrix construction (O(m^2))
    Eigen::MatrixXd log_Gamma(m, m);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        double conditional_mean = mu + phi * (midpoints(i) - mu);
        for (int j = 0; j < m; j++) {
            log_Gamma(i, j) = log_normal_pdf(midpoints(j), conditional_mean, sigma_eta) + log_b;
        }
        double log_row_sum = log_sum_exp(log_Gamma.row(i));
        log_Gamma.row(i).array() -= log_row_sum;
    }

    double log_likelihood = 0.0;
    Eigen::VectorXd log_alpha = log_delta;
    int T = static_cast<int>(y_returns.size());

    // Allocate temporary vector once outside the loop to reduce memory allocations
    Eigen::VectorXd next_log_alpha(m);

    // TIME LOOP: Must remain sequential because log_alpha(t) depends on log_alpha(t-1)
    for (int t = 0; t < T; t++) {
        double y_curr = y_returns[t];
        double y_prev = (t == 0) ? 0.0 : y_returns[t - 1];

        // 3. Parallelize observation log-likelihood updates across m grid points
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < m; j++) {
            double h_t = midpoints(j);
            double mean_y = beta_0 + beta_1 * y_prev + beta_2 * std::exp(h_t);
            double std_dev_y = std::exp(h_t / 2.0);
            log_alpha(j) += log_normal_pdf(y_curr, mean_y, std_dev_y);
        }

        // Predictive probability normalization
        double step_log_sum = log_sum_exp(log_alpha);
        if (!std::isfinite(step_log_sum)) return -1e15;

        log_likelihood += step_log_sum;
        log_alpha.array() -= step_log_sum;

        // 4. Parallelize transition step across grid states (O(m^2))
        if (t < T - 1) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < m; j++) {
                // Read-only access to log_alpha and log_Gamma.col(j) is thread-safe
                Eigen::VectorXd log_terms = log_alpha + log_Gamma.col(j);
                next_log_alpha(j) = log_sum_exp(log_terms);
            }
            log_alpha = next_log_alpha;
        }
    }

    return log_likelihood;
}

// =========================================================================
// 4. MAP ESTIMATION (L-BFGS & NUMERICAL DERIVATIVES)
// =========================================================================

double EvaluateLogPosterior(const std::vector<double>& pUnconstrained, Data* data) {
    Parameters pConstrained = transformToConstrained(pUnconstrained);
    double log_prior = evaluateLogPrior(pConstrained, data->prior_mean, data->prior_variance);
    double log_jac = evaluateLogJacobian(pUnconstrained);
    double log_lik = LikelihoodApprox(100, 4, pConstrained.beta0, pConstrained.beta1, pConstrained.beta2,
        pConstrained.mu, pConstrained.phi, pConstrained.sigma_eta, data->y_returns);
    return log_prior + log_jac + log_lik;
}

// 5-Point Central Finite Difference Gradient Approximation
void ComputeNumericalGradient(const std::vector<double>& x, std::vector<double>& grad, Data* data) {
    const size_t n = x.size();
    const double eps = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);

    for (size_t i = 0; i < n; ++i) {
        double h = eps * std::max(1.0, std::abs(x[i]));
        std::vector<double> x_p2 = x, x_p1 = x, x_m1 = x, x_m2 = x;
        x_p2[i] += 2.0 * h;
        x_p1[i] += h;
        x_m1[i] -= h;
        x_m2[i] -= 2.0 * h;

        double f_p2 = -EvaluateLogPosterior(x_p2, data);
        double f_p1 = -EvaluateLogPosterior(x_p1, data);
        double f_m1 = -EvaluateLogPosterior(x_m1, data);
        double f_m2 = -EvaluateLogPosterior(x_m2, data);

        grad[i] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * h);
    }
}

// Objective Function compatible with NLopt L-BFGS
double NLLObjectiveFunc(const std::vector<double>& paramsUnconstrained, std::vector<double>& grad, void* my_func_data) {
    Data* data = static_cast<Data*>(my_func_data);
    if (!grad.empty()) {
        ComputeNumericalGradient(paramsUnconstrained, grad, data);
    }
    double total_log_posterior = EvaluateLogPosterior(paramsUnconstrained, data);
	DOUT <<  "NLLObjectiveFunc ran successfully: " << -total_log_posterior << std::endl;
    DOUT <<  "Params: ";
    for (double p : paramsUnconstrained) { DOUT <<  p << ", "; }
    DOUT <<  std::endl;
    return -total_log_posterior; // Minimizing Negative Log Posterior
}

// Compute Hessian of Negative Log-Posterior via Central Differences
Eigen::MatrixXd ComputeHessian(const std::vector<double>& x, Data* data) {
    const size_t n = x.size();
    const double eps = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 4.0);
    Eigen::MatrixXd Hessian(n, n);
    double f_0 = -EvaluateLogPosterior(x, data);

    for (size_t i = 0; i < n; ++i) {
        double h_i = eps * std::max(1.0, std::abs(x[i]));
        for (size_t j = i; j < n; ++j) {
            double h_j = eps * std::max(1.0, std::abs(x[j]));

            if (i == j) {
                std::vector<double> x_p = x, x_m = x;
                x_p[i] += h_i;
                x_m[i] -= h_i;
                double f_p = -EvaluateLogPosterior(x_p, data);
                double f_m = -EvaluateLogPosterior(x_m, data);
                Hessian(i, i) = (f_p - 2.0 * f_0 + f_m) / (h_i * h_i);
            }
            else {
                std::vector<double> x_pp = x, x_pm = x, x_mp = x, x_mm = x;
                x_pp[i] += h_i; x_pp[j] += h_j;
                x_pm[i] += h_i; x_pm[j] -= h_j;
                x_mp[i] -= h_i; x_mp[j] += h_j;
                x_mm[i] -= h_i; x_mm[j] -= h_j;

                double f_pp = -EvaluateLogPosterior(x_pp, data);
                double f_pm = -EvaluateLogPosterior(x_pm, data);
                double f_mp = -EvaluateLogPosterior(x_mp, data);
                double f_mm = -EvaluateLogPosterior(x_mm, data);

                double d2f = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_i * h_j);
                Hessian(i, j) = d2f;
                Hessian(j, i) = d2f; // Ensure symmetry
            }
        }
    }
    // Symmetrize explicitly
    Hessian = 0.5 * (Hessian + Hessian.transpose());
    return Hessian;
}

std::vector<double> EstimateMAP(const Parameters& initial_guess, Data& data) {
    nlopt::opt opt(nlopt::LN_BOBYQA, 6);
    opt.set_min_objective(NLLObjectiveFunc, &data);
    opt.set_xtol_rel(1e-3);
    opt.set_ftol_rel(1e-3);
    opt.set_maxeval(500);

    std::vector<double> x = transformToUnconstrained(initial_guess);
    double min_nll;

    DOUT <<  "Starting L-BFGS Optimization..." << std::endl;
    nlopt::result result = opt.optimize(x, min_nll);
    DOUT <<  "Optimization converged (Code: " << result << "). Min NLL: " << min_nll << std::endl;

    return x;
}

// =========================================================================
// 5. POSTERIOR ESTIMATION (IMPORTANCE SAMPLING)
// =========================================================================

double evaluateLogProposal(const Eigen::VectorXd& theta_tilde,
    const Eigen::VectorXd& map_mean,
    const Eigen::MatrixXd& cov_inv,
    double log_det_cov) {
    Eigen::VectorXd diff = theta_tilde - map_mean;
    double quad_form = diff.transpose() * cov_inv * diff;
    return -0.5 * (6.0 * std::log(2.0 * 3.141592653589793) + log_det_cov + quad_form);
}

Parameters estimatePosteriorMean(const std::vector<double>& y_returns,
    const std::vector<double>& map_tilde_peak,
    const Eigen::MatrixXd& covariance_matrix,
    const Parameters& prior_mean,
    const Parameters& prior_variance,
    int N_samples = 1000)
{
    std::mt19937_64 rng(42);

    // Cholesky decomposition of Sigma to sample multivariate normals
    Eigen::LLT<Eigen::MatrixXd> lltOfCov(covariance_matrix);
    if (lltOfCov.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Covariance matrix is not positive-definite!");
    }
    Eigen::MatrixXd L = lltOfCov.matrixL();
    Eigen::MatrixXd cov_inv = covariance_matrix.inverse();
    double log_det_cov = 2.0 * L.diagonal().array().log().sum();

    Eigen::VectorXd map_mean_vec = Eigen::Map<const Eigen::VectorXd>(map_tilde_peak.data(), 6);

    std::vector<Parameters> sampled_params(N_samples);
    Eigen::VectorXd log_weights(N_samples);

    DOUT <<  "Starting Multivariate Importance Sampling (N = " << N_samples << ")...\n";

    int valid_samples = 0;
    for (int i = 0; i < N_samples; ++i) {
        Eigen::VectorXd z(6);
        for (int k = 0; k < 6; ++k) {
            std::normal_distribution<double> std_norm(0.0, 1.0);
            z(k) = std_norm(rng);
        }
        Eigen::VectorXd theta_tilde_i = map_mean_vec + L * z;

        std::vector<double> theta_tilde_vec(theta_tilde_i.data(), theta_tilde_i.data() + 6);
        Parameters pConstrained_i = transformToConstrained(theta_tilde_vec);
        sampled_params[i] = pConstrained_i;

        // 1. Guard against unconstrained parameter explosions before likelihood evaluation
        if (!std::isfinite(pConstrained_i.sigma_eta) || pConstrained_i.sigma_eta <= 1e-6 ||
            !std::isfinite(pConstrained_i.phi) || std::abs(pConstrained_i.phi) >= 0.999)
        {
            log_weights(i) = -1e308; // Assign virtually zero weight to invalid samples
            continue;
        }

        double log_lik = LikelihoodApprox(100, 4, pConstrained_i.beta0, pConstrained_i.beta1,
            pConstrained_i.beta2, pConstrained_i.mu,
            pConstrained_i.phi, pConstrained_i.sigma_eta, y_returns);

        // 2. Reject penalty score (-1e15) from ruining LogSumExp
        if (log_lik <= -1e14 || !std::isfinite(log_lik)) {
            log_weights(i) = -1e308;
            continue;
        }

        double log_prior = evaluateLogPrior(pConstrained_i, prior_mean, prior_variance);
        double log_jac = evaluateLogJacobian(theta_tilde_vec);
        double log_prop = evaluateLogProposal(theta_tilde_i, map_mean_vec, cov_inv, log_det_cov);

        double w = (log_lik + log_prior + log_jac) - log_prop;

        if (std::isfinite(w)) {
            log_weights(i) = w;
            valid_samples++;
        }
        else {
            log_weights(i) = -1e308;
        }
    }

    if (valid_samples == 0) {
        throw std::runtime_error("Importance Sampling failed: All sampled proposal points evaluated to invalid likelihoods!");
    }

    // Stable weight normalization via Log-Sum-Exp
    double log_sum_w = log_sum_exp(log_weights);
    Eigen::VectorXd norm_weights = (log_weights.array() - log_sum_w).exp();

    // Diagnostic: Compute Effective Sample Size (ESS)
    double ess = 1.0 / norm_weights.array().square().sum();
    DOUT <<  "Importance Sampling complete. Valid points: " << valid_samples
        << "/" << N_samples << " | ESS: " << ess << "\n";

    // 3. Compute weighted point estimate using uniform Eigen accessor syntax ()
    Parameters point_estimate = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < N_samples; ++i) {
        // Skip zero-weight invalid proposals
        if (norm_weights(i) <= 0.0 || !std::isfinite(norm_weights(i))) continue;

        point_estimate.beta0 += norm_weights(i) * sampled_params[i].beta0;
        point_estimate.beta1 += norm_weights(i) * sampled_params[i].beta1;
        point_estimate.beta2 += norm_weights(i) * sampled_params[i].beta2; // Fixed [] -> ()
        point_estimate.mu += norm_weights(i) * sampled_params[i].mu;
        point_estimate.phi += norm_weights(i) * sampled_params[i].phi;
        point_estimate.sigma_eta += norm_weights(i) * sampled_params[i].sigma_eta;
    }

    return point_estimate;
}

// =========================================================================
// 6. MAIN ROUTINE
// =========================================================================

void ReadCSVData(std::vector<double>& y_returns) {
    try {
        io::CSVReader<2> in("../data/gold18_history_log_returns.csv");
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

void ReadDataSTDIN(std::vector<double>& y_returns) {
    size_t n;
    std::cin >> n;

    y_returns.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        double x;
        if (!(std::cin >> x))
            throw std::runtime_error("Unexpected end of input.");

        y_returns.push_back(x);
    }
}

void LoadData(std::vector<double>& y_returns) {
#if NDEBUG
    return ReadDataSTDIN(y_returns);
#else
    return ReadCSVData(y_returns);
#endif // NDEBUG
}

int main() {
    try {
        std::vector<double> y_returns;
        LoadData(y_returns);
        if (y_returns.empty()) {
            throw std::runtime_error("No data loaded from CSV.");
        }

        Parameters initial_guess = { 0.0, 0.0, 0.0, 1.0, 0.6, 0.8 };
        Parameters prior_mean = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 };
        Parameters prior_variance = { 1.0, 1.0, 1.0, 4.0, 0.25, 0.25 };

        Data modeling_data = { y_returns, prior_mean, prior_variance };

        // Step 1: Estimate MAP via L-BFGS
        DOUT <<  "--- Starting MAP Estimation (L-BFGS) ---\n";
        std::vector<double> map_tilde_peak = EstimateMAP(initial_guess, modeling_data);

       
        // Step 2: Compute Symmetric Hessian at MAP Peak
        DOUT <<  "\n--- Computing Finite Difference Hessian ---\n";
        Eigen::MatrixXd Hessian = ComputeHessian(map_tilde_peak, &modeling_data);

        DOUT <<  "Final Hessian: \n" << Hessian << std::endl;

        // Regularize Hessian to prevent inversion failure due to numerical drift
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hessian);
        Eigen::VectorXd eigenvalues = es.eigenvalues();
        Eigen::MatrixXd eigenvectors = es.eigenvectors();
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (eigenvalues(i) < 1e-8) eigenvalues(i) = 1e-8; // Ensure positive definiteness
        }
        Eigen::MatrixXd reg_Hessian = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        Eigen::MatrixXd covariance_matrix = reg_Hessian.inverse();
		DOUT <<  "Final Covariance Matrix (Inverse Hessian):\n" << covariance_matrix << std::endl;

        // Step 3: Estimate Posterior Means via Multivariate Importance Sampling
        Parameters final_theta_bayes = estimatePosteriorMean(
            y_returns, map_tilde_peak, covariance_matrix, prior_mean, prior_variance, 1000
        );

        // Step 4: Output Estimated Parameters
        DOUT <<  std::fixed << std::setprecision(10);
        DOUT <<  "\n--- Final Estimated Point Parameters (Posterior Means) ---\n";
        DOUT <<  "beta0:     " << final_theta_bayes.beta0 << "\n";
        DOUT <<  "beta1:     " << final_theta_bayes.beta1 << "\n";
        DOUT <<  "beta2:     " << final_theta_bayes.beta2 << "\n";
        DOUT <<  "mu:        " << final_theta_bayes.mu << "\n";
        DOUT <<  "phi:       " << final_theta_bayes.phi << "\n";
        DOUT <<  "sigma_eta: " << final_theta_bayes.sigma_eta << "\n";

        std::cout << final_theta_bayes.beta0 << "\n";
        std::cout << final_theta_bayes.beta1 << "\n";
        std::cout << final_theta_bayes.beta2 << "\n";
        std::cout << final_theta_bayes.mu << "\n";
        std::cout << final_theta_bayes.phi << "\n";
        std::cout << final_theta_bayes.sigma_eta;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error in Execution: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}