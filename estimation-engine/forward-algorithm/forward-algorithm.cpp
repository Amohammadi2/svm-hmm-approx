#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include <random>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <map>
#include <algorithm>
#include <limits>

// Parallel Processing & Math Libraries
#include <omp.h>
#include <Eigen/Dense>
#include <nlopt.hpp>

// =============================================================================
// Constants & Mathematical Helpers
// =============================================================================
constexpr double PI_CONST = 3.14159265358979323846;
constexpr double LIKELIHOOD_PENALTY = -1e10;

namespace MathUtils {
    inline double norm_pdf(double x, double mean, double stddev) {
        double diff = x - mean;
        return (1.0 / (stddev * std::sqrt(2.0 * PI_CONST))) * std::exp(-0.5 * (diff * diff) / (stddev * stddev));
    }

    inline double norm_logpdf(double x, double mean, double stddev) {
        double diff = x - mean;
        return -0.5 * std::log(2.0 * PI_CONST) - std::log(stddev) - 0.5 * (diff * diff) / (stddev * stddev);
    }

    inline double student_t_logpdf(double y, double mu, double sigma, double nu) {
        double z = (y - mu) / sigma;
        double term1 = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0);
        double term2 = -0.5 * std::log(PI_CONST * nu) - std::log(sigma);
        double term3 = -((nu + 1.0) / 2.0) * std::log(1.0 + (z * z) / nu);
        return term1 + term2 + term3;
    }
}

// =============================================================================
// Lightweight JSON Parser and Serializer for IPC Communication
// =============================================================================
class SimpleJson {
public:
    static std::map<std::string, std::string> parse_object(const std::string& input) {
        std::map<std::string, std::string> kv;
        std::string s = input;

        // Remove whitespace and curly braces
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
            return c == '{' || c == '}' || c == '\r' || c == '\n';
            }), s.end());

        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t colon_pos = item.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = item.substr(0, colon_pos);
                std::string val = item.substr(colon_pos + 1);

                // Trim quotes
                key.erase(std::remove(key.begin(), key.end(), '\"'), key.end());
                val.erase(std::remove(val.begin(), val.end(), '\"'), val.end());

                kv[key] = val;
            }
        }
        return kv;
    }

    static std::vector<double> parse_array(const std::string& array_str) {
        std::vector<double> result;
        std::string s = array_str;
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
            return c == '[' || c == ']' || c == ' ';
            }), s.end());

        std::stringstream ss(s);
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (!val.empty()) {
                result.push_back(std::stod(val));
            }
        }
        return result;
    }
};

// =============================================================================
// IPC Manager
// =============================================================================
class IPCManager {
private:
    bool enable_logging;

public:
    IPCManager() : enable_logging(false) {}

    void set_logging(bool flag) {
        enable_logging = flag;
    }

    bool is_logging_enabled() const {
        return enable_logging;
    }

    void log_info(const std::string& msg) const {
        if (enable_logging) {
            std::cerr << "[INFO] " << msg << std::endl;
        }
    }

    void log_warn(const std::string& msg) const {
        if (enable_logging) {
            std::cerr << "[WARN] " << msg << std::endl;
        }
    }

    void log_error(const std::string& msg) const {
        // Errors are sent to stderr regardless of logging status
        std::cerr << "[ERROR] " << msg << std::endl;
    }

    void send_response(const std::string& json_payload) const {
        std::cout << json_payload << std::endl;
    }

    void send_failure(const std::string& error_message) const {
        std::stringstream ss;
        ss << "{"
            << "\"status\":\"error\","
            << "\"message\":\"" << error_message << "\""
            << "}";
        std::cout << ss.str() << std::endl;
    }
};

// =============================================================================
// Configurations and Parameters Data Structs
// =============================================================================
struct Hyperparameters {
    int m = 100;
    double b_limit = 5.0;
    int is_samples = 2000;
    double nu_min = 2.0;
    double nu_max = 40.0;
};

struct PriorConfig {
    Eigen::VectorXd mu_0;
    Eigen::VectorXd sigma_0;

    PriorConfig() {
        mu_0.resize(7);
        sigma_0.resize(7);
        mu_0 << 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0;
        sigma_0 << 10.0, 3.16, 10.0, 10.0, 1.0, 1.0, 10.0;
    }
};

struct EstimationResult {
    Eigen::VectorXd map_estimate_unc;
    std::map<std::string, double> map_estimate_con;
    Eigen::MatrixXd inv_hessian;
    std::map<std::string, double> posterior_mean_con;
    bool success;
    std::string message;
};

// =============================================================================
// Parameter Transformation Module
// =============================================================================
class ParameterTransformer {
private:
    double nu_min, nu_max, nu_a, nu_c;

public:
    ParameterTransformer(double min_nu = 2.0, double max_nu = 40.0)
        : nu_min(min_nu), nu_max(max_nu) {
        nu_a = (nu_max - nu_min) / 2.0;
        nu_c = (nu_max + nu_min) / 2.0;
    }

    Eigen::VectorXd to_constrained(const Eigen::VectorXd& theta_u) const {
        Eigen::VectorXd theta_c(7);
        double beta0 = theta_u(0);
        double gamma = theta_u(1);
        double beta2 = theta_u(2);
        double mu = theta_u(3);
        double psi = theta_u(4);
        double omega = theta_u(5);
        double xi = theta_u(6);

        double beta1 = std::tanh(gamma / 2.0);
        double phi = std::tanh(psi / 2.0);
        double sigma_eta = std::exp(omega);
        double nu = nu_a * std::tanh(xi) + nu_c;

        theta_c << beta0, beta1, beta2, mu, phi, sigma_eta, nu;
        return theta_c;
    }

    std::map<std::string, double> to_dict(const Eigen::VectorXd& theta_c) const {
        return {
            {"beta0", theta_c(0)}, {"beta1", theta_c(1)}, {"beta2", theta_c(2)},
            {"mu", theta_c(3)},    {"phi", theta_c(4)},   {"sigma_eta", theta_c(5)},
            {"nu", theta_c(6)}
        };
    }
};

// =============================================================================
// Fast Bayesian SVM Estimator Core Class
// =============================================================================
class FastBayesianSVMEstimator {
public:
    using LogPDFDensityFunc = std::function<double(double, double, double, double)>;

private:
    Eigen::VectorXd y;
    int T;
    Hyperparameters hp;
    PriorConfig priors;
    ParameterTransformer transformer;
    LogPDFDensityFunc smn_logpdf;
    IPCManager& ipc;

    Eigen::VectorXd b_grid;
    double delta_b;

    void build_hmm_matrices(const Eigen::VectorXd& theta_c, Eigen::MatrixXd& Gamma, Eigen::VectorXd& delta) const {
        double mu = theta_c(3);
        double phi = theta_c(4);
        double sigma_eta = theta_c(5);

        int m = hp.m;
        Gamma.resize(m, m);

        // Transition Matrix Parallelization
#pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                double mean_j = mu + phi * (b_grid(i) - mu);
                Gamma(i, j) = MathUtils::norm_pdf(b_grid(j), mean_j, sigma_eta) * delta_b;
            }
        }

        // Numerical resilience: Normalize rows
        for (int i = 0; i < m; ++i) {
            double row_sum = Gamma.row(i).sum();
            if (row_sum <= 0.0 || std::isnan(row_sum)) {
                Gamma.row(i).setConstant(1.0 / m);
            }
            else {
                Gamma.row(i) /= row_sum;
            }
        }

        // Initial Distribution
        double stat_var = (sigma_eta * sigma_eta) / (1.0 - phi * phi + 1e-12);
        double stat_std = std::sqrt(stat_var);

        delta.resize(m);
        for (int i = 0; i < m; ++i) {
            delta(i) = MathUtils::norm_pdf(b_grid(i), mu, stat_std) * delta_b;
        }

        double delta_sum = delta.sum();
        if (delta_sum <= 0.0 || std::isnan(delta_sum)) {
            delta.setConstant(1.0 / m);
        }
        else {
            delta /= delta_sum;
        }
    }

public:
    FastBayesianSVMEstimator(const Eigen::VectorXd& data,
        LogPDFDensityFunc density_fn,
        IPCManager& ipc_ref,
        Hyperparameters hyperparams = Hyperparameters(),
        PriorConfig prior_config = PriorConfig())
        : y(data), T(static_cast<int>(data.size())), smn_logpdf(density_fn),
        ipc(ipc_ref), hp(hyperparams), priors(prior_config),
        transformer(hyperparams.nu_min, hyperparams.nu_max) {

        b_grid.resize(hp.m);
        double step = (2.0 * hp.b_limit) / (hp.m - 1);
        for (int i = 0; i < hp.m; ++i) {
            b_grid(i) = -hp.b_limit + i * step;
        }
        delta_b = step;
    }

    double log_likelihood(const Eigen::VectorXd& theta_u) const {
        Eigen::VectorXd theta_c = transformer.to_constrained(theta_u);
        double beta0 = theta_c(0);
        double beta1 = theta_c(1);
        double beta2 = theta_c(2);
        double nu = theta_c(6);

        Eigen::MatrixXd Gamma;
        Eigen::VectorXd alpha;
        build_hmm_matrices(theta_c, Gamma, alpha);

        double log_L = 0.0;
        Eigen::VectorXd sigmas = (b_grid.array() * 0.5).exp();
        Eigen::VectorXd exp_b = b_grid.array().exp();

        Eigen::VectorXd obs_probs(hp.m);
        Eigen::VectorXd log_obs_probs(hp.m);

        for (int t = 1; t < T; ++t) {
            double y_t = y(t);
            double y_t_1 = y(t - 1);

            for (int i = 0; i < hp.m; ++i) {
                double mus_i = beta0 + beta1 * y_t_1 + beta2 * exp_b(i);
                log_obs_probs(i) = smn_logpdf(y_t, mus_i, sigmas(i), nu);
            }

            // Prevent catastrophic underflow using scaling factor shift
            double max_log = log_obs_probs.maxCoeff();
            obs_probs = (log_obs_probs.array() - max_log).exp();

            // Forward Step: alpha_t = (alpha_{t-1} * Gamma) \odot P(y_t)
            alpha = (alpha.transpose() * Gamma).array() * obs_probs.array();

            // Scaling Step
            double c_t = alpha.sum();
            if (c_t <= 0.0 || std::isnan(c_t)) {
                return LIKELIHOOD_PENALTY;
            }

            alpha /= c_t;
            log_L += std::log(c_t) + max_log;
        }

        return log_L;
    }

    double log_prior(const Eigen::VectorXd& theta_u) const {
        double lp = 0.0;
        for (int i = 0; i < 7; ++i) {
            lp += MathUtils::norm_logpdf(theta_u(i), priors.mu_0(i), priors.sigma_0(i));
        }
        return lp;
    }

    double negative_log_posterior(const Eigen::VectorXd& theta_u) const {
        double ll = log_likelihood(theta_u);
        if (ll <= LIKELIHOOD_PENALTY) {
            return -LIKELIHOOD_PENALTY;
        }
        return -(ll + log_prior(theta_u));
    }

    // Static NLopt Objective Function Wrapper
    static double nlopt_objective(const std::vector<double>& x, std::vector<double>& grad, void* func_data) {
        FastBayesianSVMEstimator* estimator = static_cast<FastBayesianSVMEstimator*>(func_data);
        Eigen::VectorXd theta_u = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());

        double obj = estimator->negative_log_posterior(theta_u);

        // Finite-difference gradient computation if required by solver
        if (!grad.empty()) {
            double eps = 1e-5;
            for (size_t i = 0; i < x.size(); ++i) {
                Eigen::VectorXd theta_plus = theta_u;
                Eigen::VectorXd theta_minus = theta_u;
                theta_plus(i) += eps;
                theta_minus(i) -= eps;

                double obj_plus = estimator->negative_log_posterior(theta_plus);
                double obj_minus = estimator->negative_log_posterior(theta_minus);
                grad[i] = (obj_plus - obj_minus) / (2.0 * eps);
            }
        }
        return obj;
    }

    Eigen::MatrixXd ensure_positive_definite(const Eigen::MatrixXd& mat, double epsilon = 1e-6) const {
        Eigen::MatrixXd result = mat;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(result);
        if (es.info() != Eigen::Success || es.eigenvalues().minCoeff() <= 0.0) {
            ipc.log_warn("Matrix is not positive definite. Applying diagonal ridge regularization.");
            double min_eig = es.eigenvalues().minCoeff();
            double add_diag = (std::abs(min_eig) + epsilon);
            result += add_diag * Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
        }
        return result;
    }

    EstimationResult estimate(const std::vector<double>& init_theta) {
        ipc.log_info("Starting L-BFGS Optimization via NLopt...");

        nlopt::opt opt(nlopt::LD_LBFGS, 7); // Low-storage BFGS derivative-free mode
        opt.set_min_objective(FastBayesianSVMEstimator::nlopt_objective, this);
        opt.set_ftol_rel(1e-6);
        opt.set_maxeval(1000);

        std::vector<double> x = init_theta;
        double min_f;

        bool success = true;
        std::string msg = "Optimization converged successfully.";

        try {
            nlopt::result status = opt.optimize(x, min_f);
            ipc.log_info("NLopt completed with code: " + std::to_string(status));
        }
        catch (const std::exception& e) {
            success = false;
            msg = std::string("NLopt Exception: ") + e.what();
            ipc.log_warn(msg);
        }

        Eigen::VectorXd map_u = Eigen::Map<Eigen::VectorXd>(x.data(), 7);

        // Compute Numerical Hessian
        ipc.log_info("Computing Numerical Hessian at MAP mode...");
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(7, 7);
        double eps = 1e-4;

        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                Eigen::VectorXd p_ij = map_u, p_i = map_u, p_j = map_u, p_base = map_u;
                p_ij(i) += eps; p_ij(j) += eps;
                p_i(i) += eps;
                p_j(j) += eps;

                double f_ij = negative_log_posterior(p_ij);
                double f_i = negative_log_posterior(p_i);
                double f_j = negative_log_posterior(p_j);
                double f_base = negative_log_posterior(p_base);

                hessian(i, j) = (f_ij - f_i - f_j + f_base) / (eps * eps);
            }
        }

        Eigen::MatrixXd inv_hessian;
        Eigen::FullPivLU<Eigen::MatrixXd> lu(hessian);
        if (lu.isInvertible()) {
            inv_hessian = lu.inverse();
        }
        else {
            ipc.log_warn("Hessian matrix inversion failed. Falling back to spherical scaling.");
            inv_hessian = Eigen::MatrixXd::Identity(7, 7) * 1e-3;
        }

        inv_hessian = ensure_positive_definite(inv_hessian);

        // Importance Sampling
        ipc.log_info("Executing multi-threaded Importance Sampling (" + std::to_string(hp.is_samples) + " samples)...");

        Eigen::LLT<Eigen::MatrixXd> llt(inv_hessian);
        Eigen::MatrixXd L = llt.matrixL();

        Eigen::VectorXd log_weights(hp.is_samples);
        Eigen::MatrixXd samples_c(hp.is_samples, 7);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64 rng(1337 + tid * 10007);
            std::normal_distribution<double> norm_dist(0.0, 1.0);

#pragma omp for schedule(dynamic)
            for (int i = 0; i < hp.is_samples; ++i) {
                Eigen::VectorXd z(7);
                for (int d = 0; d < 7; ++d) {
                    z(d) = norm_dist(rng);
                }

                Eigen::VectorXd sample_u = map_u + L * z;
                samples_c.row(i) = transformer.to_constrained(sample_u).transpose();

                double log_p = -negative_log_posterior(sample_u);

                // Normal logpdf of proposal q(theta)
                Eigen::VectorXd diff = sample_u - map_u;
                double log_q = -0.5 * 7 * std::log(2.0 * PI_CONST)
                    - 0.5 * std::log(inv_hessian.determinant())
                    - 0.5 * static_cast<double>(diff.transpose() * inv_hessian.inverse() * diff);

                log_weights(i) = log_p - log_q;
            }
        }

        // Log-Sum-Exp Trick for Weight Normalization
        double max_log_w = log_weights.maxCoeff();
        Eigen::VectorXd weights = (log_weights.array() - max_log_w).exp();
        double w_sum = weights.sum();

        if (w_sum > 0.0) {
            weights /= w_sum;
        }
        else {
            ipc.log_warn("Importance weights collapsed. Assigning uniform weights.");
            weights.setConstant(1.0 / hp.is_samples);
        }

        // Posterior Means calculation
        Eigen::VectorXd posterior_mean_vec = Eigen::VectorXd::Zero(7);
        for (int i = 0; i < hp.is_samples; ++i) {
            posterior_mean_vec += weights(i) * samples_c.row(i).transpose();
        }

        EstimationResult res;
        res.map_estimate_unc = map_u;
        res.map_estimate_con = transformer.to_dict(transformer.to_constrained(map_u));
        res.inv_hessian = inv_hessian;
        res.posterior_mean_con = transformer.to_dict(posterior_mean_vec);
        res.success = success;
        res.message = msg;

        return res;
    }
};

// =============================================================================
// Application Entry Point
// =============================================================================
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    IPCManager ipc;

    try {
        // Parse input standard stream (IPC JSON Payload)
        std::string raw_input;
        std::string line;
        while (std::getline(std::cin, line)) {
            raw_input += line;
        }

        if (raw_input.empty()) {
            ipc.send_failure("Input stdin payload was empty.");
            return 1;
        }

        auto payload = SimpleJson::parse_object(raw_input);

        // Configure IPC logging flag
        if (payload.find("enable_logging") != payload.end()) {
            ipc.set_logging(payload["enable_logging"] == "true" || payload["enable_logging"] == "1");
        }

        ipc.log_info("Native Fast Bayesian SVM C++ Engine Initialized.");

        // Data array extraction
        if (payload.find("y") == payload.end()) {
            ipc.send_failure("Missing mandatory parameter 'y' in JSON payload.");
            return 1;
        }

        std::vector<double> y_vec = SimpleJson::parse_array(payload["y"]);
        Eigen::VectorXd y_data = Eigen::Map<Eigen::VectorXd>(y_vec.data(), y_vec.size());

        // Construct Configurations
        Hyperparameters hp;
        if (payload.find("m") != payload.end()) hp.m = std::stoi(payload["m"]);
        if (payload.find("b_limit") != payload.end()) hp.b_limit = std::stod(payload["b_limit"]);
        if (payload.find("is_samples") != payload.end()) hp.is_samples = std::stoi(payload["is_samples"]);

        PriorConfig priors;
        if (payload.find("mu_0") != payload.end()) {
            std::vector<double> mu_vec = SimpleJson::parse_array(payload["mu_0"]);
            if (mu_vec.size() == 7) priors.mu_0 = Eigen::Map<Eigen::VectorXd>(mu_vec.data(), 7);
        }

        // Initialize Engine with Student-t LogPDF
        FastBayesianSVMEstimator estimator(
            y_data,
            MathUtils::student_t_logpdf,
            ipc,
            hp,
            priors
        );

        std::vector<double> init_theta = { 0.0, 0.0, 0.0, 0.0, 3.0, -1.0, 0.0 };
        EstimationResult result = estimator.estimate(init_theta);

        // Serialize output JSON response
        std::stringstream ss;
        ss << std::setprecision(8);
        ss << "{"
            << "\"status\":\"" << (result.success ? "success" : "failed") << "\","
            << "\"message\":\"" << result.message << "\","
            << "\"posterior_mean_con\":{";

        size_t idx = 0;
        for (const auto& [param, val] : result.posterior_mean_con) {
            ss << "\"" << param << "\":" << val;
            if (++idx < result.posterior_mean_con.size()) ss << ",";
        }
        ss << "},";

        ss << "\"map_estimate_con\":{";
        idx = 0;
        for (const auto& [param, val] : result.map_estimate_con) {
            ss << "\"" << param << "\":" << val;
            if (++idx < result.map_estimate_con.size()) ss << ",";
        }
        ss << "}";

        ss << "}";

        ipc.send_response(ss.str());

    }
    catch (const std::exception& ex) {
        ipc.log_error(std::string("Fatal Engine Exception: ") + ex.what());
        ipc.send_failure(ex.what());
        return 1;
    }

    return 0;
}