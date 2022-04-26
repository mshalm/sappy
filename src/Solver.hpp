#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;
using namespace std;
namespace py = pybind11;

class Solver
{
    public:
    Solver(int NC, int NV);
    void project_lorentz(VectorXd &vector);
    double project_lorentz_half_squared(VectorXd &vector);
    void root_hess_project_lorentz_half_squared(VectorXd &vector, MatrixXd &root_hess);
    std::tuple<VectorXd,VectorXd> solve(const py::EigenDRef<const MatrixXd> &J, const py::EigenDRef<const VectorXd> &q, double eps);
    std::tuple<MatrixXd,VectorXd> gradient(const py::EigenDRef<const MatrixXd> &J,
                                           const py::EigenDRef<const VectorXd> &q,
                                           const py::EigenDRef<const VectorXd> &l_sol,
                                           const py::EigenDRef<const VectorXd> &v_sol,
                                           const py::EigenDRef<const VectorXd> &backwards_grad, double eps);

    private:
    // solver instance size
    int _NC;
    int _NL;
    int _NV;

    // Eigen workspace
    VectorXd y;
    VectorXd l;
    VectorXd v;
    VectorXd J_t_l;
    VectorXd grad_loss;
    MatrixXd hess_loss;
    MatrixXd hess_projection;
    MatrixXd root_hess_projection;
    MatrixXd J_t_root_hess;
    MatrixXd J_t_hess;
    Matrix3d hess_block;
    VectorXd dv;
    VectorXd v_alpha;
    VectorXd y_alpha;
    LDLT<MatrixXd> hess_chol;
    VectorXd lt;
    VectorXd lt_hat;
    VectorXd R;
    VectorXd backwards_grad_times_hessian;
    VectorXd grad_q;
    MatrixXd grad_J;

    // projection workspace
    int idx;
    double lt_norm;
    double ln;
    double s;
    double s_over_2;
    double s_normed;

    // solver outer loop settings
    double max_iterations;
    double grad_abs_tolerance;
    double grad_rel_tolerance; 
    double cost_abs_tolerance;
    double cost_rel_tolerance;

    // line search settings
    double rho;
    double c;
    int max_ls_iterations;
    double alpha_max;

    // iteration variables
    double loss;
    double loss_prev;
    double alpha;
    double alpha_prev;
    double loss_v_alpha;
    double loss_v_alpha_prev;
    double grad_v_alpha;
    double hess_v_alpha;
    double grad_l_alpha0;

    bool momentum_stop;
    bool decrement_stop;
};