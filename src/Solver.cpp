#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "Solver.hpp"

#include <chrono>
#include <iostream>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

using std::abs;
using std::sqrt;


#define contact_index(i) 3*i
#define y_of_v(J,v,q,eps) (-(J*v + q)/eps)
#define loss_of_yv(y,v,eps) 0.5 * v.squaredNorm() + project_lorentz_half_squared(y) * eps;
#define SQRT_HALF 0.7071067811865476

Solver::Solver(int NC, int NV){
    // precompute size variables

    int NL = 3 * NC;
    _NC = NC;
    _NV = NV;
    _NL = NL;

    // preallocate workspace
    y = VectorXd::Zero(NL);
    l = VectorXd::Zero(NL);
    v = VectorXd::Zero(NV);
    J_t_l = VectorXd::Zero(NV);
    grad_loss = VectorXd::Zero(NV);
    hess_loss = MatrixXd::Zero(NV, NV);
    hess_projection = MatrixXd::Zero(NL, NL);
    root_hess_projection = MatrixXd::Zero(NL, 3);
    J_t_root_hess = MatrixXd::Zero(NV, NL);
    dv = VectorXd::Zero(NV);
    v_alpha = VectorXd::Zero(NV);
    y_alpha = VectorXd::Zero(NC);
    LDLT<MatrixXd> hess_chol(NV);
    lt = VectorXd::Zero(2);
    lt_hat = VectorXd::Zero(2);

    // set default solver settings
    max_iterations = 100;
    grad_abs_tolerance = 1.e-10;
    grad_rel_tolerance = 2.e-6; 
    cost_abs_tolerance = 1.e-10;
    cost_rel_tolerance = 1.e-6;

    rho = 0.8;
    c = 1e-4;
    max_ls_iterations = 40;
    alpha_max = 1.5;
}


void Solver::project_lorentz(VectorXd &vector)
{//projection of l onto Lorentz cone |lt_i| <= ln_i
    for (idx = 0; idx < _NL; idx += 3)
    {
        lt(0) = vector(idx);
        lt(1) = vector(idx + 1);
        ln = vector(idx + 2);
        lt_norm = lt.norm();
        if (lt_norm > ln)
        {
            if (lt_norm > -ln)
            {
                s_over_2 = (ln + lt_norm) / 2.0;
                s_normed = s_over_2 / lt_norm;

                vector(idx) *= s_normed;
                vector(idx + 1) *= s_normed;
                vector(idx + 2) = s_over_2;
            }
            else
            {
                vector(idx) = 0;
                vector(idx + 1) = 0;
                vector(idx + 2) = 0;
            }
        }
    }
}

double Solver::project_lorentz_half_squared(VectorXd &vector)
{//projection of l onto Lorentz, normend and squared, over 2.
    double norm_squared = 0.;
    for (idx = 0; idx < _NL; idx += 3)
    {   
        lt(0) = vector(idx);
        lt(1) = vector(idx + 1);
        ln = vector(idx + 2);
        lt_norm = lt.norm();
        if (lt_norm > ln)
        {
            if (lt_norm > -ln)
            {
                s_over_2 = (ln + lt_norm)/2.;
                norm_squared += 2 * s_over_2 * s_over_2;
            }
        }
        else
        {
            norm_squared += lt_norm * lt_norm + ln * ln;
        }
    }
    return norm_squared / 2;
}


void Solver::root_hess_project_lorentz_half_squared(VectorXd &vector, MatrixXd &root_hess)
{//blocks of upper triangular root U of hessian H of 0.5||project_lorentz(vector)||^2, such that, UU^T = H
    root_hess.setZero();
    for (idx = 0; idx < _NL; idx += 3)
    {
        lt(0) = vector(idx);
        lt(1) = vector(idx + 1);
        ln = vector(idx + 2);
        lt_norm = lt.norm();
        if (lt_norm > ln)
        {
            if (lt_norm > -ln)
            {   
                lt_hat = lt / lt_norm;
                s = ln + lt_norm;


                root_hess(idx, 0) = SQRT_HALF * lt_hat(0);
                root_hess(idx, 1) = -SQRT_HALF * std::sqrt(s/lt_norm) * lt_hat(1);

                root_hess(idx + 1, 0) = SQRT_HALF * lt_hat(1);
                root_hess(idx + 1, 1) = SQRT_HALF * std::sqrt(s/lt_norm) * lt_hat(0);

                root_hess(idx + 2, 0) = SQRT_HALF;
            }
        }
        else
        {   
            root_hess.block<3,3>(idx, 0).setIdentity();
        }
    }
}

static double getTime(std::chrono::high_resolution_clock::time_point* t = nullptr)
{
    using Clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>((t != nullptr ? *t : Clock::now() ).time_since_epoch()).count();
}

std::tuple<VectorXd,VectorXd> Solver::solve(const py::EigenDRef<const MatrixXd> &J, const py::EigenDRef<const VectorXd> &q, double eps) {

    loss = 0.;
    alpha = 0.;
    alpha_prev = 0.;
    loss_v_alpha = 0.;
    loss_v_alpha_prev = 0.;
    grad_v_alpha = 0.;
    hess_v_alpha = 0.;
    grad_l_alpha0 = 0.;

    momentum_stop = false;
    decrement_stop = false;

    y = y_of_v(J,v,q,eps);
    loss_prev = loss_of_yv(y,v,eps);

    for (int i = 0; i<max_iterations; i++) {
        // calculate loss value
        y = y_of_v(J,v,q,eps);
        loss = loss_of_yv(y,v,eps);

        // calculate forces l and loss gradient
        l = y;
        project_lorentz(l);
        J_t_l = J.transpose() * l;
        grad_loss = v - J_t_l;

        // inspect primal optimality for termination
        momentum_stop = grad_loss.norm() < grad_abs_tolerance + grad_rel_tolerance * (v.norm() + J_t_l.norm());

        // possible termination
        if (momentum_stop || decrement_stop) {
            break;
        }

        // calculate loss hessian
        root_hess_project_lorentz_half_squared(y, root_hess_projection);
        J_t_root_hess = J.transpose();
        for (int c = 0; c < _NL; c += 3)
        {   
            J_t_root_hess.middleCols<3>(c) *= root_hess_projection.block<3,3>(c,0);
        }
        hess_loss.triangularView<Upper>() = J_t_root_hess * J_t_root_hess.transpose() / eps;
        hess_loss.diagonal().array() += 1.;


        // calculate line search direction
        hess_chol.compute(hess_loss.selfadjointView<Upper>());
        dv = -grad_loss;
        hess_chol.solveInPlace(dv);

        // calculate line search direction gradient
        grad_l_alpha0 = dv.dot(grad_loss);

        // set line search initial_condition
        alpha = alpha_max;
        v_alpha = v + alpha * dv;
        y_alpha = y_of_v(J,v_alpha,q,eps);
        loss_v_alpha = loss_of_yv(y_alpha, v_alpha, eps);
        loss_v_alpha_prev = loss_v_alpha;
        alpha_prev = alpha;

        
        for(int j = 0; j<max_ls_iterations; j++)
        {
            alpha *= rho;
            v_alpha = v + alpha * dv;
            y_alpha = y_of_v(J,v_alpha,q,eps);
            loss_v_alpha = loss_of_yv(y_alpha, v_alpha, eps);
            if (loss_v_alpha > loss_v_alpha_prev && (loss_v_alpha < (loss + c * alpha * grad_l_alpha0))) {
                // stop
                if (loss_v_alpha_prev < (loss + c * alpha_prev * grad_l_alpha0)) {
                    alpha = alpha_prev;
                    loss_v_alpha = loss_v_alpha_prev;
                }
                break;
            }


            loss_v_alpha_prev = loss_v_alpha;
            alpha_prev = alpha;
        }
        
        // update primals and loss to best value from line search
        v += alpha * dv;
        loss = loss_v_alpha;

        // inspect primal cost decrement for termination
        decrement_stop = (std::abs(loss_prev - loss) < cost_abs_tolerance + cost_rel_tolerance * (loss + loss_prev)/2.) && alpha > 0.5;
        
        loss_prev = loss;
    }
    
    return std::make_tuple(l,v);
}