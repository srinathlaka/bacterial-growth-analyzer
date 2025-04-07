import streamlit as st
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import t as t_dist

def parse_ode(ode_expr, variables, parameters):
    """
    Convert an ODE expression string (e.g. "r*X") into a callable function f(t,y,params).
    """
    try:
        t_sym,*var_syms=sp.symbols(['t']+variables)
        param_syms=sp.symbols(parameters)
        local_dict={v: s for v,s in zip(variables,var_syms)}
        local_dict.update({p: s for p,s in zip(parameters,param_syms)})
        local_dict['t']=t_sym
        expr=sp.sympify(ode_expr, locals=local_dict)
        func=sp.lambdify((t_sym,var_syms,param_syms), expr, 'numpy')
        def ode_func(t,y,params):
            return func(t,y,params)
        return ode_func
    except Exception as e:
        st.error(f"Error parsing ODE expression '{ode_expr}': {e}")
        return None

def display_ode_equations(variables, odes):
    """
    Generate and display ODE equations in LaTeX format.
    """
    st.write("### ODE Equations")
    for var,ode in zip(variables,odes):
        try:
            ode_expr=sp.sympify(ode)
            equation=f"\\frac{{d{var}}}{{dt}} = {sp.latex(ode_expr)}"
            st.latex(equation)
        except Exception as e:
            st.error(f"Error displaying equation for variable '{var}': {e}")

def compute_ode_ci_for_X(ode_system_func, param_values, param_cov, y0, t_eval, alpha=0.05, solver_method="RK45", eps=1e-6):
    """
    Lines ~223-310 from your code: compute param-propagated CI for the X variable of an ODE solution.
    """
    sol_nom=solve_ivp(
        fun=lambda t,y: ode_system_func(t,y,param_values),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method=solver_method
    )
    if not sol_nom.success:
        st.error("ODE solver failed: "+sol_nom.message)
        return None
    nominal_X=sol_nom.y[0]
    n_time=len(t_eval)
    n_params=len(param_values)
    grad_X=np.zeros((n_time, n_params))
    for j in range(n_params):
        dp=np.zeros_like(param_values)
        dp[j]=eps
        sol_pert=solve_ivp(
            fun=lambda t,y: ode_system_func(t,y,param_values+dp),
            t_span=(t_eval[0],t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            method=solver_method
        )
        if not sol_pert.success:
            grad_X[:, j]=0.0
        else:
            grad_X[:, j]=(sol_pert.y[0]-nominal_X)/eps

    std_X=np.zeros(n_time)
    for i in range(n_time):
        g=grad_X[i,:]
        var_i=g@param_cov@g.T
        std_X[i]=np.sqrt(max(var_i,0))

    from scipy.stats import t as t_dist
    crit_val=t_dist.ppf(1-alpha/2, df=np.inf)
    CI_lower=nominal_X - crit_val*std_X
    CI_upper=nominal_X + crit_val*std_X
    return {
        "t":sol_nom.t,
        "nominal_X":nominal_X,
        "CI_lower":CI_lower,
        "CI_upper":CI_upper,
        "std":std_X
    }
