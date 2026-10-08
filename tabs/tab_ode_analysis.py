import streamlit as st
import uuid
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from utils.plotting import CHART_CONFIG, PLOT_TEMPLATE, plot_avg_sd_operated
from utils.ode_analysis import parse_ode, compute_ode_ci_for_X
from utils.plotting import plot_all_ode_fits_summary, plot_selected_wells
from utils.file_io import generate_labels
from utils.models import power_law, polynomial_func  # etc. if needed

def display_tab_ode_analysis():
    """
    Tab 5: Custom ODE Fit Analysis (CI only for X).
    Lets user define ODE intervals, parse ODEs, solve, fit parameters with minimize, etc.
    """
    st.header("📈 Custom ODE Fit Analysis (CI Only for X)")

    if "ode_fits" not in st.session_state:
        st.session_state["ode_fits"] = []

    operated_data = st.session_state.get("operated_data")
    selected_operated_wells = st.session_state.get("selected_operated_wells",[])

    if operated_data is None or not selected_operated_wells:
        st.warning("⚠️ Please complete previous steps to get operated data first.")
        return

    if "Average" not in operated_data.columns:
        operated_data["Average"] = operated_data[selected_operated_wells].mean(axis=1)

    st.subheader("Overall Operated Data")
    plot_avg_sd_operated(operated_data, selected_operated_wells)

    st.subheader("Specify Time Interval for a New ODE Fit")
    time_min = float(operated_data["Time"].min())
    time_max = float(operated_data["Time"].max())

    user_start = st.number_input("New Fit Start Time", min_value=time_min, max_value=time_max, value=time_min, step=0.1, format="%.2f")
    user_end = st.number_input("New Fit End Time", min_value=user_start, max_value=time_max, value=time_max, step=0.1, format="%.2f")

    if st.button("Add ODE Fit"):
        new_fit={
            "id":str(uuid.uuid4()),
            "fit_start":user_start,
            "fit_end":user_end,
            "variables":["X","Y"],
            "parameters":["r","k"],
            "initial_conditions":[0.1,0.5],
            "odes":["r * X","-k * Y"],
            "param_guesses":[0.2,0.3],
            "fit_results":None
        }
        st.session_state["ode_fits"].append(new_fit)
        st.success(f"Added new ODE fit for interval {user_start}–{user_end}.")

    for i,fit in enumerate(st.session_state["ode_fits"]):
        fit_id=fit["id"]
        with st.expander(f"Fit {i+1}", expanded=True):
            new_start = st.number_input(f"Start (Fit {i+1})", min_value=time_min, max_value=time_max, value=float(fit["fit_start"]), step=0.1, format="%.2f", key=f"start_{fit_id}")
            new_end = st.number_input(f"End (Fit {i+1})", min_value=new_start, max_value=time_max, value=float(fit["fit_end"]), step=0.1, format="%.2f", key=f"end_{fit_id}")
            fit["fit_start"]=new_start
            fit["fit_end"]=new_end

            st.write("#### Variables & ODEs")
            var_str=st.text_input(f"Variables (comma-separated) [Fit {i+1}]", value=", ".join(fit["variables"]), key=f"vars_{fit_id}")
            fit["variables"]=[v.strip() for v in var_str.split(",") if v.strip()]

            updated_odes=[]
            st.write("**ODE Expressions**")
            for idx,var_name in enumerate(fit["variables"]):
                default_expr=fit["odes"][idx] if idx<len(fit["odes"]) else ""
                ode_expr=st.text_input(f"d{var_name}/dt =", value=default_expr, key=f"ode_expr_{fit_id}_{var_name}")
                updated_odes.append(ode_expr)
            fit["odes"]=updated_odes

            if st.button(f"Show ODE Equations (Fit {i+1})", key=f"show_odes_{fit_id}"):
                from utils.ode_analysis import display_ode_equations
                display_ode_equations(fit["variables"], fit["odes"])

            param_str=st.text_input(f"Parameters (comma-separated) [Fit {i+1}]", value=", ".join(fit["parameters"]), key=f"params_{fit_id}")
            fit["parameters"]=[p.strip() for p in param_str.split(",") if p.strip()]

            st.write("**Initial Conditions**")
            new_inits=[]
            for idx,var_name in enumerate(fit["variables"]):
                def_init=fit["initial_conditions"][idx] if idx<len(fit["initial_conditions"]) else 1.0
                init_val=st.number_input(f"Initial {var_name}", value=def_init, step=0.01, format="%.5f", key=f"init_{fit_id}_{var_name}")
                new_inits.append(init_val)
            fit["initial_conditions"]=new_inits

            st.write("**Parameter Guesses**")
            new_guesses=[]
            for idx,pname in enumerate(fit["parameters"]):
                guess_def=fit["param_guesses"][idx] if idx<len(fit["param_guesses"]) else 1.0
                guess_val=st.number_input(f"Initial guess for {pname}", value=guess_def, step=0.01, format="%.5f", key=f"guess_{fit_id}_{pname}")
                new_guesses.append(guess_val)
            fit["param_guesses"]=new_guesses

            # Subset data
            sub_mask=(operated_data["Time"]>=fit["fit_start"]) & (operated_data["Time"]<=fit["fit_end"])
            sub_data=operated_data[sub_mask].copy()
            if sub_data.empty:
                st.warning("No data points in the selected time range.")
            else:
                st.write("#### Observed Data in this Fit Range")
                plot_selected_wells(sub_data,["Average"])

                # Parse ODEs
                from utils.ode_analysis import parse_ode
                ode_funcs=[]
                parse_failed=False
                for expr in fit["odes"]:
                    f_callable=parse_ode(expr, fit["variables"], fit["parameters"])
                    if f_callable is None:
                        parse_failed=True
                        break
                    ode_funcs.append(f_callable)

                if parse_failed:
                    st.error("One or more ODE expressions could not be parsed.")
                else:
                    def ode_system(t, y, param_vec):
                        """
                        Defines the ODE system for the solver.

                        Parameters:
                        - t: Time variable.
                        - y: State variables.
                        - param_vec: Parameter vector.

                        Returns:
                        - List of derivatives for the ODE system.
                        """
                        return [fun(t, y, param_vec) for fun in ode_funcs]

                    def cost_function(param_vec):
                        """
                        Cost function for optimization.

                        Parameters:
                        - param_vec: Parameter vector to optimize.

                        Returns:
                        - Sum of squared residuals between observed and modeled data.
                        """
                        from scipy.integrate import solve_ivp
                        sol = solve_ivp(
                            fun=lambda t, yy: ode_system(t, yy, param_vec),
                            t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                            y0=fit["initial_conditions"],
                            t_eval=sub_data["Time"].values,
                            method="RK45"
                        )
                        if not sol.success:
                            return 1e10
                        x_model = sol.y[0]
                        x_obs = sub_data["Average"].values
                        return np.sum((x_obs - x_model) ** 2)

                    fit_button_key=f"btn_fit_{fit_id}"
                    pressed_fit_button=st.button(f"Fit ODE (Fit {i+1})", key=fit_button_key)
                    if pressed_fit_button:
                        from scipy.optimize import minimize
                        try:
                            result=minimize(cost_function, x0=fit["param_guesses"], method="L-BFGS-B")
                        except Exception as e:
                            st.error(f"Exception during optimization: {e}")
                        else:
                            if not result.success:
                                st.error(f"Optimization failed: {result.message}")
                            else:
                                fitted_params=result.x
                                st.success(f"Fitted parameters: {dict(zip(fit['parameters'], fitted_params))}")

                                sol_nom=solve_ivp(
                                    fun=lambda t,yy: ode_system(t,yy,fitted_params),
                                    t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                                    y0=fit["initial_conditions"],
                                    t_eval=sub_data["Time"].values,
                                    method="RK45"
                                )
                                
                                # Add this after your solve_ivp call but before displaying results:

                                # Check for numerical issues in the solution
                                if np.any(np.isinf(sol_nom.y)) or np.any(np.isnan(sol_nom.y)) or np.any(sol_nom.y > 1e15):
                                    st.warning("⚠️ Numerical overflow detected. Try reducing simulation time, adjusting parameter values, or using a different solver method (e.g., 'BDF' instead of 'RK45').")

                                if not sol_nom.success:
                                    st.warning(f"Solver failed after fitting: {sol_nom.message}")

                                def residuals(p):
                                    tmp_sol=solve_ivp(
                                        fun=lambda t,yy: ode_system(t,yy,p),
                                        t_span=(sub_data["Time"].values[0], sub_data["Time"].values[-1]),
                                        y0=fit["initial_conditions"],
                                        t_eval=sub_data["Time"].values,
                                        method="RK45"
                                    )
                                    if not tmp_sol.success:
                                        return np.ones_like(sub_data["Average"].values)*1e5
                                    return sub_data["Average"].values - tmp_sol.y[0]

                                r0=residuals(fitted_params)
                                N=len(r0)
                                k_=len(fitted_params)
                                SSR=np.sum(r0**2)
                                dof=N-k_
                                sigma2=SSR/dof if dof>0 else SSR

                                J=np.zeros((N,k_))
                                eps=1e-6
                                for j in range(k_):
                                    dp=np.zeros_like(fitted_params)
                                    dp[j]=eps
                                    r1=residuals(fitted_params+dp)
                                    J[:, j]=(r1-r0)/eps
                                try:
                                    JTJ_inv=np.linalg.inv(J.T@J)
                                    param_cov=sigma2*JTJ_inv
                                except np.linalg.LinAlgError:
                                    param_cov=None
                                    st.warning("Jacobian is singular; cannot compute param covariance")

                                AIC=2*k_ + N*np.log(SSR/N)
                                BIC=k_*np.log(N) + N*np.log(SSR/N)

                                if param_cov is not None:
                                    std_errors=np.sqrt(np.diag(param_cov))
                                else:
                                    std_errors=np.full(k_, np.nan)

                                from scipy.stats import t as t_dist
                                t_stats=fitted_params/std_errors

                                pvals=2*(1-t_dist.cdf(np.abs(t_stats), df=dof))

                                from utils.ode_analysis import compute_ode_ci_for_X
                                ci_results=compute_ode_ci_for_X(
                                    ode_system_func=ode_system,
                                    param_values=fitted_params,
                                    param_cov=param_cov,
                                    y0=fit["initial_conditions"],
                                    t_eval=sub_data["Time"].values,
                                    alpha=0.05
                                )

                                fit_result_dict={
                                    "params":fitted_params,
                                    "param_cov":param_cov,
                                    "AIC":AIC,
                                    "BIC":BIC,
                                    "p_values":pvals,
                                    "solver_success":sol_nom.success,
                                    "time_eval":sub_data["Time"].values,
                                    "solution_y":sol_nom.y if sol_nom.success else None,
                                    "RSS":SSR
                                }
                                if ci_results is not None:
                                    fit_result_dict.update({
                                        "fit_time":ci_results["t"],
                                        "X_lower":ci_results["CI_lower"],
                                        "X_upper":ci_results["CI_upper"],
                                        "X_std":ci_results["std"]
                                    })
                                fit["fit_results"]=fit_result_dict

                                # Show summary table
                                import pandas as pd
                                param_table=pd.DataFrame({
                                    "Parameter":fit["parameters"],
                                    "Estimate":fitted_params,
                                    "Std. Error":std_errors,
                                    "t-Statistic":t_stats,
                                    "p-Value":pvals
                                })
                                st.write("### Fitted Parameters & Stats")
                                st.dataframe(param_table)

                                stats_table=pd.DataFrame({
                                    "Metric":["AIC","BIC","RSS"],
                                    "Value":[AIC,BIC,SSR]
                                })
                                st.dataframe(stats_table)

                                st.info("Fit results saved in session_state!")
                                if sol_nom.success and ci_results is not None:
                                    import plotly.graph_objects as go
                                    fig=go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=sub_data["Time"], y=sub_data["Average"],
                                        mode="markers", name="Observed X"
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=ci_results["t"],
                                        y=sol_nom.y[0],
                                        mode="lines",
                                        name="Fitted X"
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=np.concatenate([ci_results["t"], ci_results["t"][::-1]]),
                                        y=np.concatenate([ci_results["CI_upper"], ci_results["CI_lower"][::-1]]),
                                        fill="toself",
                                        fillcolor="rgba(173,216,230,0.3)",
                                        name="X 95% CI"
                                    ))
                                    if sol_nom.y.shape[0]>1:
                                        fig.add_trace(go.Scatter(
                                            x=ci_results["t"],
                                            y=sol_nom.y[1],
                                            mode="lines",
                                            name="Fitted Y"
                                        ))
                                    fig.update_layout(
                                        title=f"ODE Fit (Fit {i+1})",
                                        xaxis_title="Time",
                                        yaxis_title="Value",
                                        template=PLOT_TEMPLATE
                                    )
                                    st.plotly_chart(fig, theme=None, config=CHART_CONFIG, use_container_width=True)
                    else:
                        # Re-display old results if exist
                        old_res=fit.get("fit_results")
                        if old_res is not None:
                            st.write("### Existing Fit Results from a Previous Run")
                            old_params=old_res.get("params",[])
                            k_=len(old_params)
                            if old_res.get("param_cov") is not None:
                                std_errors_old=np.sqrt(np.diag(old_res["param_cov"]))
                            else:
                                std_errors_old=np.full(k_, np.nan)
                            import pandas as pd
                            param_table=pd.DataFrame({
                                "Parameter":fit["parameters"],
                                "Estimate":old_params,
                                "Std. Error":std_errors_old,
                                "p-Value":old_res.get("p_values",[np.nan]*k_)
                            })
                            st.dataframe(param_table)

                            stats_table=pd.DataFrame({
                                "Metric":["AIC","BIC","RSS"],
                                "Value":[
                                    old_res.get("AIC",np.nan),
                                    old_res.get("BIC",np.nan),
                                    old_res.get("RSS",np.nan)
                                ]
                            })
                            st.dataframe(stats_table)

                            x_time=old_res.get("fit_time")
                            sol_y_old=old_res.get("solution_y")
                            x_upper=old_res.get("X_upper")
                            x_lower=old_res.get("X_lower")
                            if x_time is not None and sol_y_old is not None and x_upper is not None and x_lower is not None:
                                import plotly.graph_objects as go
                                fig=go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=old_res["time_eval"],
                                    y=sub_data["Average"],
                                    mode="markers",
                                    name="Observed X"
                                ))
                                fig.add_trace(go.Scatter(
                                    x=x_time,
                                    y=sol_y_old[0],
                                    mode="lines",
                                    name="Fitted X"
                                ))
                                fig.add_trace(go.Scatter(
                                    x=np.concatenate([x_time, x_time[::-1]]),
                                    y=np.concatenate([x_upper, x_lower[::-1]]),
                                    fill="toself",
                                    fillcolor="rgba(173,216,230,0.3)",
                                    name="X 95% CI"
                                ))
                                if sol_y_old.shape[0]>1:
                                    fig.add_trace(go.Scatter(
                                        x=x_time,
                                        y=sol_y_old[1],
                                        mode="lines",
                                        name="Fitted Y"
                                    ))
                                fig.update_layout(
                                    title=f"Old ODE Fit (Fit {i+1})",
                                    xaxis_title="Time",
                                    yaxis_title="Value",
                                    template=PLOT_TEMPLATE
                                )
                                st.plotly_chart(fig, theme=None, config=CHART_CONFIG, use_container_width=True)
                            else:
                                st.info("Old results exist but are incomplete for plotting.")
                        else:
                            st.info("No old results yet for this fit.")

            # F) Delete Fit
            if st.button(f"Delete ODE Fit {i+1}", key=f"delete_fit_{fit_id}"):
                st.session_state["ode_fits"].pop(i)
                st.success(f"Deleted ODE Fit {i+1}")
                break

    st.subheader("Summary Plot of All ODE Fits (Optional)")
    if st.button("Generate Summary Plot for All ODE Fits"):
        fig_summary=plot_all_ode_fits_summary(
            ode_fits=st.session_state["ode_fits"],
            operated_data=operated_data,
            selected_operated_wells=selected_operated_wells
        )
        import plotly.graph_objects as go
        if isinstance(fig_summary, go.Figure):
            st.plotly_chart(fig_summary, theme=None, config=CHART_CONFIG, use_container_width=True)
        else:
            st.error("Summary plot did not return a valid Plotly figure.")
