# ---- Libraries
import numpy as np
import pandas as pd
import random
import multiprocessing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from deap import base, creator, tools, algorithms

# Force a renderer that works from plain scripts (outside notebooks/interactive)
pio.renderers.default = "browser"

#  Data load and cleaning 
sp_Merged = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Optimisation\New Datasets\year supply and demand.xlsx")
# pd.read_excel(r"C:\Users\msmg\OneDrive - University of Brighton\Harry-Imanpour\SolarEnergy-master\SolarEnergy\Optimisation\New Datasets\year supply and demand.xlsx")
## IMPORTANT: When you want to run this on your machine Saeed change the above path to commented out line.
## If the above line doesn't work just right click "year supply and demand.xlsx" choose copy full path and paste it in the place of the path for the dataset variable.

sp_Merged.columns = sp_Merged.columns.str.strip()
sp_Merged["Demand"] = pd.to_numeric(sp_Merged["Demand"], errors="coerce")
sp_Merged["Supply"] = pd.to_numeric(sp_Merged["Supply"], errors="coerce")
sp_Merged = sp_Merged.dropna(subset=["Demand", "Supply"])

# Constants 
oilEF = 0.73        # TCO2/MWh 
gasEF = 0.40        # TCO2/MWh 
constantEF = 0.5186 # TCO2/MWh (grid EF used for emissions)

energyCapacity = 2000.0   # MWh
max_power = 200.0         # MW
roundtripefficiency = 0.90
eta_ch = np.sqrt(roundtripefficiency)    # charge efficiency
eta_dch = np.sqrt(roundtripefficiency)   # discharge efficiency

minimum = 0.20 * energyCapacity
maximum = energyCapacity
start   = 0.50 * energyCapacity
end_bounds = (minimum, maximum)

#  Demand / Supply PV series 
demand_series = sp_Merged["Demand"].to_numpy(dtype=float)
pv_series     = sp_Merged["Supply"].to_numpy(dtype=float)
T = len(demand_series)
total_demand = float(np.sum(demand_series))
total_pv     = float(np.sum(pv_series))

def simulate_dispatch(charge_bias: float,
                      discharge_bias: float,
                      reserve_frac: float,
                      end_target_frac: float,
                      return_series: bool = False):
    """
    Dispatch that enforces:
      - Hourly energy balance
      - PV balance
      - SoC dynamics with eta_ch, eta_dch
      - Charge/discharge power <= 200 MW
      - SoC in [400, 2000] MWh
      - Start SoC = 1000 MWh; end free within [400, 2000] MWh
      - No grid charging (imports->storage = 0)
    """
    soc = start
    reserve = max(minimum, reserve_frac * energyCapacity)

    direct_list, pv2st_list, st2load_list, export_list, import_list = [], [], [], [], []
    loss_list, soc_series = [], []

    for t in range(T):
        load = demand_series[t]
        pv   = pv_series[t]

        direct = min(pv, load)  # prioritize direct PV->Load
        deficit = load - direct
        surplus = pv - direct

        pv2st = 0.0
        st2ld = 0.0
        imp   = 0.0
        exp   = 0.0
        losses = 0.0

        if surplus > 0:
            charge_req = surplus * float(np.clip(charge_bias, 0.0, 1.0))
            charge_power = min(charge_req, max_power)
            headroom = maximum - soc
            pv2st = min(charge_power, headroom)
            soc += eta_ch * pv2st
            exp = max(surplus - pv2st, 0.0)
            losses += pv2st * (1.0 - eta_ch)
        elif deficit > 0:
            avail = max(soc - reserve, 0.0)
            max_deliverable = eta_dch * avail
            discharge_req = deficit * float(np.clip(discharge_bias, 0.0, 1.0))
            st2ld = min(discharge_req, max_power, max_deliverable)
            soc -= (st2ld / eta_dch)
            imp = max(deficit - st2ld, 0.0)
            losses += st2ld * (1.0/eta_dch - 1.0)

        soc = float(np.clip(soc, minimum, maximum))
        soc_series.append(soc)

        direct_list.append(direct)
        pv2st_list.append(pv2st)
        st2load_list.append(st2ld)
        export_list.append(exp)
        import_list.append(imp)
        loss_list.append(losses)

    end_soc = soc
    delta_soc = end_soc - start

    total_imports = float(np.sum(import_list))
    total_export  = float(np.sum(export_list))
    total_losses  = float(np.sum(loss_list))
    total_direct  = float(np.sum(direct_list))
    total_pv2st   = float(np.sum(pv2st_list))
    total_st2ld   = float(np.sum(st2load_list))

    solar_served = total_direct + total_st2ld
    solar_share = 100.0 * solar_served / total_demand if total_demand > 0 else 0.0
    solar_wasted = 100.0 * total_export / total_pv if total_pv > 0 else 0.0
    emissions = total_imports * constantEF
    balance_check = (total_pv + total_imports) - (total_demand + total_export + total_losses + delta_soc)

    kpis = {
        "Grid Imports (MWh)": total_imports,
        "CO2 Emissions (tCO2)": emissions,
        "Solar Share (%)": solar_share,
        "Solar Wasted (%)": solar_wasted,
        "Solar Direct to Load (MWh)": total_direct,
        "Solar to Storage (MWh)": total_pv2st,
        "Storage to Load (MWh)": total_st2ld,
        "Storage Losses (MWh)": total_losses,
        "Solar Exported (MWh)": total_export,
        "Start Storage (MWh)": start,
        "End Storage (MWh)": end_soc,
        "Delta SoC (MWh)": delta_soc,
        "Energy Balance Check (MWh)": balance_check
    }

    if not return_series:
        return kpis

    ts_col = next((c for c in ["Timestamp", "DATE-TIME", "Date & Time", "Date"] if c in sp_Merged.columns), None)
    idx = pd.to_datetime(sp_Merged[ts_col], errors="coerce") if ts_col else pd.RangeIndex(T)

    series_df = pd.DataFrame({
        "Demand": demand_series,
        "PV": pv_series,
        "Direct": direct_list,
        "PV->Storage": pv2st_list,
        "Storage->Load": st2load_list,
        "Export": export_list,
        "Import": import_list,
        "Losses": loss_list,
        "SoC": soc_series
    }, index=idx)

    return kpis, series_df

# Models A (No Storage) and B (Heuristic)
def run_model_a_no_storage(return_series: bool = False):
    print("No Storage Model Started")
    return simulate_dispatch(charge_bias=0.0,
                             discharge_bias=0.0,
                             reserve_frac=1.0,
                             end_target_frac=0.5,
                             return_series=return_series)

def run_model_b_rule_based(high_demand_quantile: float = 0.80, return_series: bool = False):
    print("Rule Based Model Started")
    soc = start
    reserve = minimum
    direct_list, pv2st_list, st2load_list, export_list, import_list, loss_list, soc_series = [], [], [], [], [], [], []

    thr = float(np.quantile(demand_series, high_demand_quantile))
    for t in range(T):
        load = demand_series[t]
        pv   = pv_series[t]

        direct = min(pv, load)
        deficit = load - direct
        surplus = pv - direct

        pv2st = 0.0
        st2ld = 0.0
        imp   = 0.0
        exp   = 0.0
        losses = 0.0

        if surplus > 0:
            charge_power = min(surplus, max_power)
            headroom = maximum - soc
            pv2st = min(charge_power, headroom)
            soc += eta_ch * pv2st
            exp = max(surplus - pv2st, 0.0)
            losses += pv2st * (1.0 - eta_ch)

        if deficit > 0 and load >= thr:
            avail = max(soc - reserve, 0.0)
            max_deliverable = eta_dch * avail
            st2ld = min(deficit, max_power, max_deliverable)
            soc -= (st2ld / eta_dch)
            imp = max(deficit - st2ld, 0.0)
            losses += st2ld * (1.0/eta_dch - 1.0)
        elif deficit > 0:
            imp = deficit

        soc = float(np.clip(soc, minimum, maximum))
        soc_series.append(soc)

        direct_list.append(direct)
        pv2st_list.append(pv2st)
        st2load_list.append(st2ld)
        export_list.append(exp)
        import_list.append(imp)
        loss_list.append(losses)

    end_soc = soc
    delta_soc = end_soc - start
    total_imports = float(np.sum(import_list))
    total_export  = float(np.sum(export_list))
    total_losses  = float(np.sum(loss_list))
    total_direct  = float(np.sum(direct_list))
    total_pv2st   = float(np.sum(pv2st_list))
    total_st2ld   = float(np.sum(np.array(st2load_list)))

    solar_served = total_direct + total_st2ld
    solar_share = 100.0 * solar_served / total_demand if total_demand > 0 else 0.0
    solar_wasted = 100.0 * total_export / total_pv if total_pv > 0 else 0.0
    emissions = total_imports * constantEF
    balance_check = (total_pv + total_imports) - (total_demand + total_export + total_losses + delta_soc)

    kpis = {
        "Grid Imports (MWh)": total_imports,
        "CO2 Emissions (tCO2)": emissions,
        "Solar Share (%)": solar_share,
        "Solar Wasted (%)": solar_wasted,
        "Solar Direct to Load (MWh)": total_direct,
        "Solar to Storage (MWh)": total_pv2st,
        "Storage to Load (MWh)": total_st2ld,
        "Storage Losses (MWh)": total_losses,
        "Solar Exported (MWh)": total_export,
        "Start Storage (MWh)": start,
        "End Storage (MWh)": end_soc,
        "Delta SoC (MWh)": delta_soc,
        "Energy Balance Check (MWh)": balance_check
    }

    if not return_series:
        return kpis

    ts_col = next((c for c in ["Timestamp", "DATE-TIME", "Date & Time", "Date"] if c in sp_Merged.columns), None)
    idx = pd.to_datetime(sp_Merged[ts_col], errors="coerce") if ts_col else pd.RangeIndex(T)
    series_df = pd.DataFrame({
        "Demand": demand_series,
        "PV": pv_series,
        "Direct": direct_list,
        "PV->Storage": pv2st_list,
        "Storage->Load": st2load_list,
        "Export": export_list,
        "Import": import_list,
        "Losses": loss_list,
        "SoC": soc_series
    }, index=idx)
    return kpis, series_df

# DEAP NSGA-II setup
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Genes: [charge_bias, discharge_bias, reserve_frac, end_target_frac] (end_target_frac currently unused)
LOW = [0.0, 0.0, 0.20, 0.20]
UP  = [1.0, 1.0, 1.00, 1.00]

toolbox.register("attr_cb", random.uniform, LOW[0], UP[0])
toolbox.register("attr_db", random.uniform, LOW[1], UP[1])
toolbox.register("attr_rf", random.uniform, LOW[2], UP[2])
toolbox.register("attr_et", random.uniform, LOW[3], UP[3])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_cb, toolbox.attr_db, toolbox.attr_rf, toolbox.attr_et), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    cb, db, rf, et = ind
    k = simulate_dispatch(cb, db, rf, et)
    one_minus_share = 1.0 - (k["Solar Share (%)"] / 100.0)
    return (k["Grid Imports (MWh)"], k["Storage Losses (MWh)"], one_minus_share)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=10.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=15.0, indpb=0.25)
toolbox.register("select", tools.selNSGA2)

def run_model_c_nsga2(pop_size=100, ngen=40, cxpb=0.9, mutpb=0.1, use_mp=False):
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    print("NSGA-II Started")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (float(fit[0]), float(fit[1]), float(fit[2]))
    pop = toolbox.select(pop, len(pop))

    if use_mp:
        with multiprocessing.Pool() as pool:
            toolbox.register("map", pool.map)
            algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=2*pop_size,
                                      cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                      halloffame=hof, verbose=True)
    else:
        toolbox.register("map", map)
        algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=2*pop_size,
                                  cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                  halloffame=hof, verbose=True)
    return hof

# Code block for the ploting of the graphs
def compare_kpis_bar(models: dict):
    keys = ["Grid Imports (MWh)", "CO2 Emissions (tCO2)", "Solar Share (%)", "Solar Wasted (%)", "Storage Losses (MWh)"]
    x = list(models.keys())
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Bar(name=k, x=x, y=[models[m][k] for m in x]))
    fig.update_layout(barmode="group", title="Model KPI Comparison", xaxis_title="Model")
    return fig

def plot_dispatch_series(series_df: pd.DataFrame, title: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        specs=[[{"secondary_y": True}], [{}]],
                        row_heights=[0.7, 0.3], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["Demand"], name="Demand", line=dict(color="#333")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["PV"], name="PV", line=dict(color="#2ca02c")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["Import"], name="Import", marker_color="#1f77b4", opacity=0.6), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["Export"], name="Export", marker_color="#ff7f0e", opacity=0.6), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["PV->Storage"], name="PV->Storage", line=dict(dash="dot", color="#9467bd")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["Storage->Load"], name="Storage->Load", line=dict(dash="dot", color="#8c564b")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["SoC"], name="SoC (MWh)", line=dict(color="#17becf")), row=2, col=1)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["Losses"], name="Losses (MWh)", line=dict(color="#d62728")), row=2, col=1)
    fig.update_yaxes(title_text="MW/MWh", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Flows (MWh)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy (MWh)", row=2, col=1)
    fig.update_layout(title=title, barmode="overlay", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

def plot_pareto_front(hof):
    imports = [float(ind.fitness.values[0]) for ind in hof]
    losses  = [float(ind.fitness.values[1]) for ind in hof]
    one_minus_share = [float(ind.fitness.values[2]) for ind in hof]
    solar_share = [(1.0 - v) * 100.0 for v in one_minus_share]
    fig = go.Figure(data=go.Scatter(
        x=imports, y=losses, mode="markers",
        marker=dict(size=8, color=solar_share, colorscale="Viridis", colorbar=dict(title="Solar Share (%)")),
        text=[f"share={s:.1f}%" for s in solar_share],
        name="Pareto Front"
    ))
    fig.update_layout(title="NSGA-II Pareto Front", xaxis_title="Grid Imports (MWh)", yaxis_title="Storage Losses (MWh)")
    return fig

def show_figure(fig, name="figure"):
    """
    Try to display the plot in the default browser. If the active Plotly renderer
    is incompatible, fall back to writing an HTML file and auto-opening it.
    """
    try:
        fig.show(renderer="browser")
    except Exception as e:
        print(f"Plotly show() failed: {e}. Falling back to HTML.")
        outfile = f"{name}.html"
        fig.write_html(outfile, auto_open=True)
        print(f"Wrote {outfile}")

# Entry point of the script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline, rule-based, or NSGA-II optimization models.")
    parser.add_argument("--model", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--ngen", type=int, default=40)
    parser.add_argument("--cxpb", type=float, default=0.9)
    parser.add_argument("--mutpb", type=float, default=0.1)
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()

    def print_kpis(title, k):
        print(f"\n=== {title} ===")
        for key in ["Grid Imports (MWh)", "CO2 Emissions (tCO2)", "Solar Share (%)",
                    "Solar Wasted (%)", "Solar Direct to Load (MWh)",
                    "Solar to Storage (MWh)", "Storage to Load (MWh)",
                    "Storage Losses (MWh)", "Solar Exported (MWh)",
                    "Start Storage (MWh)", "End Storage (MWh)",
                    "Delta SoC (MWh)", "Energy Balance Check (MWh)"]:
            print(f"{key}: {k[key]:.4f}")

    random.seed(42); np.random.seed(42)

    k_a = k_b = k_best = None
    if args.model in ("a", "all"):
        k_a = run_model_a_no_storage(return_series=False)
        print_kpis("Model A - No Storage", k_a)

    if args.model in ("b", "all"):
        k_b = run_model_b_rule_based(high_demand_quantile=args.quantile, return_series=False)
        print_kpis(f"Model B - Rule-Based (Q={args.quantile})", k_b)

    hof = None; best = None
    if args.model in ("c", "all"):
        hof = run_model_c_nsga2(pop_size=args.pop_size, ngen=args.ngen,
                                cxpb=args.cxpb, mutpb=args.mutpb, use_mp=args.mp)
        print(f"\nPareto set size: {len(hof)}")
        best = min(hof, key=lambda ind: ind.fitness.values[0])
        k_best = simulate_dispatch(*best)
        genes = [round(float(x), 4) for x in best]
        print(f"Model C - NSGA-II Best by Imports: genes={genes}")
        print_kpis("Model C - NSGA-II KPIs", k_best)

    # Visualization
    models_for_bar = {}
    if k_a: models_for_bar["A - No Storage"] = k_a
    if k_b: models_for_bar["B - Rule-Based"] = k_b
    if k_best: models_for_bar["C - NSGA-II"] = k_best
    if models_for_bar:
        show_figure(compare_kpis_bar(models_for_bar), "kpi_comparison")

    if k_a:
        _, series_a = run_model_a_no_storage(return_series=True)
        show_figure(plot_dispatch_series(series_a, "Model A - No Storage Dispatch"), "dispatch_model_a")
    if k_b:
        _, series_b = run_model_b_rule_based(high_demand_quantile=args.quantile, return_series=True)
        show_figure(plot_dispatch_series(series_b, f"Model B - Rule-Based Dispatch (Q={args.quantile})"), "dispatch_model_b")
    if best is not None:
        _, series_c = simulate_dispatch(*best, return_series=True)
        show_figure(plot_dispatch_series(series_c, "Model C - NSGA-II Best Dispatch"), "dispatch_model_c")
        show_figure(plot_pareto_front(hof), "pareto_front")

