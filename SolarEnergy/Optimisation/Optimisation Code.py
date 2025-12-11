"""
Solar Energy Storage Optimization & Visualization

Models:
 A: Baseline (no storage dispatch)
 B: Rule-based heuristic
 C: NSGA-II (4 genes: charge_bias, discharge_bias, reserve_frac, end_target_frac[unused])
 D: PSO (scalar aggregate objective)
 E: MOPSO (Pareto archive)
 F: NSGA-III (3 genes: charge_bias, discharge_bias, reserve_frac)
 G: LP (linear program over full horizon; no grid charging; SoC dynamics)

Objective modes:
  all         -> 3 objectives (emissions, 1 - solar_share, storage_losses)
  emissions   -> duplicated single objective (emissions, emissions)
  solar_share -> duplicated (1 - solar_share, 1 - solar_share)
  losses      -> duplicated (storage_losses, storage_losses)
Capacity sweep Arguements:
 --model c --objective-set all --cap-sweep 500,1000,1500,2000 --pop-size 100 --ngen 40 --cxpb 0.9 --mutpb 0.1 NSGA-II
--model d --objective-set all --cap-sweep 500,1000,1500,2000 --pso-swarm 60 --pso-iters 60 --pso-w 0.7 --pso-c1 1.5 --pso-c2 1.5 - PSO
Note:
 - end_target_frac is currently unused in dispatch; reserved for future end-of-horizon SoC targeting.
 - Weekly aggregation: models run at the original timestep; weekly outputs are resampled views (plots/CSV) saved to a 'Weekly' subfolder when enabled via CLI.
"""
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
from deap.tools.emo import uniform_reference_points
import os

# Optional LP solver (PuLP) for Model G
try:
    import pulp as pl  # type: ignore
except Exception:
    pl = None

# Use a renderer that works in plain scripts
pio.renderers.default = "browser"

# Base export directory for HTML plots (objective-specific subfolders added at runtime)
EXPORT_DIR = r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Optimisation\HTML Files"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Objective mode (overridden by CLI): "all", "emissions", "solar_share", "losses"
OBJECTIVE_MODE = "all"

def _objective_tuple_from_kpis(k: dict, mode: str):
    """
    Convert KPI dict to objective tuple for minimization.
    Returned tuple length:
      all: 3 objectives -> (emissions, 1 - solar_share, storage_losses)
      single modes: duplicated objective (for MO compatibility):
        emissions   -> (emissions, emissions)
        solar_share -> (1 - solar_share, 1 - solar_share)
        losses      -> (storage_losses, storage_losses)
    """
    em = float(k["CO2 Emissions (tCO2)"])
    share_percent = float(k["Solar Share (%)"])
    one_minus_share = 1.0 - (share_percent / 100.0)
    losses = float(k["Storage Losses (MWh)"])
    if mode == "all":
        return (em, one_minus_share, losses)
    if mode == "emissions":
        return (em, em)
    if mode == "solar_share":
        return (one_minus_share, one_minus_share)
    return (losses, losses)

# ---- Data load & cleaning
sp_Merged = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Optimisation\New Datasets\year supply and demand.xlsx")
# For alternate machine run: update path accordingly (use full path from OS file explorer if needed).

sp_Merged.columns = sp_Merged.columns.str.strip()
sp_Merged["Demand"] = pd.to_numeric(sp_Merged["Demand"], errors="coerce")
sp_Merged["Supply"] = pd.to_numeric(sp_Merged["Supply"], errors="coerce")
sp_Merged = sp_Merged.dropna(subset=["Demand", "Supply"])

# ---- Constants
oilEF = 0.73        # TCO2/MWh (not used directly; retained for possible future fuel mix calculations)
gasEF = 0.40        # TCO2/MWh
constantEF = 0.5186 # Grid emission factor (TCO2/MWh)
END_TARGET_CONST = 0.50  # Constant end target fraction (unused in current dispatch logic)

energyCapacity = 2000.0   # MWh (max SoC)
max_power = 200.0         # MW max charge or discharge instantaneous per original timestep
roundtripefficiency = 0.90
eta_ch = np.sqrt(roundtripefficiency)      # charge efficiency
eta_dch = np.sqrt(roundtripefficiency)     # discharge efficiency

minimum = 0.20 * energyCapacity            # min SoC (400 MWh)
maximum = energyCapacity
start   = 0.50 * energyCapacity            # initial SoC (1000 MWh)
end_bounds = (minimum, maximum)

RESERVE_CONST = 0.20  # reserve fraction cap used by DEAP-based 3-gene optimizers
LOW_DEAP = [0.0, 0.0, 0.00]  # [charge_bias, discharge_bias, reserve_frac]
UP_DEAP  = [1.0, 1.0, 0.20]

# ---- Capacity sweep helpers (PSO & NSGA-II only)
def set_capacity(capacity_mwh: float):
    """
    Update global battery capacity and dependent bounds consistently.
    """
    global energyCapacity, minimum, maximum, start, end_bounds
    energyCapacity = float(capacity_mwh)
    maximum = energyCapacity
    minimum = 0.20 * energyCapacity
    start = 0.50 * energyCapacity
    end_bounds = (minimum, maximum)

# ---- Demand / PV series (original timestep; weekly views are created later for export only)
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
    Simulate storage dispatch over T timesteps (original resolution).

    Constraints / rules:
      - Direct PV serves load first.
      - Surplus PV can charge storage (bounded by max_power and headroom).
      - Deficit can be met by discharging storage (bounded by max_power and available energy above reserve).
      - SoC clipped to [minimum, maximum].
      - Round-trip efficiency applied (split evenly charge/discharge via sqrt).
      - No grid charging; imports only cover unmet load after discharge.
      - end_target_frac is currently unused (placeholder for future enforced final SoC).

    Returns:
      - If return_series = False: KPI dict only.
      - If return_series = True : (KPIs, DataFrame of time series flows at original resolution).
    """
    soc = start
    reserve = max(minimum, reserve_frac * energyCapacity)

    direct_list, pv2st_list, st2load_list = [], [], []
    export_list, import_list, loss_list, soc_series = [], [], [], []

    for t in range(T):
        load = demand_series[t]
        pv   = pv_series[t]

        # Direct PV to load
        direct = min(pv, load)
        deficit = load - direct
        surplus = pv - direct

        pv2st = st2ld = imp = exp = losses = 0.0

        # Charging (surplus)
        if surplus > 0:
            charge_req = surplus * float(np.clip(charge_bias, 0.0, 1.0))
            charge_power = min(charge_req, max_power)
            headroom = maximum - soc
            pv2st = min(charge_power, headroom)
            soc += eta_ch * pv2st
            exp = max(surplus - pv2st, 0.0)
            losses += pv2st * (1.0 - eta_ch)

        # Discharging (deficit)
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
       
    # KPI calculation (whole horizon)
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

    # Should be ~= 0 (floating point residual); (gen + imports) - (load + export + losses + delta_soc)
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

    # Build time series DataFrame (original resolution); weekly views are derived later for export only.
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

# ---- Baseline and heuristic models
def run_model_a_no_storage(return_series: bool = False):
    """
    Model A: Baseline without active storage dispatch (biases zero).
    Reserve fraction set but inert due to zero charge/discharge biases.
    """
    print("No Storage Model Started")
    return simulate_dispatch(charge_bias=0.0,
                             discharge_bias=0.0,
                             reserve_frac=0.20,
                             end_target_frac=0.5,
                             return_series=return_series)

def run_model_b_rule_based(high_demand_quantile: float = 0.80, return_series: bool = False):
    """
    Model B: Rule-based heuristic.
    Charge whenever surplus exists.
    Discharge only if demand >= high_demand_quantile threshold.
    """
    print("Rule Based Model Started")
    soc = start
    reserve = minimum
    direct_list, pv2st_list, st2load_list = [], [], []
    export_list, import_list, loss_list, soc_series = [], [], [], []

    thr = float(np.quantile(demand_series, high_demand_quantile))
    for t in range(T):
        load = demand_series[t]
        pv   = pv_series[t]

        direct = min(pv, load)
        deficit = load - direct
        surplus = pv - direct

        pv2st = st2ld = imp = exp = losses = 0.0

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

    # KPI calculation (whole horizon)
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

    # Build time series DataFrame (original resolution); weekly views are derived later for export only.
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

# ---- DEAP configuration (dynamic)
def _nobj_from_mode(mode: str) -> int:
    return 3 if mode == "all" else 2

def configure_deap(mode: str):
    from deap import creator
    for cls in ("FitnessMulti", "Individual"):
        if cls in creator.__dict__:
            del creator.__dict__[cls]
    nobj = 3 if mode == "all" else 2
    creator.create("FitnessMulti", base.Fitness, weights=tuple([-1.0] * nobj))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    global toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_cb", random.uniform, LOW_DEAP[0], UP_DEAP[0])
    toolbox.register("attr_db", random.uniform, LOW_DEAP[1], UP_DEAP[1])
    toolbox.register("attr_rf", random.uniform, LOW_DEAP[2], UP_DEAP[2])
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_cb, toolbox.attr_db, toolbox.attr_rf), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        cb, db, rf = ind
        k = simulate_dispatch(cb, db, rf, END_TARGET_CONST)
        return _objective_tuple_from_kpis(k, mode)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW_DEAP, up=UP_DEAP, eta=10.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW_DEAP, up=UP_DEAP, eta=15.0, indpb=0.25)
    toolbox.register("select", tools.selNSGA2)

# ---- NSGA-II setup (4 genes; end_target currently unused)
from deap import creator
for cls in ("FitnessMulti", "Individual"):
    if cls in creator.__dict__:
        del creator.__dict__[cls]
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
LOW = [0.0, 0.0, 0.00, 0.20]  # [cb, db, reserve_frac, end_target_frac]
UP  = [1.0, 1.0, 0.20, 1.00]

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
    return _objective_tuple_from_kpis(k, OBJECTIVE_MODE)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=10.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=15.0, indpb=0.25)
toolbox.register("select", tools.selNSGA2)

def run_model_c_nsga2(pop_size=100, ngen=40, cxpb=0.9, mutpb=0.1, use_mp=False):
    """Run NSGA-II (4-gene individual). Returns ParetoFront."""
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    print("NSGA-II Started")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = tuple(map(float, fit))
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

# ---- PSO (scalar objective)
def _pso_cost_from_kpis(k: dict, mode: str = "all", w_em: float = 0.5, w_sh: float = 0.5, w_ls: float | None = None) -> float:
    """
    Aggregate scalar for PSO minimization.
      all: w_em*emissions_norm + w_sh*(1 - solar_share) + w_ls*losses_norm
           (w_ls defaults to 1 - w_em - w_sh if not provided, truncated to >=0)
      emissions: emissions_norm
      solar_share: (1 - solar_share)
      losses: losses_norm
    """
    em = float(k["CO2 Emissions (tCO2)"])
    share_percent = float(k["Solar Share (%)"])
    losses = float(k["Storage Losses (MWh)"])
    one_minus_share = 1.0 - (share_percent / 100.0)

    denom_em = max(total_demand * constantEF, 1e-9)
    em_norm = em / denom_em
    denom_losses = max(total_pv, 1e-9)
    losses_norm = losses / denom_losses

    if mode == "emissions":
        return em_norm
    if mode == "solar_share":
        return one_minus_share
    if mode == "losses":
        return losses_norm

    if w_ls is None:
        w_ls = max(0.0, 1.0 - (w_em + w_sh))
    return w_em * em_norm + w_sh * one_minus_share + w_ls * losses_norm

def _evaluate_genes_cost(genes, mode="all", w_em=0.5, w_sh=0.5):
    cb, db, rf, et = map(float, genes)
    k = simulate_dispatch(cb, db, rf, et)
    return _pso_cost_from_kpis(k, mode, w_em, w_sh), k

def run_model_d_pso(swarm_size=60, iters=60, w=0.7, c1=1.5, c2=1.5,
                    weights=(0.5, 0.5)):
    """Run single-objective PSO; returns (best_genes, best_kpis)."""
    print("PSO Started")
    dim = 4
    low = np.array(LOW, dtype=float)
    up  = np.array(UP, dtype=float)

    pos = np.array([[np.random.uniform(low[i], up[i]) for i in range(dim)] for _ in range(swarm_size)], dtype=float)
    vel = np.zeros((swarm_size, dim), dtype=float)

    pbest_pos = pos.copy()
    pbest_cost = np.empty(swarm_size, dtype=float)
    pbest_kpis = [None] * swarm_size

    w_em, w_sh = weights
    for p in range(swarm_size):
        cost, k = _evaluate_genes_cost(pbest_pos[p], OBJECTIVE_MODE, w_em, w_sh)
        pbest_cost[p] = cost
        pbest_kpis[p] = k

    g_idx = int(np.argmin(pbest_cost))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_cost = float(pbest_cost[g_idx])
    gbest_kpis = pbest_kpis[g_idx]

    for _ in range(iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, low, up)

        for p in range(swarm_size):
            cost, k = _evaluate_genes_cost(pos[p], OBJECTIVE_MODE, w_em, w_sh)
            if cost < pbest_cost[p]:
                pbest_cost[p] = cost
                pbest_pos[p] = pos[p].copy()
                pbest_kpis[p] = k
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_pos = pos[p].copy()
                    gbest_kpis = k

    genes = [float(g) for g in gbest_pos.tolist()]
    print(f"PSO Best aggregate cost: {gbest_cost:.6f}, genes={ [round(g, 4) for g in genes] }")
    return genes, gbest_kpis

# ---- Capacity sweep wrappers for NSGA-II and PSO
def run_nsga2_for_capacity(capacity_mwh: float,
                           pop_size=100, ngen=40, cxpb=0.9, mutpb=0.1, use_mp=False):
    """
    Reconfigure capacity, run NSGA-II, and return (capacity, hof, best_genes, best_kpis).
    """
    set_capacity(capacity_mwh)
    # Keep NSGA-II toolbox/creator unchanged for consistency with the original implementation
    random.seed(42); np.random.seed(42)
    hof = run_model_c_nsga2(pop_size=pop_size, ngen=ngen, cxpb=cxpb, mutpb=mutpb, use_mp=use_mp)
    if len(hof) == 0:
        return capacity_mwh, hof, None, None
    best = min(hof, key=lambda ind: ind.fitness.values[0])
    best_genes = [float(best[0]), float(best[1]), float(best[2]), float(END_TARGET_CONST)]
    best_kpis = simulate_dispatch(best[0], best[1], best[2], END_TARGET_CONST)
    return capacity_mwh, hof, best_genes, best_kpis

def run_pso_for_capacity(capacity_mwh: float,
                         swarm_size=60, iters=60, w=0.7, c1=1.5, c2=1.5):
    """
    Reconfigure capacity, run PSO, and return (capacity, best_genes, best_kpis).
    """
    set_capacity(capacity_mwh)
    random.seed(42); np.random.seed(42)
    best_genes, best_kpis = run_model_d_pso(swarm_size=swarm_size, iters=iters, w=w, c1=c1, c2=c2)
    return capacity_mwh, best_genes, best_kpis

def run_capacity_sweep(model: str,
                       capacities: list[float],
                       nsga2_args: dict | None = None,
                       pso_args: dict | None = None):
    nsga2_args = nsga2_args or {}
    pso_args = pso_args or {}
    rows = []
    details = {}
    print(f"Capacity sweep started for model '{model}' with capacities: {capacities}")
    for cap in capacities:
        cap = float(cap)
        if model == "c":
            capacity, hof, genes, kpis = run_nsga2_for_capacity(
                capacity_mwh=cap,
                pop_size=nsga2_args.get("pop_size", 100),
                ngen=nsga2_args.get("ngen", 40),
                cxpb=nsga2_args.get("cxpb", 0.9),
                mutpb=nsga2_args.get("mutpb", 0.1),
                use_mp=nsga2_args.get("use_mp", False)
            )
            if kpis is None:
                continue
            rows.append({"Capacity (MWh)": capacity, **kpis})
            details[capacity] = {"genes": genes, "hof": hof}
            print(f"NSGA-II sweep cap={capacity:.1f} MWh -> Emissions={kpis['CO2 Emissions (tCO2)']:.4f}, Share={kpis['Solar Share (%)']:.2f}%")
        elif model == "d":
            capacity, genes, kpis = run_pso_for_capacity(
                capacity_mwh=cap,
                swarm_size=pso_args.get("swarm_size", 60),
                iters=pso_args.get("iters", 60),
                w=pso_args.get("w", 0.7),
                c1=pso_args.get("c1", 1.5),
                c2=pso_args.get("c2", 1.5)
            )
            rows.append({"Capacity (MWh)": capacity, **kpis})
            details[capacity] = {"genes": genes}
            print(f"PSO sweep cap={capacity:.1f} MWh -> Emissions={kpis['CO2 Emissions (tCO2)']:.4f}, Share={kpis['Solar Share (%)']:.2f}%")
        else:
            raise ValueError("Capacity sweep supports only 'c' (NSGA-II) or 'd' (PSO).")
    df = pd.DataFrame(rows).sort_values("Capacity (MWh)")
    return df, details

# ---- MOPSO (multi-objective PSO)
def _kpis_to_objectives(k: dict, mode: str):
    return _objective_tuple_from_kpis(k, mode)

def _dominates(a, b):
    """a dominates b if a <= b component-wise and < on at least one."""
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))

def _crowding_distance(objs: np.ndarray):
    """Crowding distance for diversity preservation."""
    n, m = objs.shape
    if n == 0:
        return np.array([])
    dist = np.zeros(n, dtype=float)
    for j in range(m):
        order = np.argsort(objs[:, j])
        sorted_vals = objs[order, j]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        vmin, vmax = sorted_vals[0], sorted_vals[-1]
        denom = vmax - vmin
        if denom <= 0:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / denom
    return dist

def _update_archive(arch_pos: np.ndarray, arch_objs: np.ndarray,
                    cand_pos: np.ndarray, cand_objs: np.ndarray, max_size: int):
    """
    Update Pareto archive:
      - Merge
      - Remove dominated
      - Remove duplicates
      - Truncate by crowding if oversized
    """
    if arch_pos.size == 0:
        all_pos = cand_pos.copy()
        all_objs = cand_objs.copy()
    else:
        all_pos = np.vstack([arch_pos, cand_pos])
        all_objs = np.vstack([arch_objs, cand_objs])

    keep = np.ones(len(all_objs), dtype=bool)
    for i in range(len(all_objs)):
        if not keep[i]:
            continue
        for j in range(len(all_objs)):
            if i == j or not keep[j]:
                continue
            if _dominates(all_objs[j], all_objs[i]):
                keep[i] = False
                break
    nd_pos = all_pos[keep]
    nd_objs = all_objs[keep]

    unique_map = {}
    uniq_indices = []
    for idx, obj in enumerate(map(tuple, nd_objs.tolist())):
        if obj not in unique_map:
            unique_map[obj] = idx
            uniq_indices.append(idx)
    nd_pos = nd_pos[uniq_indices]
    nd_objs = nd_objs[uniq_indices]

    if len(nd_objs) > max_size:
        cd = _crowding_distance(nd_objs)
        idxs = list(range(len(nd_objs)))
        while len(idxs) > max_size:
            finite = [(i, cd[i]) for i in idxs if not np.isinf(cd[i])]
            if not finite:
                idxs.pop(np.random.randint(len(idxs)))
                continue
            i_min = min(finite, key=lambda t: t[1])[0]
            idxs.remove(i_min)
        nd_pos = nd_pos[idxs]
        nd_objs = nd_objs[idxs]

    return nd_pos, nd_objs

def _select_leader(arch_objs: np.ndarray):
    """Select archive leader biased toward sparse region (crowding distance weighting)."""
    cd = _crowding_distance(arch_objs)
    weights = np.array([1e6 if np.isinf(v) else max(v, 1e-12) for v in cd], dtype=float)
    probs = weights / np.sum(weights)
    return int(np.random.choice(len(arch_objs), p=probs))

def run_model_e_mopso(swarm_size=80, iters=80, w=0.7, c1=1.5, c2=1.5,
                      archive_size=100, vmax_frac=0.2):
    """
    MOPSO minimizing:
      all: (emissions, 1 - solar_share, storage_losses)
      single objective modes: duplicated objective (2D).
    Archive update currently called twice per iteration (kept; could be consolidated).
    """
    print("MOPSO Started")
    dim = 4
    low = np.array(LOW, dtype=float)
    up  = np.array(UP, dtype=float)
    span = up - low
    vmax = span * float(max(vmax_frac, 1e-6))

    pos = np.array([[np.random.uniform(low[i], up[i]) for i in range(dim)] for _ in range(swarm_size)], dtype=float)
    vel = np.zeros((swarm_size, dim), dtype=float)

    m = len(_kpis_to_objectives(simulate_dispatch(*pos[0]), OBJECTIVE_MODE))
    pbest_pos = pos.copy()
    pbest_objs = np.zeros((swarm_size, m), dtype=float)
    cur_objs = np.zeros((swarm_size, m), dtype=float)

    for i in range(swarm_size):
        k = simulate_dispatch(*pos[i])
        cur_objs[i] = _kpis_to_objectives(k, OBJECTIVE_MODE)
    pbest_objs[:] = cur_objs

    arch_pos, arch_objs = _update_archive(np.empty((0, dim)), np.empty((0, m)), pos, cur_objs, archive_size)

    # Iterate
    for _ in range(iters):
        if len(arch_objs) == 0:
            arch_pos, arch_objs = _update_archive(np.empty((0, dim)), np.empty((0, m)), pos, cur_objs, archive_size)
        leader_idx = _select_leader(arch_objs)

        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)

        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (arch_pos[leader_idx] - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, low, up)

        for i in range(swarm_size):
            k = simulate_dispatch(*pos[i])
            cur = _kpis_to_objectives(k, OBJECTIVE_MODE)

            if _dominates(cur, pbest_objs[i]):
                pbest_objs[i] = cur
                pbest_pos[i] = pos[i].copy()
            elif not _dominates(pbest_objs[i], cur) and np.random.rand() < 0.3:
                pbest_objs[i] = cur
                pbest_pos[i] = pos[i].copy()

        arch_pos, arch_objs = _update_archive(arch_pos, arch_objs, pos, cur_objs, archive_size)
        eval_objs = np.array([_kpis_to_objectives(simulate_dispatch(*p), OBJECTIVE_MODE) for p in pos])
        arch_pos, arch_objs = _update_archive(arch_pos, arch_objs, pos, eval_objs, archive_size)
        cur_objs = eval_objs

    # Pick representative best by first objective
    if len(arch_objs) == 0:
        idx = int(np.argmin(pbest_objs[:, 0]))
        best_genes = pbest_pos[idx].tolist()
        best_kpis = simulate_dispatch(*best_genes)
    else:
        idx = int(np.argmin(arch_objs[:, 0]))
        best_genes = arch_pos[idx].tolist()
        best_kpis = simulate_dispatch(*best_genes)

    genes_rounded = [round(float(x), 4) for x in best_genes]
    print(f"MOPSO archive size: {len(arch_objs)}")
    print(f"MOPSO Best by first objective: genes={genes_rounded}")
    return arch_pos, arch_objs, best_genes, best_kpis

# ---- LP (Model G)
def run_model_g_lp(mode: str = None,
                   reserve_frac: float = RESERVE_CONST,
                   end_target_frac: float | None = None,
                   weights: tuple[float, float] = (0.5, 0.5),
                   return_series: bool = False):
    """
    Model G: Linear Program over the full horizon with storage physics.
      - No grid charging (only PV->Storage).
      - SoC dynamics with efficiencies.
      - Bounds, reserve floor, and max power.
      - PV split and load balance per step.
    Objective:
      - emissions: minimize total imports
      - solar_share: maximize (Direct + Storage->Load) <=> minimize -(Direct+Storage->Load)
      - losses: minimize storage losses
      - all: weighted linear sum (same normalization idea as PSO)
    """
    if mode is None:
        mode = OBJECTIVE_MODE
    if pl is None:
        raise RuntimeError("PuLP is required for Model G (LP). Install with: pip install pulp")

    print("LP Model Started")

    reserve = max(minimum, float(reserve_frac) * energyCapacity)
    soc_lb = reserve
    soc_ub = maximum

    # Create LP
    prob = pl.LpProblem("LP_Dispatch", pl.LpMinimize)

    # Variables (per timestep, full horizon)
    soc = {t: pl.LpVariable(f"soc_{t}", lowBound=soc_lb, upBound=soc_ub) for t in range(T + 1)}
    direct = {t: pl.LpVariable(f"direct_{t}", lowBound=0.0) for t in range(T)}
    pv2st = {t: pl.LpVariable(f"pv2st_{t}", lowBound=0.0, upBound=max_power) for t in range(T)}
    st2ld = {t: pl.LpVariable(f"st2ld_{t}", lowBound=0.0, upBound=max_power) for t in range(T)}
    exp = {t: pl.LpVariable(f"export_{t}", lowBound=0.0) for t in range(T)}
    imp = {t: pl.LpVariable(f"import_{t}", lowBound=0.0) for t in range(T)}

    # Initial SoC
    prob += soc[0] == start, "soc_initial"

    # Constraints per timestep (match simulate_dispatch physics)
    for t in range(T):
        # PV split: direct + pv2st + export = PV
        prob += direct[t] + pv2st[t] + exp[t] == float(pv_series[t]), f"pv_split_{t}"

        # Load balance: direct + storage_to_load + import = Demand
        prob += direct[t] + st2ld[t] + imp[t] == float(demand_series[t]), f"load_balance_{t}"

        # SoC dynamics: SoC[t+1] = SoC[t] + eta_ch*pv2st - st2ld/eta_dch
        prob += soc[t + 1] == soc[t] + float(eta_ch) * pv2st[t] - (st2ld[t] / float(eta_dch)), f"soc_dyn_{t}"

        # Headroom limit on charging (conservative, matches simulate_dispatch)
        prob += pv2st[t] <= maximum - soc[t], f"charge_headroom_{t}"

        # Discharge availability limited by energy above reserve (matches simulate_dispatch)
        prob += st2ld[t] <= float(eta_dch) * (soc[t] - reserve), f"discharge_available_{t}"

        # SoC bounds are enforced via variable bounds

    # Optional end-of-horizon SoC target (if provided)
    if end_target_frac is not None:
        target = float(end_target_frac) * energyCapacity
        lb, ub = end_bounds
        prob += soc[T] >= max(lb, target), "soc_end_lb"
        prob += soc[T] <= ub, "soc_end_ub"

    # Objective
    denom_em = max(total_demand * constantEF, 1e-9)
    denom_losses = max(total_pv, 1e-9)
    w_em, w_sh = float(weights[0]), float(weights[1])
    w_ls = max(0.0, 1.0 - (w_em + w_sh))

    if mode == "emissions":
        obj = pl.lpSum(imp[t] for t in range(T))  # proportional to emissions
    elif mode == "solar_share":
        # minimize -(Direct + Storage->Load) / total_demand
        obj = - (1.0 / max(total_demand, 1e-9)) * pl.lpSum(direct[t] + st2ld[t] for t in range(T))
    elif mode == "losses":
        obj = pl.lpSum(
            pv2st[t] * (1.0 - float(eta_ch)) + st2ld[t] * (1.0 / float(eta_dch) - 1.0)
            for t in range(T)
        )
    else:
        # weighted normalized linear sum (constants omitted)
        obj = (
            (w_em * constantEF / denom_em) * pl.lpSum(imp[t] for t in range(T))
            + (- w_sh / max(total_demand, 1e-9)) * pl.lpSum(direct[t] + st2ld[t] for t in range(T))
            + (w_ls / denom_losses) * pl.lpSum(
                pv2st[t] * (1.0 - float(eta_ch)) + st2ld[t] * (1.0 / float(eta_dch) - 1.0)
                for t in range(T)
            )
        )
    prob += obj

    # Solve
    solver = pl.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pl.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP did not find an optimal solution. Status: {pl.LpStatus[status]}")

    # Extract solution (original resolution); weekly views are derived later for export only.
    direct_list = [pl.value(direct[t]) for t in range(T)]
    pv2st_list = [pl.value(pv2st[t]) for t in range(T)]
    st2load_list = [pl.value(st2ld[t]) for t in range(T)]
    export_list = [pl.value(exp[t]) for t in range(T)]
    import_list = [pl.value(imp[t]) for t in range(T)]
    soc_series = [pl.value(soc[t + 1]) for t in range(T)]
    loss_list = [
        pv2st_list[t] * (1.0 - float(eta_ch)) + st2load_list[t] * (1.0 / float(eta_dch) - 1.0)
        for t in range(T)
    ]

    # KPI calculation (whole horizon)
    end_soc = float(pl.value(soc[T]))
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

# ---- NSGA-III
def run_model_f_nsga3(pop_size=100, ngen=40, cxpb=0.9, mutpb=0.1, divisions=12, use_mp=False):
    """Run NSGA-III on 3-gene individuals (cb, db, reserve_frac)."""
    nobj = _nobj_from_mode(OBJECTIVE_MODE)

    toolbox3 = base.Toolbox()
    toolbox3.register("attr_cb", random.uniform, LOW_DEAP[0], UP_DEAP[0])
    toolbox3.register("attr_db", random.uniform, LOW_DEAP[1], UP_DEAP[1])
    toolbox3.register("attr_rf", random.uniform, LOW_DEAP[2], UP_DEAP[2])
    toolbox3.register("individual", tools.initCycle, creator.Individual,
                      (toolbox3.attr_cb, toolbox3.attr_db, toolbox3.attr_rf), n=1)
    toolbox3.register("population", tools.initRepeat, list, toolbox3.individual)

    def eval3(ind):
        cb, db, rf = ind
        k = simulate_dispatch(cb, db, rf, END_TARGET_CONST)
        return _objective_tuple_from_kpis(k, OBJECTIVE_MODE)

    toolbox3.register("evaluate", eval3)
    toolbox3.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=10.0)
    toolbox3.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=15.0, indpb=0.25)

    ref_points = uniform_reference_points(nobj=nobj, p=divisions)
    toolbox3.register("select", tools.selNSGA3, ref_points=ref_points)

    pop = toolbox3.population(n=pop_size)
    hof3 = tools.ParetoFront()
    print("NSGA-III Started")

    fitnesses = list(map(toolbox3.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = tuple(map(float, fit))
    pop = toolbox3.select(pop, len(pop))

    if use_mp:
        with multiprocessing.Pool() as pool:
            toolbox3.register("map", pool.map)
            algorithms.eaMuPlusLambda(
                pop, toolbox3, mu=pop_size, lambda_=2*pop_size,
                cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof3, verbose=True
            )
    else:
        toolbox3.register("map", map)
        algorithms.eaMuPlusLambda(
            pop, toolbox3, mu=pop_size, lambda_=2*pop_size,
            cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof3, verbose=True
        )

    return hof3

# ---- Visualization helpers (unchanged)
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
                        row_heights=[0.6, 0.4], vertical_spacing=0.07)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["Demand"], name="Demand", marker_color="#333"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["PV"], name="PV", marker_color="#2ca02c", opacity=0.7), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["Import"], name="Import", marker_color="#1f77b4"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=series_df.index, y=series_df["Export"], name="Export", marker_color="#ff7f0e"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["PV->Storage"], name="PV->Storage", line=dict(dash="dot", color="#9467bd")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["Storage->Load"], name="Storage->Load", line=dict(dash="dot", color="#8c564b")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["SoC"], name="SoC (MWh)", line=dict(color="#17becf")), row=2, col=1)
    fig.add_trace(go.Scatter(x=series_df.index, y=series_df["Losses"], name="Losses (MWh)", line=dict(color="#d62728")), row=2, col=1)
    fig.update_yaxes(title_text="MW/MWh", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Flows (MWh)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy (MWh)", row=2, col=1)
    fig.update_layout(title=title, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

def plot_pareto_front(hof, title: str = "NSGA-II Pareto Front", mode: str = "all"):
    vals = [tuple(map(float, ind.fitness.values)) for ind in hof]

    if mode == "all":
        x_vals = [v[0] for v in vals]
        y_vals = [v[2] for v in vals]
        solar_share = [(1.0 - v[1]) * 100.0 for v in vals]
        fig = go.Figure(data=go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(size=8, color=solar_share, colorscale="Viridis",
                        colorbar=dict(title="Solar Share (%)")),
            text=[f"share={s:.1f}%" for s in solar_share],
            name="Pareto Front"
        ))
        fig.update_layout(title=title, xaxis_title="CO2 Emissions (tCO2)", yaxis_title="Storage Losses (MWh)")
        return fig

    x_vals = [v[0] for v in vals]
    y_vals = [v[1] for v in vals]
    if mode == "emissions":
        xt, yt = "CO2 Emissions (tCO2)", "CO2 Emissions (tCO2)"
    elif mode == "solar_share":
        xt, yt = "1 - Solar Share", "1 - Solar Share"
    elif mode == "losses":
        xt, yt = "Storage Losses (MWh)", "Storage Losses (MWh)"
    else:
        return plot_pareto_front(hof, title, "all")

    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="markers", name="Pareto Front",
                                    marker=dict(size=8, color="#1f77b4")))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt)
    return fig

def plot_pareto_front_mopso(archive_objs: np.ndarray, mode: str = "all"):
    objs = np.asarray(archive_objs, dtype=float)

    if mode == "all":
        x_vals = objs[:, 0].tolist()
        y_vals = objs[:, 2].tolist()
        solar_share = [(1.0 - v) * 100.0 for v in objs[:, 1].tolist()]
        fig = go.Figure(data=go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(size=8, color=solar_share, colorscale="Plasma",
                        colorbar=dict(title="Solar Share (%)")),
            text=[f"share={s:.1f}%" for s in solar_share],
            name="MOPSO Pareto Front"
        ))
        fig.update_layout(title="MOPSO Pareto Front", xaxis_title="CO2 Emissions (tCO2)", yaxis_title="Storage Losses (MWh)")
        return fig

    x_vals = objs[:, 0].tolist()
    y_vals = objs[:, 1].tolist()
    if mode == "emissions":
        title, xt, yt = "MOPSO Pareto Front (Emissions)", "CO2 Emissions (tCO2)", "CO2 Emissions (tCO2)"
    elif mode == "solar_share":
        title, xt, yt = "MOPSO Pareto Front (Solar Share)", "1 - Solar Share", "1 - Solar Share"
    elif mode == "losses":
        title, xt, yt = "MOPSO Pareto Front (Losses)", "Storage Losses (MWh)", "Storage Losses (MWh)"
    fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode="markers", name="MOPSO Pareto Front",
                                    marker=dict(size=8, color="#9467bd")))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt)
    return fig

def show_figure(fig, name="figure", offline=True):
    """
    Show figure and save standalone HTML.
    offline=True embeds plotly.js for local viewing without internet.
    """
    try:
        fig.show(renderer="browser")
    finally:
        outfile = os.path.join(EXPORT_DIR, f"{name}.html")
        fig.write_html(
            outfile,
            include_plotlyjs=True if offline else "cdn",
            full_html=True,
            auto_open=False
        )
        print(f"Saved plot to: {outfile}")

# ---- Weekly aggregation utilities (unchanged)
def aggregate_weekly(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample original-resolution dispatch series to weekly aggregates for readability/export.
      - Energy flows summed per week; end-of-week SoC captured as the last value.
      - Derived weekly Solar Share and Solar Wasted computed for visualization.
    """
    if not isinstance(series_df.index, pd.DatetimeIndex):
        raise ValueError("Weekly aggregation requires a datetime index.")
    # Week ending Sunday; change to 'W-MON' for Monday weeks if preferred
    week_df = series_df.resample("W").agg({
        "Demand": "sum",
        "PV": "sum",
        "Direct": "sum",
        "PV->Storage": "sum",
        "Storage->Load": "sum",
        "Export": "sum",
        "Import": "sum",
        "Losses": "sum",
        "SoC": "last"
    })
    week_df.rename(columns={"SoC": "End SoC"}, inplace=True)
    # Derived weekly KPIs (guard against division by zero)
    demand_nonzero = week_df["Demand"].replace(0, np.nan)
    pv_nonzero = week_df["PV"].replace(0, np.nan)
    week_df["Solar Share (%)"] = 100.0 * (week_df["Direct"] + week_df["Storage->Load"]) / demand_nonzero
    week_df["Solar Wasted (%)"] = 100.0 * week_df["Export"] / pv_nonzero
    week_df.fillna(0.0, inplace=True)
    return week_df

def plot_dispatch_series_weekly(week_df: pd.DataFrame, title: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        specs=[[{"secondary_y": True}], [{}]],
                        row_heights=[0.6, 0.4], vertical_spacing=0.07)
    fig.add_trace(go.Bar(x=week_df.index, y=week_df["Demand"], name="Demand", marker_color="#333"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=week_df.index, y=week_df["PV"], name="PV", marker_color="#2ca02c", opacity=0.7), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=week_df.index, y=week_df["Import"], name="Import", marker_color="#1f77b4"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=week_df.index, y=week_df["Export"], name="Export", marker_color="#ff7f0e"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=week_df.index, y=week_df["PV->Storage"], name="PV->Storage", line=dict(dash="dot", color="#9467bd")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=week_df.index, y=week_df["Storage->Load"], name="Storage->Load", line=dict(dash="dot", color="#8c564b")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=week_df.index, y=week_df["End SoC"], name="End SoC (MWh)", line=dict(color="#17becf")), row=2, col=1)
    fig.add_trace(go.Scatter(x=week_df.index, y=week_df["Losses"], name="Losses (MWh)", line=dict(color="#d62728")), row=2, col=1)
    fig.update_yaxes(title_text="Weekly Energy (MWh)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Weekly Flows (MWh)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Energy / Losses (MWh)", row=2, col=1)
    fig.update_layout(title=title, barmode="group",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

def show_weekly_figure(fig, name: str, weekly_dir: str, offline: bool = True):
    try:
        fig.show(renderer="browser")
    finally:
        outfile = os.path.join(weekly_dir, f"{name}.html")
        fig.write_html(outfile, include_plotlyjs=True if offline else "cdn",
                       full_html=True, auto_open=False)
        print(f"Saved weekly plot to: {outfile}")

# ---- Script entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline, heuristic, or evolutionary/swarm optimization models.")
    parser.add_argument("--model", choices=["a", "b", "c", "d", "e", "f", "g", "all"], default="all")
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--ngen", type=int, default=40)
    parser.add_argument("--cxpb", type=float, default=0.9)
    parser.add_argument("--mutpb", type=float, default=0.1)
    parser.add_argument("--mp", action="store_true")
    # PSO
    parser.add_argument("--pso-swarm", type=int, default=60)
    parser.add_argument("--pso-iters", type=int, default=60)
    parser.add_argument("--pso-w", type=float, default=0.7)
    parser.add_argument("--pso-c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", type=float, default=1.5)
    # MOPSO
    parser.add_argument("--mopso-swarm", type=int, default=80)
    parser.add_argument("--mopso-iters", type=int, default=80)
    parser.add_argument("--mopso-w", type=float, default=0.7)
    parser.add_argument("--mopso-c1", type=float, default=1.5)
    parser.add_argument("--mopso-c2", type=float, default=1.5)
    parser.add_argument("--mopso-archive", type=int, default=100)
    parser.add_argument("--mopso-vmax", type=float, default=0.2, help="Max velocity as fraction of range per dimension")
    # NSGA-III
    parser.add_argument("--nsga3-div", type=int, default=12, help="Divisions for NSGA-III reference points")
    # Objectives
    parser.add_argument("--objective-set",
                        choices=["all", "emissions", "solar_share", "losses"],
                        default="all",
                        help="Objectives selection: all=3 objectives; others duplicate single objective.")
    # Output interval (raw vs weekly view-only)
    parser.add_argument("--interval",
                        choices=["raw", "weekly"],
                        default="raw",
                        help="Export granularity for dispatch series: raw=original timestep; weekly=aggregated view exported to 'Weekly' subfolder.")
    # Capacity sweep (NSGA-II 'c' and PSO 'd' only)
    parser.add_argument("--cap-sweep", type=str, default="",
                        help="Comma-separated capacities in MWh to sweep (e.g., 500,1000,2000). Applies to models 'c' and 'd'.")
    parser.add_argument("--cap-range", type=str, default="",
                        help="Range sweep 'start,end,step' in MWh (e.g., 500,2000,250). Applies to models 'c' and 'd'.")

    args = parser.parse_args()

    OBJECTIVE_MODE = args.objective_set
    subdir = {"all": "all", "emissions": "emissions", "solar_share": "solar share", "losses": "losses"}[OBJECTIVE_MODE]
    EXPORT_DIR = os.path.join(EXPORT_DIR, subdir)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"Objective mode: {OBJECTIVE_MODE}. Exporting figures to: {EXPORT_DIR}")

    # Weekly subfolder (view-only aggregation)
    WEEKLY_DIR = None
    if args.interval == "weekly":
        WEEKLY_DIR = os.path.join(EXPORT_DIR, "Weekly")
        os.makedirs(WEEKLY_DIR, exist_ok=True)
        print(f"Weekly aggregation enabled. Weekly figures to: {WEEKLY_DIR}")

    configure_deap(OBJECTIVE_MODE)

    def print_kpis(title, k):
        print(f"\n=== {title} ===")
        order = ["Grid Imports (MWh)", "CO2 Emissions (tCO2)", "Solar Share (%)",
                 "Solar Wasted (%)", "Solar Direct to Load (MWh)",
                 "Solar to Storage (MWh)", "Storage to Load (MWh)",
                 "Storage Losses (MWh)", "Solar Exported (MWh)",
                 "Start Storage (MWh)", "End Storage (MWh)",
                 "Delta SoC (MWh)", "Energy Balance Check (MWh)"]
        for key in order:
            print(f"{key}: {k[key]:.4f}")

    random.seed(42); np.random.seed(42)

    k_a = k_b = k_best = None
    k_pso = None
    pso_best_genes = None
    k_mopso = None
    mopso_archive_pos = None
    mopso_archive_objs = None
    mopso_best_genes = None
    k_lp = None

    # Parse capacity sweep inputs
    sweep_caps: list[float] = []
    if args.cap_sweep.strip():
        try:
            sweep_caps = [float(x) for x in args.cap_sweep.split(",")]
        except Exception:
            print("Invalid --cap-sweep format. Expected comma-separated numbers.")
            sweep_caps = []
    elif args.cap_range.strip():
        try:
            s, e, st = [float(x) for x in args.cap_range.split(",")]
            if st == 0:
                raise ValueError("Step must be non-zero.")
            n = int(np.floor((e - s) / st)) + 1
            sweep_caps = [s + i * st for i in range(max(n, 0))]
        except Exception:
            print("Invalid --cap-range format. Expected 'start,end,step' with numeric values.")
            sweep_caps = []

    # ---- Capacity sweep execution FIRST (before other models)
    if sweep_caps:
        if args.model not in ("c", "d", "all"):
            print("Capacity sweep requested but model is not NSGA-II ('c') or PSO ('d'). Sweep skipped.")
        else:
            selected_model = args.model if args.model in ("c", "d") else "c"  # default to NSGA-II on 'all'
            if args.model == "all":
                print("Note: --model=all with capacity sweep will run NSGA-II sweep only to avoid extended runtimes.")
            nsga2_args = dict(pop_size=args.pop_size, ngen=args.ngen, cxpb=args.cxpb, mutpb=args.mutpb, use_mp=args.mp)
            pso_args = dict(swarm_size=args.pso_swarm, iters=args.pso_iters, w=args.pso_w, c1=args.pso_c1, c2=args.pso_c2)

            sweep_df, sweep_details = run_capacity_sweep(selected_model, sweep_caps, nsga2_args=nsga2_args, pso_args=pso_args)

            # Print tabular summary
            print("\n=== Capacity Sweep Summary ===")
            cols = ["Capacity (MWh)", "CO2 Emissions (tCO2)", "Solar Share (%)", "Storage Losses (MWh)",
                    "Grid Imports (MWh)", "Solar Wasted (%)", "Solar Direct to Load (MWh)",
                    "Solar to Storage (MWh)", "Storage to Load (MWh)"]
            for _, row in sweep_df[cols].iterrows():
                print(f"Cap={row['Capacity (MWh)']:.1f} | Em={row['CO2 Emissions (tCO2)']:.4f} "
                      f"| Share={row['Solar Share (%)']:.2f}% | Losses={row['Storage Losses (MWh)']:.2f} "
                      f"| Imports={row['Grid Imports (MWh)']:.2f} | Wasted={row['Solar Wasted (%)']:.2f}%")

            # Export CSV summary to the objective subfolder
            sweep_csv = os.path.join(EXPORT_DIR, "capacity_sweep_summary.csv")
            sweep_df.to_csv(sweep_csv, index=False)
            print(f"Saved capacity sweep summary CSV to: {sweep_csv}")

            # Simple plot: emissions and solar share vs capacity
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sweep_df["Capacity (MWh)"], y=sweep_df["CO2 Emissions (tCO2)"],
                    mode="lines+markers", name="Emissions"
                ))
                fig.add_trace(go.Scatter(
                    x=sweep_df["Capacity (MWh)"], y=sweep_df["Solar Share (%)"],
                    mode="lines+markers", name="Solar Share (%)", yaxis="y2"
                ))
                fig.update_layout(
                    title="Capacity Sweep: Emissions and Solar Share",
                    xaxis_title="Battery Capacity (MWh)",
                    yaxis=dict(title="CO2 Emissions (tCO2)"),
                    yaxis2=dict(title="Solar Share (%)", overlaying="y", side="right")
                )
                show_figure(fig, "capacity_sweep")
            except Exception as e:
                print(f"Plotting failed for capacity sweep: {e}")

    # ---- Run selected models (original resolution) AFTER sweep
    if args.model in ("a", "all"):
        k_a = run_model_a_no_storage(return_series=False)
        print_kpis("Model A - No Storage", k_a)

    if args.model in ("b", "all"):
        k_b = run_model_b_rule_based(high_demand_quantile=args.quantile, return_series=False)
        print_kpis(f"Model B - Rule-Based (Q={args.quantile})", k_b)

    hof = None; best = None
    hof3 = None; best3 = None
    if args.model in ("c", "all"):
        configure_deap(OBJECTIVE_MODE)
        hof = run_model_c_nsga2(pop_size=args.pop_size, ngen=args.ngen,
                                cxpb=args.cxpb, mutpb=args.mutpb, use_mp=args.mp)
        print(f"\nPareto set size: {len(hof)}")
        best = min(hof, key=lambda ind: ind.fitness.values[0])
        k_best = simulate_dispatch(best[0], best[1], best[2], END_TARGET_CONST)
        genes = [round(float(x), 4) for x in best]
        if OBJECTIVE_MODE == "solar_share":
            label = "NSGA-II Best by (1 - Share)"
        elif OBJECTIVE_MODE == "losses":
            label = "NSGA-II Best by Losses"
        else:
            label = "NSGA-II Best by Emissions"
        print(f"Model C - {label}: genes={genes}")
        print_kpis("Model C - NSGA-II KPIs", k_best)

    if args.model in ("d", "all"):
        pso_best_genes, k_pso = run_model_d_pso(
            swarm_size=args.pso_swarm,
            iters=args.pso_iters,
            w=args.pso_w,
            c1=args.pso_c1,
            c2=args.pso_c2
        )
        print_kpis("Model D - PSO KPIs", k_pso)

    if args.model in ("e", "all"):
        mopso_archive_pos, mopso_archive_objs, mopso_best_genes, k_mopso = run_model_e_mopso(
            swarm_size=args.mopso_swarm,
            iters=args.mopso_iters,
            w=args.mopso_w,
            c1=args.mopso_c1,
            c2=args.mopso_c2,
            archive_size=args.mopso_archive,
            vmax_frac=args.mopso_vmax
        )
        print_kpis("Model E - MOPSO KPIs (Best by first objective)", k_mopso)

    if args.model in ("f", "all"):
        hof3 = run_model_f_nsga3(
            pop_size=args.pop_size, ngen=args.ngen,
            cxpb=args.cxpb, mutpb=args.mutpb,
            divisions=args.nsga3_div, use_mp=args.mp
        )
        print(f"\nNSGA-III Pareto set size: {len(hof3)}")
        best3 = min(hof3, key=lambda ind: ind.fitness.values[0])
        k_best3 = simulate_dispatch(best3[0], best3[1], best3[2], END_TARGET_CONST)
        genes3 = [round(float(x), 4) for x in best3]
        print(f"Model F - NSGA-III Best by First Objective: genes={genes3}")
        print_kpis("Model F - NSGA-III KPIs", k_best3)

    if args.model in ("g", "all"):
        k_lp = run_model_g_lp(mode=OBJECTIVE_MODE, reserve_frac=RESERVE_CONST, end_target_frac=None, weights=(0.5, 0.5), return_series=False)
        print_kpis("Model G - LP KPIs", k_lp)

    # ---- Visualization (KPI comparison bar)
    models_for_bar = {}
    if k_a: models_for_bar["A - No Storage"] = k_a
    if k_b: models_for_bar["B - Rule-Based"] = k_b
    if 'k_best' in locals() and k_best: models_for_bar["C - NSGA-II"] = k_best
    if k_pso: models_for_bar["D - PSO"] = k_pso
    if k_mopso: models_for_bar["E - MOPSO"] = k_mopso
    if 'k_best3' in locals() and k_best3: models_for_bar["F - NSGA-III"] = k_best3
    if k_lp: models_for_bar["G - LP"] = k_lp

    if models_for_bar:
        show_figure(compare_kpis_bar(models_for_bar), "kpi_comparison")

    # ---- Raw and optional weekly plotting/export helpers
    def export_dispatch_with_optional_weekly(series_df: pd.DataFrame, base_name: str):
        """
        Export raw (original resolution) dispatch plots to objective subfolder.
        If weekly export enabled, also create aggregated weekly plots and CSV under 'Weekly' subfolder.
        """
        show_figure(plot_dispatch_series(series_df, f"{base_name} Dispatch"), base_name.lower().replace(" ", "_"))
        if WEEKLY_DIR:
            week_df = aggregate_weekly(series_df)
            week_fig = plot_dispatch_series_weekly(week_df, f"{base_name} Weekly Aggregated Dispatch")
            show_weekly_figure(week_fig, base_name.lower().replace(" ", "_") + "_weekly", WEEKLY_DIR)
            csv_path = os.path.join(WEEKLY_DIR, base_name.lower().replace(" ", "_") + "_weekly.csv")
            week_df.to_csv(csv_path)
            print(f"Saved weekly CSV to: {csv_path}")

    # ---- Per-model series exports
    if args.model in ("a", "all"):
        _, series_a = run_model_a_no_storage(return_series=True)
        export_dispatch_with_optional_weekly(series_a, "Model A - No Storage")
    if args.model in ("b", "all"):
        _, series_b = run_model_b_rule_based(high_demand_quantile=args.quantile, return_series=True)
        export_dispatch_with_optional_weekly(series_b, f"Model B - Rule-Based (Q={args.quantile})")
    if args.model in ("c", "all") and 'best' in locals() and best is not None:
        _, series_c = simulate_dispatch(best[0], best[1], best[2], END_TARGET_CONST, return_series=True)
        export_dispatch_with_optional_weekly(series_c, "Model C - NSGA-II Best")
        show_figure(plot_pareto_front(hof, title="NSGA-II Pareto Front", mode=OBJECTIVE_MODE), "pareto_front")
    if args.model in ("d", "all") and pso_best_genes is not None:
        _, series_d = simulate_dispatch(*pso_best_genes, return_series=True)
        export_dispatch_with_optional_weekly(series_d, "Model D - PSO Best")
    if args.model in ("e", "all") and mopso_best_genes is not None:
        _, series_e = simulate_dispatch(*mopso_best_genes, return_series=True)
        export_dispatch_with_optional_weekly(series_e, "Model E - MOPSO Best-by-First-Objective")
        if mopso_archive_objs is not None and len(mopso_archive_objs) > 0:
            show_figure(plot_pareto_front_mopso(mopso_archive_objs, mode=OBJECTIVE_MODE), "pareto_front_mopso")
    if args.model in ("f", "all") and 'best3' in locals() and best3 is not None:
        _, series_f = simulate_dispatch(best3[0], best3[1], best3[2], END_TARGET_CONST, return_series=True)
        export_dispatch_with_optional_weekly(series_f, "Model F - NSGA-III Best")
        show_figure(plot_pareto_front(hof3, "NSGA-III Pareto Front", mode=OBJECTIVE_MODE), "pareto_front_nsga3")
    if args.model in ("g", "all") and k_lp is not None:
        k_lp_vals, series_lp = run_model_g_lp(mode=OBJECTIVE_MODE, reserve_frac=RESERVE_CONST, end_target_frac=None, weights=(0.5, 0.5), return_series=True)
        export_dispatch_with_optional_weekly(series_lp, "Model G - LP Optimal Dispatch")
