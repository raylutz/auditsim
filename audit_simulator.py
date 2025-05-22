# audit_simulator.py

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pprint

from typing import List, Dict, Any, Tuple, Optional, Union, Set, cast, Type, Iterable, Callable
def fake_function(a: Optional[List[Dict[str, Tuple[Union[Iterable, Callable, Set, Type],int,Any]]]] = None) -> Optional[int]:
    a = cast(a, int)
    return None

T_dodi = Dict[str, Dict[str, int]]
T_doda = Dict[str, Dict[str, Any]]
T_lor  = List[range]


def main(ui_keys: list[str], test_state: dict=None):

    changes, op_state = get_state(ui_keys, test_state)
    
    op_plan = gen_op_plan(changes, test_state)
    
    gen_election_model  (op_plan, op_state)
    gen_samples         (op_plan, op_state)
    gen_stats           (op_plan, op_state)
    gen_plots           (op_plan, op_state)

    if test_state:
        return op_state['fig']

    fig = op_state['fig']
    del op_state['fig']
    if 'votes_dodi' in op_state:
        del op_state['votes_dodi']  # don't save the votes once stats are generated.
    if 'bin_samples' in op_state:
        del op_state['bin_samples']
    
    st.session_state["op_state"] = op_state  # remember state for next pass
    print(pprint.pformat(op_state, sort_dicts=False))
    
    st.plotly_chart(fig, use_container_width=True)

    return None
    

def get_state(
        ui_keys: list[str],
        test_state: dict | None
        ) -> Tuple[dict[str, bool], dict]:     # changes, op_state
    """
    Detect changes to input parameters by comparing Streamlit inputs to previous state.
    
    Args:
        previous_state: Prior parameter state (or None on first run).
    
    Returns:
        - changes: dict[str, bool] showing which keys changed
        - current_state: new snapshot of all tracked parameters
    """
    # current_state = {
        # 'n_total':    st.session_state['n_total'],
        # 'margin_pct': st.session_state['margin_pct'],
        # 'noise_pct':  st.session_state['noise_pct'],
        # 'hack_pct':   st.session_state['hack_pct'],
        # 'n_samples':  st.session_state['n_samples'],
        # 'n_trials':   st.session_state['n_trials'],
        # 'risk_sd':    st.session_state['risk_sd'],
        # 'risk_label': st.session_state['risk_label']
    # }
    
    if test_state:
        current_state = test_state
        prev_state = {}
    else:
        # only consider ui_keys
        current_state = {k: v for k, v in st.session_state.items() if k in ui_keys}
        prev_state    = dict(st.session_state.get('prev_state', {}))
        print(f"Recovered session_state:\n{pprint.pformat(current_state, sort_dicts=False)}")


    if not prev_state:
        changes = {k: True for k in current_state}
        op_state = current_state
    else:
        changes = {
            k: current_state[k] != prev_state.get(k)
                for k in current_state
            }
        op_state = st.session_state.get('op_state', current_state)
        
    if not test_state:
        st.session_state['prev_state'] = current_state
        
    return changes, op_state
    

def gen_op_plan(changes: dict[str, bool], test_state: dict | None) -> dict[str, bool]:
    """
    Interpret field-level changes into execution flags.
    Determines which parts of the pipeline must run.
    """
    model_keys  = ['n_total', 'margin_pct', 'noise_pct', 'hack_pct']
    sample_keys = ['n_samples', 'n_trials']
    plot_keys   = ['plot_H0_trials', 'plot_H1_trials', 'plot_H0_contours', 'plot_H1_contours']

    if test_state:
        op_plan = {}
        for key in ['gen_model_flag', 'gen_samples_flag', 'gen_stats_flag'] + plot_keys:
            op_plan[key] = True
        return op_plan
    
    gen_model_flag      = any(changes.get(k, False) for k in model_keys)
    gen_samples_flag    = gen_model_flag or any(changes.get(k, False) for k in sample_keys)
    gen_stats_flag      = gen_samples_flag

    op_plan = {
        'gen_model_flag':   gen_model_flag,
        'gen_samples_flag': gen_samples_flag,
        'gen_stats_flag':   gen_stats_flag,
        }

    for key in plot_keys:
        op_plan[key] = bool(st.session_state.get(key, False))
        
    return op_plan    


# === Step 1: Generate election model ===
def gen_election_model(
        op_plan,
        op_state,
        ) -> None: # updates op_state    
    """
    Generate both H0 (null) and H1 (flipped) election models.

    updates
        op_state['votes_dodi']: {"H0": votes_H0, "H1": votes_H1}
        op_state['meta_doda']:  {"H0": meta_H0,  "H1": meta_H1}
    """
    if not op_plan['gen_model_flag']:
        return
    
    votes_dodi = {}
    meta_doda  = {}

    for hyp in ("H0", "H1"):
        votes_dodi[hyp], meta_doda[hyp] = gen_election_instance(
            op_state    = op_state,
            is_flipped  = bool(hyp == 'H1')
            )
    op_state['votes_dodi'] = votes_dodi
    op_state['meta_doda']  = meta_doda
    return
    
def gen_election_instance(
        op_state: dict,
        is_flipped: bool,
        ) -> tuple[dict, dict]:  # votes_di, meta_da
    """
    Generate a single election instance (H0 or H1) using block sizes and cumulative bin boundaries.
    """
    n_total         = op_state.get('n_total',       110_000)
    A_votes         = op_state.get('A_votes',        51_500)
    B_votes         = op_state.get('B_votes',        48_500)
    margin_pct      = op_state.get('margin_pct',          3)
    noise1_pct      = op_state.get('noise1_pct',         .2)
    hack2_pct       = op_state.get('hack2_pct',           0)

    margin_frac = margin_pct / 100
    noise1_frac = noise1_pct / 100
    hack2_frac  = hack2_pct  / 100

    # Step 1: Compute V — total ballots with actual A or B vote (excluding N)
    V = A_votes + B_votes
    #V = int(np.ceil(n_total * (1 - noise1_frac) / (1 + ((margin_frac * (1 - hack2_frac)) / 2))))

    o1_total = round(V * margin_frac)                   # total overstatement effect needed
    o2_hack = round(o1_total * hack2_frac / 2)          # 2-vote overstatements (each contributes 2)
    o1_hack = o1_total - 2 * o2_hack                    # remaining 1-vote overstatements

    noise1 = round(n_total * noise1_frac)
    onAN = onNB = unNA = unBN = round(noise1 / 2)
    onAB = noise2 = 0                                   # noise2 is not used; set to 0
    unBA = onAB
    noise2 = noise2                                     # fool linter

    # For H1 (flipped): add actual manipulation overstatements
    ohAN = ohNB = ohAB = 0
    if is_flipped:
        ohAN = round(o1_hack / 2)
        ohNB = round(o1_hack / 2)
        ohAB = o2_hack

    # Reported vote totals (before noise or hacking)
    A_rep = round(V * (1 + margin_frac) / 2)
    B_rep = V - A_rep

    H_0A = A_rep - onAB - onAN - (ohAN + ohAB if is_flipped else 0)
    H_0B = B_rep - unBA - unBN
    H_0N = 0 if is_flipped else round(n_total - V - noise1)  # no-vote block, only in H0

    # note that the most commmon bins are placed first as a speed optimization.
    votes_di = {
        "H_0A": H_0A,
        "H_0B": H_0B,
        "H_0N": H_0N,
        "oNB":  onNB + ohNB,
        "oAN":  onAN + ohAN,
        "unNA": unNA,
        "unBN": unBN,
        "oAB":  onAB + ohAB,
        "unBA": unBA,
    }

    # Metadata
    A_total = onAB + onAN + H_0A
    B_total = unBN + unBA + H_0B if not is_flipped else onAB + ohAB + onNB + ohNB + H_0B
    margin = A_total - B_total
    a_frac = round(A_total / V, 4)
    b_frac = round(B_total / V, 4)

    meta_da = {
        "A_total": A_total,
        "B_total": B_total,
        "A_frac": a_frac,
        "B_frac": b_frac,
        "margin": round(margin / V, 4),
        "V": V,
        "n_total": n_total,
        "bin2os": np.array([0, 0, 0, +1, +1, -1, -1, +2, -2]),
    }

    return votes_di, meta_da
    


# === Step 2: Generate audit samples ===
def gen_samples(
        op_plan,
        op_state,
        ) -> None: # updates op_state    
    """
    Generate audit trials by sampling bin numbers from each election model (H0, H1).

    Returns:
        bin_samples: {"H0": np.ndarray, "H1": np.ndarray}, shape = (trials, n_samples)
    """
    if not op_plan['gen_samples_flag']:
        return
    diagnose = False

    n_samples   = op_state.get('n_samples', 500)
    n_trials    = op_state.get('n_trials', 1000)
    replace     = op_state.get('replace', False)
    meta_doda   = op_state.get('meta_doda', {})
    votes_dodi  = op_state.get('votes_dodi', {})
    
    print(f"Creating {n_trials} audit trials, each with {n_samples} samples per case...")

    bin_samples_donpa = {"H0": np.zeros((n_trials, n_samples), dtype=int),
                         "H1": np.zeros((n_trials, n_samples), dtype=int)}

    for hyp in ["H0", "H1"]:
        if diagnose:
            print(f"{hyp}")
        n_total = meta_doda[hyp]['n_total']
        if diagnose:
            print(f"{n_total}")

        # bins_lor = build_lor_from_blocks(block_sizes = votes_dodi[hyp].values(), n_total = n_total)
        # if diagnose:
            # print(f"{bins_lor}")
        thresholds = np.cumsum(list(votes_dodi[hyp].values()))

        #idx_to_bin = build_reverse_index_array(bins_lor, n_total)
        
        # if hyp == 'H1':
            # breakpoint()
            # pass
        
        for trial in range(n_trials):
            if diagnose:
                print(f"trial:{trial}")
            sample_indices = np.random.choice(n_total, size=n_samples, replace=replace)
            if diagnose:
                print(f"{sample_indices=}\n, starting convert to bins...")
            #bin_ids = idx_to_bin[sample_indices]
            bin_ids = classify_bins_9bins(sample_indices, thresholds)
            # bin_ids = convert_vector_to_bins(sample_indices, bins_lor)
            if diagnose:
                print(f"{bin_ids=}")
            bin_samples_donpa[hyp][trial] = bin_ids
            
    op_state['bin_samples'] = bin_samples_donpa
    return None
    

def classify_bins_9bins(B: np.ndarray, thresholds: list[int]) -> np.ndarray:
    
    A = ((B < thresholds[0])).astype(int) * 0 \
      + ((B >= thresholds[0]) & (B < thresholds[1])).astype(int) * 1 \
      + ((B >= thresholds[1]) & (B < thresholds[2])).astype(int) * 2 \
      + ((B >= thresholds[2]) & (B < thresholds[3])).astype(int) * 3 \
      + ((B >= thresholds[3]) & (B < thresholds[4])).astype(int) * 4 \
      + ((B >= thresholds[4]) & (B < thresholds[5])).astype(int) * 5 \
      + ((B >= thresholds[5]) & (B < thresholds[6])).astype(int) * 6 \
      + ((B >= thresholds[6]) & (B < thresholds[7])).astype(int) * 7 \
      + ((B >= thresholds[7])).astype(int) * 8
    return A


# === Step 3: Compute stats ===
def gen_stats(
        op_plan,
        op_state,
        ) -> None: # updates op_state    
    """
    Generate OS (overstatements), CS (cumulative sums), Mean, and SD across trials.

    Args:
        bin_samples: {"H0": array, "H1": array} of shape (trials, n_samples), containing bin indices
        bin2os: array of overstatement values per bin index (length 9)

    Returns:
        stats: dict with keys "H0" and "H1", each containing:
            - OS: overstatements per sample
            - CS: cumulative overstatements
            - Mean: mean cumulative overstatement at each sample index
            - SD: std dev across trials at each sample index
    """
    if not op_plan['gen_stats_flag']:
        return
        
    print("Generating stats: OS, CS, Mean, SD...")
    
    bin_samples = op_state['bin_samples']
    meta_doda   = op_state['meta_doda']

    stats = {}

    for hyp in ["H0", "H1"]:
        # breakpoint()
        # pass
        
        bins = bin_samples[hyp]     # shape: (trials, n_samples)
        bin2os = meta_doda[hyp]['bin2os']
        
        OS = bin2os[bins]           # shape: same as bins — values in [-2, 2]
        CS = np.cumsum(OS, axis=1)  # cumulative sum over sample axis

        Mean = np.mean(CS, axis=0)  # mean at each sample step
        SD = np.std(CS, axis=0, ddof=1)  # sample SD (unbiased)

        stats[hyp] = {
            "OS": OS,
            "CS": CS,
            "Mean": Mean,
            "SD": SD
        }

    op_state['stats'] = stats
    return None


# === Step 4: Plot results ===
def gen_plots(
        op_plan,
        op_state,
        ) -> None:   

    # from scipy.stats import norm

    RiskText    = ["20%",  "10%",  "5%",   "2.5%", "1%",   "0.5%", "0.1%"]                #, "5sigma"]
    #RiskFrac    = [.20,    .1,     .05,    0.025,  0.01,   0.005,  0.001]                 #,  0.0000003]
    #sd_mult     = [round(norm.ppf(1 - r), 4) for r in RiskFrac]                          # dynamically generated.
    SD_mult     = [0.8416, 1.2816, 1.6449, 1.9600, 2.3263, 2.5758, 3.0902]                #,  5)
    nRisks      = len(RiskText)
    riskidx0_1percent   = RiskText.index("0.1%")
    riskidx5percent     = RiskText.index("5%")


    n_total     = op_state.get('n_total',       100_000)
    margin_pct  = op_state.get('margin_pct',    3)
    noise1_pct  = op_state.get('noise1_pct',    .2)
    hack2_pct   = op_state.get('hack2_pct',     0)
    n_samples   = op_state.get('n_samples',     500)
    n_trials    = op_state.get('n_trials',      1000)
    replace     = op_state.get('replace',       False)
    
    stats       = op_state.get('stats', {})
    
    x_axis      = np.array(range(n_samples))

    fig = go.Figure()

    # Plot individual trials (scissors)
    for hyp, color in [("H0", "green"), ("H1", "red")]:
        if not op_plan[f'plot_{hyp}_trials']:
            continue
            
        CS = stats[hyp]["CS"]
        for r in range(min(n_trials, CS.shape[0])):
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=CS[r],
                mode='lines',
                line=dict(color=color, width=1.5),
                opacity=0.4,
                showlegend=False,
            ))

    # Plot contours
    for hyp, color in [("H0", "yellow"), ("H1", "white")]:
        if not op_plan[f"plot_{hyp}_contours"]:
            continue

        mean = stats[hyp]["Mean"]
        sd = stats[hyp]["SD"]

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=mean,
            mode='lines',
            name=f"{hyp} Contours",
            line=dict(color=color, width=2),
        ))

        for riskidx in range(nRisks):
            
            if hyp == 'H0' and riskidx not in [riskidx0_1percent, riskidx5percent]:
                continue

            SD_mult_value = SD_mult[riskidx]
            
            if hyp == 'H0':
                mid_x = 7 * len(mean) // 8
            else:    
                mid_x = 3 * len(mean) // 4
            
            fig.add_trace(go.Scatter(
                x   = x_axis,
                y   = mean - SD_mult_value * sd,
                mode = 'lines',
                name = f"{hyp} Risk",
                line = dict(color=color, width=1),
                showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x   = x_axis,
                y   = mean + SD_mult_value * sd,
                mode = 'lines',
                name = f"{hyp} Risk",
                line = dict(color=color, width=1),
                showlegend=False,
                ))
            y1 = mean[mid_x] - SD_mult_value * sd[mid_x]     # upper
            y2 = mean[mid_x] + SD_mult_value * sd[mid_x]     # lower
            fig.add_annotation(
                x=mid_x, y=y1,
                text=f"{RiskText[riskidx]}",
                showarrow=False,
                font=dict(color=color, size=10),
                yanchor="bottom",
            )
            fig.add_annotation(
                x=mid_x, y=y2,
                text=f"{RiskText[riskidx]}",
                showarrow=False,
                font=dict(color=color, size=10),
                yanchor="bottom",
            )

        mean = stats[hyp]["Mean"]
        sd = stats[hyp]["SD"]


    for riskidx in range(nRisks):
        upper_H0 = stats['H0']['Mean'] + SD_mult[riskidx] * stats['H0']['SD']
        lower_H1 = stats['H1']['Mean'] - SD_mult[riskidx] * stats['H1']['SD']
        # Find the first index i where the H0 upper bound exceeds or meets
        # the H1 lower bound — i.e., where the two risk contours overlap or cross.
        # np.where(...) returns a tuple of arrays; [0] extracts the index array.
        
        crossing = np.where(upper_H0 <= lower_H1)[0]
        if crossing.size > 0:
            min_samples_this_risk = int(crossing[0])
            fig.add_shape(
                type="line",
                x0=min_samples_this_risk, x1=min_samples_this_risk,
                xref="x",
                yref="paper",   # use relative Y coordinates from 0 (bottom) to 1 (top)
                y0=0,
                y1=1,
                line=dict(color="orange", dash="dash")
            )        
            fig.add_annotation(
                x=min_samples_this_risk + 5, 
                y=0,                   # bottom of plot
                xref="x",
                yref="paper",          # relative y=0 (bottom edge of chart area)
                text=f"{RiskText[riskidx]}:{min_samples_this_risk} samples",
                showarrow=False,
                font=dict(color="orange", size=10),
                yanchor="bottom",      # attach bottom of text box to y=0
                xanchor="left"         # align text to the right of the vertical line
            )

    fig.update_layout(
        title=f"Ballot Comparison RLA -- {st.session_state.get('election_name', '')}",
        xaxis_title="Ballot Samples",
        yaxis_title="Cumulative Net Overstatements",
        height=600,
        margin=dict(l=40, r=20, t=60, b=40),
        )
        
    fig.add_annotation(
        text=f"nTotalBallots:{n_total}; margin:{margin_pct:.2f}%; noise1pct:{noise1_pct:.2f}%; hack2pct:{hack2_pct:.2f}%; trials:{n_trials}; replace:{replace}",
        xref="paper", yref="paper",
        x=0, y=10,  # bottom-left corner
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=10)
        )

    op_state['fig'] = fig



if __name__ == "__main__":
    import sys

    # Optional: accept 'test' as CLI argument
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        pass
        
        # import pprint
        # print("Running basic test case for audit simulator...")

        # test_state = dict(
            # n_total=100_000,
            # margin_pct=3.0,
            # noise1_pct=0.2,
            # hack2_pct=0.0,
            # n_samples=500,
            # n_trials=1000,
            # )
            
        # votes, meta = gen_election_model(
            # op_state = test_state
        # )

        # for hyp in ('H0', 'H1'):
            # print(f"\nVotes ({hyp}):")
            # pprint.pprint(votes[hyp], sort_dicts=False)
            # print(f"\nMetadata ({hyp}):")
            # pprint.pprint(meta[hyp], sort_dicts=False)
        
        # print("Generating samples")
        # bin_samples = generate_samples(votes, meta, n_samples=1000, n_trials=1000, replace=False)
        
        # pprint.pprint(bin_samples, sort_dicts=False)
    

    elif len(sys.argv) > 1 and sys.argv[1] == "run_full_test":
        print("Running full audit simulation test...")

        test_state = dict(
            n_total      = 110_000,
            A_votes      = 51_500,
            B_votes      = 45_500,
            margin_pct   = 3.0,
            noise1_pct   = 0.2,
            hack2_pct    = 0.0,
            n_samples    = 500,
            n_trials     = 1000,
            )
            
        fig = main(test_state)

        fig.write_html("audit_sim_output.html")
        print("Plot saved to audit_sim_output.html")    
    else:
        print("No action specified. Run with 'test' or 'run_full_test' argument to execute a test case.")
        
        