# audit_simulator.py

import numpy as np
import markdown
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import pprint

from typing import List, Dict, Any, Tuple, Optional, Union, Set, cast, Type, Iterable, Callable
def fake_function(a: Optional[List[Dict[str, Tuple[Union[Iterable, Callable, Set, Type],int,Any]]]] = None) -> Optional[int]:
    a = cast(a, int)
    return None

T_dodi = Dict[str, Dict[str, int]]
T_doda = Dict[str, Dict[str, Any]]
T_lor  = List[range]


def main_simulator(ui_keys: list[str], status_box: DeltaGenerator, test_state: dict=None):

    changes, op_state = get_state(ui_keys, test_state)
    
    op_plan = gen_op_plan(changes, test_state)
    
    gen_election_model  (op_plan, op_state, status_box)
    gen_samples         (op_plan, op_state, status_box)
    gen_stats           (op_plan, op_state, status_box)
    gen_plots           (op_plan, op_state, status_box)

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
    
    return fig
    

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
        # 'noise1_pct': st.session_state['noise1_pct'],
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
    plot_keys   = ['plot_H0_trials', 'plot_H1_trials', 'plot_H0_contours', 'plot_H1_contours', 'n_samples_disp']

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
        status_box,
        ) -> None: # updates op_state    
    """
    Generate both H0 (null) and H1 (flipped) election models.

    updates
        op_state['votes_dodi']: {"H0": votes_H0, "H1": votes_H1}
        op_state['meta_doda']:  {"H0": meta_H0,  "H1": meta_H1}
    """
    if not op_plan['gen_model_flag']:
        return        
    msg = "Creating Election model..."
    status_box.text(msg)
    print(msg)    

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
    A_votes         = op_state.get('A_votes',       51_500)
    B_votes         = op_state.get('B_votes',       48_500)
    margin_pct      = op_state.get('margin_pct',    3)
    noise1_pct      = op_state.get('noise1_pct',    .2)
    hack2_pct       = op_state.get('hack2_pct',     0)

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
        status_box,
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
    
    msg = f"Creating {n_trials} audit trials, each with {n_samples} samples per case..."
    status_box.text(msg)
    print(msg)    

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
        status_box,
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
    msg = f"Creating stats..."
    status_box.text(msg)
    print(msg)    
    
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
        status_box,
        ) -> None:   

    msg = f"Creating plots..."
    status_box.text(msg)
    print(msg)    

    # from scipy.stats import norm

    RiskText    = ["20%",  "10%",  "5%",   "2.5%", "1%",   "0.5%", "0.1%"]                #, "5sigma"]
    #RiskFrac    = [.20,    .1,     .05,    0.025,  0.01,   0.005,  0.001]                 #,  0.0000003]
    #sd_mult     = [round(norm.ppf(1 - r), 4) for r in RiskFrac]                          # dynamically generated.
    SD_mult     = [0.8416, 1.2816, 1.6449, 1.9600, 2.3263, 2.5758, 3.0902]                #,  5)
    nRisks      = len(RiskText)
    riskidx0_1percent   = RiskText.index("0.1%")
    riskidx5percent     = RiskText.index("5%")


    n_total         = op_state.get('n_total',       100_000)
    margin_pct      = op_state.get('margin_pct',    3)
    noise1_pct      = op_state.get('noise1_pct',    .2)
    hack2_pct       = op_state.get('hack2_pct',     0)
    n_samples       = op_state.get('n_samples',     500)
    n_trials        = op_state.get('n_trials',      1000)
    replace         = op_state.get('replace',       False)
    n_samples_disp  = op_state.get('n_samples_disp', n_samples)
    
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
                x=x_axis,                               # [0:n_samples_disp],
                y=CS[r],                                # [0:n_samples_disp],
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
            x=x_axis,                                   # [0:n_samples_disp],
            y=mean,                                     # [0:n_samples_disp],
            mode='lines',
            name=f"{hyp} Contours",
            line=dict(color=color, width=2),
        ))

        for riskidx in range(nRisks):
            
            if hyp == 'H0' and riskidx not in [riskidx0_1percent, riskidx5percent]:
                continue

            SD_mult_value = SD_mult[riskidx]
            
            width = len(mean)                           # n_samples_disp
            if hyp == 'H0':
                text_x = 7 * width // 8
            else:    
                text_x = 3 * width // 4
            
            fig.add_trace(go.Scatter(
                x   = x_axis,                           # [0:n_samples_disp],
                y   = (mean - SD_mult_value * sd),      # [0:n_samples_disp],
                mode = 'lines',
                name = f"{hyp} Risk",
                line = dict(color=color, width=1),
                showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x   = x_axis,                           # [0:n_samples_disp],
                y   = (mean + SD_mult_value * sd),      # [0:n_samples_disp],
                mode = 'lines',
                name = f"{hyp} Risk",
                line = dict(color=color, width=1),
                showlegend=False,
                ))
            y1 = mean[text_x] - SD_mult_value * sd[text_x]     # upper
            y2 = mean[text_x] + SD_mult_value * sd[text_x]     # lower
            fig.add_annotation(
                x=text_x, y=y1,
                text=f"{RiskText[riskidx]}",
                showarrow=False,
                font=dict(color=color, size=10),
                yanchor="bottom",
            )
            fig.add_annotation(
                x=text_x, y=y2,
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



def markdown_with_tables(md_text: str) -> None:
    """
    Render Markdown text including GitHub-style tables using Streamlit.

    Args:
        md_text (str): Markdown-formatted string (including table syntax)
    """
    html = markdown.markdown(md_text, extensions=["tables"])
    st.markdown(html, unsafe_allow_html=True)
    
def main():
        
    # ------------------------
    # Page setup and UI

    # ------------------------

    st.set_page_config(page_title="RLA Simulator", layout="wide")

    st.title("RLA Simulator")
    with st.expander("ℹ️ About This Simulator (click to show/hide)"):
        markdown_with_tables("""   
This tool simulates risk-limiting audits (RLAs) using Monte Carlo methods.
Adjust the parameters below and click Run Simulation to visualize how audits 
perform under different election scenarios. A Monte Carlo method allows determination of
statistical values without the use of sometimes complex equations, particular when 
samples are not replaced.

Currently, only a "ballot comparison" audit can be simulated. This type of audit requires
that each ballot can be individually referenced in storage and also by the CVR (cast vote record)
so they can be compared.

## How It Works
The simulator models two versions of an election:

1. H0 (Null hypothesis): The reported results are correct — the stated winner actually won.
2. H1 (Alternative hypothesis): The true outcome is flipped — the reported loser actually won by one vote.

Each ballot is classified into one of nine categories, depending on how it affects the reported margin,
where A is the reported winning candidate and B is the reported loser.

|  Case  |  Stated Results  |  True Results  |  Comment                     |
|:------:|:----------------:|:--------------:|:-----------------------------|
|   1.   |   A: 1; B: 0     |   A: 1; B: 0   | No Change (vote for winner)  |
|   2.   |   A: 0; B: 1     |   A: 0; B: 1   | No Change (vote for loser)   |
|   3.   |   A: 0; B: 0     |   A: 0; B: 0   | No Change (vote for neither) |
|   4.   |   A: 0; B: 0     |   A: 0; B: 1   | +1 Overstatement  (B +1)     |
|   5.   |   A: 1; B: 0     |   A: 0; B: 0   | +1 Overstatement  (A -1)     |
|   6.   |   A: 0; B: 1     |   A: 0; B: 0   | -1 Understatement (B -1)     |
|   7.   |   A: 0; B: 0     |   A: 1; B: 0   | -1 Understatement (A +1)     |
|   8.   |   A: 1; B: 0     |   A: 0; B: 1   | +2 Overstatement  (A -1, B +1)     |
|   9.   |   A: 0; B: 1     |   A: 1; B: 0   | -2 Understatement (A +1, B -1)     |

Note that to remove a vote from a candidate, the vote must be given to the No-vote group or another candidate.

## Noise and Trial Simulation
You can optionally add noise: random misinterpretations or marking errors that don't 
systematically favor either side but may affect totals.

The simulator runs a number of audit trials (typically 1000), each with a given 
number of sampled ballots. It visualizes the resulting distributions under H0 and H1.

The goal is to determine whether a given sample size is sufficient to:

- Reject H1 if the election was honest, and
- Reject H0 if the election was manipulated.

## Sample Thresholds
There are a number of sampling thresholds associated with different risk limits. These are represented as vertical dashed lines.
These thresholds are set according to the crossing of distribution profiles. For example, the 5% threshold is set to the number of
samples such that the total number of overstatements has a 95% chance of being in the H0 distribution and a 5% chance of being in 
the H1 distribution.

## Application Status
This application is a port of a app originally written in "R" in 2019. Not all functions are fully operational but do not detract 
from the usefulness of this app.

- Samples displayed is not fully functional. This would normally allow the user to focus in on a sub range for clarity.
- Settigs 'Actual RLA samples' and 'Net Overstatements in the RLA' are currently not functional. These would normally plot a
marker at the actual number of samples.
- The number of samples requires for a given risk limit does differ between this app and other calculations which are overly 
aggressive. This application provides sample counts that are more conservative and are reflective of the Monte Carlo analysis.

""")

    col0, col1 = st.columns(2)

    ELECTION_PRESETS = {
        "Choose a preset":   {"label": "Default", 
                                "n_total": 110000,  "A_votes": 51500,  "B_votes": 48500,  'RLA_samples':0,   'RLA_net_OS':0},
        "NV 2024 President": {"label": "NV 2024 Presidential Contest", 
                                "n_total": 1487887, "A_votes": 751205, "B_votes": 705197, 'RLA_samples':220, 'RLA_net_OS':0},
        }
                                

    with col0:
        election_choice = st.selectbox("Select election preset:", options=list(ELECTION_PRESETS.keys()))
        preset = ELECTION_PRESETS[election_choice]

    with col1:
        # Input controls, saved to st.session_state
        st.text_input  (label="Election Name, Date and Contest Name", value=preset['label'], help=None, key='election_name')

    col0, col1, col2, col3 = st.columns(4)

    with col0:
        n_total = st.number_input("Total Ballots Cast", value=preset['n_total'], key='n_total',         
                        min_value=100, max_value=100_000_000, step=1000, 
                        help="Enter the total ballots cast in the district which includes the contest of interest.")
    with col1:                                
        A_votes = st.number_input("Votes for the stated winner (A)", value=preset['A_votes'], key='A_votes',
                    min_value=100, max_value=min(100_000_000, n_total), step=1000, 
                    help="Enter the official reported number of ballots for the winner.")
    with col2:                
        B_votes = st.number_input("Votes for the stated loser (B)", value=preset['B_votes'], key='B_votes', 
                    min_value=100, max_value=min(100_000_000, n_total - A_votes), step=1000, 
                    help="Enter the official reported number of ballots for the loser.")
    # Compute margin percentage
    A_and_B_votes = A_votes + B_votes
    if A_and_B_votes > 0:
        margin_pct = 100 * (A_votes - B_votes) / A_and_B_votes
    else:
        margin_pct = 0.0

    with col3:
        st.number_input("Margin %",         
                    min_value=0.001, max_value=99.0, value=margin_pct,     
                    disabled=True,  key='margin_pct',
                    help="The calculated margin of victory among the two leading candidates." 
                    )

    with col0:
        RLA_samples = st.number_input("Actual RLA_samples", value=preset['RLA_samples'], key='RLA_samples', 
                    min_value=0, max_value=100_000, step=1, 
                    help="Enter the actual number of ballot samples in the RLA.")

    with col1:                                
        RLA_net_OS  = st.number_input("Net overstatements detected in RLA.", value=preset['RLA_net_OS'], key='RLA_net_OS', 
                    min_value=-100, max_value=100, step=1, 
                    help="Enter the official reported number of ballots for the loser.")


    col0, col1, col2, col3 = st.columns(4)
    with col0:
        st.slider("Noise %",          min_value=0.0,  max_value=10.,  value=0.2,  step=0.1, key='noise1_pct',
                    help="Typically, about 0.2% noise is expected due to voter errors. These are expressed as an equal number of "
                    "overstatements and understatements.")
                    
    with col1:
        st.slider("Flip Hack %",      min_value=0.0,  max_value=10.,  value=0.0,  step=0.1, key='hack_pct',
                    help="% of alteration expressed as 2-vote flips. Normal (not malicious) errors result in 1-vote over or "
                    "under statements. Even one flipped vote on a ballot "
                    "should prompt a full hand count because it is likely due to a malicious act. Thus, leave this at 0.")

    with col2:
        n_samples = st.slider("Samples per trial", min_value=100, max_value=5000, value=1000, step=100, key='n_samples',
                    help="Depending on how close the election is, the number of samples should be at least twice "
                    "the number of samples expected in the audit. The number of sampled does not affect how many "
                    "ballot samples are needed in the actual audit, and are only for the visualization.")

    with col3:
        st.slider("Number of trials",  min_value=100, max_value=1000, value=1000, step=100, key='n_trials',
                    help="To form a good visualization, about 1000 trails per hypothesis is normal, but the "
                    "simulation will run faster if this is reduced.")

    # Plot controls
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        st.checkbox("Plot H0 trials",   value=True, key='plot_H0_trials',
                    help="Include the green H0 'null hypothesis' audit trails cloud in the visualization")
    with col1:
        st.checkbox("Plot H1 trials",   value=True, key='plot_H1_trials',
                    help="Include the red H1 'hacked hypothesis' audit trails cloud in the visualization")

    with col2:
        st.slider("Samples displayed", min_value=10, max_value=n_samples, value=min(1000, n_samples), step=10, key='n_samples_disp',
                    help="Depending on how close the election is, the number of samples should be at least twice "
                    "the number of samples expected in the audit. The number of sampled does not affect how many "
                    "ballot samples are needed in the actual audit, and are only for the visualization.")

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        st.checkbox("Plot H0 contours", value=True, key='plot_H0_contours',
                    help="Include the mean and risk contours in for the H0 null hypothesis")
    with col1:
        st.checkbox("Plot H1 contours", value=True, key='plot_H1_contours',
                    help="Include the mean and risk contours in for the H1 hacked hypothesis")

    ui_keys = ['election_name', 'n_total', 'A_votes', 'B_votes', 'margin_pct',
                    'RLA_samples', 'RLA_net_OS',
                    'noise1_pct', 'hack_pct', 'n_samples', 'n_trials', 'n_samples_disp',
                    'plot_H0_trials', 'plot_H1_trials', 'plot_H0_contours', 'plot_H1_contours']

    # Run button
    if st.button("Run Simulation"):
        progress = st.progress(0)
        status_box = st.empty()
        status_box.text("🔄 Starting simulation...")
        fig = main_simulator(ui_keys=ui_keys, status_box=status_box)  # This function should internally call st.plotly_chart()

        st.plotly_chart(fig, use_container_width=True)
        

# if __name__ == "__main__":
    # import sys

    # # Optional: accept 'test' as CLI argument
    # if len(sys.argv) > 1 and sys.argv[1] == "test":
        # pass
        
        # # import pprint
        # # print("Running basic test case for audit simulator...")

        # # test_state = dict(
            # # n_total=100_000,
            # # margin_pct=3.0,
            # # noise1_pct=0.2,
            # # hack2_pct=0.0,
            # # n_samples=500,
            # # n_trials=1000,
            # # )
            
        # # votes, meta = gen_election_model(
            # # op_state = test_state
        # # )

        # # for hyp in ('H0', 'H1'):
            # # print(f"\nVotes ({hyp}):")
            # # pprint.pprint(votes[hyp], sort_dicts=False)
            # # print(f"\nMetadata ({hyp}):")
            # # pprint.pprint(meta[hyp], sort_dicts=False)
        
        # # print("Generating samples")
        # # bin_samples = generate_samples(votes, meta, n_samples=1000, n_trials=1000, replace=False)
        
        # # pprint.pprint(bin_samples, sort_dicts=False)
    

    # elif len(sys.argv) > 1 and sys.argv[1] == "run_full_test":
        # print("Running full audit simulation test...")

        # test_state = dict(
            # n_total      = 110_000,
            # A_votes      = 51_500,
            # B_votes      = 45_500,
            # margin_pct   = 3.0,
            # noise1_pct   = 0.2,
            # hack2_pct    = 0.0,
            # n_samples    = 500,
            # n_trials     = 1000,
            # )
            
        # fig = main_simulator(test_state)

        # fig.write_html("audit_sim_output.html")
        # print("Plot saved to audit_sim_output.html")    
    # else:
        # print("No action specified. Run with 'test' or 'run_full_test' argument to execute a test case.")
        

        