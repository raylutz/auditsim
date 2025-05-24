import streamlit as st
from audit_simulator import main

# ------------------------
# Page setup and UI

# ------------------------

st.set_page_config(page_title="RLA Simulator", layout="wide")

st.title("RLA Simulator")
with st.expander("ℹ️ About This Simulator (click to show/hide)"):
    st.markdown("""
    
This tool simulates risk-limiting audits (RLAs) using Monte Carlo methods.
Adjust the parameters below and click Run Simulation to visualize how audits 
perform under different election scenarios.

## How It Works
The simulator models two versions of an election:

1. H0 (Null hypothesis): The reported results are correct — the stated winner actually won.
2. H1 (Alternative hypothesis): The true outcome is flipped — the reported loser actually won by one vote.

Each ballot is classified into one of nine categories, depending on how it affects the reported margin,
where A is the reported winning candidate and B is the reported loser.

|  Case  |  Stated Results  |  True Results  |  Comment                     |
+:------:+:----------------:+:--------------:+:-----------------------------+
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
    fig = main(ui_keys=ui_keys, status_box=status_box)  # This function should internally call st.plotly_chart()

    st.plotly_chart(fig, use_container_width=True)

