import random
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import model   # your abm_model module
from model import Organization

st.set_page_config(page_title="ABM Simulator", layout="wide")
st.title("Agent‐Based Model Simulator")

# Initialize session state
if "efficiency_data" not in st.session_state:
    st.session_state.efficiency_data = None
    st.session_state.agents_data     = None
    st.session_state.df_combined     = None
    st.session_state.model_data      = None
    st.session_state.strategies      = None

# ——————————————————————————
# Block 0: Editable ROLE_PROFILES
# ——————————————————————————
with st.container():
    st.header("0. ROLE_PROFILES")
    SKILLS = ["tech", "management", "compliance", "soft_skills"]
    new_roles = {}
    for lvl in sorted(model.ROLE_PROFILES):
        cols = st.columns(4)
        new_roles[lvl] = {}
        st.markdown(f"**Level {lvl}**")
        for i, skill in enumerate(SKILLS):
            default = model.ROLE_PROFILES[lvl].get(skill, 0.0)
            new_roles[lvl][skill] = cols[i].number_input(
                f"{skill}", min_value=0.0, max_value=1.0,
                value=float(default), step=0.01,
                key=f"rp_{lvl}_{skill}"
            )
    model.ROLE_PROFILES = new_roles

# ——————————————————————————
# Block 1: Controls & Combined Efficiency
# ——————————————————————————
with st.container():
    st.header("1. Simulation Controls & Efficiency")
    strategies = st.multiselect(
        "Promotion strategies",
        ["merit", "seniority", "hybrid", "random", "tournament"],
        default=["merit"],
    )
    num_agents = st.number_input(
        "Number of agents", min_value=100, max_value=500_000,
        value=100_000, step=100
    )
    steps = st.number_input(
        "Simulation steps", min_value=1, max_value=1_000,
        value=100, step=1
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run = st.button("Run simulation")

    if run:
        if not strategies:
            st.error("Select at least one strategy.")
        else:
            with st.spinner("Simulation running..."):
                efficiency_data = {}
                agents_data     = {}
                model_data      = {}

                for strat in strategies:
                    # re-seed all RNGs
                    random.seed(seed)
                    np.random.seed(seed)

                    m = Organization(
                        num_agents=num_agents,
                        promotion_strategy=strat,
                        seed=seed,
                    )
                    # record initial model-vars
                    m.datacollector.collect(m)

                    eff = [m.get_efficiency()]
                    for _ in range(steps):
                        m.step()
                        eff.append(m.get_efficiency())

                    efficiency_data[strat] = eff
                    agents_data[strat]     = m.datacollector.get_agent_vars_dataframe()
                    model_data[strat]      = m.datacollector.get_model_vars_dataframe()

                # save results
                st.session_state.efficiency_data = efficiency_data
                st.session_state.agents_data     = agents_data
                st.session_state.df_combined     = pd.DataFrame(efficiency_data)
                st.session_state.model_data      = model_data
                st.session_state.strategies      = strategies

            st.success("Simulation complete!")

    # combined efficiency plot
    if st.session_state.efficiency_data:
        fig, ax = plt.subplots()
        all_vals = [
            v for eff in st.session_state.efficiency_data.values() for v in eff
        ]
        lo, hi = min(all_vals), max(all_vals)
        delta = (hi - lo) * 0.1 if hi > lo else 0.05

        for strat, eff in st.session_state.efficiency_data.items():
            sns.lineplot(x=range(len(eff)), y=eff, label=strat.capitalize(), ax=ax)

        ax.set_ylim(lo - delta, hi + delta)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Efficiency")
        ax.legend(title="Strategy")
        st.pyplot(fig)
    else:
        st.info("Set parameters and click **Run simulation**.")

# ——————————————————————————
# Block 2: Filterable Model-level Data
# ——————————————————————————
with st.container():
    st.header("2. Model-level Data (filterable)")

    if st.session_state.model_data:
        # 1) Pick the strategy and “metric”
        strat2 = st.selectbox(
            "Choose strategy",
            st.session_state.strategies,
            key="block2_strat",
        )
        metric = st.selectbox(
            "View",
            ["Attrited by Level", "Promoted by Level", "Raw DataFrame"],
            key="block2_metric",
        )

        df_model = st.session_state.model_data[strat2]
        df_model.index.name = "Timestep"

        if metric == "Raw DataFrame":
            st.subheader(f"Full model DataFrame for '{strat2}'")
            st.dataframe(df_model)

        else:
            # unpack the dict-of-IDs column into a DataFrame of lists
            df_lists = df_model[metric].apply(pd.Series)

            # filter controls
            levels = sorted(df_lists.columns)
            levels_sel = st.multiselect(
                "Levels to display", levels, default=levels, key="block2_levels"
            )

            t_min, t_max = int(df_model.index.min()), int(df_model.index.max())
            time_range = st.slider(
                "Timestep range",
                t_min,
                t_max,
                (t_min, t_max),
                key="block2_timerange"
            )

            view_mode = st.radio(
                "View as",
                ["Counts", "Agent IDs"],
                key="block2_viewmode"
            )

            # slice the DataFrame
            sub = df_lists.loc[time_range[0]:time_range[1], levels_sel]

            if view_mode == "Counts":
                st.subheader(f"{metric} counts")
                sub_counts = sub.applymap(len)
                st.dataframe(sub_counts)
            else:
                st.subheader(f"{metric} agent IDs")
                st.dataframe(sub)

    else:
        st.info("Run a simulation first to see model-level data.")



# ——————————————————————————
# Block 3: Agent-level Performance
# ——————————————————————————
with st.container():
    st.header("3. Agent-level Performance")
    if st.session_state.agents_data:
        strat3 = st.selectbox(
            "Strategy for agent history",
            st.session_state.strategies,
            key="strat3"
        )
        aid = st.number_input("Agent ID to track", min_value=0, value=5, step=1)
        df_agents = st.session_state.agents_data[strat3]
        try:
            agent_df = df_agents.xs(aid, level="AgentID")
            st.dataframe(agent_df)

            fig2, ax2 = plt.subplots()
            sns.lineplot(x=agent_df.index, y=agent_df["Performance"], ax=ax2)
            ax2.set_title(f"Agent {aid} Performance — {strat3.capitalize()}")
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Performance")
            st.pyplot(fig2)
        except KeyError:
            st.error(f"No data for AgentID {aid} under '{strat3}'.")

    else:
        st.info("Run a simulation to inspect agent-level history.")
