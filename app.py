"""
Streamlit Salary Predictor App
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Tech Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .big-metric { font-size: 2.8rem; font-weight: 700; color: #4ade80; }
  .sub-metric { font-size: 1.1rem; color: #94a3b8; margin-top: -8px; }
  .card {
    background: #1e293b; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 1rem;
  }
  .stButton>button {
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    color: #0f172a; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.6rem 2rem; font-size: 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Load model + encoders + metadata ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("models/salary_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return model, encoders, meta


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/salaries.csv")


try:
    model, encoders, meta = load_artifacts()
    df_raw = load_raw_data()
    READY = True
except FileNotFoundError:
    READY = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Salary Predictor")
    st.caption("Levels.fyi-style · Tech Industry")
    st.divider()

    if READY:
        st.success(f"Model: **{meta['model_name']}**")
        m = meta["test_metrics"]
        st.metric("Test R²", f"{m['r2']:.3f}")
        st.metric("Median Error", f"~{m['mape']:.1f}%")
        st.metric("MAE", f"${m['mae']:,.0f}")
    else:
        st.error("Models not found. Run `python generate_data.py` then `python train.py` first.")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
    1. Fill in your profile →  
    2. Model predicts **Total Compensation**  
    3. See salary ranges & benchmarks
    """)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("💰 Tech Salary Predictor")
st.markdown("*Estimate your market value based on Levels.fyi-style data*")

if not READY:
    st.error("Please run the setup scripts first (see README).")
    st.stop()

# ── Shared prediction helper ─────────────────────────────────────────────────
def seniority(r):
    r = r.lower()
    if "principal" in r: return 5
    if "staff" in r:     return 4
    if "senior" in r or "manager" in r: return 3
    if "ml" in r or "data scientist" in r: return 2
    return 1

def enc(col, val):
    le = encoders[col]
    return le.transform([str(val)])[0] if str(val) in le.classes_ else -1

def predict_tc(role, location, company_tier, education, yoe, remote):
    yoe_bucket = pd.cut(
        [yoe], bins=[-1,2,5,10,15,100],
        labels=["0-2","3-5","6-10","11-15","15+"]
    )[0]
    row = {
        "role_enc":            enc("role", role),
        "location_enc":        enc("location", location),
        "education_enc":       enc("education", education),
        "company_tier_enc":    enc("company_tier", company_tier),
        "yoe_bucket_enc":      enc("yoe_bucket", yoe_bucket),
        "years_of_experience": yoe,
        "seniority_level":     seniority(role),
        "is_faang":            1 if company_tier == "FAANG" else 0,
        "is_big_tech":         1 if company_tier in ["FAANG","Tier2"] else 0,
        "remote_work":         int(remote),
    }
    X = pd.DataFrame([row])[meta["feature_cols"]]
    return np.expm1(model.predict(X)[0])

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict",
    "💬 Negotiation Tips",
    "⚖️ What-If Comparison",
    "📊 Market Explorer",
    "🔍 Model Insights",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 · PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Your Profile")
        role         = st.selectbox("Role", sorted(df_raw["role"].unique()))
        location     = st.selectbox("Location", sorted(df_raw["location"].unique()))
        company_tier = st.selectbox("Company Tier", ["FAANG","Tier2","Tier3","Startup"],
                                    help="FAANG=Google/Meta/Apple/Amazon/Netflix · Tier2=Microsoft/Stripe/etc.")
        education    = st.selectbox("Education", ["Bachelor's","Master's","PhD","Bootcamp","Self-taught"])
        yoe          = st.slider("Years of Experience", 0, 25, 5)
        remote       = st.checkbox("Remote work", value=False)
        predict_btn  = st.button("🚀 Predict My Salary", use_container_width=True)

    with col_result:
        if predict_btn:
            pred   = predict_tc(role, location, company_tier, education, yoe, remote)
            lo, hi = pred * 0.85, pred * 1.15

            st.subheader("💡 Prediction")
            st.markdown(f'<div class="big-metric">${pred:,.0f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-metric">Estimated Total Compensation / year</div>',
                        unsafe_allow_html=True)
            st.caption(f"Range: **${lo:,.0f}** – **${hi:,.0f}**")
            st.divider()

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pred,
                number={"prefix": "$", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, 600000]},
                    "bar":  {"color": "#4ade80"},
                    "steps": [
                        {"range": [0,      120000], "color": "#1e293b"},
                        {"range": [120000, 250000], "color": "#334155"},
                        {"range": [250000, 400000], "color": "#475569"},
                        {"range": [400000, 600000], "color": "#64748b"},
                    ],
                    "threshold": {"line": {"color": "#22d3ee","width":3},
                                  "thickness":0.75,"value":pred},
                },
            ))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=30,b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            peers = df_raw[
                (df_raw["role"] == role) &
                (df_raw["company_tier"] == company_tier)
            ]["total_compensation"]
            if len(peers) > 5:
                pct = (peers < pred).mean() * 100
                st.metric("Your Percentile vs Peers", f"{pct:.0f}th")
                st.progress(int(pct) / 100)
        else:
            st.info("Fill in your profile and click **Predict** →")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 · NEGOTIATION TIPS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("💬 Salary Negotiation Tips")
    st.markdown("Enter your current situation to get personalised negotiation advice.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_role     = st.selectbox("Your Role", sorted(df_raw["role"].unique()), key="n_role")
        n_tier     = st.selectbox("Company Tier", ["FAANG","Tier2","Tier3","Startup"], key="n_tier")
    with c2:
        n_location = st.selectbox("Location", sorted(df_raw["location"].unique()), key="n_loc")
        n_edu      = st.selectbox("Education", ["Bachelor's","Master's","PhD","Bootcamp","Self-taught"], key="n_edu")
    with c3:
        n_yoe      = st.slider("Years of Experience", 0, 25, 5, key="n_yoe")
        n_current  = st.number_input("Your Current TC ($, 0 if new offer)", min_value=0,
                                      max_value=2000000, value=0, step=5000, key="n_curr")
        n_remote   = st.checkbox("Remote", key="n_rem")

    if st.button("💡 Generate Negotiation Tips", use_container_width=True, key="neg_btn"):
        market_pred = predict_tc(n_role, n_location, n_tier, n_edu, n_yoe, n_remote)

        # Peer percentile
        peers = df_raw[
            (df_raw["role"] == n_role) &
            (df_raw["company_tier"] == n_tier)
        ]["total_compensation"]
        pct = (peers < market_pred).mean() * 100 if len(peers) > 5 else 50

        gap = market_pred - n_current if n_current > 0 else 0
        gap_pct = (gap / n_current * 100) if n_current > 0 else 0

        # ── Metrics row ──────────────────────────────────────────────────────
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Market Rate (Predicted)", f"${market_pred:,.0f}")
        if n_current > 0:
            mc2.metric("Your Current TC", f"${n_current:,.0f}",
                       delta=f"${gap:+,.0f} vs market")
            mc3.metric("Market Gap", f"{gap_pct:+.1f}%",
                       delta="above market" if gap_pct < 0 else "below market")
        else:
            mc2.metric("Peer Percentile", f"{pct:.0f}th")
            mc3.metric("Suggested Ask", f"${market_pred * 1.10:,.0f}",
                       delta="+10% above market")

        st.divider()

        # ── Contextual tips ──────────────────────────────────────────────────
        def tip_card(emoji, title, body):
            st.markdown(f"""
            <div class="card">
              <strong>{emoji} {title}</strong><br>
              <span style="color:#cbd5e1">{body}</span>
            </div>
            """, unsafe_allow_html=True)

        # Anchor number
        ask = market_pred * 1.12
        tip_card("🎯", "Set Your Anchor High",
                 f"Open with <b>${ask:,.0f}</b> — ~12% above market. "
                 "Research shows first offers anchor the final number. "
                 "They'll negotiate down; you want room to land at market rate.")

        # Leverage tips based on percentile
        if pct >= 75:
            tip_card("🔥", "You're in the Top 25% — Leverage It",
                     f"Your predicted TC is at the <b>{pct:.0f}th percentile</b> for "
                     f"{n_role}s at {n_tier}. Lead with this: <i>'Based on my research, "
                     "I'm targeting the upper quartile for this role.'</i>")
        elif pct >= 50:
            tip_card("📈", "You're at Market — Push for More",
                     f"You're at the <b>{pct:.0f}th percentile</b>. "
                     "This is the baseline, not the ceiling. Ask for top-quartile: "
                     f"<b>${peers.quantile(0.75):,.0f}</b> is achievable for strong candidates.")
        else:
            tip_card("⚡", "Below Market — Strong Case to Negotiate",
                     f"You're at the <b>{pct:.0f}th percentile</b>, which means "
                     "significant room to negotiate up. Use market data as your evidence: "
                     f"<i>'Glassdoor and Levels.fyi show market rate is ${market_pred:,.0f}.'</i>")

        # Stock/bonus tip
        if n_tier in ["FAANG", "Tier2"]:
            tip_card("📦", "Don't Forget Equity",
                     "Big tech often has more flexibility on RSU grants than base salary. "
                     "If base is capped, ask: <i>'Can we increase the equity package or "
                     "add a sign-on bonus to bridge the gap?'</i>")
        else:
            tip_card("🚀", "Equity Upside at Startups",
                     "Startups pay less cash but equity can be worth more. Ask for: "
                     "vesting schedule, cliff period, strike price, last 409A valuation, "
                     "and total shares outstanding to calculate real ownership %.")

        # YOE-based tip
        if n_yoe < 3:
            tip_card("🌱", "Early Career: Focus on Growth + Total Package",
                     "Negotiate for learning budget ($2–5k/yr), conference attendance, "
                     "and fast review cycles (6mo instead of annual). These compound "
                     "more than a $5k base bump early in your career.")
        elif n_yoe >= 10:
            tip_card("👑", "Senior Talent: You Have the Leverage",
                     "Senior engineers are hard to replace. Mention competing offers "
                     "(real or in-progress). Even hinting at other conversations raises "
                     "urgency. Ask for a <i>retention package</i> if at your current job.")

        # Location tip
        if n_location == "Remote":
            tip_card("🌍", "Remote: Negotiate Location-Neutral Pay",
                     "Push back on location-based pay cuts. Say: <i>'My output and "
                     "skills are the same regardless of where I work. I'd like to be "
                     "compensated on the role, not my zip code.'</i>")

        # Closing script
        st.divider()
        st.markdown("**📝 Sample Negotiation Script**")
        st.code(f"""
"Thank you for the offer — I'm genuinely excited about the role.
Based on my research on Levels.fyi and Glassdoor, and considering
my {n_yoe} years of experience in {n_role.lower()} positions,
the market rate is around ${market_pred:,.0f}.

I'd like to target ${ask:,.0f} in total compensation.
Is there flexibility to get closer to that number, 
either through base, equity, or a sign-on bonus?"
""", language="text")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 · WHAT-IF COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚖️ What-If Comparison")
    st.markdown("Compare how your salary changes across **different scenarios** side by side.")

    # ── Shared profile ───────────────────────────────────────────────────────
    with st.expander("🔧 Base Profile (shared across scenarios)", expanded=True):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            w_role = st.selectbox("Role", sorted(df_raw["role"].unique()), key="w_role")
            w_edu  = st.selectbox("Education", ["Bachelor's","Master's","PhD","Bootcamp","Self-taught"], key="w_edu")
        with wc2:
            w_yoe    = st.slider("Years of Experience", 0, 25, 5, key="w_yoe")
            w_remote = st.checkbox("Remote", key="w_rem")
        with wc3:
            w_loc = st.selectbox("Base Location", sorted(df_raw["location"].unique()), key="w_loc")

    st.markdown("---")
    st.markdown("### 🆚 Define Up to 4 Scenarios")

    TIERS     = ["FAANG","Tier2","Tier3","Startup"]
    LOCATIONS = sorted(df_raw["location"].unique())

    scenarios = []
    scol1, scol2, scol3, scol4 = st.columns(4)
    scenario_cols = [scol1, scol2, scol3, scol4]
    defaults = [
        ("FAANG",   w_loc),
        ("Tier2",   w_loc),
        ("Tier3",   w_loc),
        ("Startup", w_loc),
    ]
    colors = ["#4ade80","#22d3ee","#f59e0b","#f87171"]

    for i, (col, (def_tier, def_loc)) in enumerate(zip(scenario_cols, defaults)):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            s_tier = st.selectbox("Tier", TIERS, index=TIERS.index(def_tier), key=f"s_tier_{i}")
            s_loc  = st.selectbox("Location", LOCATIONS,
                                   index=LOCATIONS.index(def_loc) if def_loc in LOCATIONS else 0,
                                   key=f"s_loc_{i}")
            s_name = st.text_input("Label", value=f"{s_tier} · {s_loc.split(',')[0]}", key=f"s_name_{i}")
            scenarios.append({"name": s_name, "tier": s_tier, "loc": s_loc, "color": colors[i]})

    if st.button("🔄 Run Comparison", use_container_width=True, key="whatif_btn"):
        results = []
        for s in scenarios:
            tc = predict_tc(w_role, s["loc"], s["tier"], w_edu, w_yoe, w_remote)
            results.append({"Scenario": s["name"], "TC": tc, "color": s["color"]})

        res_df = pd.DataFrame(results).sort_values("TC", ascending=False)
        best   = res_df.iloc[0]
        worst  = res_df.iloc[-1]
        diff   = best["TC"] - worst["TC"]

        # ── Summary metrics ──────────────────────────────────────────────────
        mc = st.columns(len(results))
        for i, row in res_df.iterrows():
            delta = row["TC"] - res_df["TC"].mean()
            mc[res_df.index.get_loc(i)].metric(
                row["Scenario"],
                f"${row['TC']:,.0f}",
                delta=f"{delta:+,.0f} vs avg"
            )

        st.divider()

        # ── Bar chart ────────────────────────────────────────────────────────
        fig_bar = go.Figure()
        for _, row in res_df.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["Scenario"]], y=[row["TC"]],
                marker_color=row["color"],
                text=f"${row['TC']:,.0f}",
                textposition="outside",
                name=row["Scenario"],
            ))
        fig_bar.update_layout(
            title=f"Total Compensation Comparison · {w_role} · {w_yoe} YOE",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=420, showlegend=False,
            yaxis=dict(title="Total Comp ($)", tickformat="$,.0f"),
            xaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── YOE progression chart ─────────────────────────────────────────────
        st.markdown("#### 📈 TC Growth by YOE across Scenarios")
        yoe_range = list(range(0, 26, 2))
        fig_line = go.Figure()
        for s, color in zip(scenarios, colors):
            tcs = [predict_tc(w_role, s["loc"], s["tier"], w_edu, y, w_remote)
                   for y in yoe_range]
            fig_line.add_trace(go.Scatter(
                x=yoe_range, y=tcs, mode="lines+markers",
                name=s["name"], line=dict(color=color, width=2),
                marker=dict(size=5),
            ))
        fig_line.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            xaxis_title="Years of Experience",
            yaxis=dict(title="Total Comp ($)", tickformat="$,.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Insight callout ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card">
          <strong>💡 Key Insight</strong><br>
          <span style="color:#cbd5e1">
          Choosing <b>{best['Scenario']}</b> over <b>{worst['Scenario']}</b>
          would earn you an extra <b>${diff:,.0f}/year</b>
          (${diff*5:,.0f} over 5 years before taxes & investment returns).
          </span>
        </div>
        """, unsafe_allow_html=True)


with tab4:
    st.subheader("📊 Market Salary Explorer")

    c1, c2 = st.columns(2)
    with c1:
        selected_roles = st.multiselect(
            "Filter by Role", options=sorted(df_raw["role"].unique()),
            default=["Software Engineer", "Senior Software Engineer", "Staff Engineer"]
        )
    with c2:
        selected_tiers = st.multiselect(
            "Filter by Tier", options=["FAANG","Tier2","Tier3","Startup"],
            default=["FAANG","Tier2","Tier3"]
        )

    filtered = df_raw[
        df_raw["role"].isin(selected_roles) &
        df_raw["company_tier"].isin(selected_tiers)
    ]

    if len(filtered) == 0:
        st.warning("No data for selected filters.")
    else:
        # Box plot by role
        fig_box = px.box(
            filtered, x="role", y="total_compensation", color="company_tier",
            title="Total Comp by Role & Tier",
            color_discrete_map={"FAANG":"#4ade80","Tier2":"#22d3ee",
                                 "Tier3":"#f59e0b","Startup":"#f87171"},
            template="plotly_dark",
            labels={"total_compensation": "Total Comp ($)", "role": ""},
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420,
                               legend_title_text="Tier")
        st.plotly_chart(fig_box, use_container_width=True)

        # YOE vs TC scatter
        fig_sc = px.scatter(
            filtered.sample(min(1000, len(filtered))),
            x="years_of_experience", y="total_compensation",
            color="company_tier", size_max=8, opacity=0.6,
            title="Experience vs Total Comp",
            color_discrete_map={"FAANG":"#4ade80","Tier2":"#22d3ee",
                                  "Tier3":"#f59e0b","Startup":"#f87171"},
            template="plotly_dark",
            labels={"total_compensation": "Total Comp ($)",
                    "years_of_experience": "Years of Experience"},
        )
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Location heatmap
        loc_agg = (
            df_raw.groupby("location")["total_compensation"]
            .median().reset_index()
            .sort_values("total_compensation", ascending=True)
        )
        fig_loc = px.bar(
            loc_agg, x="total_compensation", y="location", orientation="h",
            title="Median TC by Location", template="plotly_dark",
            color="total_compensation", color_continuous_scale="Viridis",
            labels={"total_compensation": "Median Total Comp ($)", "location": ""},
        )
        fig_loc.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False)
        st.plotly_chart(fig_loc, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 · MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔍 Model Insights")

    c1, c2, c3 = st.columns(3)
    m = meta["test_metrics"]
    c1.metric("R² Score", f"{m['r2']:.3f}", help="1.0 = perfect")
    c2.metric("MAE", f"${m['mae']:,.0f}", help="Mean absolute error in $")
    c3.metric("MAPE", f"{m['mape']:.1f}%", help="Mean absolute % error")

    st.divider()

    # CV results table
    st.markdown("**Cross-Validation Results**")
    cv_df = pd.DataFrame(meta["cv_results"]).T.reset_index()
    cv_df.columns = ["Model", "CV R² Mean", "CV R² Std"]
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.divider()

    # Feature importance
    st.markdown("**Feature Importances**")
    fi = meta["feature_importances"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True)

    # Clean labels
    fi_df["Feature"] = fi_df["Feature"].str.replace("_enc", "").str.replace("_", " ").str.title()

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        template="plotly_dark", color="Importance",
        color_continuous_scale="Teal",
        title="What drives your salary prediction?",
    )
    fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400,
                          showlegend=False, yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Based on synthetic Levels.fyi-style data for demonstration. Not financial advice.")