from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "processed" / "HRDataset_v14_engineered.csv"

DEPARTMENT_ORDER = ["Executive Office", "Admin Offices", "IT/IS", "Sales", "Software Engineering", "Production"]
SALARY_ORDER = ["0-50k", "50k-100k", "100k-150k", "150k-200k", "200k+"]
GENERATION_ORDER = ["Baby Boomers", "Generation X", "Millennials", "Generation Z", "Silent"]
PERFORMANCE_ORDER = ["PIP", "Needs Improvement", "Fully Meets", "Exceeds"]
ATTRITION_STATUS_ORDER = ["Active", "Voluntarily Terminated", "Terminated for Cause"]


st.set_page_config(
    page_title="HR Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def assign_segment(row: pd.Series) -> str:
    if row["termd"] == 1 or row["performance_score"] in {"PIP", "Needs Improvement"}:
        return "At-Risk"
    if bool(row["high_engagement_flag"]) and row["performance_score"] in {"Exceeds", "Fully Meets"} and row["lateness_flag"] is False:
        return "High Performer"
    if row["engagement_survey"] >= 4.1 and row["days_late_last_30"] <= 1:
        return "Stable Core"
    return "Developing"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["employment_status"] = pd.Categorical(frame["employment_status"], categories=ATTRITION_STATUS_ORDER, ordered=False)
    frame["salary_band"] = pd.Categorical(frame["salary_band"], categories=SALARY_ORDER, ordered=True)
    frame["generation"] = pd.Categorical(frame["generation"], categories=GENERATION_ORDER, ordered=False)
    frame["department"] = pd.Categorical(frame["department"], categories=DEPARTMENT_ORDER, ordered=False)
    frame["performance_score"] = pd.Categorical(frame["performance_score"], categories=PERFORMANCE_ORDER, ordered=True)
    frame["engagement_group"] = pd.cut(
        frame["engagement_survey"],
        bins=[0, 3.5, 4.25, 5.1],
        labels=["Watch", "Healthy", "Strong"],
        include_lowest=True,
        right=False,
    )
    frame["segment"] = frame.apply(assign_segment, axis=1)
    return frame


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def make_bar_chart(frame: pd.DataFrame, x: str, y: str, title: str, color: str = "#1f4e79"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x=x, y=y, ax=ax, color=color)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.18)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def make_count_chart(frame: pd.DataFrame, column: str, title: str, order: list[str] | None = None, color: str = "#0f766e"):
    counts = frame[column].value_counts(dropna=False)
    if order is not None:
        counts = counts.reindex(order).dropna()
    counts = counts.reset_index()
    counts.columns = [column, "count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x=column, y="count", ax=ax, color=color)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Employees")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def make_heatmap(frame: pd.DataFrame, title: str):
    numeric_columns = [
        "salary",
        "engagement_survey",
        "emp_satisfaction",
        "special_projects_count",
        "days_late_last_30",
        "absences",
        "perf_score_id",
        "tenure_years",
        "months_since_review",
        "age",
        "age_at_hire",
        "absent_rate",
    ]
    corr = frame[numeric_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        annot=False,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(title, fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


def safe_percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean() * 100)


def summary_card(label: str, value: str, delta: str | None = None):
    st.metric(label, value, delta)


def apply_sidebar_filters(frame: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filters")
    st.sidebar.caption("Scope the dashboard to the parts of the workforce you want to review.")

    department_options = [value for value in DEPARTMENT_ORDER if value in frame["department"].dropna().unique()]
    salary_options = [value for value in SALARY_ORDER if value in frame["salary_band"].dropna().unique()]
    generation_options = [value for value in GENERATION_ORDER if value in frame["generation"].dropna().unique()]
    source_options = sorted(frame["recruitment_source"].dropna().unique().tolist())
    status_options = [value for value in ATTRITION_STATUS_ORDER if value in frame["employment_status"].dropna().unique()]

    selected_departments = st.sidebar.multiselect("Department", department_options, default=department_options)
    selected_bands = st.sidebar.multiselect("Salary band", salary_options, default=salary_options)
    selected_generations = st.sidebar.multiselect("Generation", generation_options, default=generation_options)
    selected_sources = st.sidebar.multiselect("Recruitment source", source_options, default=source_options)
    selected_status = st.sidebar.multiselect("Employment status", status_options, default=status_options)

    filtered = frame[
        frame["department"].isin(selected_departments)
        & frame["salary_band"].isin(selected_bands)
        & frame["generation"].isin(selected_generations)
        & frame["recruitment_source"].isin(selected_sources)
        & frame["employment_status"].isin(selected_status)
    ].copy()

    st.sidebar.divider()
    st.sidebar.markdown("### Notebook signals baked into this dashboard")
    st.sidebar.write("Production and Sales are the most useful engagement watchlist.")
    st.sidebar.write("Attrition is concentrated in Production and in a few recruiting sources.")
    st.sidebar.write("The segmentation notebook points to an at-risk group, a high performer group, and a developing group.")

    return filtered


def overview_tab(frame: pd.DataFrame):
    st.subheader("Overview")
    st.write("A compact executive readout of the workforce, attrition, and performance mix.")

    total_headcount = len(frame)
    attrition_rate = safe_percentage(frame["termd"])
    avg_engagement = frame["engagement_survey"].mean()
    avg_tenure = frame["tenure_years"].mean()
    avg_perf_index = frame["perf_score_id"].mean()
    at_risk_perf_share = safe_percentage(frame["performance_score"].isin(["PIP", "Needs Improvement"]))

    metric_cols = st.columns(6)
    with metric_cols[0]:
        summary_card("Headcount", f"{total_headcount}")
    with metric_cols[1]:
        summary_card("Attrition rate", format_pct(attrition_rate))
    with metric_cols[2]:
        summary_card("Avg engagement", format_float(avg_engagement))
    with metric_cols[3]:
        summary_card("Avg performance index", format_float(avg_perf_index))
    with metric_cols[4]:
        summary_card("Avg tenure (years)", format_float(avg_tenure))
    with metric_cols[5]:
        summary_card("At-risk performance", format_pct(at_risk_perf_share))

    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        dept_attrition = frame.groupby("department", observed=True)["termd"].mean().sort_values(ascending=False).mul(100).reset_index()
        fig = make_bar_chart(dept_attrition, x="termd", y="department", title="Attrition by Department", color="#8b1e3f")
        st.pyplot(fig, clear_figure=True)
    with right_col:
        fig = make_count_chart(frame, "performance_score", "Performance Mix", order=PERFORMANCE_ORDER, color="#2457a6")
        st.pyplot(fig, clear_figure=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        engagement_by_dept = frame.groupby("department", observed=True)["engagement_survey"].mean().sort_values(ascending=False).reset_index()
        fig = make_bar_chart(engagement_by_dept, x="engagement_survey", y="department", title="Average Engagement by Department", color="#15616d")
        st.pyplot(fig, clear_figure=True)
    with lower_right:
        source_risk = frame.groupby("recruitment_source")["termd"].mean().sort_values(ascending=False).mul(100).reset_index().head(8)
        fig = make_bar_chart(source_risk, x="termd", y="recruitment_source", title="Attrition by Recruitment Source", color="#b06b00")
        st.pyplot(fig, clear_figure=True)


def workforce_tab(frame: pd.DataFrame):
    st.subheader("Workforce Profile")
    st.write("Composition, distribution, and reach across the core demographic and tenure cuts used in the notebooks.")

    top_left, top_right = st.columns(2)
    with top_left:
        fig = make_count_chart(frame, "generation", "Generation Mix", order=GENERATION_ORDER, color="#4f6d7a")
        st.pyplot(fig, clear_figure=True)
    with top_right:
        fig = make_count_chart(frame, "salary_band", "Salary Band Mix", order=SALARY_ORDER, color="#2f6690")
        st.pyplot(fig, clear_figure=True)

    second_left, second_right = st.columns(2)
    with second_left:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(frame["age"], bins=12, kde=True, color="#356f6f", ax=ax)
        ax.set_title("Age Distribution", fontsize=14, weight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Employees")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with second_right:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(frame["tenure_years"], bins=12, kde=True, color="#8c5a3c", ax=ax)
        ax.set_title("Tenure Distribution", fontsize=14, weight="bold")
        ax.set_xlabel("Tenure (years)")
        ax.set_ylabel("Employees")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    third_left, third_right = st.columns(2)
    with third_left:
        gender_df = frame["sex"].value_counts().reset_index()
        gender_df.columns = ["sex", "count"]
        fig = make_bar_chart(gender_df, x="count", y="sex", title="Gender Mix", color="#5663a4")
        st.pyplot(fig, clear_figure=True)
    with third_right:
        state_df = frame["state"].value_counts().head(12).reset_index()
        state_df.columns = ["state", "count"]
        fig = make_bar_chart(state_df, x="count", y="state", title="Top States by Headcount", color="#0e7490")
        st.pyplot(fig, clear_figure=True)


def attrition_tab(frame: pd.DataFrame):
    st.subheader("Attrition & Risk")
    st.write("Focus on where exits concentrate and which factors align with higher churn.")

    left_col, right_col = st.columns(2)
    with left_col:
        attrition_by_salary = frame.groupby("salary_band", observed=True)["termd"].mean().sort_values(ascending=False).mul(100).reset_index()
        fig = make_bar_chart(attrition_by_salary, x="termd", y="salary_band", title="Attrition by Salary Band", color="#9b2226")
        st.pyplot(fig, clear_figure=True)
    with right_col:
        attrition_by_generation = frame.groupby("generation", observed=True)["termd"].mean().sort_values(ascending=False).mul(100).reset_index()
        fig = make_bar_chart(attrition_by_generation, x="termd", y="generation", title="Attrition by Generation", color="#bb3e03")
        st.pyplot(fig, clear_figure=True)

    second_left, second_right = st.columns(2)
    with second_left:
        lateness = frame.groupby("lateness_flag", dropna=False)["termd"].mean().mul(100).reset_index()
        lateness["lateness_flag"] = lateness["lateness_flag"].map({True: "Late", False: "Not late"})
        fig = make_bar_chart(lateness, x="termd", y="lateness_flag", title="Attrition vs Lateness Flag", color="#6d597a")
        st.pyplot(fig, clear_figure=True)
    with second_right:
        high_engagement = frame.groupby("high_engagement_flag", dropna=False)["termd"].mean().mul(100).reset_index()
        high_engagement["high_engagement_flag"] = high_engagement["high_engagement_flag"].map({True: "High engagement", False: "Other"})
        fig = make_bar_chart(high_engagement, x="termd", y="high_engagement_flag", title="Attrition vs High Engagement Flag", color="#1d7874")
        st.pyplot(fig, clear_figure=True)

    manager_table = (
        frame.groupby(["manager_name", "department"], observed=True)
        .agg(headcount=("employee_id", "size"), attrition_rate=("termd", "mean"), avg_engagement=("engagement_survey", "mean"))
        .query("headcount >= 5")
        .sort_values(["attrition_rate", "headcount"], ascending=[False, False])
        .head(10)
        .reset_index()
    )
    manager_table["attrition_rate"] = manager_table["attrition_rate"].mul(100).round(1)
    manager_table["avg_engagement"] = manager_table["avg_engagement"].round(2)
    st.markdown("#### Watchlist managers")
    st.dataframe(manager_table, use_container_width=True, hide_index=True)


def engagement_tab(frame: pd.DataFrame):
    st.subheader("Engagement & Performance")
    st.write("The notebooks highlight engagement gaps, especially in Production and Sales, and the dashboard should keep those visible.")

    left_col, right_col = st.columns(2)
    with left_col:
        engagement_by_dept = frame.groupby("department", observed=True)["engagement_survey"].mean().sort_values(ascending=False).reset_index()
        fig = make_bar_chart(engagement_by_dept, x="engagement_survey", y="department", title="Average Engagement by Department", color="#0b7285")
        st.pyplot(fig, clear_figure=True)
    with right_col:
        satisfaction_by_dept = frame.groupby("department", observed=True)["emp_satisfaction"].mean().sort_values(ascending=False).reset_index()
        fig = make_bar_chart(satisfaction_by_dept, x="emp_satisfaction", y="department", title="Average Satisfaction by Department", color="#7c3f2d")
        st.pyplot(fig, clear_figure=True)

    scatter_col, heatmap_col = st.columns([1.1, 1])
    with scatter_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=frame,
            x="engagement_survey",
            y="perf_score_id",
            hue="employment_status",
            palette={"Active": "#1f77b4", "Voluntarily Terminated": "#d95f02", "Terminated for Cause": "#7570b3"},
            alpha=0.85,
            ax=ax,
        )
        ax.set_title("Engagement vs Performance Index", fontsize=14, weight="bold")
        ax.set_xlabel("Engagement survey")
        ax.set_ylabel("Performance index")
        ax.legend(title="Employment status", loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with heatmap_col:
        fig = make_heatmap(frame, "Numeric Signal Correlations")
        st.pyplot(fig, clear_figure=True)


def dei_tab(frame: pd.DataFrame):
    st.subheader("Diversity & Hiring")
    st.write("A practical diversity lens built from the fields used in the notebooks: gender, race, source, and the diversity-job-fair indicator.")

    left_col, right_col = st.columns(2)
    with left_col:
        sex_by_dept = frame.groupby(["department", "sex"], observed=True).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=sex_by_dept, x="department", y="count", hue="sex", ax=ax)
        ax.set_title("Gender Mix by Department", fontsize=14, weight="bold")
        ax.set_xlabel("Department")
        ax.set_ylabel("Employees")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with right_col:
        race_mix = frame["race_desc"].value_counts().head(8).reset_index()
        race_mix.columns = ["race_desc", "count"]
        fig = make_bar_chart(race_mix, x="count", y="race_desc", title="Top Race/Ethnicity Categories", color="#5f6caf")
        st.pyplot(fig, clear_figure=True)

    source_quality = (
        frame.groupby("recruitment_source")
        .agg(headcount=("employee_id", "size"), attrition_rate=("termd", "mean"), avg_engagement=("engagement_survey", "mean"))
        .sort_values(["attrition_rate", "headcount"], ascending=[False, False])
        .reset_index()
    )
    source_quality["attrition_rate"] = source_quality["attrition_rate"].mul(100).round(1)
    source_quality["avg_engagement"] = source_quality["avg_engagement"].round(2)
    st.markdown("#### Recruitment source quality")
    st.dataframe(source_quality, use_container_width=True, hide_index=True)

    diversity_cols = st.columns(3)
    with diversity_cols[0]:
        summary_card("Diversity job fair share", format_pct(safe_percentage(frame["from_diversity_job_fair_id"])))
    with diversity_cols[1]:
        summary_card("Unique states", f"{frame['state'].nunique()}")
    with diversity_cols[2]:
        summary_card("Unique managers", f"{frame['manager_name'].nunique()}")


def segments_tab(frame: pd.DataFrame):
    st.subheader("Segmentation")
    st.write("The segmentation notebook found three meaningful employee groups. This dashboard operationalizes that idea with a transparent rule-based profile built from the engineered fields.")

    segment_summary = (
        frame.groupby("segment")
        .agg(
            headcount=("employee_id", "size"),
            attrition_rate=("termd", "mean"),
            avg_engagement=("engagement_survey", "mean"),
            avg_perf_index=("perf_score_id", "mean"),
            avg_tenure=("tenure_years", "mean"),
            lateness_rate=("lateness_flag", "mean"),
        )
        .sort_values("headcount", ascending=False)
        .reset_index()
    )
    segment_summary["attrition_rate"] = segment_summary["attrition_rate"].mul(100).round(1)
    segment_summary["lateness_rate"] = segment_summary["lateness_rate"].mul(100).round(1)
    segment_summary["avg_engagement"] = segment_summary["avg_engagement"].round(2)
    segment_summary["avg_perf_index"] = segment_summary["avg_perf_index"].round(2)
    segment_summary["avg_tenure"] = segment_summary["avg_tenure"].round(2)

    left_col, right_col = st.columns([1, 1.1])
    with left_col:
        fig = make_count_chart(frame, "segment", "Segment Mix", order=["At-Risk", "Developing", "Stable Core", "High Performer"], color="#c44536")
        st.pyplot(fig, clear_figure=True)
    with right_col:
        st.dataframe(segment_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Segment profiles")
    detail_cols = st.columns(4)
    profiles = {
        "At-Risk": "Low engagement, recent performance concern, or termination history. Use for targeted intervention and manager review.",
        "Developing": "Employees with uneven signals who need coaching, mobility planning, or better role-fit support.",
        "Stable Core": "The main operating base: steady engagement and acceptable performance, but not yet a top-tier retention signal.",
        "High Performer": "Strong engagement and strong performance, with low lateness. Protect with recognition and growth pathways.",
    }
    for column, (segment_name, description) in zip(detail_cols, profiles.items()):
        with column:
            st.markdown(f"**{segment_name}**")
            st.write(description)


def executive_highlights(frame: pd.DataFrame):
    insights = []
    if not frame.empty:
        dept_attrition = frame.groupby("department", observed=True)["termd"].mean().sort_values(ascending=False)
        if not dept_attrition.empty:
            insights.append(f"Highest attrition department: {dept_attrition.index[0]} ({dept_attrition.iloc[0] * 100:.1f}%).")
        dept_engagement = frame.groupby("department", observed=True)["engagement_survey"].mean().sort_values()
        if not dept_engagement.empty:
            insights.append(f"Lowest engagement department: {dept_engagement.index[0]} ({dept_engagement.iloc[0]:.2f}/5).")
        source_risk = frame.groupby("recruitment_source")["termd"].mean().sort_values(ascending=False)
        if not source_risk.empty:
            insights.append(f"Highest-risk recruiting source: {source_risk.index[0]} ({source_risk.iloc[0] * 100:.1f}% attrition).")
    if insights:
        st.info("\n".join(insights))


def main():
    sns.set_theme(style="whitegrid", context="talk")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7fb 0%, #ffffff 25%, #f8fafc 100%);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
            color: white;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: white;
            margin: 0;
            font-size: 2.1rem;
        }
        .hero p {
            color: rgba(255, 255, 255, 0.88);
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>HR Intelligence Dashboard</h1>
            <p>Built from the engineered HR dataset and aligned to the notebook workflow across EDA, engineering, statistics, modeling, and segmentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()

    frame = load_data()
    filtered = apply_sidebar_filters(frame)

    if filtered.empty:
        st.warning("No rows match the current filters. Adjust the sidebar selections and try again.")
        return

    executive_highlights(filtered)

    tabs = st.tabs(["Overview", "Workforce", "Attrition & Risk", "Engagement & Performance", "Diversity & Hiring", "Segmentation"])

    with tabs[0]:
        overview_tab(filtered)
    with tabs[1]:
        workforce_tab(filtered)
    with tabs[2]:
        attrition_tab(filtered)
    with tabs[3]:
        engagement_tab(filtered)
    with tabs[4]:
        dei_tab(filtered)
    with tabs[5]:
        segments_tab(filtered)


if __name__ == "__main__":
    main()