import pandas as pd
import altair as alt


def load_data(path):
    """
    Load the processed heart disease dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)


def compute_summary_statistics(df):
    """
    Compute summary statistics for the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Summary statistics including numerical and categorical features.
    """
    return df.describe(include="all")


def plot_target_distribution(df):
    """
    Generate a bar chart showing counts of heart disease vs no heart disease.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing a 'target' column.

    Returns
    -------
    alt.Chart
        Bar chart with labels.
    """
    counts = df.groupby("target").size().reset_index(name="count")

    bar = (
        alt.Chart(counts)
        .mark_bar(stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("target:N", title="Heart Disease"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("target:N", title="Heart Disease"),
            tooltip=[alt.Tooltip("count:Q", title="Count")],
        )
        .properties(title="Cases of Heart Disease", width=300, height=300)
    )

    bar_labels = bar.mark_text(dy=-5, size=14).encode(text="count:Q")

    return bar + bar_labels


def plot_numerical_distributions(df, num_cols):
    """
    Generate histograms for numerical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the numerical features.

    num_cols : list of str
        List of numerical column names.

    Returns
    -------
    alt.VConcatChart
        Faceted vertical concatenation of histograms.
    """
    charts = []

    for col in num_cols:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count()", title="Count"),
                tooltip=[
                    alt.Tooltip(f"{col}:Q", title=col),
                    alt.Tooltip("count()", title="Count"),
                ],
            )
            .properties(title=f"Distribution of {col}", width=300, height=250)
        )
        charts.append(chart)

    rows = []
    for i in range(0, len(charts), 2):
        rows.append(alt.hconcat(*charts[i : i + 2]))

    return alt.vconcat(*rows).configure_legend(orient="top")


def plot_boxplots(df, num_cols):
    """
    Create boxplots showing numerical features vs heart disease target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing target and numerical features.

    num_cols : list of str
        List of numerical column names.

    Returns
    -------
    alt.VConcatChart
        Boxplots arranged as vertical concatenation.
    """
    charts = []

    for col in num_cols:
        chart = (
            alt.Chart(df)
            .mark_boxplot(size=20)
            .encode(
                x=alt.X(f"{col}:Q", title=col),
                y=alt.Y("target:N", title="Heart Disease"),
                color=alt.Color("target:N", title="Heart Disease"),
            )
            .properties(title=f"{col} vs Heart Disease", width=300, height=250)
        )
        charts.append(chart)

    rows = []
    for i in range(0, len(charts), 2):
        rows.append(alt.hconcat(*charts[i : i + 2]))

    return alt.vconcat(*rows).configure_legend(orient="top")


def plot_categorical_vs_target(df, axis_titles):
    """
    Create bar charts showing categorical features vs heart disease target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing categorical features.

    axis_titles : dict
        Mapping of column names to readable axis labels.

    Returns
    -------
    alt.VConcatChart
        Vertical stack of bar charts.
    """
    charts = []

    for col, title in axis_titles.items():
        chart = (
            alt.Chart(df)
            .mark_bar(size=30)
            .encode(
                x=alt.X(
                    f"{col}:N",
                    title=title,
                    scale=alt.Scale(paddingInner=0.5, paddingOuter=0.5)  
                ),
                xOffset="target:N",
                y=alt.Y("count()", title="Count"),
                color=alt.Color("target:N", title="Heart Disease"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(title=f"{col} vs Heart Disease", width=300, height=250)
        )
        charts.append(chart)

    rows = []
    for i in range(0, len(charts), 2):
        rows.append(alt.hconcat(*charts[i : i + 2]))

    return alt.vconcat(*rows).configure_legend(orient="top")


def plot_correlation_heatmap(df, num_cols, cat_cols, target_col='target'):
    """
    Generate a correlation heatmap for all numerical and categorical features with the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numerical and categorical features along with the target column.
    num_cols : list of str
        List of numerical feature column names.
    cat_cols : list of str
        List of categorical feature column names.
    target_col : str, optional
        Name of the target column. Default is 'target'.

    Returns
    -------
    alt.Chart
        Correlation heatmap chart with correlation values as text.
    """
    df_corr = df.copy()
    if df_corr[target_col].dtype == 'O' or df_corr[target_col].dtype.name == 'category':
        df_corr[target_col] = df_corr[target_col].replace({
            'Heart Disease': 1,
            'No Heart Disease': 0
        })
    
    corr_matrix = df_corr[num_cols + cat_cols + [target_col]].corr()

    corr_long = corr_matrix.reset_index().melt(id_vars='index')
    corr_long.columns = ['feature_x', 'feature_y', 'correlation']
    
    base = alt.Chart(corr_long).encode(
        x=alt.X('feature_x:N', title='Feature'),
        y=alt.Y('feature_y:N', title='Feature')
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
        tooltip=['feature_x', 'feature_y', 'correlation']
    )
    text = base.mark_text(
        fontSize=12,
        color='black'
    ).encode(
        text=alt.Text('correlation:Q', format='.2f')
    )

    final_chart = (heatmap + text).properties(
        title='Correlation Heatmap of All Features with Target',
        width=600,
        height=600
    )

    return final_chart