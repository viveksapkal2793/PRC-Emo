import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TRAIN_CSV = "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/train_sent_emo.csv"
DEFAULT_DEV_CSV = "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/dev_sent_emo.csv"
DEFAULT_TEST_CSV = "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/test_sent_emo.csv"

TARGET_COLUMNS = [
    "Emotion",
    "Sentiment",
    "Speaker",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MELD metadata distributions and categorical associations."
    )
    parser.add_argument("--train_csv", type=str, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--dev_csv", type=str, default=DEFAULT_DEV_CSV)
    parser.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis/meld_dataset_statistics",
        help="Directory where CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="How many groups to show in top-K distribution tables.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation and only write tabular outputs.",
    )
    return parser.parse_args()


def load_split(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["split"] = split
    df["Dialogue_ID"] = df["Dialogue_ID"].astype(str)
    df["Utterance_ID"] = df["Utterance_ID"].astype(str)
    df["Season"] = df["Season"].astype(str)
    df["Episode"] = df["Episode"].astype(str)
    df["Season_Episode"] = df["Season"] + "_E" + df["Episode"]
    df["Speaker"] = df["Speaker"].fillna("UNKNOWN")
    df["Emotion"] = df["Emotion"].fillna("UNKNOWN")
    df["Sentiment"] = df["Sentiment"].fillna("UNKNOWN")
    return df


def ensure_output_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_plot_dir(output_dir: str) -> str:
    plot_dir = os.path.join(output_dir, "plots")
    ensure_output_dir(plot_dir)
    return plot_dir


def normalized_distribution(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    table = pd.crosstab(df[row_col], df[col_col], normalize="index") * 100.0
    return table.round(2)


def count_distribution(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    return pd.crosstab(df[row_col], df[col_col])


def cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    contingency = pd.crosstab(df[col_a], df[col_b])
    if contingency.empty:
        return 0.0

    observed = contingency.to_numpy(dtype=np.float64)
    n = observed.sum()
    if n == 0:
        return 0.0

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    mask = expected > 0
    chi2 = np.sum(((observed - expected) ** 2 / expected)[mask])

    r, k = observed.shape
    if r <= 1 or k <= 1:
        return 0.0

    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1.0, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1.0, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1.0, n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def grouped_purity(df: pd.DataFrame, feature_col: str, target_col: str) -> dict:
    counts = df.groupby([feature_col, target_col]).size().reset_index(name="count")
    totals = counts.groupby(feature_col)["count"].sum().rename("group_total")
    dominant = counts.groupby(feature_col)["count"].max().rename("dominant_count")
    merged = pd.concat([totals, dominant], axis=1).reset_index()
    merged["purity"] = merged["dominant_count"] / merged["group_total"]
    weighted_purity = float(
        (merged["purity"] * merged["group_total"]).sum() / max(1, merged["group_total"].sum())
    )
    return {
        "groups": int(len(merged)),
        "mean_group_size": float(merged["group_total"].mean()),
        "median_group_size": float(merged["group_total"].median()),
        "weighted_purity": weighted_purity,
    }


def per_group_top_distribution(
    df: pd.DataFrame, group_col: str, target_col: str, top_k: int
) -> pd.DataFrame:
    top_groups = df[group_col].value_counts().head(top_k).index
    subset = df[df[group_col].isin(top_groups)].copy()
    counts = count_distribution(subset, group_col, target_col)
    proportions = normalized_distribution(subset, group_col, target_col)
    counts.columns = [f"{c}_count" for c in counts.columns]
    proportions.columns = [f"{c}_pct" for c in proportions.columns]
    totals = subset[group_col].value_counts().rename("total_count")
    merged = pd.concat([totals, counts, proportions], axis=1).reset_index()
    merged = merged.rename(columns={"index": group_col})
    return merged.sort_values("total_count", ascending=False)


def emotion_sentiment_alignment(df: pd.DataFrame) -> pd.DataFrame:
    counts = count_distribution(df, "Emotion", "Sentiment")
    props = normalized_distribution(df, "Emotion", "Sentiment")
    counts.columns = [f"{c}_count" for c in counts.columns]
    props.columns = [f"{c}_pct" for c in props.columns]
    totals = df["Emotion"].value_counts().rename("total_count")
    result = pd.concat([totals, counts, props], axis=1).reset_index()
    result = result.rename(columns={"index": "Emotion"})
    return result.sort_values("total_count", ascending=False)


def pairwise_cramers_v(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for left in columns:
        row = {"column": left}
        for right in columns:
            row[right] = round(cramers_v(df, left, right), 4) if left != right else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def basic_summary(df: pd.DataFrame) -> dict:
    return {
        "num_rows": int(len(df)),
        "num_dialogues": int(df["Dialogue_ID"].nunique()),
        "num_speakers": int(df["Speaker"].nunique()),
        "num_seasons": int(df["Season"].nunique()),
        "num_episodes": int(df["Season_Episode"].nunique()),
        "emotion_distribution": df["Emotion"].value_counts().to_dict(),
        "sentiment_distribution": df["Sentiment"].value_counts().to_dict(),
    }


def build_association_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["Sentiment", "Speaker", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "Season_Episode"]
    rows = []
    for feature_col in feature_cols:
        purity = grouped_purity(df, feature_col, "Emotion")
        rows.append(
            {
                "feature": feature_col,
                "num_unique_values": int(df[feature_col].nunique()),
                "cramers_v_with_emotion": round(cramers_v(df, feature_col, "Emotion"), 4),
                "cramers_v_with_sentiment": round(cramers_v(df, feature_col, "Sentiment"), 4),
                "mean_group_size": round(purity["mean_group_size"], 2),
                "median_group_size": round(purity["median_group_size"], 2),
                "weighted_dominant_emotion_purity": round(purity["weighted_purity"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v_with_emotion", ascending=False)


def console_print_section(title: str):
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_figure(fig: plt.Figure, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_split_emotion_distribution(split_frames: dict[str, pd.DataFrame], plot_dir: str):
    rows = []
    for split, df in split_frames.items():
        counts = df["Emotion"].value_counts()
        for emotion, count in counts.items():
            rows.append({"split": split, "Emotion": emotion, "count": count})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Emotion", y="count", hue="split", ax=ax)
    ax.set_title("Emotion Distribution by Split")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, os.path.join(plot_dir, "emotion_distribution_by_split.png"))


def plot_split_sentiment_distribution(split_frames: dict[str, pd.DataFrame], plot_dir: str):
    rows = []
    for split, df in split_frames.items():
        counts = df["Sentiment"].value_counts()
        for sentiment, count in counts.items():
            rows.append({"split": split, "Sentiment": sentiment, "count": count})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Sentiment", y="count", hue="split", ax=ax)
    ax.set_title("Sentiment Distribution by Split")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    save_figure(fig, os.path.join(plot_dir, "sentiment_distribution_by_split.png"))


def plot_emotion_sentiment_heatmap(df: pd.DataFrame, plot_dir: str, prefix: str):
    heatmap_df = pd.crosstab(df["Emotion"], df["Sentiment"], normalize="index") * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_df.round(1), annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Emotion vs Sentiment ({prefix})")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Emotion")
    save_figure(fig, os.path.join(plot_dir, f"{prefix}_emotion_vs_sentiment_heatmap.png"))


def plot_association_summary(association_df: pd.DataFrame, plot_dir: str):
    melted = association_df.melt(
        id_vars="feature",
        value_vars=["cramers_v_with_emotion", "cramers_v_with_sentiment"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="feature", y="value", hue="metric", ax=ax)
    ax.set_title("Categorical Association Strength")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cramer's V")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, os.path.join(plot_dir, "association_strength_barplot.png"))


def plot_pairwise_cramers_v(pairwise_df: pd.DataFrame, plot_dir: str):
    heatmap_df = pairwise_df.set_index("column")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Pairwise Cramer's V")
    ax.set_xlabel("Column")
    ax.set_ylabel("Column")
    save_figure(fig, os.path.join(plot_dir, "pairwise_cramers_v_heatmap.png"))


def plot_top_group_distribution(df: pd.DataFrame, group_col: str, target_col: str, top_k: int, plot_dir: str, filename: str):
    top_groups = df[group_col].value_counts().head(top_k).index
    subset = df[df[group_col].isin(top_groups)].copy()
    counts = pd.crosstab(subset[group_col], subset[target_col])
    counts = counts.loc[top_groups]
    proportions = counts.div(counts.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(14, 7))
    proportions.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"{target_col} Mix Across Top {top_k} {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Percentage")
    ax.legend(title=target_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, os.path.join(plot_dir, filename))


def plot_dialogue_length_distribution(dialogue_df: pd.DataFrame, plot_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(dialogue_df["total_utterances"], bins=30, kde=True, ax=ax)
    ax.set_title("Dialogue Length Distribution")
    ax.set_xlabel("Number of Utterances in Dialogue")
    ax.set_ylabel("Count")
    save_figure(fig, os.path.join(plot_dir, "dialogue_length_distribution.png"))


def generate_plots(
    split_frames: dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    association_all: pd.DataFrame,
    pairwise_all: pd.DataFrame,
    dialogue_size: pd.DataFrame,
    top_k: int,
    output_dir: str,
):
    plot_dir = make_plot_dir(output_dir)
    plot_split_emotion_distribution(split_frames, plot_dir)
    plot_split_sentiment_distribution(split_frames, plot_dir)
    plot_emotion_sentiment_heatmap(full_df, plot_dir, "all")
    for split, df in split_frames.items():
        plot_emotion_sentiment_heatmap(df, plot_dir, split)
    plot_association_summary(association_all, plot_dir)
    plot_pairwise_cramers_v(pairwise_all, plot_dir)
    plot_top_group_distribution(full_df, "Speaker", "Emotion", top_k, plot_dir, "top_speakers_emotion_mix.png")
    plot_top_group_distribution(full_df, "Season_Episode", "Emotion", top_k, plot_dir, "top_season_episode_emotion_mix.png")
    plot_top_group_distribution(full_df, "Utterance_ID", "Emotion", top_k, plot_dir, "top_utterance_id_emotion_mix.png")
    plot_dialogue_length_distribution(dialogue_size, plot_dir)


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    split_to_path = {
        "train": args.train_csv,
        "dev": args.dev_csv,
        "test": args.test_csv,
    }
    split_frames = {split: load_split(path, split) for split, path in split_to_path.items()}
    full_df = pd.concat(split_frames.values(), ignore_index=True)

    summary = {
        "train": basic_summary(split_frames["train"]),
        "dev": basic_summary(split_frames["dev"]),
        "test": basic_summary(split_frames["test"]),
        "all": basic_summary(full_df),
    }

    association_all = build_association_summary(full_df)
    pairwise_all = pairwise_cramers_v(
        full_df,
        ["Emotion", "Sentiment", "Speaker", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "Season_Episode"],
    )
    emotion_vs_sentiment_all = emotion_sentiment_alignment(full_df)
    speaker_emotion_top = per_group_top_distribution(full_df, "Speaker", "Emotion", args.top_k)
    season_emotion_top = per_group_top_distribution(full_df, "Season", "Emotion", args.top_k)
    episode_emotion_top = per_group_top_distribution(full_df, "Season_Episode", "Emotion", args.top_k)
    utterance_emotion_top = per_group_top_distribution(full_df, "Utterance_ID", "Emotion", args.top_k)
    dialogue_size = (
        full_df.groupby("Dialogue_ID")
        .agg(
            total_utterances=("Dialogue_ID", "size"),
            unique_speakers=("Speaker", "nunique"),
            dominant_emotion=("Emotion", lambda s: s.value_counts().index[0]),
            dominant_emotion_share=("Emotion", lambda s: float(s.value_counts(normalize=True).iloc[0])),
            dominant_sentiment=("Sentiment", lambda s: s.value_counts().index[0]),
        )
        .reset_index()
        .sort_values("total_utterances", ascending=False)
    )

    for split, df in split_frames.items():
        split_prefix = os.path.join(args.output_dir, f"{split}")
        write_dataframe(
            emotion_sentiment_alignment(df),
            f"{split_prefix}_emotion_vs_sentiment.csv",
        )
        write_dataframe(
            build_association_summary(df),
            f"{split_prefix}_association_summary.csv",
        )
        write_dataframe(
            pairwise_cramers_v(
                df,
                ["Emotion", "Sentiment", "Speaker", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "Season_Episode"],
            ),
            f"{split_prefix}_pairwise_cramers_v.csv",
        )

    write_dataframe(association_all, os.path.join(args.output_dir, "all_association_summary.csv"))
    write_dataframe(pairwise_all, os.path.join(args.output_dir, "all_pairwise_cramers_v.csv"))
    write_dataframe(emotion_vs_sentiment_all, os.path.join(args.output_dir, "all_emotion_vs_sentiment.csv"))
    write_dataframe(speaker_emotion_top, os.path.join(args.output_dir, "top_speakers_emotion_distribution.csv"))
    write_dataframe(season_emotion_top, os.path.join(args.output_dir, "top_seasons_emotion_distribution.csv"))
    write_dataframe(episode_emotion_top, os.path.join(args.output_dir, "top_season_episode_emotion_distribution.csv"))
    write_dataframe(utterance_emotion_top, os.path.join(args.output_dir, "top_utterance_id_emotion_distribution.csv"))
    write_dataframe(dialogue_size, os.path.join(args.output_dir, "dialogue_level_summary.csv"))

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.no_plots:
        generate_plots(
            split_frames=split_frames,
            full_df=full_df,
            association_all=association_all,
            pairwise_all=pairwise_all,
            dialogue_size=dialogue_size,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )

    console_print_section("MELD Split Summary")
    for split_name, split_summary in summary.items():
        print(
            f"{split_name}: rows={split_summary['num_rows']}, dialogues={split_summary['num_dialogues']}, "
            f"speakers={split_summary['num_speakers']}, seasons={split_summary['num_seasons']}, "
            f"season_episodes={split_summary['num_episodes']}"
        )
        print(f"  emotions={split_summary['emotion_distribution']}")
        print(f"  sentiments={split_summary['sentiment_distribution']}")

    console_print_section("Association Summary")
    print(association_all.to_string(index=False))

    console_print_section("Emotion vs Sentiment")
    print(emotion_vs_sentiment_all.to_string(index=False))

    console_print_section("Top Speaker Emotion Mix")
    print(speaker_emotion_top.to_string(index=False))

    console_print_section("Top Season-Episode Emotion Mix")
    print(episode_emotion_top.to_string(index=False))

    print(f"\nSaved detailed outputs to: {args.output_dir}")
    if not args.no_plots:
        print(f"Saved plots to: {os.path.join(args.output_dir, 'plots')}")


if __name__ == "__main__":
    main()
